"""
opportunities_backtest.py
=========================
Historical version of `opportunities.py`. Replays every prob >= PROB_MIN
signal in the last HISTORICAL_YEARS years, fully realizes T+1..T+5 for
each, and writes the same per-signal table the live tracker emits — plus
hit-rate / expected-return aggregates per (regime, prob_bucket).

For each historical (symbol, signal_date) we compute (per `_evolve_one_signal`):
  - daily OC% return on each of T+1..T+5
  - on each day: did any 5-min bar close above the static T+1 ORB-15 high?
                 did the last bar of that day also close above? (held EOD)
  - intraday MAE/MFE per day
  - naive entry (T+1 open) cum return + MAE/MFE
  - ORB-held entry (first day where held-EOD confirms) cum return + MAE/MFE

Aggregations per (regime, prob_bucket):
  n, broke_t1_orh_t1_pct, broke_t1_orh_anyday_pct, held_t1_orh_anyday_pct,
  mean_naive_ret, mean_naive_ret_net, mean_orb_ret_taken, mean_orb_ret_net_taken,
  hit_sl3_pct (naive), hit_tp3_pct (naive), hit_tp5_pct (naive),
  mean_naive_mae, mean_naive_mfe, win_rate_naive_pct,
  hit_sl3_pct_orb_taken, hit_tp3_pct_orb_taken, hit_tp5_pct_orb_taken,
  selection_benefit_per_trade  (naive ret on held-anyday subset minus
                                 naive ret on NOT-held-anyday subset)

OUTPUT
------
<BASE_DIR>/opportunities/
    backtest_long_<YYYYMMDD>.parquet     (one row per signal x day_offset)
    backtest_wide_<YYYYMMDD>.parquet     (one row per signal)
    backtest_by_bucket_<YYYYMMDD>.csv
    backtest_summary_<YYYYMMDD>.xlsx     (Summary, Wide, By_Bucket, Long_head)

USAGE
-----
    python opportunities_backtest.py
    python opportunities_backtest.py --historical-years 3 --prob-min 0.70
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import watchlist_followup as WF
import hybrid_entry as HE
from opportunities import (
    _evolve_one_signal,
    OUT_DIR,
    PROB_MIN,
    MAX_HORIZON_DAYS,
    SL_PCT,
    TP_PCTS,
)

warnings.filterwarnings("ignore")

HISTORICAL_YEARS = 3.0
TOTAL_COST_PCT   = WF.TOTAL_COST_PCT


# ============================================================================
# Build the historical per-signal cohort
# ============================================================================

def build_backtest(prob_min: float, historical_years: float,
                   max_horizon: int = MAX_HORIZON_DAYS
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (long_df, wide_df). Wide = one row per signal."""
    print(f"[bt] scoring panel; window={historical_years}y, prob_min={prob_min}")
    panel = WF.score_recent(prob_min, lookback_days=10,
                            historical_years=historical_years)
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame()

    sigs = panel[panel["prob"] >= prob_min].copy()
    if sigs.empty:
        return pd.DataFrame(), pd.DataFrame()

    sigs = sigs.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    print(f"[bt] {len(sigs):,} signals to simulate")

    intra_cache: Dict[str, Optional[pd.DataFrame]] = {}
    long_rows: List[Dict] = []
    wide_rows: List[Dict] = []
    skipped = 0

    n = len(sigs)
    for i, sig in enumerate(sigs.itertuples(index=False), start=1):
        if i % 1000 == 0:
            print(f"  {i:,}/{n:,}")
        sym = sig.symbol
        if sym not in intra_cache:
            intra_cache[sym] = WF._load_intraday(sym)
        intra = intra_cache[sym]
        if intra is None or intra.empty:
            skipped += 1
            continue

        evo = _evolve_one_signal(intra, pd.Timestamp(sig.timestamp),
                                 max_horizon=max_horizon)
        if evo is None:
            skipped += 1
            continue
        # Backtest requires the full horizon to be realized
        if evo["summary"]["n_days_realized"] < max_horizon:
            skipped += 1
            continue

        bucket, rule = HE.route_prob(float(sig.prob))

        for r in evo["long_rows"]:
            long_rows.append({
                "symbol":         sym,
                "signal_date":    pd.Timestamp(sig.timestamp),
                "signal_prob":    float(sig.prob),
                "signal_regime":  sig.stock_regime,
                "prob_bucket":    bucket,
                "rule":           rule,
                **r,
            })

        wide = {
            "symbol":              sym,
            "signal_date":         pd.Timestamp(sig.timestamp),
            "signal_prob":         float(sig.prob),
            "signal_regime":       sig.stock_regime,
            "signal_close":        float(sig.close),
            "prob_bucket":         bucket,
            "rule":                rule,
            **evo["summary"],
        }
        # Per-day flat columns
        for r in evo["long_rows"]:
            d = r["day_offset"]
            wide[f"d{d}_ret_oc_pct"]      = r["ret_oc_pct"]
            wide[f"d{d}_broke_t1_orh"]    = r["broke_t1_orh_today"]
            wide[f"d{d}_held_t1_orh_eod"] = r["held_t1_orh_eod"]
        wide_rows.append(wide)

    if skipped:
        print(f"[bt] skipped {skipped:,} signals (no data or short window)")

    return pd.DataFrame(long_rows), pd.DataFrame(wide_rows)


# ============================================================================
# Aggregations
# ============================================================================

def _bucket_aggregate(wide: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """Per (regime, prob_bucket) hit rates and expected returns."""
    if wide.empty:
        return pd.DataFrame()

    # broke-on-T+1 flag has to come from the long table
    if not long.empty:
        t1 = (long[long["day_offset"] == 1]
                .groupby(["symbol", "signal_date"])["broke_t1_orh_today"]
                .first().rename("broke_t1_orh_t1"))
        wide = wide.merge(t1.reset_index(), on=["symbol", "signal_date"], how="left")
    else:
        wide["broke_t1_orh_t1"] = False

    rows = []
    for (reg, bk), g in wide.groupby(["signal_regime", "prob_bucket"]):
        n = len(g)
        n_held = int(g["held_t1_orh_anyday"].sum())
        held = g[g["held_t1_orh_anyday"] == True]              # noqa: E712
        not_held = g[g["held_t1_orh_anyday"] == False]         # noqa: E712

        naive_mae = g["naive_mae_pct"]
        naive_mfe = g["naive_mfe_pct"]
        naive_ret = g["naive_cum_ret_pct"]

        # Per-trade returns of the ORB-held entry on the triggered subset
        orb_taken = g[g["orb_triggered"] == True]              # noqa: E712

        row = {
            "regime":       reg,
            "prob_bucket":  bk,
            "rule":         g["rule"].iloc[0],
            "n":            n,
            "broke_t1_t1_pct":     round(g["broke_t1_orh_t1"].mean() * 100, 2),
            "broke_t1_anyday_pct": round(g["broke_t1_orh_anyday"].mean() * 100, 2),
            "held_t1_anyday_pct":  round(g["held_t1_orh_anyday"].mean() * 100, 2),

            "mean_naive_ret":      round(naive_ret.mean(), 3),
            "mean_naive_ret_net":  round(naive_ret.mean() - TOTAL_COST_PCT, 3),
            "median_naive_ret":    round(naive_ret.median(), 3),
            "win_rate_naive_pct":  round((naive_ret > 0).mean() * 100, 2),
            "mean_naive_mae":      round(naive_mae.mean(), 3),
            "mean_naive_mfe":      round(naive_mfe.mean(), 3),
            "naive_hit_sl3_pct":   round((naive_mae <= SL_PCT).mean() * 100, 2),
        }
        for tp in TP_PCTS:
            row[f"naive_hit_tp{int(tp)}_pct"] = round((naive_mfe >= tp).mean() * 100, 2)

        if len(orb_taken):
            orb_ret = orb_taken["orb_cum_ret_pct"]
            orb_mae = orb_taken["orb_mae_pct"]
            orb_mfe = orb_taken["orb_mfe_pct"]
            row.update({
                "n_orb_taken":          len(orb_taken),
                "mean_orb_ret":         round(orb_ret.mean(), 3),
                "mean_orb_ret_net":     round(orb_ret.mean() - TOTAL_COST_PCT, 3),
                "win_rate_orb_pct":     round((orb_ret > 0).mean() * 100, 2),
                "orb_taken_hit_sl3_pct": round((orb_mae <= SL_PCT).mean() * 100, 2),
            })
            for tp in TP_PCTS:
                row[f"orb_taken_hit_tp{int(tp)}_pct"] = round((orb_mfe >= tp).mean() * 100, 2)
        else:
            row.update({"n_orb_taken": 0,
                        "mean_orb_ret": float("nan"),
                        "mean_orb_ret_net": float("nan"),
                        "win_rate_orb_pct": float("nan")})

        # The selection-benefit diagnostic that drove the [0.70,0.75) finding
        if len(held) and len(not_held):
            row["selection_benefit_pct"] = round(
                held["naive_cum_ret_pct"].mean() -
                not_held["naive_cum_ret_pct"].mean(), 3)
        else:
            row["selection_benefit_pct"] = float("nan")

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["regime", "prob_bucket"]).reset_index(drop=True)
    return out


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob-min", type=float, default=PROB_MIN)
    ap.add_argument("--historical-years", type=float, default=HISTORICAL_YEARS)
    ap.add_argument("--max-horizon", type=int, default=MAX_HORIZON_DAYS)
    args = ap.parse_args()

    long_df, wide_df = build_backtest(
        prob_min=args.prob_min,
        historical_years=args.historical_years,
        max_horizon=args.max_horizon,
    )

    if wide_df.empty:
        print("[bt] no signals materialized; nothing to write")
        return

    by_bucket = _bucket_aggregate(wide_df, long_df)

    stamp = pd.Timestamp(wide_df["signal_date"].max()).strftime("%Y%m%d")
    long_pq  = OUT_DIR / f"backtest_long_{stamp}.parquet"
    wide_pq  = OUT_DIR / f"backtest_wide_{stamp}.parquet"
    bk_csv   = OUT_DIR / f"backtest_by_bucket_{stamp}.csv"
    xlsx     = OUT_DIR / f"backtest_summary_{stamp}.xlsx"

    long_df.to_parquet(long_pq, index=False)
    print(f"[out] wrote {long_pq}  ({len(long_df):,} rows)")
    wide_df.to_parquet(wide_pq, index=False)
    print(f"[out] wrote {wide_pq}  ({len(wide_df):,} rows)")
    by_bucket.to_csv(bk_csv, index=False)
    print(f"[out] wrote {bk_csv}")

    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
            by_bucket.to_excel(xw, sheet_name="By_Bucket", index=False)
            wide_df.head(50_000).to_excel(xw, sheet_name="Wide", index=False)
            long_df.head(50_000).to_excel(xw, sheet_name="Long_head", index=False)
        print(f"[out] wrote {xlsx}")
    except Exception as e:
        print(f"[warn] could not write xlsx: {e}")

    print("\n=== Backtest by regime x prob bucket ===")
    show_cols = [c for c in [
        "regime", "prob_bucket", "rule", "n",
        "broke_t1_t1_pct", "broke_t1_anyday_pct", "held_t1_anyday_pct",
        "mean_naive_ret_net", "win_rate_naive_pct",
        "naive_hit_sl3_pct", "naive_hit_tp3_pct",
        "n_orb_taken", "mean_orb_ret_net", "win_rate_orb_pct",
        "selection_benefit_pct",
    ] if c in by_bucket.columns]
    print(by_bucket[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
