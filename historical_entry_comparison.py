"""
historical_entry_comparison.py
==============================
ONE-OFF BACKTEST.  Run this once to settle the entry-execution debate
(open vs ORB) and produce a static lookup table that the daily
`watchlist_followup.py` will join on every morning.

For every prob >= 0.70 signal in the past `--years` years, this script
fetches the intraday 5-min cache and computes per-signal forward stats:
    * entry_at_open      = T+1 first 5-min bar's open
    * entry_at_orb       = first close > T+1 15-min ORB high (or NaN)
    * MAE / MFE over the next HOLD_DAYS trading days, from each entry
    * first-touch hit flags for TP={2, 3, 4, 5, 7, 10}%
                              SL={-2, -3, -5}%
    * OCO outcomes for every (TP, SL) pair
    * realized return at T+5 close

Aggregates per (regime x prob_bucket x entry_method).

OUTPUT
------
<BASE_DIR>/watchlist_followup/
    aggregate_entry_comparison.csv     <-- the lookup table
    per_signal_historical.parquet      <-- full per-signal detail
    historical_summary.xlsx

USAGE
-----
    python historical_entry_comparison.py
    python historical_entry_comparison.py --years 3 --prob-min 0.70
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR      = Path(r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full")
PANEL_PATH    = BASE_DIR / "panel_cache.parquet"
FEATURES_PATH = BASE_DIR / "features_train.json"
ROUTER_PATH   = BASE_DIR / "models" / "m5_regime_router.joblib"
INTRADAY_DIR  = Path(r"C:\Users\karanvsi\Desktop\Pycharm\Cache\intraday_5min")

OUT_DIR       = BASE_DIR / "watchlist_followup"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IST = "Asia/Kolkata"

PROB_MIN          = 0.70
HOLD_DAYS         = 5
HISTORICAL_YEARS  = 2
SIGNAL_REGIMES    = ["bull_trend", "bear_trend"]

ORB_END_TIME      = "09:30"
ORB_BARS_NEEDED   = 3

COST_BPS_RT       = 25
SLIP_BPS_PER_SIDE = 5
TOTAL_COST_PCT    = (COST_BPS_RT + 2 * SLIP_BPS_PER_SIDE) / 100.0

TP_LEVELS_PCT = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
SL_LEVELS_PCT = [-2.0, -3.0, -5.0]


# ============================================================================
# Intraday helpers
# ============================================================================

def _load_intraday(symbol: str) -> Optional[pd.DataFrame]:
    fp = INTRADAY_DIR / f"{symbol}.parquet"
    if not fp.exists():
        return None
    try:
        df = pd.read_parquet(fp)
    except Exception:
        return None
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(IST)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(IST)
    return df.sort_values("timestamp").reset_index(drop=True)


def _market_hours(bars: pd.DataFrame) -> pd.DataFrame:
    ts = bars["timestamp"]
    return bars[((ts.dt.hour > 9) | ((ts.dt.hour == 9) & (ts.dt.minute >= 15))) &
                ((ts.dt.hour < 15) | ((ts.dt.hour == 15) & (ts.dt.minute <= 30)))]


def _orb_cutoff(day: pd.Timestamp) -> pd.Timestamp:
    h, m = map(int, ORB_END_TIME.split(":"))
    return day.normalize().replace(hour=h, minute=m)


# ============================================================================
# Score the historical panel
# ============================================================================

def score_history(prob_min: float, years: float) -> pd.DataFrame:
    from joblib import load

    print(f"[score] reading panel: {PANEL_PATH}")
    panel = pd.read_parquet(PANEL_PATH)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    if panel["timestamp"].dt.tz is None:
        panel["timestamp"] = panel["timestamp"].dt.tz_localize(IST)
    if "stock_regime" not in panel.columns:
        raise SystemExit("FATAL: panel missing stock_regime")
    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    panel["avg20_vol"] = (panel.groupby("symbol")["volume"]
                                .transform(lambda s: s.rolling(20, min_periods=1).mean()))
    panel = panel[(panel["close"] >= 2.0) & (panel["avg20_vol"] >= 200_000)]
    panel = panel[panel["stock_regime"].isin(SIGNAL_REGIMES)].reset_index(drop=True)

    latest = panel["timestamp"].max()
    cutoff = latest - pd.Timedelta(days=int(years * 365))
    panel = panel[panel["timestamp"] >= cutoff].copy()

    schema = json.loads(FEATURES_PATH.read_text())
    FEATURES = schema["features"]
    IMPUTE = {k: float(v) for k, v in schema["impute"].items()}

    print(f"[score] scoring {len(panel):,} rows over the last {years} years...")
    router = load(ROUTER_PATH)
    X = panel.reindex(columns=FEATURES).copy()
    for c in FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(IMPUTE.get(c, 0.0))
    panel["prob"] = router.predict_proba_by_regime(X, panel["stock_regime"])
    return panel


# ============================================================================
# Per-signal forward stats
# ============================================================================

def _per_signal(sig_row, hold_days: int, intra: pd.DataFrame) -> Optional[Dict]:
    sig_date = pd.Timestamp(sig_row.timestamp)
    sig_norm = sig_date.normalize()

    after = _market_hours(
        intra[intra["timestamp"].dt.normalize() > sig_norm]
    ).sort_values("timestamp").reset_index(drop=True)
    if after.empty:
        return None
    days_after = sorted(after["timestamp"].dt.normalize().unique())
    if len(days_after) < hold_days:
        return None

    t1 = days_after[0]
    t1_bars = after[after["timestamp"].dt.normalize() == t1].reset_index(drop=True)
    if t1_bars.empty:
        return None

    cutoff_t1 = _orb_cutoff(t1)
    orb_bars  = t1_bars[t1_bars["timestamp"] <= cutoff_t1]
    if len(orb_bars) < ORB_BARS_NEEDED:
        return None
    orb_high = float(orb_bars["high"].max())

    entry_open_px = float(t1_bars.iloc[0]["open"])

    entry_orb_px   = np.nan
    entry_orb_idx  = -1
    entry_orb_time = None
    after_orb_t1 = t1_bars[t1_bars["timestamp"] > cutoff_t1].reset_index(drop=True)
    for j, b in after_orb_t1.iterrows():
        if b["close"] > orb_high:
            entry_orb_px   = float(b["close"])
            entry_orb_time = b["timestamp"].strftime("%H:%M")
            entry_orb_idx  = j
            break

    hold_window = days_after[:hold_days]
    fwd = after[after["timestamp"].dt.normalize().isin(hold_window)] \
            .sort_values("timestamp").reset_index(drop=True)

    rec: Dict = {
        "symbol": sig_row.symbol,
        "signal_date": sig_date,
        "regime": sig_row.stock_regime,
        "prob": float(sig_row.prob),
        "prev_close": float(sig_row.close),
        "entry_open_px": entry_open_px,
        "entry_orb_px": entry_orb_px,
        "entry_orb_time": entry_orb_time,
        "orb_high_t1": orb_high,
        "orb_broke_t1": pd.notna(entry_orb_px),
        "slip_open_to_orb_pct": ((entry_orb_px / entry_open_px - 1) * 100
                                  if pd.notna(entry_orb_px) else np.nan),
        "slip_signal_to_orb_pct": ((entry_orb_px / sig_row.close - 1) * 100
                                    if pd.notna(entry_orb_px) else np.nan),
        "n_forward_days": len(hold_window),
    }

    for label, e_px, walk_df in [
        ("open", entry_open_px, fwd),
        ("orb",  entry_orb_px,
                 (pd.concat([
                     after_orb_t1.iloc[entry_orb_idx:] if entry_orb_idx >= 0
                                                       else after_orb_t1.iloc[0:0],
                     fwd[fwd["timestamp"].dt.normalize() > t1],
                 ]).sort_values("timestamp").reset_index(drop=True)
                 if pd.notna(entry_orb_px) else pd.DataFrame())),
    ]:
        if pd.isna(e_px) or walk_df.empty:
            continue

        lows  = walk_df["low"].values.astype(float)
        highs = walk_df["high"].values.astype(float)
        closes = walk_df["close"].values.astype(float)
        n = len(lows)

        rec[f"{label}_mae_pct"] = (lows.min()  / e_px - 1) * 100
        rec[f"{label}_mfe_pct"] = (highs.max() / e_px - 1) * 100
        rec[f"{label}_t5_close_ret_pct"] = (closes[-1] / e_px - 1) * 100

        first_tp_idx: Dict[float, int] = {}
        for tp in TP_LEVELS_PCT:
            tp_px = e_px * (1 + tp / 100.0)
            hits = np.where(highs >= tp_px)[0]
            first_tp_idx[tp] = int(hits[0]) if len(hits) else n
            rec[f"{label}_hit_tp{tp:g}"] = first_tp_idx[tp] < n

        first_sl_idx: Dict[float, int] = {}
        for sl in SL_LEVELS_PCT:
            sl_px = e_px * (1 + sl / 100.0)
            hits = np.where(lows <= sl_px)[0]
            first_sl_idx[sl] = int(hits[0]) if len(hits) else n
            rec[f"{label}_hit_sl{abs(sl):g}"] = first_sl_idx[sl] < n

        t5_ret = rec[f"{label}_t5_close_ret_pct"]
        for tp in TP_LEVELS_PCT:
            for sl in SL_LEVELS_PCT:
                ftp = first_tp_idx[tp]
                fsl = first_sl_idx[sl]
                if ftp == n and fsl == n:
                    outcome, ret = "neither", t5_ret
                elif fsl <= ftp:
                    outcome, ret = "sl", sl
                else:
                    outcome, ret = "tp", tp
                rec[f"{label}_oco_tp{tp:g}_sl{abs(sl):g}_outcome"] = outcome
                rec[f"{label}_oco_tp{tp:g}_sl{abs(sl):g}_ret"] = ret

    return rec


def run_historical(panel: pd.DataFrame, prob_min: float,
                   hold_days: int) -> pd.DataFrame:
    sig = panel[panel["prob"] >= prob_min].copy()
    print(f"[historical] {len(sig):,} signals to follow up on")
    sig = sig.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    rows: List[Dict] = []
    cur_sym: Optional[str] = None
    intra: Optional[pd.DataFrame] = None
    for i, row in enumerate(sig.itertuples(index=False), 1):
        if i % 500 == 0:
            print(f"  {i}/{len(sig)}  kept={len(rows)}")
        if row.symbol != cur_sym:
            cur_sym = row.symbol
            intra = _load_intraday(cur_sym)
        if intra is None:
            continue
        rec = _per_signal(row, hold_days, intra)
        if rec is not None:
            rows.append(rec)
    return pd.DataFrame(rows)


# ============================================================================
# Aggregate
# ============================================================================

def aggregate(per_signal: pd.DataFrame) -> pd.DataFrame:
    if per_signal.empty:
        return pd.DataFrame()
    df = per_signal.copy()
    df["prob_bucket"] = pd.cut(
        df["prob"],
        bins=[0.70, 0.75, 0.85, 1.01],
        right=False,
        labels=["[0.70, 0.75)", "[0.75, 0.85)", "[0.85, 1.01)"],
    ).astype(str)

    rows = []
    for entry in ["open", "orb"]:
        ret_col = f"{entry}_t5_close_ret_pct"
        mae_col = f"{entry}_mae_pct"
        mfe_col = f"{entry}_mfe_pct"
        if ret_col not in df.columns:
            continue
        for (regime, bucket), grp in df.groupby(["regime", "prob_bucket"]):
            valid = grp[grp[ret_col].notna()]
            if valid.empty:
                continue
            r = {
                "entry": entry, "regime": regime, "prob_bucket": bucket,
                "n": len(valid),
                "trigger_pct": (100 * grp["orb_broke_t1"].mean()
                                if entry == "orb" else 100.0),
                "mean_mae_pct": valid[mae_col].mean(),
                "mean_mfe_pct": valid[mfe_col].mean(),
                "mean_t5_ret_pct": valid[ret_col].mean(),
                "mean_t5_ret_net_pct": valid[ret_col].mean() - TOTAL_COST_PCT,
                "median_t5_ret_pct": valid[ret_col].median(),
            }
            for tp in TP_LEVELS_PCT:
                col = f"{entry}_hit_tp{tp:g}"
                if col in valid.columns:
                    r[f"hit_tp{tp:g}_pct"] = 100 * valid[col].mean()
            for sl in SL_LEVELS_PCT:
                col = f"{entry}_hit_sl{abs(sl):g}"
                if col in valid.columns:
                    r[f"hit_sl{abs(sl):g}_pct"] = 100 * valid[col].mean()
            outcome_col = f"{entry}_oco_tp3_sl3_outcome"
            ret_oco_col = f"{entry}_oco_tp3_sl3_ret"
            if outcome_col in valid.columns:
                r["oco_3x3_tp_pct"] = 100 * (valid[outcome_col] == "tp").mean()
                r["oco_3x3_sl_pct"] = 100 * (valid[outcome_col] == "sl").mean()
                r["oco_3x3_neither_pct"] = 100 * (valid[outcome_col] == "neither").mean()
                r["oco_3x3_mean_ret"] = valid[ret_oco_col].mean()
                r["oco_3x3_mean_net"] = valid[ret_oco_col].mean() - TOTAL_COST_PCT
            rows.append(r)
    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob-min", type=float, default=PROB_MIN)
    ap.add_argument("--hold-days", type=int, default=HOLD_DAYS)
    ap.add_argument("--years", type=float, default=HISTORICAL_YEARS)
    args = ap.parse_args()

    panel  = score_history(args.prob_min, args.years)
    per_sig = run_historical(panel, args.prob_min, args.hold_days)
    agg     = aggregate(per_sig)

    if not per_sig.empty:
        per_sig.to_parquet(OUT_DIR / "per_signal_historical.parquet", index=False)
        print(f"[out] per_signal_historical.parquet  ({len(per_sig)} rows)")

    if not agg.empty:
        agg.to_csv(OUT_DIR / "aggregate_entry_comparison.csv", index=False)
        print(f"[out] aggregate_entry_comparison.csv  (the lookup table)")

    try:
        with pd.ExcelWriter(OUT_DIR / "historical_summary.xlsx",
                            engine="openpyxl") as xw:
            if not agg.empty:
                agg.sort_values(["regime", "prob_bucket", "entry"]) \
                   .to_excel(xw, sheet_name="Entry_Comparison", index=False)
            if not per_sig.empty:
                per_sig.head(50_000).to_excel(xw, sheet_name="Per_Signal",
                                              index=False)
    except Exception as e:
        print(f"[warn] could not write xlsx: {e}")

    if not agg.empty:
        print("\n=== Entry comparison: open vs orb ===")
        show_cols = ["entry", "regime", "prob_bucket", "n", "trigger_pct",
                     "mean_mae_pct", "mean_mfe_pct",
                     "mean_t5_ret_pct", "mean_t5_ret_net_pct",
                     "hit_tp2_pct", "hit_tp3_pct", "hit_tp4_pct", "hit_tp5_pct",
                     "hit_sl2_pct", "hit_sl3_pct", "hit_sl5_pct",
                     "oco_3x3_tp_pct", "oco_3x3_sl_pct", "oco_3x3_mean_net"]
        show_cols = [c for c in show_cols if c in agg.columns]
        print(agg.sort_values(["regime", "prob_bucket", "entry"])[show_cols]
                 .to_string(index=False))


if __name__ == "__main__":
    main()
