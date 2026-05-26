"""
opportunities.py
================
Live opportunity tracker. For every (symbol, signal_date) in the last
LOOKBACK_DAYS trading days where prob >= PROB_MIN on the signal day,
follow the name day-by-day through T+1..T+5 and report:

  - signal-day prob, regime, close
  - prob today (rescored on the most-recent panel bar)
  - daily return on each of T+1..T+5  (open->close % and close->close %)
  - T+1 ORB-15 high (the static reference; the held-anyday rule keys off it)
  - on each day: did any 5-min bar close above T+1 ORH?  did the last bar
    of that day also close above T+1 ORH? (the held-EOD confirmation)
  - intraday MAE / MFE per day from that day's open
  - cumulative naive return  (entry = T+1 open)
  - cumulative ORB return    (entry = first held-EOD breakout close)
  - distance-to-T+1-ORH  (helpful while the breakout is still pending)
  - distance-to-SL  (-3 %), distance-to-TP  (3 / 5 %)
  - a routing recommendation per the hybrid rule
  - a status flag per signal:
      WAITING_T1        signal is today; T+1 has not yet started
      WATCHING_ORB      ORB-routed; not yet held-EOD on any day
      TAKEN_NAIVE       naive bucket; trade is on, ride to T+5
      TAKEN_ORB         ORB held-EOD on a day in T+1..T+5; entry recorded
      EXPIRED_NO_ORB    5 days elapsed without held-EOD breakout (no trade)
      STOPPED_OUT       MAE since entry <= -3 %  (informational; risk view)

OUTPUT (multi-tab xlsx + parquet/csv)
-------------------------------------
<BASE_DIR>/opportunities/
    summary_<YYYYMMDD>.xlsx      (Summary, Long, Wide, By_Bucket)
    long_<YYYYMMDD>.parquet      (one row per (symbol, signal_date, day_offset))
    long_<YYYYMMDD>.csv
    wide_<YYYYMMDD>.csv          (one row per (symbol, signal_date))

USAGE
-----
    python opportunities.py
    python opportunities.py --prob-min 0.70 --lookback-days 5
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import watchlist_followup as WF       # canonical loaders, helpers, constants
import hybrid_entry as HE             # bucket router

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR     = WF.BASE_DIR
INTRADAY_DIR = WF.INTRADAY_DIR
IST          = WF.IST

OUT_DIR = BASE_DIR / "opportunities"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROB_MIN          = 0.70
LOOKBACK_DAYS     = 5
MAX_HORIZON_DAYS  = 5

ORB_END_TIME    = WF.ORB_END_TIME
ORB_BARS_NEEDED = WF.ORB_BARS_NEEDED
TOTAL_COST_PCT  = WF.TOTAL_COST_PCT

SL_PCT = -3.0
TP_PCTS = [3.0, 5.0]


# ============================================================================
# Per-signal evolution: the workhorse, shared with the backtest
# ============================================================================

def _evolve_one_signal(intra: pd.DataFrame,
                       sig_date: pd.Timestamp,
                       max_horizon: int = MAX_HORIZON_DAYS) -> Optional[Dict]:
    """
    For a single (symbol, signal_date), walk the intraday tape across the
    next up-to `max_horizon` trading days and return:

        {
          "long_rows": [ {day-level dict} for each day actually present ],
          "summary":   {aggregate fields described above},
        }

    Returns None only if there is no intraday data after the signal day or
    fewer than ORB_BARS_NEEDED pre-09:30 bars on T+1.
    """
    sig_norm = sig_date.normalize()
    after = WF._market_hours(
        intra[intra["timestamp"].dt.normalize() > sig_norm]
    ).sort_values("timestamp").reset_index(drop=True)
    if after.empty:
        return None

    days = sorted(after["timestamp"].dt.normalize().unique())
    if not days:
        return None

    t1 = days[0]
    t1_bars = after[after["timestamp"].dt.normalize() == t1].reset_index(drop=True)
    if t1_bars.empty:
        return None

    cutoff_t1 = WF._orb_cutoff(t1)
    orb_bars_t1 = t1_bars[t1_bars["timestamp"] <= cutoff_t1]
    if len(orb_bars_t1) < ORB_BARS_NEEDED:
        return None
    orh_t1 = float(orb_bars_t1["high"].max())
    orl_t1 = float(orb_bars_t1["low"].min())

    naive_entry_px = float(t1_bars.iloc[0]["open"])

    # Walk each day in T+1..T+max_horizon (whichever days exist in the tape)
    horizon_days = days[:max_horizon]
    long_rows: List[Dict] = []

    orb_entry_px: float = float("nan")
    orb_entry_day: Optional[pd.Timestamp] = None
    orb_entry_time: Optional[pd.Timestamp] = None
    orb_entry_idx: int = -1

    naive_high_so_far = naive_entry_px
    naive_low_so_far  = naive_entry_px

    for idx, d in enumerate(horizon_days, start=1):
        d_bars = after[after["timestamp"].dt.normalize() == d].reset_index(drop=True)
        if d_bars.empty:
            continue

        d_open  = float(d_bars.iloc[0]["open"])
        d_close = float(d_bars.iloc[-1]["close"])
        d_high  = float(d_bars["high"].max())
        d_low   = float(d_bars["low"].min())

        # this day's own ORB (informational; the held_anyday rule keys off T+1)
        cutoff_d = WF._orb_cutoff(d)
        orb_d = d_bars[d_bars["timestamp"] <= cutoff_d]
        day_orh = float(orb_d["high"].max()) if len(orb_d) >= ORB_BARS_NEEDED else float("nan")
        day_orl = float(orb_d["low"].min())  if len(orb_d) >= ORB_BARS_NEEDED else float("nan")

        # T+1 ORH break check (any 5-min bar after the ORB window on T+1,
        # or any bar at all on subsequent days)
        if d == t1:
            scan = d_bars[d_bars["timestamp"] > cutoff_t1].reset_index(drop=True)
        else:
            scan = d_bars
        broke_today_hits = scan[scan["close"] > orh_t1] if not scan.empty else scan
        broke_today = not broke_today_hits.empty
        first_break_time = (broke_today_hits.iloc[0]["timestamp"]
                            if broke_today else None)
        first_break_px = (float(broke_today_hits.iloc[0]["close"])
                          if broke_today else float("nan"))

        held_eod = float(d_bars.iloc[-1]["close"]) > orh_t1

        # Intraday MAE/MFE from that day's open
        intraday_mae = (d_low / d_open - 1.0) * 100.0
        intraday_mfe = (d_high / d_open - 1.0) * 100.0

        # Daily returns
        ret_oc_pct = (d_close / d_open - 1.0) * 100.0
        if idx == 1:
            ret_cc_pct = float("nan")  # no prev-close available in window
        else:
            prev_close = long_rows[-1]["daily_close"]
            ret_cc_pct = (d_close / prev_close - 1.0) * 100.0

        # If ORB entry has not yet been taken, check held-EOD on this day
        if not np.isfinite(orb_entry_px) and held_eod and broke_today:
            orb_entry_px   = first_break_px
            orb_entry_day  = pd.Timestamp(d)
            orb_entry_time = first_break_time
            orb_entry_idx  = int(after.index[after["timestamp"] == first_break_time][0])

        # Cumulative naive return (T+1 open -> this day's close)
        naive_cum_ret_pct = (d_close / naive_entry_px - 1.0) * 100.0
        naive_high_so_far = max(naive_high_so_far, d_high)
        naive_low_so_far  = min(naive_low_so_far, d_low)

        long_rows.append({
            "day_offset":      idx,
            "trade_date":      pd.Timestamp(d),
            "daily_open":      d_open,
            "daily_close":     d_close,
            "daily_high":      d_high,
            "daily_low":       d_low,
            "ret_oc_pct":      ret_oc_pct,
            "ret_cc_pct":      ret_cc_pct,
            "t1_orh":          orh_t1,
            "t1_orl":          orl_t1,
            "day_orh":         day_orh,
            "day_orl":         day_orl,
            "broke_t1_orh_today":  bool(broke_today),
            "first_break_time":    (first_break_time.strftime("%H:%M")
                                    if first_break_time is not None else None),
            "first_break_px":      first_break_px,
            "held_t1_orh_eod":     bool(held_eod),
            "intraday_mae_pct":    intraday_mae,
            "intraday_mfe_pct":    intraday_mfe,
            "naive_cum_ret_pct":   naive_cum_ret_pct,
        })

    if not long_rows:
        return None

    last_row = long_rows[-1]
    last_close = last_row["daily_close"]

    # Naive MAE/MFE from T+1 open across the realized window
    naive_mae_pct = (naive_low_so_far  / naive_entry_px - 1.0) * 100.0
    naive_mfe_pct = (naive_high_so_far / naive_entry_px - 1.0) * 100.0

    # ORB-entry path metrics (only if triggered)
    if np.isfinite(orb_entry_px) and orb_entry_idx >= 0:
        from_entry = after.iloc[orb_entry_idx:].copy()
        from_entry = from_entry[from_entry["timestamp"].dt.normalize() <= horizon_days[-1]]
        if not from_entry.empty:
            orb_high = float(from_entry["high"].max())
            orb_low  = float(from_entry["low"].min())
            orb_last_close = float(from_entry.iloc[-1]["close"])
            orb_mfe_pct = (orb_high / orb_entry_px - 1.0) * 100.0
            orb_mae_pct = (orb_low  / orb_entry_px - 1.0) * 100.0
            orb_cum_ret_pct = (orb_last_close / orb_entry_px - 1.0) * 100.0
        else:
            orb_mfe_pct = orb_mae_pct = orb_cum_ret_pct = float("nan")
    else:
        orb_mfe_pct = orb_mae_pct = orb_cum_ret_pct = float("nan")

    # Distance to T+1 ORH using the latest available close (negative if above)
    dist_to_orh_pct = (orh_t1 / last_close - 1.0) * 100.0

    summary = {
        "n_days_realized":     len(long_rows),
        "t1_open_px":          naive_entry_px,
        "t1_orh":              orh_t1,
        "t1_orl":              orl_t1,
        "latest_close":        last_close,
        "latest_date":         pd.Timestamp(last_row["trade_date"]),
        "naive_cum_ret_pct":   last_row["naive_cum_ret_pct"],
        "naive_cum_ret_net_pct": last_row["naive_cum_ret_pct"] - TOTAL_COST_PCT,
        "naive_mae_pct":       naive_mae_pct,
        "naive_mfe_pct":       naive_mfe_pct,
        "orb_triggered":       bool(np.isfinite(orb_entry_px)),
        "orb_entry_day":       orb_entry_day,
        "orb_entry_time":      (orb_entry_time.strftime("%Y-%m-%d %H:%M")
                                if orb_entry_time is not None else None),
        "orb_entry_px":        orb_entry_px,
        "orb_cum_ret_pct":     orb_cum_ret_pct,
        "orb_cum_ret_net_pct": (orb_cum_ret_pct - TOTAL_COST_PCT
                                if np.isfinite(orb_cum_ret_pct) else float("nan")),
        "orb_mae_pct":         orb_mae_pct,
        "orb_mfe_pct":         orb_mfe_pct,
        "dist_to_t1_orh_pct":  dist_to_orh_pct,
        "broke_t1_orh_anyday": any(r["broke_t1_orh_today"] for r in long_rows),
        "held_t1_orh_anyday":  any(r["held_t1_orh_eod"]    for r in long_rows),
    }

    return {"long_rows": long_rows, "summary": summary}


# ============================================================================
# Live tracker
# ============================================================================

def _classify_status(summary: Dict, rule: str, n_days_realized: int,
                     max_horizon: int) -> str:
    if n_days_realized == 0:
        return "WAITING_T1"
    if rule == "naive_open":
        # Naive bucket: entry was at T+1 open. Live trade until T+5.
        if summary["naive_mae_pct"] <= SL_PCT:
            return "STOPPED_OUT"
        return "TAKEN_NAIVE"
    if rule == "orb_15_held_anyday":
        if summary["orb_triggered"]:
            if summary["orb_mae_pct"] <= SL_PCT:
                return "STOPPED_OUT"
            return "TAKEN_ORB"
        if n_days_realized >= max_horizon:
            return "EXPIRED_NO_ORB"
        return "WATCHING_ORB"
    return "UNKNOWN"


def _today_prob_lookup(panel_recent: pd.DataFrame) -> Dict[str, Tuple[float, pd.Timestamp]]:
    """Map symbol -> (latest_prob, latest_panel_timestamp) on the most-recent
    panel bar where the symbol appears. Used to compute prob_today."""
    if panel_recent.empty:
        return {}
    latest = panel_recent["timestamp"].max().normalize()
    today = panel_recent[panel_recent["timestamp"].dt.normalize() == latest]
    if today.empty:
        return {}
    return {row["symbol"]: (float(row["prob"]), row["timestamp"])
            for _, row in today[["symbol", "prob", "timestamp"]].iterrows()}


def build_opportunities(prob_min: float, lookback_days: int,
                        max_horizon: int = MAX_HORIZON_DAYS
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (long_df, wide_df, summary_df_for_console).
    """
    print(f"[opps] scoring panel; lookback={lookback_days} days, prob_min={prob_min}")
    panel_recent = WF.score_recent(prob_min, lookback_days, historical_years=0.0)

    if panel_recent.empty:
        print("[opps] empty panel; nothing to do")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    latest_panel_date = panel_recent["timestamp"].max().normalize()
    cutoff = latest_panel_date - pd.Timedelta(days=lookback_days * 2)
    # Take signals that fired in the last `lookback_days` *trading* days.
    trading_days = sorted(panel_recent["timestamp"].dt.normalize().unique())
    eligible_dates = set(trading_days[-lookback_days:])
    sigs = panel_recent[(panel_recent["prob"] >= prob_min) &
                        (panel_recent["timestamp"].dt.normalize().isin(eligible_dates))].copy()

    if sigs.empty:
        print(f"[opps] no signals at or above {prob_min} in last "
              f"{lookback_days} trading days")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(f"[opps] {len(sigs):,} signals to follow")
    today_lookup = _today_prob_lookup(panel_recent)
    intra_cache: Dict[str, Optional[pd.DataFrame]] = {}

    long_rows: List[Dict] = []
    wide_rows: List[Dict] = []
    skipped = 0

    sigs = sigs.sort_values(["timestamp", "prob"], ascending=[False, False]).reset_index(drop=True)
    for sig in sigs.itertuples(index=False):
        sym = sig.symbol
        if sym not in intra_cache:
            intra_cache[sym] = WF._load_intraday(sym)
        intra = intra_cache[sym]
        if intra is None or intra.empty:
            skipped += 1
            continue

        evo = _evolve_one_signal(intra, pd.Timestamp(sig.timestamp), max_horizon=max_horizon)
        if evo is None:
            # Either zero forward bars (signal date is today) or insufficient
            # T+1 ORB. Emit a placeholder waiting-row anyway.
            base = {
                "symbol":         sym,
                "signal_date":    pd.Timestamp(sig.timestamp),
                "signal_prob":    float(sig.prob),
                "signal_regime":  sig.stock_regime,
                "signal_close":   float(sig.close),
            }
            prob_today, prob_today_date = today_lookup.get(sym, (float("nan"), None))
            bucket, rule = HE.route_prob(float(sig.prob))
            wide_rows.append({
                **base,
                "prob_today":          prob_today,
                "prob_today_date":     prob_today_date,
                "prob_change":         (prob_today - float(sig.prob)
                                        if np.isfinite(prob_today) else float("nan")),
                "prob_bucket":         bucket,
                "rule":                rule,
                "n_days_realized":     0,
                "status":              "WAITING_T1",
                "recommendation":      _recommendation(rule, "WAITING_T1", float("nan")),
                "t1_open_px":          float("nan"),
                "t1_orh":              float("nan"),
                "latest_close":        float("nan"),
                "naive_cum_ret_pct":   float("nan"),
                "orb_triggered":       False,
                "broke_t1_orh_anyday": False,
                "held_t1_orh_anyday":  False,
                "dist_to_t1_orh_pct":  float("nan"),
            })
            continue

        prob_today, prob_today_date = today_lookup.get(sym, (float("nan"), None))
        bucket, rule = HE.route_prob(float(sig.prob))
        status = _classify_status(evo["summary"], rule,
                                  evo["summary"]["n_days_realized"], max_horizon)

        # Long rows
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

        # Wide row
        wide = {
            "symbol":              sym,
            "signal_date":         pd.Timestamp(sig.timestamp),
            "signal_prob":         float(sig.prob),
            "signal_regime":       sig.stock_regime,
            "signal_close":        float(sig.close),
            "prob_today":          prob_today,
            "prob_today_date":     prob_today_date,
            "prob_change":         (prob_today - float(sig.prob)
                                    if np.isfinite(prob_today) else float("nan")),
            "prob_bucket":         bucket,
            "rule":                rule,
            "status":              status,
            "recommendation":      _recommendation(rule, status,
                                                   evo["summary"]["dist_to_t1_orh_pct"]),
            **evo["summary"],
        }
        # Per-day flat columns d1_..d5_ for quick scanning
        for r in evo["long_rows"]:
            d = r["day_offset"]
            wide[f"d{d}_date"]            = r["trade_date"]
            wide[f"d{d}_ret_oc_pct"]      = r["ret_oc_pct"]
            wide[f"d{d}_broke_t1_orh"]    = r["broke_t1_orh_today"]
            wide[f"d{d}_held_t1_orh_eod"] = r["held_t1_orh_eod"]
            wide[f"d{d}_intraday_mae"]    = r["intraday_mae_pct"]
            wide[f"d{d}_intraday_mfe"]    = r["intraday_mfe_pct"]
        wide_rows.append(wide)

    if skipped:
        print(f"[opps] skipped {skipped:,} signals (missing intraday data)")

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)

    # Console summary frame
    summary_cols = ["symbol", "signal_date", "signal_prob", "prob_today",
                    "signal_regime", "prob_bucket", "rule", "status",
                    "n_days_realized", "naive_cum_ret_pct",
                    "orb_triggered", "orb_cum_ret_pct",
                    "broke_t1_orh_anyday", "held_t1_orh_anyday",
                    "dist_to_t1_orh_pct", "recommendation"]
    summary_cols = [c for c in summary_cols if c in wide_df.columns]
    summary = wide_df[summary_cols].copy() if not wide_df.empty else pd.DataFrame()

    return long_df, wide_df, summary


def _recommendation(rule: str, status: str, dist_to_orh_pct: float) -> str:
    if rule == "naive_open":
        if status == "WAITING_T1":
            return "BUY at T+1 09:15 open; hold to T+5 close"
        if status == "TAKEN_NAIVE":
            return "Holding; ride to T+5 close. Stop at -3% from entry."
        if status == "STOPPED_OUT":
            return "Risk view: MAE through -3% since entry."
        return "Hold to T+5 close."
    if rule == "orb_15_held_anyday":
        if status == "WAITING_T1":
            return ("Tomorrow: compute T+1 09:15-09:30 ORH; if any 5-min "
                    "close > ORH AND last bar of that day also > ORH, enter.")
        if status == "WATCHING_ORB":
            if np.isfinite(dist_to_orh_pct):
                if dist_to_orh_pct > 0:
                    return (f"Pending. {dist_to_orh_pct:+.2f}% to ORH; "
                            "wait for 5-min close > ORH held into EOD.")
                return ("Above ORH intraday but EOD not yet confirmed; "
                        "do not enter mid-day.")
            return "Pending ORB-15 held-EOD confirmation."
        if status == "TAKEN_ORB":
            return "Holding from breakout. Stop at -3% from entry; hold to T+5 close."
        if status == "EXPIRED_NO_ORB":
            return "5 days elapsed without held-EOD breakout. No trade."
        if status == "STOPPED_OUT":
            return "Risk view: MAE through -3% since ORB entry."
        return "Watch."
    return "No-trade bucket."


# ============================================================================
# Bucket roll-up across the live cohort (snapshot)
# ============================================================================

def _by_bucket(wide_df: pd.DataFrame) -> pd.DataFrame:
    if wide_df.empty:
        return pd.DataFrame()
    g = wide_df.groupby(["signal_regime", "prob_bucket"])
    out = pd.DataFrame({
        "n":                       g.size(),
        "n_taken":                 g.apply(lambda d: int(((d["status"] == "TAKEN_NAIVE") |
                                                          (d["status"] == "TAKEN_ORB")).sum())),
        "n_watching":              g.apply(lambda d: int((d["status"] == "WATCHING_ORB").sum())),
        "n_expired":               g.apply(lambda d: int((d["status"] == "EXPIRED_NO_ORB").sum())),
        "broke_t1_orh_anyday_pct": g["broke_t1_orh_anyday"].mean().mul(100).round(2),
        "held_t1_orh_anyday_pct":  g["held_t1_orh_anyday"].mean().mul(100).round(2),
        "mean_naive_cum_ret":      g["naive_cum_ret_pct"].mean().round(2),
        "mean_orb_cum_ret":        g["orb_cum_ret_pct"].mean().round(2),
    }).reset_index()
    return out


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob-min", type=float, default=PROB_MIN)
    ap.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    ap.add_argument("--max-horizon", type=int, default=MAX_HORIZON_DAYS)
    args = ap.parse_args()

    long_df, wide_df, summary = build_opportunities(
        prob_min=args.prob_min,
        lookback_days=args.lookback_days,
        max_horizon=args.max_horizon,
    )

    if wide_df.empty:
        return

    stamp = pd.Timestamp(wide_df["signal_date"].max()).strftime("%Y%m%d")
    long_pq  = OUT_DIR / f"long_{stamp}.parquet"
    long_csv = OUT_DIR / f"long_{stamp}.csv"
    wide_csv = OUT_DIR / f"wide_{stamp}.csv"
    xlsx     = OUT_DIR / f"summary_{stamp}.xlsx"

    if not long_df.empty:
        long_df.to_parquet(long_pq, index=False)
        long_df.to_csv(long_csv, index=False)
        print(f"[out] wrote {long_pq}")
        print(f"[out] wrote {long_csv}")

    wide_df.to_csv(wide_csv, index=False)
    print(f"[out] wrote {wide_csv}")

    by_bucket = _by_bucket(wide_df)
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
            summary.to_excel(xw, sheet_name="Summary", index=False)
            wide_df.to_excel(xw, sheet_name="Wide", index=False)
            if not long_df.empty:
                long_df.head(50_000).to_excel(xw, sheet_name="Long", index=False)
            if not by_bucket.empty:
                by_bucket.to_excel(xw, sheet_name="By_Bucket", index=False)
        print(f"[out] wrote {xlsx}")
    except Exception as e:
        print(f"[warn] could not write xlsx: {e}")

    # Console teaser
    print("\n=== Opportunities snapshot (last "
          f"{args.lookback_days} trading days, prob>={args.prob_min}) ===")
    show_cols = [c for c in [
        "symbol", "signal_date", "signal_prob", "prob_today",
        "signal_regime", "prob_bucket", "status",
        "naive_cum_ret_pct", "orb_cum_ret_pct",
        "broke_t1_orh_anyday", "held_t1_orh_anyday",
        "dist_to_t1_orh_pct", "recommendation",
    ] if c in summary.columns]
    print(summary[show_cols].to_string(index=False))

    if not by_bucket.empty:
        print("\n=== Cohort by regime x prob bucket ===")
        print(by_bucket.to_string(index=False))


if __name__ == "__main__":
    main()
