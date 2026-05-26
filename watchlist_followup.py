"""
watchlist_followup.py
=====================
Two-in-one tool for the >=0.70 probability watchlist.

Part A  -- LIVE WATCHLIST (default mode, fast)
----------------------------------------------
For the last N (=10) trading days, find every (symbol, day) where the
model rates prob >= 0.70. For every qualifying symbol, output a per-day
timeline over those N days that includes:
    * daily prob, regime
    * OHLC, prev_close
    * close-to-close ret%, open-to-close ret%, gap%
    * 15-min ORB high / low for that day
    * a comment on the day if/when the ORB broke
      (e.g. "broke 442.10 at 09:35 px 444.50 (slip +0.81%)" or
            "no break (ORB 442.10)" or
            "gap above ORB - opened at 446.00")
    * intraday MAE / MFE from open

Part B  -- HISTORICAL EXECUTION STATS (with --with-historical)
-------------------------------------------------------------
For every prob >= 0.70 signal in the past `--historical-years` years,
fetch the intraday cache and compute per-signal forward stats:
    * entry_at_open      = T+1 first 5-min bar's open
    * entry_at_orb       = first close > T+1 15-min ORB high (or NaN)
    * MAE / MFE over the next HOLD_DAYS trading days, from each entry
    * first-touch hit flags for TP={2, 3, 4, 5, 7, 10}%
                              SL={-2, -3, -5}%
    * OCO outcomes (tp first, sl first, neither) for every (TP, SL) combo
    * realized return at T+5 close

Aggregate result: per (regime, prob_bucket, entry_method) the historical
MAE, MFE, hit rates and OCO-outcome rates. THIS is the lookup table you
apply to the live watchlist when sizing / setting stops.

OUTPUT
------
<BASE_DIR>/watchlist_followup/
    watchlist_followup_<YYYYMMDD>.xlsx   (multi-tab workbook)
    recent_signals_<YYYYMMDD>.csv
    daily_timeline_<YYYYMMDD>.csv
    aggregate_entry_comparison.csv          (with --with-historical)
    per_signal_historical.parquet            (with --with-historical)

USAGE
-----
    python watchlist_followup.py                      # fast: just watchlist
    python watchlist_followup.py --with-historical    # also full backtest
    python watchlist_followup.py --prob-min 0.70 --lookback-days 10
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG  -- mirrors orb_machine_v2_updated.py / orb_execution_quality.py
# ============================================================================

BASE_DIR      = Path(r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full")
PANEL_PATH    = BASE_DIR / "panel_cache.parquet"
FEATURES_PATH = BASE_DIR / "features_train.json"
ROUTER_PATH   = BASE_DIR / "models" / "m5_regime_router.joblib"
INTRADAY_DIR  = Path(r"C:\Users\karanvsi\Desktop\Pycharm\Cache\intraday_5min")

OUT_DIR       = BASE_DIR / "watchlist_followup"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IST = "Asia/Kolkata"

# Defaults (overridable from CLI)
PROB_MIN          = 0.70
LOOKBACK_DAYS     = 10
HOLD_DAYS         = 5
HISTORICAL_YEARS  = 2
SIGNAL_REGIMES    = ["bull_trend", "bear_trend"]

# 15-min ORB
ORB_END_TIME      = "09:30"
ORB_BARS_NEEDED   = 3            # 09:15, 09:20, 09:25 close out the 15-min window

# Costs
COST_BPS_RT       = 25
SLIP_BPS_PER_SIDE = 5
TOTAL_COST_PCT    = (COST_BPS_RT + 2 * SLIP_BPS_PER_SIDE) / 100.0

# TP / SL ladders (user requested 2,3,4,5%)
TP_LEVELS_PCT = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
SL_LEVELS_PCT = [-2.0, -3.0, -5.0]


# ============================================================================
# Helpers
# ============================================================================

def _load_intraday(symbol: str) -> Optional[pd.DataFrame]:
    """Read the 5-min intraday parquet for a symbol, normalize timezone."""
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
# 1) Score recent panel
# ============================================================================

def score_recent(prob_min: float, lookback_days: int,
                 historical_years: float = 0.0) -> pd.DataFrame:
    """Score the panel; return rows with prob, regime, OHLC, ret%."""
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
    # Determine cutoff: max of the lookback window and the historical window
    days_needed = max(lookback_days * 2, historical_years * 365)
    cutoff = latest - pd.Timedelta(days=int(days_needed))
    recent = panel[panel["timestamp"] >= cutoff].copy()

    schema = json.loads(FEATURES_PATH.read_text())
    FEATURES = schema["features"]
    IMPUTE = {k: float(v) for k, v in schema["impute"].items()}

    print(f"[score] scoring {len(recent):,} rows...")
    router = load(ROUTER_PATH)
    X = recent.reindex(columns=FEATURES).copy()
    for c in FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(IMPUTE.get(c, 0.0))
    recent["prob"] = router.predict_proba_by_regime(X, recent["stock_regime"])

    # Daily ret %
    recent["prev_close"] = recent.groupby("symbol")["close"].shift(1)
    recent["ret_cc_pct"] = (recent["close"] / recent["prev_close"] - 1) * 100
    recent["ret_oc_pct"] = (recent["close"] / recent["open"]       - 1) * 100
    recent["gap_pct"]    = (recent["open"]  / recent["prev_close"] - 1) * 100

    return recent


# ============================================================================
# 2) Per-day ORB break detection (used by both watchlist and follow-up)
# ============================================================================

def _detect_orb_for_day(bars_full: pd.DataFrame, day: pd.Timestamp,
                        prev_close: float) -> Dict:
    """
    Given the full intraday DataFrame and a day, compute:
        orb_high, orb_low, broke (bool), break_time, break_price,
        intraday_open, intraday_close, intraday_high, intraday_low,
        intraday_mae_from_open_pct, intraday_mfe_from_open_pct
    """
    out = {
        "orb_high": np.nan, "orb_low": np.nan,
        "broke": False, "break_time": None, "break_price": np.nan,
        "intraday_open": np.nan, "intraday_close": np.nan,
        "intraday_high": np.nan, "intraday_low": np.nan,
        "intraday_mae_from_open_pct": np.nan,
        "intraday_mfe_from_open_pct": np.nan,
    }
    day_norm = day.normalize()
    bars = bars_full[bars_full["timestamp"].dt.normalize() == day_norm]
    bars = _market_hours(bars).sort_values("timestamp").reset_index(drop=True)
    if len(bars) < ORB_BARS_NEEDED:
        return out

    cutoff = _orb_cutoff(day_norm)
    orb_bars = bars[bars["timestamp"] <= cutoff]
    if len(orb_bars) < ORB_BARS_NEEDED:
        return out

    out["orb_high"]      = float(orb_bars["high"].max())
    out["orb_low"]       = float(orb_bars["low"].min())
    out["intraday_open"] = float(bars.iloc[0]["open"])
    out["intraday_close"] = float(bars.iloc[-1]["close"])
    out["intraday_high"] = float(bars["high"].max())
    out["intraday_low"]  = float(bars["low"].min())

    op = out["intraday_open"]
    out["intraday_mae_from_open_pct"] = (out["intraday_low"]  / op - 1) * 100
    out["intraday_mfe_from_open_pct"] = (out["intraday_high"] / op - 1) * 100

    after = bars[bars["timestamp"] > cutoff]
    for _, b in after.iterrows():
        if b["close"] > out["orb_high"]:
            out["broke"]       = True
            out["break_time"]  = b["timestamp"].strftime("%H:%M")
            out["break_price"] = float(b["close"])
            break
    return out


# ============================================================================
# 3) Build live watchlist + daily timeline
# ============================================================================

def build_watchlist(recent: pd.DataFrame,
                    prob_min: float,
                    lookback_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (recent_signals, daily_timeline). recent_signals is the flat list
    of (symbol, day) where prob >= prob_min in the last N days; daily_timeline
    is the full per-day panel for those symbols."""
    all_days = sorted(recent["timestamp"].dt.normalize().unique())[-lookback_days:]
    window = recent[recent["timestamp"].dt.normalize().isin(all_days)].copy()

    # Symbols that hit prob >= prob_min any day in the window
    qualifying = (window[window["prob"] >= prob_min]["symbol"].unique().tolist())
    print(f"[watchlist] {len(qualifying)} symbols hit prob>={prob_min} in the "
          f"last {lookback_days} trading days")

    if not qualifying:
        return pd.DataFrame(), pd.DataFrame()

    timeline = window[window["symbol"].isin(qualifying)].copy()

    # Detect ORB break for every (symbol, day) cell
    print(f"[watchlist] detecting ORB breaks for {len(timeline)} cells...")
    seen_intraday: Dict[str, Optional[pd.DataFrame]] = {}
    orb_records = []
    for i, row in enumerate(timeline.itertuples(index=False), 1):
        if i % 200 == 0:
            print(f"  {i}/{len(timeline)}")
        sym = row.symbol
        if sym not in seen_intraday:
            seen_intraday[sym] = _load_intraday(sym)
        intra = seen_intraday[sym]
        if intra is None:
            orb_records.append({
                "orb_high": np.nan, "orb_low": np.nan,
                "broke": False, "break_time": None, "break_price": np.nan,
                "intraday_open": np.nan, "intraday_close": np.nan,
                "intraday_high": np.nan, "intraday_low": np.nan,
                "intraday_mae_from_open_pct": np.nan,
                "intraday_mfe_from_open_pct": np.nan,
            })
        else:
            orb_records.append(_detect_orb_for_day(intra, row.timestamp,
                                                   row.prev_close))
    orb_df = pd.DataFrame(orb_records)
    timeline = pd.concat([timeline.reset_index(drop=True), orb_df], axis=1)

    # ORB comment column
    def _comment(r):
        if pd.isna(r["orb_high"]):
            return "no intraday data"
        if r["broke"]:
            slip = ((r["break_price"] / r["prev_close"] - 1) * 100
                    if pd.notna(r["prev_close"]) else np.nan)
            slip_str = f" (slip {slip:+.2f}%)" if pd.notna(slip) else ""
            return (f"broke ORB {r['orb_high']:.2f} at {r['break_time']} "
                    f"px {r['break_price']:.2f}{slip_str}")
        if (pd.notna(r["intraday_open"]) and
            r["intraday_open"] > r["orb_high"]):
            return (f"gap above ORB ({r['orb_high']:.2f}) - "
                    f"opened at {r['intraday_open']:.2f}")
        return f"no break (ORB {r['orb_high']:.2f})"

    timeline["orb_comment"]  = timeline.apply(_comment, axis=1)
    timeline["is_signal_day"] = timeline["prob"] >= prob_min

    cols = [
        "symbol", "timestamp", "stock_regime", "prob", "is_signal_day",
        "prev_close", "open", "high", "low", "close",
        "ret_cc_pct", "ret_oc_pct", "gap_pct",
        "orb_high", "orb_low", "broke", "break_time", "break_price",
        "intraday_mae_from_open_pct", "intraday_mfe_from_open_pct",
        "orb_comment",
    ]
    timeline = timeline[[c for c in cols if c in timeline.columns]]
    timeline = timeline.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    recent_signals = (
        window[window["prob"] >= prob_min]
        .sort_values(["timestamp", "prob"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return recent_signals, timeline


# ============================================================================
# 4) Historical per-signal stats  (TP/SL hit rates, MAE, MFE, OCO)
# ============================================================================

def _per_signal_followup(sig_row, hold_days: int,
                         intra: pd.DataFrame) -> Optional[Dict]:
    """
    Compute forward stats for one (symbol, signal_date). Returns None if
    insufficient intraday data.
    """
    sig_date = pd.Timestamp(sig_row.timestamp)
    sig_norm = sig_date.normalize()

    after = _market_hours(
        intra[intra["timestamp"].dt.normalize() > sig_norm]
    ).sort_values("timestamp").reset_index(drop=True)
    if after.empty:
        return None
    days_after = sorted(after["timestamp"].dt.normalize().unique())
    if not days_after:
        return None
    if len(days_after) < hold_days:
        return None  # not enough forward data

    t1 = days_after[0]
    t1_bars = after[after["timestamp"].dt.normalize() == t1].reset_index(drop=True)
    if t1_bars.empty:
        return None

    # ORB on T+1
    cutoff_t1 = _orb_cutoff(t1)
    orb_bars  = t1_bars[t1_bars["timestamp"] <= cutoff_t1]
    if len(orb_bars) < ORB_BARS_NEEDED:
        return None
    orb_high = float(orb_bars["high"].max())

    entry_open_px = float(t1_bars.iloc[0]["open"])

    # Entry-at-ORB: first close > ORB high after the ORB period
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

    # Forward bars over hold window
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

    # Walk-forward stats for each entry method
    for label, e_px, walk_df in [
        ("open", entry_open_px, fwd),
        ("orb",  entry_orb_px,
                 pd.concat([
                     # bars on T+1 from the breakout bar onwards
                     after_orb_t1.iloc[entry_orb_idx:] if entry_orb_idx >= 0
                                                       else after_orb_t1.iloc[0:0],
                     # all bars on subsequent days in the hold window
                     fwd[fwd["timestamp"].dt.normalize() > t1],
                 ]).sort_values("timestamp").reset_index(drop=True)
                 if pd.notna(entry_orb_px) else pd.DataFrame()),
    ]:
        if pd.isna(e_px) or walk_df.empty:
            continue

        lows  = walk_df["low"].values.astype(float)
        highs = walk_df["high"].values.astype(float)
        closes = walk_df["close"].values.astype(float)
        n = len(lows)

        # MAE / MFE (no exit)
        mae = (lows.min()  / e_px - 1) * 100
        mfe = (highs.max() / e_px - 1) * 100
        rec[f"{label}_mae_pct"] = mae
        rec[f"{label}_mfe_pct"] = mfe
        rec[f"{label}_t5_close_ret_pct"] = (closes[-1] / e_px - 1) * 100

        # First-hit indices
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

        # OCO outcomes
        t5_ret = rec[f"{label}_t5_close_ret_pct"]
        for tp in TP_LEVELS_PCT:
            for sl in SL_LEVELS_PCT:
                ftp = first_tp_idx[tp]
                fsl = first_sl_idx[sl]
                if ftp == n and fsl == n:
                    outcome = "neither"
                    ret = t5_ret
                elif fsl <= ftp:
                    outcome = "sl"
                    ret = sl
                else:
                    outcome = "tp"
                    ret = tp
                rec[f"{label}_oco_tp{tp:g}_sl{abs(sl):g}_outcome"] = outcome
                rec[f"{label}_oco_tp{tp:g}_sl{abs(sl):g}_ret"] = ret

    return rec


def historical_followup(panel_recent: pd.DataFrame, prob_min: float,
                        hold_days: int) -> pd.DataFrame:
    """For every prob >= prob_min signal in panel_recent, compute forward stats."""
    sig = panel_recent[panel_recent["prob"] >= prob_min].copy()
    print(f"[historical] {len(sig):,} signals to follow up on")
    sig = sig.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    rows: List[Dict] = []
    cur_sym = None
    intra: Optional[pd.DataFrame] = None
    for i, row in enumerate(sig.itertuples(index=False), 1):
        if i % 200 == 0:
            print(f"  {i}/{len(sig)}")
        if row.symbol != cur_sym:
            cur_sym = row.symbol
            intra = _load_intraday(cur_sym)
        if intra is None:
            continue
        rec = _per_signal_followup(row, hold_days, intra)
        if rec is not None:
            rows.append(rec)
    return pd.DataFrame(rows)


# ============================================================================
# 5) Aggregate: per (regime, prob_bucket, entry) what to expect
# ============================================================================

def aggregate_followup(per_signal: pd.DataFrame) -> pd.DataFrame:
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
            # OCO at TP=3 / SL=-3 as the headline balanced exit
            outcome_col = f"{entry}_oco_tp3_sl3_outcome"
            ret_oco_col = f"{entry}_oco_tp3_sl3_ret"
            if outcome_col in valid.columns:
                r["oco_3x3_tp_pct"] = 100 * (valid[outcome_col] == "tp").mean()
                r["oco_3x3_sl_pct"] = 100 * (valid[outcome_col] == "sl").mean()
                r["oco_3x3_neither_pct"] = 100 * (valid[outcome_col] == "neither").mean()
                r["oco_3x3_mean_ret"]    = valid[ret_oco_col].mean()
                r["oco_3x3_mean_net"]    = valid[ret_oco_col].mean() - TOTAL_COST_PCT
            rows.append(r)
    return pd.DataFrame(rows)


# ============================================================================
# 6) MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob-min", type=float, default=PROB_MIN)
    ap.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    ap.add_argument("--hold-days", type=int, default=HOLD_DAYS)
    ap.add_argument("--with-historical", action="store_true",
                    help="also run the historical TP/SL hit-rate backtest")
    ap.add_argument("--historical-years", type=float, default=HISTORICAL_YEARS,
                    help="how many years back to scan for historical signals")
    args = ap.parse_args()

    hist_years = args.historical_years if args.with_historical else 0.0
    recent = score_recent(args.prob_min, args.lookback_days,
                          historical_years=hist_years)

    recent_signals, timeline = build_watchlist(recent,
                                               args.prob_min,
                                               args.lookback_days)

    latest_str = recent["timestamp"].max().strftime("%Y%m%d")
    out_xlsx = OUT_DIR / f"watchlist_followup_{latest_str}.xlsx"

    per_sig = pd.DataFrame()
    agg = pd.DataFrame()
    if args.with_historical:
        per_sig = historical_followup(recent, args.prob_min, args.hold_days)
        agg = aggregate_followup(per_sig)
        if not per_sig.empty:
            per_sig.to_parquet(OUT_DIR / "per_signal_historical.parquet",
                                index=False)
        if not agg.empty:
            agg.to_csv(OUT_DIR / "aggregate_entry_comparison.csv", index=False)

    print(f"[out] writing {out_xlsx}")
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
            if not recent_signals.empty:
                cols = [c for c in [
                    "symbol", "timestamp", "stock_regime", "prob",
                    "prev_close", "close", "ret_cc_pct"
                ] if c in recent_signals.columns]
                recent_signals[cols].to_excel(xw, sheet_name="Recent_Signals",
                                              index=False)
            if not timeline.empty:
                timeline.to_excel(xw, sheet_name="Daily_Timeline", index=False)
            if not agg.empty:
                agg.sort_values(["regime", "prob_bucket", "entry"]) \
                   .to_excel(xw, sheet_name="Entry_Comparison", index=False)
            if not per_sig.empty:
                # cap at 50k rows for excel, full data is in parquet
                per_sig.head(50_000).to_excel(xw, sheet_name="Per_Signal",
                                              index=False)
    except Exception as e:
        print(f"[warn] could not write xlsx: {e}")

    if not recent_signals.empty:
        recent_signals.to_csv(OUT_DIR / f"recent_signals_{latest_str}.csv",
                              index=False)
    if not timeline.empty:
        timeline.to_csv(OUT_DIR / f"daily_timeline_{latest_str}.csv", index=False)

    # Console teaser
    if not recent_signals.empty:
        print(f"\n=== Recent signals (prob>={args.prob_min}, last "
              f"{args.lookback_days} days)  -- top 20 by date / prob ===")
        print(recent_signals[["symbol", "timestamp", "stock_regime",
                              "prob", "close"]].head(20).to_string(index=False))

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
