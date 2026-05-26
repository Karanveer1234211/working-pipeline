"""
watchlist_followup.py
=====================
DAILY TOOL.  Run every morning (or whenever your panel updates).

For the last N (=10) trading days, find every (symbol, day) where the
model rates prob >= 0.70.  For every qualifying symbol, write a
per-day timeline including:
    * daily prob, regime, is_signal_day flag
    * OHLC, prev_close
    * close-to-close ret%, open-to-close ret%, gap%
    * 15-min ORB high / low for that day
    * a comment on the day if/when the ORB broke
    * intraday MAE / MFE from open

If a historical lookup CSV is present (produced by
`historical_entry_comparison.py`, run once and forgotten), this script
also enriches each live signal with the historical expected
MAE / MFE / TP-hit / SL-hit rates for that (regime, prob_bucket).

OUTPUT
------
<BASE_DIR>/watchlist_followup/
    watchlist_followup_<YYYYMMDD>.xlsx
    recent_signals_<YYYYMMDD>.csv
    daily_timeline_<YYYYMMDD>.csv

USAGE
-----
    python watchlist_followup.py
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
# CONFIG
# ============================================================================

BASE_DIR      = Path(r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full")
PANEL_PATH    = BASE_DIR / "panel_cache.parquet"
FEATURES_PATH = BASE_DIR / "features_train.json"
ROUTER_PATH   = BASE_DIR / "models" / "m5_regime_router.joblib"
INTRADAY_DIR  = Path(r"C:\Users\karanvsi\Desktop\Pycharm\Cache\intraday_5min")

OUT_DIR       = BASE_DIR / "watchlist_followup"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# If this file exists (built once by historical_entry_comparison.py), the
# script will join its rows onto each live signal so you can see the
# historically expected hit-rates next to today's pick.
HISTORICAL_LOOKUP_CSV = OUT_DIR / "aggregate_entry_comparison.csv"

IST = "Asia/Kolkata"

PROB_MIN          = 0.70
LOOKBACK_DAYS     = 10
SIGNAL_REGIMES    = ["bull_trend", "bear_trend"]

# 15-min ORB
ORB_END_TIME      = "09:30"
ORB_BARS_NEEDED   = 3            # 09:15, 09:20, 09:25 close out the 15-min window


# ============================================================================
# Intraday helpers
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


def _detect_orb_for_day(bars_full: pd.DataFrame, day: pd.Timestamp) -> Dict:
    """Compute ORB high/low and detect first close>ORB for one (symbol, day)."""
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

    out["orb_high"]       = float(orb_bars["high"].max())
    out["orb_low"]        = float(orb_bars["low"].min())
    out["intraday_open"]  = float(bars.iloc[0]["open"])
    out["intraday_close"] = float(bars.iloc[-1]["close"])
    out["intraday_high"]  = float(bars["high"].max())
    out["intraday_low"]   = float(bars["low"].min())

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
# Score panel for the recent window
# ============================================================================

def score_recent(prob_min: float, lookback_days: int) -> pd.DataFrame:
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
    cutoff = latest - pd.Timedelta(days=lookback_days * 2)  # buffer for non-trading days
    recent = panel[panel["timestamp"] >= cutoff].copy()

    schema = json.loads(FEATURES_PATH.read_text())
    FEATURES = schema["features"]
    IMPUTE = {k: float(v) for k, v in schema["impute"].items()}

    print(f"[score] scoring {len(recent):,} rows (last ~{lookback_days} trading days)...")
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
# Build watchlist + daily timeline
# ============================================================================

def build_watchlist(recent: pd.DataFrame,
                    prob_min: float,
                    lookback_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_days = sorted(recent["timestamp"].dt.normalize().unique())[-lookback_days:]
    window = recent[recent["timestamp"].dt.normalize().isin(all_days)].copy()

    qualifying = (window[window["prob"] >= prob_min]["symbol"].unique().tolist())
    print(f"[watchlist] {len(qualifying)} symbols hit prob>={prob_min} in the "
          f"last {lookback_days} trading days")

    if not qualifying:
        return pd.DataFrame(), pd.DataFrame()

    timeline = window[window["symbol"].isin(qualifying)].copy()

    print(f"[watchlist] detecting ORB breaks for {len(timeline)} cells...")
    seen_intraday: Dict[str, Optional[pd.DataFrame]] = {}
    orb_records = []
    empty_orb = {
        "orb_high": np.nan, "orb_low": np.nan,
        "broke": False, "break_time": None, "break_price": np.nan,
        "intraday_open": np.nan, "intraday_close": np.nan,
        "intraday_high": np.nan, "intraday_low": np.nan,
        "intraday_mae_from_open_pct": np.nan,
        "intraday_mfe_from_open_pct": np.nan,
    }
    for i, row in enumerate(timeline.itertuples(index=False), 1):
        if i % 200 == 0:
            print(f"  {i}/{len(timeline)}")
        sym = row.symbol
        if sym not in seen_intraday:
            seen_intraday[sym] = _load_intraday(sym)
        intra = seen_intraday[sym]
        if intra is None:
            orb_records.append(dict(empty_orb))
        else:
            orb_records.append(_detect_orb_for_day(intra, row.timestamp))
    orb_df = pd.DataFrame(orb_records)
    timeline = pd.concat([timeline.reset_index(drop=True), orb_df], axis=1)

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

    timeline["orb_comment"]   = timeline.apply(_comment, axis=1)
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
# Optional enrichment from historical lookup (if file exists)
# ============================================================================

def enrich_with_historical(recent_signals: pd.DataFrame) -> pd.DataFrame:
    if not HISTORICAL_LOOKUP_CSV.exists():
        print(f"[enrich] no historical lookup at {HISTORICAL_LOOKUP_CSV}; "
              f"run historical_entry_comparison.py once to enable")
        return recent_signals
    print(f"[enrich] joining historical expectations from {HISTORICAL_LOOKUP_CSV}")
    hist = pd.read_csv(HISTORICAL_LOOKUP_CSV)
    if hist.empty or "entry" not in hist.columns:
        return recent_signals
    # Use the entry='open' rows as the headline expectation
    h = hist[hist["entry"] == "open"].copy()
    keep = ["regime", "prob_bucket", "n",
            "mean_mae_pct", "mean_mfe_pct",
            "mean_t5_ret_pct", "mean_t5_ret_net_pct",
            "hit_tp2_pct", "hit_tp3_pct", "hit_tp4_pct", "hit_tp5_pct",
            "hit_sl3_pct", "hit_sl5_pct"]
    keep = [c for c in keep if c in h.columns]
    h = h[keep].rename(columns={c: f"hist_open_{c}" for c in keep
                                if c not in ("regime", "prob_bucket")})

    df = recent_signals.copy()
    df["prob_bucket"] = pd.cut(
        df["prob"],
        bins=[0.70, 0.75, 0.85, 1.01],
        right=False,
        labels=["[0.70, 0.75)", "[0.75, 0.85)", "[0.85, 1.01)"],
    ).astype(str)
    return df.merge(h, left_on=["stock_regime", "prob_bucket"],
                    right_on=["regime", "prob_bucket"], how="left") \
             .drop(columns=["regime"], errors="ignore")


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob-min", type=float, default=PROB_MIN)
    ap.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    args = ap.parse_args()

    recent = score_recent(args.prob_min, args.lookback_days)
    recent_signals, timeline = build_watchlist(recent,
                                               args.prob_min,
                                               args.lookback_days)
    recent_signals = enrich_with_historical(recent_signals)

    latest_str = recent["timestamp"].max().strftime("%Y%m%d")
    out_xlsx   = OUT_DIR / f"watchlist_followup_{latest_str}.xlsx"

    print(f"[out] writing {out_xlsx}")
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
            if not recent_signals.empty:
                recent_signals.to_excel(xw, sheet_name="Recent_Signals",
                                        index=False)
            if not timeline.empty:
                timeline.to_excel(xw, sheet_name="Daily_Timeline", index=False)
    except Exception as e:
        print(f"[warn] could not write xlsx: {e}")

    if not recent_signals.empty:
        recent_signals.to_csv(OUT_DIR / f"recent_signals_{latest_str}.csv",
                              index=False)
    if not timeline.empty:
        timeline.to_csv(OUT_DIR / f"daily_timeline_{latest_str}.csv", index=False)

    if not recent_signals.empty:
        print(f"\n=== Recent signals (prob>={args.prob_min}, last "
              f"{args.lookback_days} days)  -- top 20 by date / prob ===")
        show = ["symbol", "timestamp", "stock_regime", "prob", "close"]
        print(recent_signals[show].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
