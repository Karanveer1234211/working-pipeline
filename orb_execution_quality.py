"""
orb_execution_quality.py
========================
Execution-quality backtest for the >=0.65-probability watchlist.

WHAT THIS ANSWERS
-----------------
1. If we keep only stocks the model rates >= 65% on day T, and we wait for an
   ORB break on T+1..T+N (any day), how do those trades actually behave?
2. For each trade we record:
     * MAE  (max adverse excursion in % from entry, over hold window)
     * MFE  (max favorable excursion)
     * Whether each TP in {+1, +2, +3, +5, +7, +10} % and each SL in
       {-1, -2, -3, -5, -range_low, -2x range, -ATR(14)} fired first (OCO).
     * Net return after costs.
3. "If the watchlist signal is at 100 but ORB breaks at 108, is it still
   worth taking?"  -> answered by the slippage bucket pivot:
     slippage_from_signal_pct = (orb_entry_price / prev_close_at_T0 - 1) * 100
   We bucket every triggered trade by slippage and report median net return
   per bucket, broken down by (probability bucket x regime).
4. "Is the ORB wait even worth it?"  -> we simulate, on the SAME signals, a
   naive next-open-to-T+5-close entry, with the same TP/SL ladder, and report
   the lift  (orb_net  -  naive_net).
5. The bear_trend >= 0.70 edge from the four-regime pivot is preserved as its
   own column / filter so we can confirm execution is genuinely secondary
   for that bucket (median ~ +2%) and only matters near 0.65.

INPUTS
------
The script reads the SAME `extracted_paths_v2.parquet` that
`orb_machine_v2_updated.py` produces, so you do NOT need to re-fetch any
intraday data. If that parquet does not exist yet, run the v2 ORB extractor
once with SIGNAL_PROB_MIN lowered to 0.65 (override below) and it will be
created.

OUTPUTS  (under <BASE_DIR>/orb_execution_quality/)
--------
  per_trade.parquet                 - one row per (signal, variant) outcome
  01_trigger_rate.csv               - % of signals that ever break ORB
  02_tp_sl_ladder.csv               - OCO hit rates for every (TP, SL) combo
  03_mae_mfe_distribution.csv       - MAE / MFE percentiles
  04_orb_vs_naive_lift.csv          - lift of waiting for ORB vs T+1 open
  05_slippage_analysis.csv          - return as a function of paying up
  06_per_regime_summary.csv         - bear_trend >=0.70 vs bull_trend etc.
  summary.xlsx                      - all of the above as tabs

USAGE
-----
    python orb_execution_quality.py                # run with defaults
    python orb_execution_quality.py --reuse-v2     # use existing extraction

NOTE
----
This file matches the data conventions of `orb_machine_v2_updated.py`:
  - 5-min intraday parquet at INTRADAY_DIR / "<symbol>.parquet"
  - panel_cache.parquet, features_train.json, m5_regime_router.joblib
  - identical bar arrays:  bar_timestamps / bar_opens / bar_highs /
    bar_lows / bar_closes / bar_days   (bar_days is the trading-day index
    where 0 = T+1 entry day, 1 = T+2, ...).
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG  (mirror orb_machine_v2_updated.py)
# =============================================================================

BASE_DIR = Path(r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full")
PANEL_PATH = BASE_DIR / "panel_cache.parquet"
FEATURES_PATH = BASE_DIR / "features_train.json"
ROUTER_PATH = BASE_DIR / "models" / "m5_regime_router.joblib"
INTRADAY_DIR = Path(r"C:\Users\karanvsi\Desktop\Pycharm\Cache\intraday_5min")

V2_PATHS_FILE = BASE_DIR / "orb_machine_results_v2" / "extracted_paths_v2.parquet"

OUT_DIR = BASE_DIR / "orb_execution_quality"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IST = "Asia/Kolkata"

# ---- signal filter ----------------------------------------------------------
SIGNAL_PROB_MIN = 0.65          # the user asked for >=65 %
SIGNAL_REGIMES = ["bull_trend", "bear_trend"]
PROB_BUCKETS = [(0.65, 0.70), (0.70, 0.75), (0.75, 0.85), (0.85, 1.01)]

# ---- holding / extraction window -------------------------------------------
HOLDING_DAYS = 5      # how long we keep a position alive
EXTENDED_DAYS = 10    # extra bars for rolling exits / late MFE/MAE

# ---- costs ------------------------------------------------------------------
COST_BPS_ROUND_TRIP = 25
SLIPPAGE_BPS_PER_SIDE = 5
TOTAL_COST_PCT = (COST_BPS_ROUND_TRIP + 2 * SLIPPAGE_BPS_PER_SIDE) / 100.0  # ~0.35 %

# ---- ORB variants -----------------------------------------------------------
# small, focused catalog; covers the "any day on T+1..T+5" question the user
# asked about, plus a same-day baseline for comparison.

VARIANTS: Dict[str, Dict] = {
    "ORB_15_t1_only": {
        "label": "15-min range, close above, T+1 only",
        "range_def": "15min", "watch_mode": "t1_only", "confirm": "close_above",
    },
    "ORB_15_anyday_5d": {
        "label": "15-min range, close above, T+1..T+5 (static T+1 range)",
        "range_def": "15min", "watch_mode": "static_t1_5d", "confirm": "close_above",
    },
    "ORB_15_anyday_dyn": {
        "label": "15-min range, close above, T+1..T+5 (each day own range)",
        "range_def": "15min", "watch_mode": "dynamic_each_day", "confirm": "close_above",
    },
    "ORB_15_held_anyday": {
        "label": "15-min range, held-EOD, T+1..T+5 (static)",
        "range_def": "15min", "watch_mode": "static_t1_5d", "confirm": "close_held_eod",
    },
    "ORB_5_t1_only": {
        "label": "5-min range, close above, T+1 only",
        "range_def": "5min", "watch_mode": "t1_only", "confirm": "close_above",
    },
}

RANGE_DEFINITIONS = {
    "5min":  {"end_time": "09:20", "bars_needed": 1},
    "15min": {"end_time": "09:30", "bars_needed": 3},
    "30min": {"end_time": "09:45", "bars_needed": 6},
    "60min": {"end_time": "10:15", "bars_needed": 12},
}

# ---- TP / SL ladder ---------------------------------------------------------
TP_LEVELS_PCT = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
SL_LEVELS_PCT = [-1.0, -2.0, -3.0, -5.0]   # also: range_low and 2x range below

# =============================================================================
# SIGNAL GENERATION  (lower-threshold version of v2)
# =============================================================================

def generate_signals_lowthresh(prob_min: float = SIGNAL_PROB_MIN) -> pd.DataFrame:
    """Same as orb_machine_v2_updated.generate_signals() but with a configurable
    threshold and prob_bucket / cost-margin columns added up-front."""
    from joblib import load

    print(f"[signals] reading panel: {PANEL_PATH}")
    panel = pd.read_parquet(PANEL_PATH)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    if panel["timestamp"].dt.tz is None:
        panel["timestamp"] = panel["timestamp"].dt.tz_localize(IST)

    if "stock_regime" not in panel.columns:
        raise SystemExit("FATAL: panel missing stock_regime")

    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    panel["avg20_vol"] = (
        panel.groupby("symbol")["volume"]
             .transform(lambda s: s.rolling(20, min_periods=1).mean())
    )
    panel = panel[(panel["close"] >= 2.0) & (panel["avg20_vol"] >= 200_000)]
    panel = panel[panel["stock_regime"].isin(SIGNAL_REGIMES)].reset_index(drop=True)

    schema = json.loads(FEATURES_PATH.read_text())
    FEATURES = schema["features"]
    IMPUTE = {k: float(v) for k, v in schema["impute"].items()}

    print(f"[signals] scoring {len(panel):,} rows...")
    router = load(ROUTER_PATH)
    X = panel.reindex(columns=FEATURES).copy()
    for c in FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(IMPUTE.get(c, 0.0))
    panel["prob"] = router.predict_proba_by_regime(X, panel["stock_regime"])

    sigs = panel[panel["prob"] >= prob_min].copy()
    sigs = sigs[["symbol", "timestamp", "close", "prob", "stock_regime",
                 "high", "low", "open"]].copy()
    sigs.columns = ["symbol", "signal_date", "prev_close", "probability",
                    "regime", "prev_high", "prev_low", "prev_open"]
    sigs["prob_bucket"] = pd.cut(
        sigs["probability"],
        bins=[lo for lo, hi in PROB_BUCKETS] + [PROB_BUCKETS[-1][1]],
        right=False,
        labels=[f"[{lo:.2f},{hi:.2f})" for lo, hi in PROB_BUCKETS],
    ).astype(str)
    print(f"[signals] kept {len(sigs):,} signals  "
          f"(prob >= {prob_min:.2f}, regimes={SIGNAL_REGIMES})")
    return sigs

# =============================================================================
# REUSE  v2's path-extraction (it's already correct)
# =============================================================================

def load_v2_paths_or_extract(reuse_v2: bool, prob_min: float) -> pd.DataFrame:
    """
    Prefer the extracted_paths_v2.parquet that orb_machine_v2_updated.py
    already produces; fall back to running the same extractor here if needed.
    """
    if reuse_v2 and V2_PATHS_FILE.exists():
        print(f"[paths] reusing {V2_PATHS_FILE}")
        df = pd.read_parquet(V2_PATHS_FILE)
        # the v2 extractor was run with prob >= 0.75, so if we want lower we
        # need to extend.  Detect that case:
        if "probability" in df.columns:
            min_in_file = float(df["probability"].min())
            if min_in_file > prob_min + 1e-6:
                print(f"[paths] v2 file only has prob>={min_in_file:.2f}; "
                      f"need >={prob_min:.2f} -> extracting the missing rows")
                missing = generate_signals_lowthresh(prob_min)
                missing = missing[missing["probability"] < min_in_file]
                extra = _extract_paths(missing)
                df = pd.concat([df, extra], ignore_index=True)
        return df

    print("[paths] extracting fresh paths from intraday cache")
    sigs = generate_signals_lowthresh(prob_min)
    return _extract_paths(sigs)


def _extract_paths(signals: pd.DataFrame) -> pd.DataFrame:
    """Local copy of v2's extract_signal_paths. Identical bar schema."""
    from datetime import datetime as _dt

    def time_to_ts(date: pd.Timestamp, time_str: str) -> pd.Timestamp:
        h, m = map(int, time_str.split(":"))
        return date.replace(hour=h, minute=m, second=0, microsecond=0)

    def filter_market_hours(b: pd.DataFrame) -> pd.DataFrame:
        ts = b["timestamp"]
        return b[((ts.dt.hour > 9) | ((ts.dt.hour == 9) & (ts.dt.minute >= 15))) &
                ((ts.dt.hour < 15) | ((ts.dt.hour == 15) & (ts.dt.minute <= 30)))]

    def daily_bars(bars: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
        return bars[bars["timestamp"].dt.normalize() == day.normalize()]

    def load_bars(symbol: str, start, end) -> Optional[pd.DataFrame]:
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
        m = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        df = df[m].sort_values("timestamp").reset_index(drop=True)
        df = filter_market_hours(df)
        return df if not df.empty else None

    rows: List[Dict] = []
    n_total = len(signals)
    for i, sig in enumerate(signals.itertuples(index=False), 1):
        if i % 200 == 0:
            print(f"  {i}/{n_total}")
        sd = sig.signal_date
        bars = load_bars(sig.symbol,
                         sd,
                         sd + pd.Timedelta(days=EXTENDED_DAYS + 7))
        if bars is None or bars.empty:
            continue
        after = bars[bars["timestamp"].dt.normalize() > sd.normalize()]
        if after.empty:
            continue
        unique_days = sorted(after["timestamp"].dt.normalize().unique())
        if len(unique_days) < HOLDING_DAYS:
            continue
        ext_days = unique_days[: min(EXTENDED_DAYS, len(unique_days))]
        ext = bars[bars["timestamp"].dt.normalize().isin(ext_days)].reset_index(drop=True)
        entry_day = ext_days[0]
        e1 = daily_bars(ext, entry_day)
        if len(e1) < 50:
            continue
        rec: Dict = {
            "symbol": sig.symbol,
            "signal_date": sd,
            "entry_day": entry_day,
            "probability": sig.probability,
            "regime": sig.regime,
            "prev_close": sig.prev_close,
            "prob_bucket": sig.prob_bucket,
        }
        # daily levels + per-day OR ranges
        for d_idx, d in enumerate(ext_days[:HOLDING_DAYS], start=1):
            db = daily_bars(ext, d)
            if db.empty:
                continue
            rec[f"d{d_idx}_open"] = db.iloc[0]["open"]
            rec[f"d{d_idx}_high"] = db["high"].max()
            rec[f"d{d_idx}_low"] = db["low"].min()
            rec[f"d{d_idx}_close"] = db.iloc[-1]["close"]
            for rn, rd in RANGE_DEFINITIONS.items():
                rng = db[db["timestamp"] <= time_to_ts(d, rd["end_time"])]
                if len(rng) >= rd["bars_needed"]:
                    rec[f"d{d_idx}_orh_{rn}"] = rng["high"].max()
                    rec[f"d{d_idx}_orl_{rn}"] = rng["low"].min()
        for rn, rd in RANGE_DEFINITIONS.items():
            rng = e1[e1["timestamp"] <= time_to_ts(entry_day, rd["end_time"])]
            if len(rng) >= rd["bars_needed"]:
                rec[f"t1_orh_{rn}"] = rng["high"].max()
                rec[f"t1_orl_{rn}"] = rng["low"].min()
        ext = ext.sort_values("timestamp").reset_index(drop=True)
        d2i = {d.normalize(): i for i, d in enumerate(ext_days)}
        rec["bar_timestamps"] = ext["timestamp"].astype("int64").tolist()
        rec["bar_opens"] = ext["open"].tolist()
        rec["bar_highs"] = ext["high"].tolist()
        rec["bar_lows"] = ext["low"].tolist()
        rec["bar_closes"] = ext["close"].tolist()
        rec["bar_days"] = [d2i.get(t.normalize(), -1)
                           for t in ext["timestamp"]]
        rows.append(rec)
    return pd.DataFrame(rows)

# =============================================================================
# CORE: simulate one trade and capture every metric we care about
# =============================================================================

@dataclass
class TradeOutcome:
    triggered: bool = False
    entry_idx: int = -1
    entry_time: Optional[pd.Timestamp] = None
    entry_price: float = np.nan
    entry_day_offset: int = -1
    orh: float = np.nan
    orl: float = np.nan
    # raw forward path stats from entry to end of HOLDING_DAYS
    mae_pct: float = np.nan        # max adverse excursion
    mfe_pct: float = np.nan        # max favorable excursion
    fwd_return_to_t5_close_pct: float = np.nan
    # OCO ladder outcomes:  dict of (tp,sl) -> realized pct return
    oco: Optional[Dict[Tuple[float, float], Dict]] = None
    # naive entry comparison
    naive_entry_price: float = np.nan          # T+1 open
    naive_mae_pct: float = np.nan
    naive_mfe_pct: float = np.nan
    naive_fwd_t5_close_ret_pct: float = np.nan


def _find_breakout(bar_highs, bar_lows, bar_closes, bar_days,
                   bar_ts, variant, sig, range_bars_needed) -> Tuple[int, float, float, float]:
    """Return (entry_idx, entry_price, orh, orl) or (-1, ...)."""
    watch_mode = variant["watch_mode"]
    confirm = variant["confirm"]
    range_def = variant["range_def"]

    if watch_mode == "t1_only":
        watch_max = 1
        dynamic = False
    elif watch_mode == "static_t1_5d":
        watch_max = HOLDING_DAYS
        dynamic = False
    elif watch_mode == "dynamic_each_day":
        watch_max = HOLDING_DAYS
        dynamic = True
    else:
        return -1, np.nan, np.nan, np.nan

    if not dynamic:
        orh = sig.get(f"t1_orh_{range_def}")
        orl = sig.get(f"t1_orl_{range_def}")
        if pd.isna(orh) or pd.isna(orl) or orh <= orl:
            return -1, np.nan, np.nan, np.nan

    n = len(bar_ts)
    for i in range(n):
        d = int(bar_days[i])
        if d < 0 or d >= watch_max:
            continue
        if dynamic:
            day_label = d + 1
            day_orh = sig.get(f"d{day_label}_orh_{range_def}")
            day_orl = sig.get(f"d{day_label}_orl_{range_def}")
            if pd.isna(day_orh) or pd.isna(day_orl) or day_orh <= day_orl:
                continue
            same_day_before = int(np.sum(bar_days[:i] == d))
            if same_day_before < range_bars_needed:
                continue
            cur_h, cur_l = day_orh, day_orl
        else:
            if d == 0:
                same_day_before = int(np.sum(bar_days[:i] == 0))
                if same_day_before < range_bars_needed:
                    continue
            cur_h, cur_l = orh, orl

        triggered = False
        if confirm == "close_above":
            triggered = bar_closes[i] > cur_h
        elif confirm == "touch_above":
            triggered = bar_highs[i] > cur_h
        elif confirm == "close_held_eod":
            if bar_closes[i] > cur_h:
                same_day = np.where(bar_days == d)[0]
                if len(same_day) and bar_closes[same_day[-1]] > cur_h:
                    triggered = True

        if triggered:
            return i, float(bar_closes[i]), float(cur_h), float(cur_l)
    return -1, np.nan, np.nan, np.nan


def _walk_oco(bar_highs, bar_lows, bar_closes, bar_days,
              entry_idx: int, entry_price: float, orl: float,
              orh: float, holding_days: int) -> Tuple[Dict, float, float, float]:
    """
    Walk forward from entry_idx for `holding_days` trading days; for each
    (TP, SL) combo return:
      hit_tp / hit_sl / hit_neither, signed_ret_pct, bars_to_exit
    Also return the (raw, no-exit) MAE, MFE and T+5-close return from entry.
    """
    n = len(bar_highs)
    end_day = int(bar_days[entry_idx]) + holding_days
    raw_mae = 0.0   # most negative excursion (in %)
    raw_mfe = 0.0   # most positive excursion (in %)
    last_idx = entry_idx
    for j in range(entry_idx + 1, n):
        if bar_days[j] > end_day - 1:
            break
        last_idx = j
        lo_pct = (bar_lows[j] / entry_price - 1) * 100
        hi_pct = (bar_highs[j] / entry_price - 1) * 100
        raw_mae = min(raw_mae, lo_pct)
        raw_mfe = max(raw_mfe, hi_pct)
    fwd_to_t5_close = (bar_closes[last_idx] / entry_price - 1) * 100

    # OCO ladder: for every (tp, sl) check who fires first
    oco: Dict[Tuple[float, float], Dict] = {}
    sl_special = [("range_low", orl),
                  ("two_range", entry_price - 2 * (orh - orl))]
    sl_pct_levels = [(lvl, entry_price * (1 + lvl / 100)) for lvl in SL_LEVELS_PCT]
    sl_levels_all = sl_pct_levels + [(name, px) for name, px in sl_special if px < entry_price]

    for tp_pct in TP_LEVELS_PCT:
        tp_px = entry_price * (1 + tp_pct / 100)
        for sl_name, sl_px in sl_levels_all:
            hit_tp = False
            hit_sl = False
            exit_idx = last_idx
            for j in range(entry_idx + 1, last_idx + 1):
                # stop check first (conservative)
                if bar_lows[j] <= sl_px:
                    hit_sl = True
                    exit_idx = j
                    break
                if bar_highs[j] >= tp_px:
                    hit_tp = True
                    exit_idx = j
                    break
            if hit_tp:
                ret = tp_pct
            elif hit_sl:
                ret = (sl_px / entry_price - 1) * 100
            else:
                ret = fwd_to_t5_close
            oco[(tp_pct, sl_name)] = {
                "hit_tp": hit_tp, "hit_sl": hit_sl,
                "ret_pct": ret,
                "bars_to_exit": exit_idx - entry_idx,
            }
    return oco, raw_mae, raw_mfe, fwd_to_t5_close


def simulate_signal(sig: pd.Series, variant_id: str, variant: Dict) -> TradeOutcome:
    out = TradeOutcome()
    bts = sig.get("bar_timestamps")
    if not isinstance(bts, (list, np.ndarray)) or len(bts) < 20:
        return out
    bar_ts = pd.to_datetime(bts, unit="ns")
    bar_opens = np.asarray(sig["bar_opens"], dtype=float)
    bar_highs = np.asarray(sig["bar_highs"], dtype=float)
    bar_lows = np.asarray(sig["bar_lows"], dtype=float)
    bar_closes = np.asarray(sig["bar_closes"], dtype=float)
    bar_days = np.asarray(sig["bar_days"], dtype=int)

    rd = variant["range_def"]
    range_bars_needed = RANGE_DEFINITIONS[rd]["bars_needed"]

    # 1) ORB-based entry --------------------------------------------------------
    e_idx, e_px, orh, orl = _find_breakout(
        bar_highs, bar_lows, bar_closes, bar_days,
        bar_ts, variant, sig, range_bars_needed,
    )
    if e_idx >= 0:
        out.triggered = True
        out.entry_idx = e_idx
        out.entry_time = bar_ts[e_idx]
        out.entry_price = e_px
        out.entry_day_offset = int(bar_days[e_idx]) + 1
        out.orh, out.orl = orh, orl
        oco, mae, mfe, fwd = _walk_oco(
            bar_highs, bar_lows, bar_closes, bar_days,
            e_idx, e_px, orl, orh, HOLDING_DAYS,
        )
        out.oco = oco
        out.mae_pct = mae
        out.mfe_pct = mfe
        out.fwd_return_to_t5_close_pct = fwd

    # 2) Naive T+1-open benchmark on the same signal ----------------------------
    #    (so we can compute lift = orb_net - naive_net per trade)
    same_d1 = np.where(bar_days == 0)[0]
    if len(same_d1):
        n_idx = same_d1[0]
        n_px = float(bar_opens[n_idx])
        out.naive_entry_price = n_px
        _, n_mae, n_mfe, n_fwd = _walk_oco(
            bar_highs, bar_lows, bar_closes, bar_days,
            n_idx, n_px, n_px * 0.95, n_px * 1.05, HOLDING_DAYS,
        )
        out.naive_mae_pct = n_mae
        out.naive_mfe_pct = n_mfe
        out.naive_fwd_t5_close_ret_pct = n_fwd
    return out


# =============================================================================
# RUN ALL VARIANTS x ALL SIGNALS  ->  per_trade DataFrame
# =============================================================================

def run_all(paths_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    n = len(paths_df)
    for var_id, variant in VARIANTS.items():
        print(f"[sim] {var_id}: {variant['label']}")
        triggered_n = 0
        for i, sig in enumerate(paths_df.itertuples(index=False)):
            if i % 1000 == 0 and i:
                print(f"   {i}/{n}  triggered_so_far={triggered_n}")
            sig_d = sig._asdict()
            res = simulate_signal(pd.Series(sig_d), var_id, variant)
            base = {
                "variant": var_id,
                "symbol": sig_d["symbol"],
                "signal_date": sig_d["signal_date"],
                "regime": sig_d["regime"],
                "probability": sig_d["probability"],
                "prob_bucket": sig_d.get("prob_bucket", ""),
                "prev_close": sig_d["prev_close"],
                "triggered": res.triggered,
                "entry_day_offset": res.entry_day_offset,
                "entry_price": res.entry_price,
                "orh": res.orh, "orl": res.orl,
                "slippage_from_signal_pct":
                    (res.entry_price / sig_d["prev_close"] - 1) * 100
                    if res.triggered else np.nan,
                "mae_pct": res.mae_pct,
                "mfe_pct": res.mfe_pct,
                "fwd_return_to_t5_close_pct": res.fwd_return_to_t5_close_pct,
                "naive_entry_price": res.naive_entry_price,
                "naive_mae_pct": res.naive_mae_pct,
                "naive_mfe_pct": res.naive_mfe_pct,
                "naive_fwd_t5_close_ret_pct": res.naive_fwd_t5_close_ret_pct,
            }
            # flatten OCO ladder
            if res.oco:
                triggered_n += 1
                for (tp, sl), o in res.oco.items():
                    base[f"tp{tp:g}_sl{sl}_ret"] = o["ret_pct"]
                    base[f"tp{tp:g}_sl{sl}_hittp"] = int(o["hit_tp"])
                    base[f"tp{tp:g}_sl{sl}_hitsl"] = int(o["hit_sl"])
            rows.append(base)
        print(f"   trigger rate {var_id}: {triggered_n}/{n} = "
              f"{100*triggered_n/max(n,1):.1f}%")
    return pd.DataFrame(rows)

# =============================================================================
# AGGREGATIONS  (the actual answers to the user's questions)
# =============================================================================

def _net(x: pd.Series) -> pd.Series:
    """Apply round-trip cost (only to triggered trades)."""
    return x - TOTAL_COST_PCT


def trigger_rate(per_trade: pd.DataFrame) -> pd.DataFrame:
    g = (per_trade.groupby(["variant", "regime", "prob_bucket"])
                  .agg(n_signals=("triggered", "size"),
                       n_triggered=("triggered", "sum"))
                  .reset_index())
    g["trigger_rate_pct"] = 100 * g["n_triggered"] / g["n_signals"].clip(lower=1)
    return g


def tp_sl_ladder(per_trade: pd.DataFrame) -> pd.DataFrame:
    """
    For every (TP, SL) combo, hit-rates and median net return per
    (variant, regime, prob_bucket). Triggered trades only.
    """
    df = per_trade[per_trade["triggered"]].copy()
    out = []
    tp_cols = [c for c in df.columns
               if c.startswith("tp") and c.endswith("_ret")]
    for col in tp_cols:
        # parse  "tp3_sl-2.0_ret"  ->  tp=3, sl="-2.0"
        body = col[len("tp"):-len("_ret")]
        tp_str, sl_str = body.split("_sl", 1)
        tp = float(tp_str)
        sl = sl_str
        hittp_col = col.replace("_ret", "_hittp")
        hitsl_col = col.replace("_ret", "_hitsl")
        agg = (df.groupby(["variant", "regime", "prob_bucket"])
                 .agg(n=(col, "size"),
                      hit_tp_pct=(hittp_col, lambda x: 100*x.mean()),
                      hit_sl_pct=(hitsl_col, lambda x: 100*x.mean()),
                      median_gross_pct=(col, "median"),
                      mean_gross_pct=(col, "mean"))
                 .reset_index())
        agg["tp"] = tp
        agg["sl"] = sl
        agg["mean_net_pct"] = agg["mean_gross_pct"] - TOTAL_COST_PCT
        out.append(agg)
    res = pd.concat(out, ignore_index=True)
    return res[["variant", "regime", "prob_bucket", "tp", "sl",
                "n", "hit_tp_pct", "hit_sl_pct",
                "median_gross_pct", "mean_gross_pct", "mean_net_pct"]]


def mae_mfe_distribution(per_trade: pd.DataFrame) -> pd.DataFrame:
    df = per_trade[per_trade["triggered"]].copy()
    g = (df.groupby(["variant", "regime", "prob_bucket"])
           .agg(n=("mae_pct", "size"),
                mae_p25=("mae_pct", lambda s: np.nanpercentile(s, 25)),
                mae_p50=("mae_pct", lambda s: np.nanpercentile(s, 50)),
                mae_p75=("mae_pct", lambda s: np.nanpercentile(s, 75)),
                mfe_p25=("mfe_pct", lambda s: np.nanpercentile(s, 25)),
                mfe_p50=("mfe_pct", lambda s: np.nanpercentile(s, 50)),
                mfe_p75=("mfe_pct", lambda s: np.nanpercentile(s, 75)),
                fwd_t5_mean_pct=("fwd_return_to_t5_close_pct", "mean"),
                fwd_t5_median_pct=("fwd_return_to_t5_close_pct", "median"))
           .reset_index())
    g["fwd_t5_net_mean_pct"] = g["fwd_t5_mean_pct"] - TOTAL_COST_PCT
    return g


def orb_vs_naive_lift(per_trade: pd.DataFrame) -> pd.DataFrame:
    """Lift of waiting for ORB vs taking the naive T+1 open. Triggered only."""
    df = per_trade[per_trade["triggered"]].copy()
    df["lift_pct"] = (df["fwd_return_to_t5_close_pct"]
                      - df["naive_fwd_t5_close_ret_pct"])
    g = (df.groupby(["variant", "regime", "prob_bucket"])
           .agg(n=("lift_pct", "size"),
                lift_mean_pct=("lift_pct", "mean"),
                lift_median_pct=("lift_pct", "median"),
                orb_mean_pct=("fwd_return_to_t5_close_pct", "mean"),
                naive_mean_pct=("naive_fwd_t5_close_ret_pct", "mean"))
           .reset_index())
    return g


def slippage_analysis(per_trade: pd.DataFrame) -> pd.DataFrame:
    """
    *** This is the answer to "watchlist signal at 100 vs ORB break at 108". ***
    Bucket triggered trades by slippage_from_signal_pct and report mean net
    return; if the high-slippage buckets are still positive after costs the
    answer is "yes worth it"; if they cross zero the answer is "no".
    """
    df = per_trade[per_trade["triggered"]].copy()
    df["slip_bucket"] = pd.cut(
        df["slippage_from_signal_pct"],
        bins=[-1e9, 0, 1, 2, 3, 5, 8, 12, 1e9],
        labels=["<=0%", "0-1%", "1-2%", "2-3%", "3-5%", "5-8%", "8-12%", ">12%"],
    ).astype(str)
    g = (df.groupby(["variant", "regime", "prob_bucket", "slip_bucket"])
           .agg(n=("fwd_return_to_t5_close_pct", "size"),
                mean_gross_pct=("fwd_return_to_t5_close_pct", "mean"),
                median_gross_pct=("fwd_return_to_t5_close_pct", "median"),
                naive_mean_gross_pct=("naive_fwd_t5_close_ret_pct", "mean"))
           .reset_index())
    g["mean_net_pct"] = g["mean_gross_pct"] - TOTAL_COST_PCT
    g["lift_vs_naive_pct"] = (g["mean_gross_pct"]
                              - g["naive_mean_gross_pct"])
    return g


def per_regime_summary(per_trade: pd.DataFrame) -> pd.DataFrame:
    """
    Single table you can read top-to-bottom:
      bear_trend [0.65, 0.70)  -> n, hit_rates, mean_net
      bear_trend [0.70, 0.75)  -> ...
      bull_trend [0.65, 0.70)  -> ...
      ...
    Uses the (TP=3 %, SL=-3 %) row from the OCO ladder as a representative
    "balanced" exit; you can recompute with any other (tp, sl) you want.
    """
    df = per_trade[per_trade["triggered"]].copy()
    col_ret = "tp3_sl-3.0_ret"
    col_hittp = "tp3_sl-3.0_hittp"
    col_hitsl = "tp3_sl-3.0_hitsl"
    if col_ret not in df.columns:
        return pd.DataFrame()
    g = (df.groupby(["variant", "regime", "prob_bucket"])
           .agg(n=(col_ret, "size"),
                hit_tp3_pct=(col_hittp, lambda x: 100*x.mean()),
                hit_sl3_pct=(col_hitsl, lambda x: 100*x.mean()),
                mean_net_pct=(col_ret, lambda x: x.mean() - TOTAL_COST_PCT),
                median_pct=(col_ret, "median"),
                mean_mae_pct=("mae_pct", "mean"),
                mean_mfe_pct=("mfe_pct", "mean"),
                mean_slippage_from_signal_pct=("slippage_from_signal_pct", "mean"),
                lift_vs_naive_mean_pct=
                    ("fwd_return_to_t5_close_pct",
                     lambda s: (s.mean()
                                - df.loc[s.index, "naive_fwd_t5_close_ret_pct"].mean())))
           .reset_index())
    return g


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-v2", action="store_true",
                    help="reuse extracted_paths_v2.parquet from the v2 ORB run")
    ap.add_argument("--prob-min", type=float, default=SIGNAL_PROB_MIN)
    args = ap.parse_args()

    paths = load_v2_paths_or_extract(args.reuse_v2, args.prob_min)
    paths = paths[paths["probability"] >= args.prob_min].reset_index(drop=True)
    print(f"[main] {len(paths)} signals to simulate")

    per_trade = run_all(paths)
    pt_path = OUT_DIR / "per_trade.parquet"
    per_trade.to_parquet(pt_path, index=False)
    print(f"[out] {pt_path}")

    tab1 = trigger_rate(per_trade)
    tab2 = tp_sl_ladder(per_trade)
    tab3 = mae_mfe_distribution(per_trade)
    tab4 = orb_vs_naive_lift(per_trade)
    tab5 = slippage_analysis(per_trade)
    tab6 = per_regime_summary(per_trade)

    tab1.to_csv(OUT_DIR / "01_trigger_rate.csv", index=False)
    tab2.to_csv(OUT_DIR / "02_tp_sl_ladder.csv", index=False)
    tab3.to_csv(OUT_DIR / "03_mae_mfe_distribution.csv", index=False)
    tab4.to_csv(OUT_DIR / "04_orb_vs_naive_lift.csv", index=False)
    tab5.to_csv(OUT_DIR / "05_slippage_analysis.csv", index=False)
    tab6.to_csv(OUT_DIR / "06_per_regime_summary.csv", index=False)

    try:
        with pd.ExcelWriter(OUT_DIR / "summary.xlsx", engine="openpyxl") as xw:
            tab1.to_excel(xw, sheet_name="01_trigger_rate", index=False)
            tab2.to_excel(xw, sheet_name="02_tp_sl_ladder", index=False)
            tab3.to_excel(xw, sheet_name="03_mae_mfe", index=False)
            tab4.to_excel(xw, sheet_name="04_orb_vs_naive", index=False)
            tab5.to_excel(xw, sheet_name="05_slippage", index=False)
            tab6.to_excel(xw, sheet_name="06_regime", index=False)
    except Exception as e:
        print(f"[warn] could not write xlsx: {e}")

    # Quick teaser to stdout
    print("\n=== 06_per_regime_summary (head) ===")
    print(tab6.sort_values(["variant", "regime", "prob_bucket"]).to_string(index=False))
    print("\n=== 05_slippage_analysis (head 30) ===")
    print(tab5.sort_values(["variant", "regime", "prob_bucket", "slip_bucket"])
              .head(30).to_string(index=False))


if __name__ == "__main__":
    main()
