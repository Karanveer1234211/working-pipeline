"""
hybrid_entry.py
===============
Single live strategy module that routes each signal through the right
entry rule based on its model-probability bucket. Wraps the validated
finding from entry_strategy_comparison.py:

    prob in [0.70, 0.75)  ->  ORB-15 held-anyday filter
                              (skip unless any 5-min bar in T+1..T+5
                               closes above the static T+1 ORB-15 high
                               AND that same day's last bar also closes
                               above it; entry at the breakout bar close)

    prob in [0.75, 0.85)  ->  buy at T+1 first 5-min open
    prob in [0.85, 1.01)  ->  buy at T+1 first 5-min open

Hold to T+5 close. Costs = 0.35 % round trip (matches the rest of the
pipeline). Universe filter: bull_trend or bear_trend, close >= 2,
avg20_vol >= 200_000 (live-trading version).

Two modes
---------
--mode live      (default) Score the latest panel, emit a routing
                 sheet listing every signal in lookback_days with its
                 bucket, rule, action and the parameters the desk
                 needs to act on it tomorrow morning.

--mode backtest  Replay every signal in the historical window, simulate
                 each per its bucket's rule, write per-trade parquet
                 plus a Bucket_Compare tab so you can verify the hybrid
                 reproduces the +6.01 / +5.14 net edge you observed.

Outputs
-------
<BASE_DIR>/hybrid_entry/
    routing_<YYYYMMDD>.csv               (live mode)
    routing_<YYYYMMDD>.xlsx              (live mode)
    per_trade_<YYYYMMDD>.parquet         (backtest mode)
    summary_<YYYYMMDD>.csv               (backtest mode)
    summary_<YYYYMMDD>.xlsx              (backtest mode, multi-tab)

Usage
-----
    python hybrid_entry.py
    python hybrid_entry.py --mode live --prob-min 0.70 --lookback-days 10
    python hybrid_entry.py --mode backtest --historical-years 3
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse all the canonical helpers — same identifiers, same tz, same IO.
import watchlist_followup as WF

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIG  -- mirrors watchlist_followup.py / orb_execution_quality.py
# ============================================================================

BASE_DIR     = WF.BASE_DIR
INTRADAY_DIR = WF.INTRADAY_DIR
IST          = WF.IST

OUT_DIR = BASE_DIR / "hybrid_entry"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Defaults (overridable from CLI)
PROB_MIN          = 0.70
LOOKBACK_DAYS     = 10
HOLD_DAYS         = 5
HISTORICAL_YEARS  = 3.0

ORB_END_TIME    = WF.ORB_END_TIME       # "09:30"
ORB_BARS_NEEDED = WF.ORB_BARS_NEEDED    # 3

TOTAL_COST_PCT  = WF.TOTAL_COST_PCT     # 0.35

# Costs are split per-side for non-triggered ORB skip slots (zero exposure
# means zero cost); for taken trades we charge the full round-trip on the
# realized return.

# ---- the routing rule the comparison study selected -----------------------

BUCKETS: List[Dict] = [
    {"lo": 0.70, "hi": 0.75, "label": "[0.70, 0.75)", "rule": "orb_15_held_anyday"},
    {"lo": 0.75, "hi": 0.85, "label": "[0.75, 0.85)", "rule": "naive_open"},
    {"lo": 0.85, "hi": 1.01, "label": "[0.85, 1.01)", "rule": "naive_open"},
]

# Sentinel rule for prob below the lowest bucket (we should not be trading
# these but the router may emit them under --prob-min < 0.70; flag clearly).
RULE_NONE = "no_trade"


def route_prob(prob: float) -> Tuple[str, str]:
    """Return (bucket_label, rule_name) for a given probability."""
    if not np.isfinite(prob):
        return ("nan", RULE_NONE)
    for b in BUCKETS:
        if b["lo"] <= prob < b["hi"]:
            return (b["label"], b["rule"])
    if prob < BUCKETS[0]["lo"]:
        return ("<0.70", RULE_NONE)
    return (">=1.01", RULE_NONE)


# ============================================================================
# Backtest simulation primitives
# ============================================================================

def _eod_close_above(day_bars: pd.DataFrame, level: float) -> bool:
    """True if the last 5-min bar of `day_bars` closes above `level`."""
    if day_bars.empty or not np.isfinite(level):
        return False
    return float(day_bars.iloc[-1]["close"]) > level


def _simulate_naive_open(intra: pd.DataFrame, sig_date: pd.Timestamp,
                         hold_days: int) -> Optional[Dict]:
    """Buy at T+1 first 5-min open; hold to T+(hold_days) close."""
    sig_norm = sig_date.normalize()
    after = WF._market_hours(
        intra[intra["timestamp"].dt.normalize() > sig_norm]
    ).sort_values("timestamp").reset_index(drop=True)
    if after.empty:
        return None
    days = sorted(after["timestamp"].dt.normalize().unique())
    if len(days) < hold_days:
        return None

    t1_bars = after[after["timestamp"].dt.normalize() == days[0]]
    if t1_bars.empty:
        return None
    entry_px = float(t1_bars.iloc[0]["open"])

    fwd = after[after["timestamp"].dt.normalize().isin(days[:hold_days])] \
            .reset_index(drop=True)
    exit_px = float(fwd.iloc[-1]["close"])

    highs = fwd["high"].values.astype(float)
    lows  = fwd["low"].values.astype(float)
    mfe = (highs.max() / entry_px - 1.0) * 100.0
    mae = (lows.min()  / entry_px - 1.0) * 100.0
    ret = (exit_px / entry_px - 1.0) * 100.0

    return {
        "triggered": True,
        "rule": "naive_open",
        "entry_day": pd.Timestamp(days[0]),
        "entry_time": t1_bars.iloc[0]["timestamp"],
        "entry_px": entry_px,
        "exit_day": pd.Timestamp(days[hold_days - 1]),
        "exit_px": exit_px,
        "ret_pct": ret,
        "ret_net_pct": ret - TOTAL_COST_PCT,
        "mae_pct": mae,
        "mfe_pct": mfe,
        "orh_t1": np.nan,
    }


def _simulate_orb_15_held_anyday(intra: pd.DataFrame, sig_date: pd.Timestamp,
                                 hold_days: int) -> Optional[Dict]:
    """
    Static T+1 ORB-15 high. For each day in T+1..T+(hold_days):
      - find the first 5-min bar whose close > ORH
      - if that day's LAST 5-min bar also closes > ORH, take entry at the
        breakout bar's close (hold-into-EOD confirms the breakout)
    First qualifying day wins. Skip if no day qualifies.
    """
    sig_norm = sig_date.normalize()
    after = WF._market_hours(
        intra[intra["timestamp"].dt.normalize() > sig_norm]
    ).sort_values("timestamp").reset_index(drop=True)
    if after.empty:
        return None
    days = sorted(after["timestamp"].dt.normalize().unique())
    if len(days) < hold_days:
        return None

    t1 = days[0]
    t1_bars = after[after["timestamp"].dt.normalize() == t1].reset_index(drop=True)
    if t1_bars.empty:
        return None

    cutoff_t1 = WF._orb_cutoff(t1)
    orb_bars = t1_bars[t1_bars["timestamp"] <= cutoff_t1]
    if len(orb_bars) < ORB_BARS_NEEDED:
        return None
    orh = float(orb_bars["high"].max())

    entry_idx_global = -1
    entry_px = np.nan
    entry_day = None
    entry_time = None

    for d in days[:hold_days]:
        d_bars = after[after["timestamp"].dt.normalize() == d].reset_index(drop=True)
        if d_bars.empty:
            continue
        # On T+1, the breakout has to come AFTER the ORB window;
        # other days, any bar in regular hours is eligible.
        if d == t1:
            scan = d_bars[d_bars["timestamp"] > cutoff_t1].reset_index(drop=True)
        else:
            scan = d_bars
        # Hold-into-EOD confirmation: last bar of the day must close above.
        if not _eod_close_above(d_bars, orh):
            continue
        hits = scan[scan["close"] > orh]
        if hits.empty:
            continue
        first = hits.iloc[0]
        entry_px   = float(first["close"])
        entry_day  = pd.Timestamp(d)
        entry_time = first["timestamp"]
        # Locate the global index in `after` for forward path slicing
        entry_idx_global = int(after.index[after["timestamp"] == first["timestamp"]][0])
        break

    if entry_idx_global < 0:
        return {
            "triggered": False,
            "rule": "orb_15_held_anyday",
            "entry_day": None, "entry_time": None,
            "entry_px": np.nan, "exit_day": None, "exit_px": np.nan,
            "ret_pct": np.nan, "ret_net_pct": np.nan,
            "mae_pct": np.nan, "mfe_pct": np.nan,
            "orh_t1": orh,
        }

    # Hold from entry to T+(hold_days) EOD
    hold_end = days[hold_days - 1]
    fwd = after.iloc[entry_idx_global:]
    fwd = fwd[fwd["timestamp"].dt.normalize() <= hold_end].reset_index(drop=True)
    if fwd.empty:
        return None
    exit_px = float(fwd.iloc[-1]["close"])
    highs = fwd["high"].values.astype(float)
    lows  = fwd["low"].values.astype(float)
    mfe = (highs.max() / entry_px - 1.0) * 100.0
    mae = (lows.min()  / entry_px - 1.0) * 100.0
    ret = (exit_px / entry_px - 1.0) * 100.0

    return {
        "triggered": True,
        "rule": "orb_15_held_anyday",
        "entry_day": entry_day,
        "entry_time": entry_time,
        "entry_px": entry_px,
        "exit_day": pd.Timestamp(hold_end),
        "exit_px": exit_px,
        "ret_pct": ret,
        "ret_net_pct": ret - TOTAL_COST_PCT,
        "mae_pct": mae,
        "mfe_pct": mfe,
        "orh_t1": orh,
    }


_RULE_DISPATCH = {
    "naive_open":          _simulate_naive_open,
    "orb_15_held_anyday":  _simulate_orb_15_held_anyday,
}


# ============================================================================
# Backtest driver
# ============================================================================

def backtest_hybrid(signals: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    """
    For each signal row, dispatch to the rule for its prob bucket and
    simulate. Returns a per-trade DataFrame.

    `signals` schema (matches watchlist_followup.score_recent output):
        symbol, timestamp, stock_regime, prob, close (= signal-day close)
    """
    if signals.empty:
        return pd.DataFrame()

    rows: List[Dict] = []
    skipped_no_data = 0
    intra_cache: Dict[str, Optional[pd.DataFrame]] = {}

    sigs = signals.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    n = len(sigs)
    print(f"[backtest] simulating {n:,} signals...")

    for i, sig in enumerate(sigs.itertuples(index=False), start=1):
        if i % 500 == 0:
            print(f"  {i:,}/{n:,}")

        prob = float(getattr(sig, "prob"))
        bucket_label, rule = route_prob(prob)
        if rule == RULE_NONE:
            continue

        sym = sig.symbol
        if sym not in intra_cache:
            intra_cache[sym] = WF._load_intraday(sym)
        intra = intra_cache[sym]
        if intra is None or intra.empty:
            skipped_no_data += 1
            continue

        sim = _RULE_DISPATCH[rule](intra, pd.Timestamp(sig.timestamp), hold_days)
        if sim is None:
            skipped_no_data += 1
            continue

        rows.append({
            "symbol":       sym,
            "signal_date":  pd.Timestamp(sig.timestamp),
            "regime":       getattr(sig, "stock_regime"),
            "prob":         prob,
            "prob_bucket":  bucket_label,
            "rule":         rule,
            "prev_close":   float(getattr(sig, "close")),
            **sim,
        })

    if skipped_no_data:
        print(f"[backtest] skipped {skipped_no_data:,} signals "
              f"(missing or short intraday data)")

    return pd.DataFrame(rows)


# ============================================================================
# Backtest aggregation: reproduces the user's headline tables
# ============================================================================

def _portfolio_sharpe(rets_pct: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe of a daily-basket return series."""
    r = rets_pct.dropna() / 100.0
    if r.std(ddof=1) == 0 or len(r) < 2:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def _max_drawdown_pct(rets_pct: pd.Series) -> float:
    """Max drawdown of cumulative product of (1 + r/100), reported as %."""
    r = rets_pct.dropna() / 100.0
    if r.empty:
        return float("nan")
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1.0)
    return float(dd.min() * 100.0)


def _bucket_metrics(df: pd.DataFrame, label: str) -> Dict:
    """Single-row summary for a (regime, bucket) cell."""
    n = len(df)
    if n == 0:
        return {"label": label, "n": 0}

    taken = df[df["triggered"] == True]            # noqa: E712
    n_taken = len(taken)
    n_skipped = n - n_taken
    trigger_rate = (n_taken / n) * 100.0 if n else 0.0

    # Per-trade stats on triggered subset
    if n_taken:
        mean_ret_taken     = taken["ret_pct"].mean()
        mean_ret_net_taken = taken["ret_net_pct"].mean()
        mean_mae_taken     = taken["mae_pct"].mean()
        mean_mfe_taken     = taken["mfe_pct"].mean()
        hit_sl3_taken_pct  = (taken["mae_pct"] <= -3.0).mean() * 100.0
    else:
        mean_ret_taken = mean_ret_net_taken = float("nan")
        mean_mae_taken = mean_mfe_taken = float("nan")
        hit_sl3_taken_pct = float("nan")

    # Portfolio-level: per signal (skipped slots = 0% return for ORB-filter
    # buckets; for naive buckets every signal is taken so taken == all)
    per_sig_ret = df["ret_net_pct"].fillna(0.0)
    portfolio_mean = per_sig_ret.mean()

    # Daily-basket aggregation for Sharpe and max DD
    daily = (df.assign(_day=df["entry_day"].fillna(df["signal_date"])
                                            .dt.normalize())
                .groupby("_day")["ret_net_pct"].mean())
    sharpe = _portfolio_sharpe(daily)
    max_dd = _max_drawdown_pct(daily)

    return {
        "label":              label,
        "n":                  n,
        "n_taken":            n_taken,
        "n_skipped":          n_skipped,
        "trigger_pct":        round(trigger_rate, 2),
        "mean_ret_taken":     mean_ret_taken,
        "mean_ret_net_taken": mean_ret_net_taken,
        "mean_mae_taken":     mean_mae_taken,
        "mean_mfe_taken":     mean_mfe_taken,
        "hit_sl3_taken_pct":  hit_sl3_taken_pct,
        "portfolio_mean_per_sig":  portfolio_mean,
        "portfolio_sharpe":   sharpe,
        "portfolio_max_dd_pct": max_dd,
    }


def summarize(per_trade: pd.DataFrame) -> pd.DataFrame:
    """Per (regime, prob_bucket) summary -- the table you've been reading."""
    if per_trade.empty:
        return pd.DataFrame()
    out = []
    for (reg, bk), g in per_trade.groupby(["regime", "prob_bucket"]):
        row = _bucket_metrics(g, f"{reg} {bk}")
        row["regime"]       = reg
        row["prob_bucket"]  = bk
        row["rule"]         = g["rule"].iloc[0]
        out.append(row)
    df = pd.DataFrame(out)
    cols = ["regime", "prob_bucket", "rule"] + \
           [c for c in df.columns if c not in ("regime", "prob_bucket", "rule", "label")]
    return df[cols].sort_values(["regime", "prob_bucket"]).reset_index(drop=True)


# ============================================================================
# Live routing
# ============================================================================

def build_routing_sheet(signals: pd.DataFrame) -> pd.DataFrame:
    """
    For every signal in `signals`, emit one row describing:
      - which bucket it falls into
      - the rule the module would apply
      - the action language the desk needs

    The T+1 ORB-15 high cannot be known until 09:30 IST tomorrow, so for
    ORB-routed names we emit instructions, not a price. The desk computes
    ORH from the live tape, then the rule check ('any 5-min bar in T+1..T+5
    closes above ORH AND that day's last bar also closes above') is exactly
    what the backtest validates.
    """
    if signals.empty:
        return pd.DataFrame()

    rows: List[Dict] = []
    for sig in signals.itertuples(index=False):
        prob = float(getattr(sig, "prob"))
        bucket_label, rule = route_prob(prob)

        if rule == "naive_open":
            action = "BUY at T+1 first 5-min open; hold to T+5 close"
            entry_note = "market order at 09:15 open"
        elif rule == "orb_15_held_anyday":
            action = ("WATCH ORB-15 T+1..T+5; enter on any 5-min close "
                      "above T+1 ORB high IF that day's last bar also "
                      "closes above; hold to T+5 close")
            entry_note = ("ORH = max(high) over T+1 09:15-09:30; "
                          "confirm-into-EOD before entering")
        else:
            action = "NO TRADE (prob outside hybrid buckets)"
            entry_note = ""

        rows.append({
            "symbol":       sig.symbol,
            "signal_date":  pd.Timestamp(sig.timestamp),
            "stock_regime": getattr(sig, "stock_regime"),
            "prob":         round(prob, 4),
            "prob_bucket":  bucket_label,
            "rule":         rule,
            "action":       action,
            "entry_note":   entry_note,
            "prev_close":   float(getattr(sig, "close")),
            "hold_days":    HOLD_DAYS,
            "stop_pct":     -3.0,           # informational; matches diagnostic table
            "cost_pct":     TOTAL_COST_PCT,
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["signal_date", "prob"], ascending=[False, False]) \
             .reset_index(drop=True)


# ============================================================================
# MAIN
# ============================================================================

def _stamp(latest: pd.Timestamp) -> str:
    return pd.Timestamp(latest).strftime("%Y%m%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "backtest"], default="live")
    ap.add_argument("--prob-min", type=float, default=PROB_MIN,
                    help="lowest prob to consider a signal (default 0.70)")
    ap.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS,
                    help="(live) how many recent trading days to scan")
    ap.add_argument("--hold-days", type=int, default=HOLD_DAYS)
    ap.add_argument("--historical-years", type=float, default=HISTORICAL_YEARS,
                    help="(backtest) how many years back to scan")
    args = ap.parse_args()

    if args.mode == "live":
        recent = WF.score_recent(args.prob_min, args.lookback_days,
                                 historical_years=0.0)
        signals = recent[recent["prob"] >= args.prob_min].copy()
        if signals.empty:
            print("[live] no signals at or above prob_min in the lookback window")
            return

        routing = build_routing_sheet(signals)
        stamp = _stamp(recent["timestamp"].max())
        csv_path  = OUT_DIR / f"routing_{stamp}.csv"
        xlsx_path = OUT_DIR / f"routing_{stamp}.xlsx"
        routing.to_csv(csv_path, index=False)
        print(f"[out] wrote {csv_path}")
        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
                routing.to_excel(xw, sheet_name="Routing", index=False)
                # Bucket-counts summary tab
                counts = (routing.groupby(["stock_regime", "prob_bucket", "rule"])
                                  .size().rename("n").reset_index())
                counts.to_excel(xw, sheet_name="Bucket_Counts", index=False)
            print(f"[out] wrote {xlsx_path}")
        except Exception as e:
            print(f"[warn] could not write xlsx: {e}")

        # Console teaser
        print("\n=== Routing decisions (top 20 by date / prob) ===")
        show = routing[["symbol", "signal_date", "stock_regime",
                        "prob", "prob_bucket", "rule", "action"]].head(20)
        print(show.to_string(index=False))

        print("\n=== Bucket counts ===")
        print(routing.groupby(["stock_regime", "prob_bucket", "rule"])
                     .size().rename("n").reset_index().to_string(index=False))
        return

    # ---- backtest ---------------------------------------------------------
    recent = WF.score_recent(args.prob_min, args.lookback_days,
                             historical_years=args.historical_years)
    signals = recent[recent["prob"] >= args.prob_min].copy()
    if signals.empty:
        print("[backtest] no signals in window")
        return

    per_trade = backtest_hybrid(signals, args.hold_days)
    summary   = summarize(per_trade)

    stamp = _stamp(recent["timestamp"].max())
    pt_path  = OUT_DIR / f"per_trade_{stamp}.parquet"
    csv_path = OUT_DIR / f"summary_{stamp}.csv"
    xlsx_path = OUT_DIR / f"summary_{stamp}.xlsx"

    if not per_trade.empty:
        per_trade.to_parquet(pt_path, index=False)
        print(f"[out] wrote {pt_path}  ({len(per_trade):,} rows)")
    if not summary.empty:
        summary.to_csv(csv_path, index=False)
        print(f"[out] wrote {csv_path}")

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
            if not summary.empty:
                summary.to_excel(xw, sheet_name="Bucket_Compare", index=False)
            if not per_trade.empty:
                per_trade.head(50_000).to_excel(xw, sheet_name="Per_Trade",
                                                index=False)
        print(f"[out] wrote {xlsx_path}")
    except Exception as e:
        print(f"[warn] could not write xlsx: {e}")

    # Console teaser
    if not summary.empty:
        print("\n=== Hybrid bucket comparison ===")
        cols = ["regime", "prob_bucket", "rule", "n", "n_taken", "trigger_pct",
                "mean_ret_net_taken", "hit_sl3_taken_pct",
                "portfolio_mean_per_sig", "portfolio_sharpe",
                "portfolio_max_dd_pct"]
        cols = [c for c in cols if c in summary.columns]
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
