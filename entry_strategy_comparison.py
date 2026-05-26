"""
entry_strategy_comparison.py
============================
Apples-to-apples portfolio comparison of the two entry strategies on the
same signal universe (every prob >= PROB_MIN signal):

  A. NAIVE PORTFOLIO          : take every signal at T+1 open
  B. ORB-FILTERED PORTFOLIO   : take only the signals that break ORB on
                                T+1, at the breakout price; non-triggered
                                signals are skipped (cash for that slot)

This SUPERSEDES the lift metric in 04_orb_vs_naive_lift.csv.  That metric
was conditioned on triggered=True and therefore could only ever measure
the price of paying up for the breakout.  By design it could not see the
ORB selection benefit -- the trades ORB would have skipped were absent
from both sides of the subtraction.

What this script computes per (variant, regime, prob_bucket)
------------------------------------------------------------
Three blocks on the same N signals:

  NAIVE-ALL        : every signal, entry = T+1 open, exit = T+5 close
  NAIVE-on-TAKEN   : the subset where ORB triggered, naive entry
  NAIVE-on-SKIPPED : the subset where ORB did NOT trigger, naive entry
  ORB-TAKEN        : the triggered subset, entry = ORB break price

Per block we report:
  n, mean_ret, median_ret, win%, hit_sl3%, hit_sl5%, hit_tp3%, hit_tp5%,
  mean_mae, mean_mfe, daily-basket annualized Sharpe

Then the two derived diagnostics:
  selection_benefit = mean_ret(NAIVE-on-TAKEN) - mean_ret(NAIVE-on-SKIPPED)
                      (>0 means the skipped names were genuinely worse,
                       ORB filter has value)
  entry_cost        = mean_ret(NAIVE-on-TAKEN) - mean_ret(ORB-TAKEN)
                      (the old 'lift' metric, but correctly named)
  net_orb_advantage = selection_benefit - entry_cost
                      (>0 -> ORB-FILTERED beats NAIVE on the same universe)

And the headline portfolio-level metrics:
  naive_portfolio_sharpe vs orb_portfolio_sharpe
  naive_portfolio_max_dd vs orb_portfolio_max_dd
  naive_hit_sl3_pct      vs orb_hit_sl3_pct  (over the FULL universe;
                                              ORB skipped slots count as
                                              "no SL hit" since no exposure)

USAGE
-----
    python entry_strategy_comparison.py
    python entry_strategy_comparison.py --per-trade /path/to/per_trade.parquet

Output:
    <BASE_DIR>/orb_execution_quality/entry_strategy_comparison.csv
    <BASE_DIR>/orb_execution_quality/entry_strategy_summary.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full")
DEFAULT_PER_TRADE = BASE_DIR / "orb_execution_quality" / "per_trade.parquet"
OUT_DIR = BASE_DIR / "orb_execution_quality"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# round-trip cost in % (matches orb_execution_quality.py: 25 bps + 2*5 bps slip)
TOTAL_COST_PCT = 0.35

# Prob buckets to report (matches the existing module)
PROB_BUCKETS = [(0.65, 0.70), (0.70, 0.75), (0.75, 0.85), (0.85, 1.01)]


# ---------------------------------------------------------------------------
# Block-level metrics
# ---------------------------------------------------------------------------

def _block_metrics(df: pd.DataFrame, ret_col: str,
                   mae_col: Optional[str] = None,
                   mfe_col: Optional[str] = None,
                   prefix: str = "") -> Dict:
    """
    Summary stats for a slice of trades evaluated under one entry method.
    For SL/TP hits we use first-touch from MAE/MFE since the parquet stores
    those for both ORB and naive entries.
    """
    out: Dict[str, float] = {f"{prefix}n": int(len(df))}
    if df.empty:
        return out

    ret = pd.to_numeric(df[ret_col], errors="coerce")
    out[f"{prefix}mean_ret_pct"]   = float(ret.mean())
    out[f"{prefix}median_ret_pct"] = float(ret.median())
    out[f"{prefix}win_pct"]        = float(100 * (ret > 0).mean())
    out[f"{prefix}mean_net_pct"]   = float(ret.mean() - TOTAL_COST_PCT)

    if mae_col and mae_col in df.columns:
        mae = pd.to_numeric(df[mae_col], errors="coerce")
        out[f"{prefix}mean_mae_pct"]  = float(mae.mean())
        out[f"{prefix}hit_sl2_pct"]   = float(100 * (mae <= -2).mean())
        out[f"{prefix}hit_sl3_pct"]   = float(100 * (mae <= -3).mean())
        out[f"{prefix}hit_sl5_pct"]   = float(100 * (mae <= -5).mean())
        out[f"{prefix}hit_sl10_pct"]  = float(100 * (mae <= -10).mean())
    if mfe_col and mfe_col in df.columns:
        mfe = pd.to_numeric(df[mfe_col], errors="coerce")
        out[f"{prefix}mean_mfe_pct"]  = float(mfe.mean())
        out[f"{prefix}hit_tp2_pct"]   = float(100 * (mfe >= 2).mean())
        out[f"{prefix}hit_tp3_pct"]   = float(100 * (mfe >= 3).mean())
        out[f"{prefix}hit_tp5_pct"]   = float(100 * (mfe >= 5).mean())
        out[f"{prefix}hit_tp10_pct"]  = float(100 * (mfe >= 10).mean())
    return out


def _daily_sharpe(per_trade_subset: pd.DataFrame, ret_col: str,
                  date_col: str = "signal_date") -> float:
    """
    Aggregate per signal_date with equal weighting, then annualize.
    Returns NaN if the series is empty / degenerate.
    """
    if per_trade_subset.empty:
        return float("nan")
    s = (per_trade_subset
         .groupby(per_trade_subset[date_col].dt.normalize())[ret_col]
         .mean()
         .sort_index())
    if len(s) < 5 or s.std(ddof=0) == 0:
        return float("nan")
    return float(np.sqrt(252) * s.mean() / s.std(ddof=0))


def _max_drawdown(per_trade_subset: pd.DataFrame, ret_col: str,
                  date_col: str = "signal_date") -> float:
    """
    Cumulative (compounded) per-day basket equity curve, then max drawdown.
    Returns the worst peak-to-trough drawdown in % (negative number).
    """
    if per_trade_subset.empty:
        return float("nan")
    daily = (per_trade_subset
             .groupby(per_trade_subset[date_col].dt.normalize())[ret_col]
             .mean()
             .sort_index())
    if daily.empty:
        return float("nan")
    eq = (1.0 + daily / 100.0).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1.0) * 100.0
    return float(dd.min())


# ---------------------------------------------------------------------------
# One (variant, regime, prob_bucket) cell
# ---------------------------------------------------------------------------

def _row_for_cell(cell: pd.DataFrame) -> Dict:
    """Compute the four blocks + diagnostics for one (variant, regime, prob_bucket)."""
    n_total = int(len(cell))
    n_trig = int(cell["triggered"].sum())
    n_skip = n_total - n_trig

    triggered  = cell[cell["triggered"]].copy()
    skipped    = cell[~cell["triggered"]].copy()

    # NAIVE-ALL: every signal, naive entry
    block_naive_all = _block_metrics(
        cell, "naive_fwd_t5_close_ret_pct",
        mae_col="naive_mae_pct", mfe_col="naive_mfe_pct",
        prefix="naive_all_",
    )
    # NAIVE-on-TAKEN: the triggered subset, naive entry
    block_naive_taken = _block_metrics(
        triggered, "naive_fwd_t5_close_ret_pct",
        mae_col="naive_mae_pct", mfe_col="naive_mfe_pct",
        prefix="naive_taken_",
    )
    # NAIVE-on-SKIPPED: the non-triggered subset, naive entry
    block_naive_skipped = _block_metrics(
        skipped, "naive_fwd_t5_close_ret_pct",
        mae_col="naive_mae_pct", mfe_col="naive_mfe_pct",
        prefix="naive_skipped_",
    )
    # ORB-TAKEN: triggered subset, ORB entry
    block_orb_taken = _block_metrics(
        triggered, "fwd_return_to_t5_close_pct",
        mae_col="mae_pct", mfe_col="mfe_pct",
        prefix="orb_taken_",
    )

    # Daily-basket portfolio metrics on the SAME UNIVERSE
    # NAIVE portfolio   = mean naive return across all signals on each day
    # ORB-FILTERED port = mean (triggered ? orb_ret : 0) across all on each day
    cell_for_orb = cell.copy()
    cell_for_orb["orb_signal_ret"] = np.where(
        cell_for_orb["triggered"],
        cell_for_orb["fwd_return_to_t5_close_pct"],
        0.0,                # cash for skipped slots
    )
    naive_sharpe = _daily_sharpe(cell, "naive_fwd_t5_close_ret_pct")
    orb_sharpe   = _daily_sharpe(cell_for_orb, "orb_signal_ret")
    naive_dd     = _max_drawdown(cell, "naive_fwd_t5_close_ret_pct")
    orb_dd       = _max_drawdown(cell_for_orb, "orb_signal_ret")

    # Diagnostics (the actual answers to the debate)
    selection_benefit = (block_naive_taken.get("naive_taken_mean_ret_pct", np.nan)
                         - block_naive_skipped.get("naive_skipped_mean_ret_pct", np.nan))
    entry_cost = (block_naive_taken.get("naive_taken_mean_ret_pct", np.nan)
                  - block_orb_taken.get("orb_taken_mean_ret_pct", np.nan))
    net_orb_adv = selection_benefit - entry_cost

    # Portfolio mean return per signal slot (skipped contributes 0 to ORB)
    naive_port_mean = block_naive_all.get("naive_all_mean_ret_pct", np.nan)
    orb_port_mean   = (cell_for_orb["orb_signal_ret"].mean()
                       if not cell_for_orb.empty else np.nan)

    row = {
        "n_signals":  n_total,
        "n_taken":    n_trig,
        "n_skipped":  n_skip,
        "trigger_pct": (100 * n_trig / n_total) if n_total else np.nan,
        # diagnostics first (these are the headline numbers)
        "selection_benefit_pct": float(selection_benefit) if pd.notna(selection_benefit) else np.nan,
        "entry_cost_pct":        float(entry_cost)        if pd.notna(entry_cost) else np.nan,
        "net_orb_advantage_pct": float(net_orb_adv)       if pd.notna(net_orb_adv) else np.nan,
        # portfolio means on the same universe
        "naive_portfolio_mean_pct": float(naive_port_mean) if pd.notna(naive_port_mean) else np.nan,
        "orb_portfolio_mean_pct":   float(orb_port_mean)   if pd.notna(orb_port_mean) else np.nan,
        # portfolio sharpe / drawdown
        "naive_portfolio_sharpe": naive_sharpe,
        "orb_portfolio_sharpe":   orb_sharpe,
        "naive_portfolio_max_dd_pct": naive_dd,
        "orb_portfolio_max_dd_pct":   orb_dd,
    }
    row.update(block_naive_all)
    row.update(block_naive_taken)
    row.update(block_naive_skipped)
    row.update(block_orb_taken)
    return row


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def aggregate(per_trade: pd.DataFrame) -> pd.DataFrame:
    if per_trade.empty:
        return pd.DataFrame()

    df = per_trade.copy()
    if "signal_date" in df.columns:
        df["signal_date"] = pd.to_datetime(df["signal_date"])

    rows: List[Dict] = []
    for (variant, regime, bucket), cell in df.groupby(
            ["variant", "regime", "prob_bucket"], dropna=False):
        row = {
            "variant":     variant,
            "regime":      regime,
            "prob_bucket": bucket,
        }
        row.update(_row_for_cell(cell))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-trade", type=Path, default=DEFAULT_PER_TRADE,
                    help=f"path to per_trade.parquet "
                         f"(default: {DEFAULT_PER_TRADE})")
    args = ap.parse_args()

    if not args.per_trade.exists():
        raise SystemExit(
            f"FATAL: per-trade parquet not found at {args.per_trade}.  "
            f"Run orb_execution_quality.py first to produce it."
        )
    print(f"[load] {args.per_trade}")
    per_trade = pd.read_parquet(args.per_trade)
    print(f"[load] {len(per_trade):,} rows, "
          f"{per_trade['variant'].nunique()} variants")

    res = aggregate(per_trade)
    out_csv = OUT_DIR / "entry_strategy_comparison.csv"
    res.to_csv(out_csv, index=False)
    print(f"[out] {out_csv}")

    try:
        with pd.ExcelWriter(OUT_DIR / "entry_strategy_summary.xlsx",
                            engine="openpyxl") as xw:
            res.sort_values(["variant", "regime", "prob_bucket"]) \
               .to_excel(xw, sheet_name="Comparison", index=False)
    except Exception as e:
        print(f"[warn] could not write xlsx: {e}")

    # Console teaser: show the diagnostics line per cell
    print("\n=== Net ORB advantage by cell (>0 means ORB beats naive) ===")
    show = ["variant", "regime", "prob_bucket",
            "n_signals", "n_taken", "n_skipped", "trigger_pct",
            "naive_portfolio_mean_pct", "orb_portfolio_mean_pct",
            "selection_benefit_pct", "entry_cost_pct", "net_orb_advantage_pct",
            "naive_portfolio_sharpe", "orb_portfolio_sharpe",
            "naive_portfolio_max_dd_pct", "orb_portfolio_max_dd_pct"]
    show = [c for c in show if c in res.columns]
    print(res.sort_values(["variant", "regime", "prob_bucket"])[show]
             .to_string(index=False))

    # And the SL-hit-rate comparison (the user's specific intuition)
    print("\n=== SL hit rates: NAIVE-on-TAKEN vs NAIVE-on-SKIPPED "
          "(if SKIPPED is higher, ORB filter has selection value) ===")
    sl_cols = ["variant", "regime", "prob_bucket",
               "naive_taken_hit_sl3_pct", "naive_skipped_hit_sl3_pct",
               "naive_taken_hit_sl5_pct", "naive_skipped_hit_sl5_pct",
               "naive_taken_mean_mae_pct", "naive_skipped_mean_mae_pct"]
    sl_cols = [c for c in sl_cols if c in res.columns]
    print(res.sort_values(["variant", "regime", "prob_bucket"])[sl_cols]
             .to_string(index=False))


if __name__ == "__main__":
    main()
