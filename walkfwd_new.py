#!/usr/bin/env python3
"""
=============================================================================
WALK-FORWARD ROBUSTNESS TEST (Option B — true retraining per fold)
=============================================================================
"""

import os
import sys
import json
import time
import gc
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump, load

warnings.filterwarnings("ignore")

# ----- import the user's pipeline -----
sys.path.insert(0, r"C:\Users\karanvsi\PyCharmMiscProject")
import New_model as NM

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(r"C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full")
PANEL_PATH = BASE_DIR / "panel_cache.parquet"
FEATURES_PATH = BASE_DIR / "features_train.json"
REGIME_FEATURES = BASE_DIR / "feature_diagnostics" / "regime_features.json"

WF_DIR = BASE_DIR / "walkforward_results"
CKPT_DIR = WF_DIR / "checkpoints"
WF_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

IST = "Asia/Kolkata"
RET_COL = "ret_5d_oc_pct"

MIN_CLOSE = getattr(NM, "TRAIN_MIN_CLOSE", 2.0)
MIN_AVG20_VOL = getattr(NM, "TRAIN_MIN_AVG20_VOL", 100_000)

WF_MEMBERS = 3
EV_TARGET = "cc"
EMBARGO_DAYS = getattr(NM, "EMBARGO_DAYS", 5)

# Fold definitions
FOLDS = [
    (1, "2021-12-31", "2022-01-01", "2022-12-31"),
    (2, "2022-12-31", "2023-01-01", "2023-12-31"),
    (3, "2023-12-31", "2024-01-01", "2024-12-31"),
    (4, "2024-12-31", "2025-01-01", "2099-12-31"),
]

PROB_BUCKETS = [
    (0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
    (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90),
    (0.90, 0.95), (0.95, 1.001),
]

REGIMES = ["bull_trend", "bear_trend", "bull_range", "bear_range"]
THRESHOLDS = [0.65, 0.70]

# =============================================================================
# HELPERS
# =============================================================================

def log(m):
    print(f"[WF] {m}", flush=True)


def load_feature_lists():
    schema = json.loads(Path(FEATURES_PATH).read_text())
    feats = schema["features"]
    impute = {k: float(v) for k, v in schema["impute"].items()}
    global_feats = feats

    if REGIME_FEATURES.exists():
        rf = json.loads(REGIME_FEATURES.read_text())
        global_feats = rf.get("global", feats) or feats
        log(f"Loaded regime_features.json: global keep={len(global_feats)}")

    return feats, impute, global_feats


def prep_X(panel, feats, impute):
    X = panel.reindex(columns=feats).copy()
    for c in feats:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(impute.get(c, 0.0))
    return X


def bucket_stats(sub, prob_col):
    if len(sub) == 0:
        return None

    r = sub[RET_COL].values
    wins = r > 0

    return dict(
        n=int(len(sub)),
        win_rate_pct=round(100 * wins.mean(), 2),
        pred_prob=round(sub[prob_col].mean(), 4),
        calib_gap=round(sub[prob_col].mean() - wins.mean(), 4),
        avg_ret_pct=round(float(r.mean()), 3),
        median_ret_pct=round(float(np.median(r)), 3),
        std_pct=round(float(r.std()), 3),
        min_ret=round(float(r.min()), 2),
        max_ret=round(float(r.max()), 2),
    )

# =============================================================================
# MAIN
# =============================================================================

def main():

    t_all = time.perf_counter()

    log("Loading enriched panel ONCE...")
    panel = pd.read_parquet(PANEL_PATH)

    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    if panel["timestamp"].dt.tz is None:
        panel["timestamp"] = panel["timestamp"].dt.tz_localize(IST)
    else:
        panel["timestamp"] = panel["timestamp"].dt.tz_convert(IST)

    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    for need in ("stock_regime", RET_COL):
        if need not in panel.columns:
            raise SystemExit(f"FATAL: panel missing '{need}'")

    LABEL_COL = "top20_vs_bot20_5d"

    if LABEL_COL not in panel.columns:
        log(f"Building label via NM.build_5d_rank_quant_labels...")
        panel = NM.build_5d_rank_quant_labels(panel, ev_target=EV_TARGET)

    # ---- Universe filter ----
    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    panel["close"] = pd.to_numeric(panel["close"], errors="coerce")
    panel["volume"] = pd.to_numeric(panel["volume"], errors="coerce")

    panel["avg20_vol"] = panel.groupby("symbol")["volume"].transform(
        lambda s: s.rolling(20, min_periods=1).mean()
    )

    panel = panel[
        (panel["close"] >= MIN_CLOSE) &
        (panel["avg20_vol"] >= MIN_AVG20_VOL)
    ].reset_index(drop=True)

    feats, impute, global_feats = load_feature_lists()

    all_rows = []

    for fold_id, train_end, test_start, test_end in FOLDS:

        ckpt_results = CKPT_DIR / f"fold{fold_id}_buckets.csv"

        if ckpt_results.exists():
            log(f"Fold {fold_id}: checkpoint exists → SKIP")
            all_rows.append(pd.read_csv(ckpt_results))
            continue

        te = pd.Timestamp(train_end, tz=IST)
        ts0 = pd.Timestamp(test_start, tz=IST)
        ts1 = pd.Timestamp(test_end, tz=IST)

        train_panel = panel[panel["timestamp"] <= te].copy()
        test_panel = panel[
            (panel["timestamp"] >= ts0) &
            (panel["timestamp"] <= ts1)
        ].copy()

        log(f"Fold {fold_id}: train={len(train_panel):,} test={len(test_panel):,}")

        ckpt_model = CKPT_DIR / f"fold{fold_id}_pooled.joblib"

        if ckpt_model.exists():
            pooled = load(ckpt_model)
        else:
            out = NM.fit_regime_ensembles(
                train_panel,
                feats,
                EV_TARGET,
                n_members=WF_MEMBERS,
                train_regime_specialists=False,
                global_feats=global_feats
            )
            pooled = out[3]
            dump(pooled, ckpt_model)

        Xte = prep_X(test_panel, global_feats, impute)

        p = pooled.predict_proba(Xte)
        p = p[:, 1] if p.ndim == 2 else p

        test_panel = test_panel.assign(prob=p)
        test_panel = test_panel[test_panel[RET_COL].notna()]

        rows = []

        scopes = [("GLOBAL", test_panel)] + [
            (rg, test_panel[test_panel["stock_regime"] == rg])
            for rg in REGIMES
        ]

        for scope, dfx in scopes:
            for thr in THRESHOLDS:
                s = bucket_stats(dfx[dfx["prob"] >= thr], "prob")
                if s:
                    s.update(
                        fold=fold_id,
                        test_year=test_start[:4],
                        scope=scope,
                        threshold=f">={thr:.2f}"
                    )
                    rows.append(s)

        fold_df = pd.DataFrame(rows)
        fold_df.to_csv(ckpt_results, index=False)
        all_rows.append(fold_df)

        del train_panel, test_panel, Xte
        gc.collect()

    if not all_rows:
        return

    agg = pd.concat(all_rows, ignore_index=True)
    agg.to_csv(WF_DIR / "walkforward_all_folds.csv", index=False)

    log("DONE")


if __name__ == "__main__":
    main()