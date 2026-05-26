# feature_imp v4 → v5 changes

`feature_imp_v5.py` is a drop-in replacement for `feature imp.py` (v4).
Run it the same way; outputs land in the same directory.

## v5.1 patch (2026-05-22)

Fixes the load-time crash:
```
pyarrow.lib.ArrowInvalid: No match for FieldRef.Name(M_nifty_ret) ...
```

Root cause: `New_model.py` writes `panel_cache.parquet` *before* it joins macros and computes `regime_*` / `stock_regime`, so those columns are in `features_train.json` (186 features) but not in the parquet (~172 features).

v5.1 handles this by:

1. Reading the parquet schema first via `pyarrow.parquet.read_schema` and only requesting columns that actually exist.
2. Calling a new `_ensure_panel_features()` helper that:
   - Recomputes `regime_market_trend / regime_high_vol / regime_dispersion` from cross-sectional 1d returns
   - Recomputes `stock_regime` from `D_sma200` + `D_adx14`, leaving NaN for early-history rows where SMA200/ADX are unknown (fixes the audit issue where early bars wrongly fell into `bear_trend`)
   - Loads macros from `MACRO_CACHE_PATH` (defaults to `C:\Users\karanvsi\Desktop\Pycharm\Cache\macro_cache.parquet`, override with `MACRO_CACHE_PATH` env var)
   - Rebuilds `top20_vs_bot20_5d` from `ret_5d_oc_pct` + ATR% if missing
3. After all recomputation, drops any features still missing from the panel and warns; the rest of the pipeline runs on whatever was successfully loaded/recomputed.

If you set `MACRO_CACHE_PATH` correctly, v5.1 will give you the full 186-feature analysis. If macros aren't reachable, you'll get ~179 features (everything except `M_*`) and a clear warning.

The right long-term fix is in `New_model.py` itself — write the augmented panel back to disk after macros + regime are added — but that's a separate PR.

## Why v5 exists

v4 produced KEEP / DROP / REVIEW recommendations anchored to an incorrect
ground truth. Acting on those recommendations could remove features that
genuinely generalize out-of-time and keep features that only "work" because
the test split was a different set of stocks, not a different set of dates.

## Correctness fixes

| # | What v4 did | What v5 does |
|---|---|---|
| 1 | `tr_idx = np.arange(0, 0.8*N)` on a panel sorted by `(symbol, timestamp)`. First 80% of rows = first 80% of *symbols*. | `time_split_by_date()` — sorts by timestamp, takes the last `TEST_FRAC=0.20` of *calendar dates* as test, with `EMBARGO_DAYS=6` (horizon+1) gap. |
| 2 | Per-regime permutation re-used the same row-index split inside the regime subset. | Same time-based split applied per regime. |
| 3 | Diagnostic LightGBM was 400 trees @ LR=0.05, num_leaves=63, min_data_in_leaf=100 — nothing like production's 3000 @ 0.005, 31, 500. | Diagnostic uses 1500 trees @ 0.01 with `num_leaves=31`, `min_data_in_leaf=300`, `feature_fraction=0.7`, `extra_trees=True`, `reg_alpha=0.3`, `reg_lambda=10` — same regularization profile as prod, ~2× faster. |
| 4 | Single shuffle per feature → AUC-drop std ~ 0.001, on the same order as the KEEP/DROP threshold. | `permute_auc_multi()` averages over `PERM_SHUFFLES=5` shuffles per feature; both mean and std reported in the CSV. |
| 5 | Imputation used global medians (computed across train+test). | Train-only medians, with the production schema's `impute` dict as fallback for features absent in train. |

## New analyses

| | What it does |
|---|---|
| **Harmful feature detection** | Any feature whose mean perm-drop is `< -0.0005` in any regime is tagged `DROP_HARMFUL` and pulled out into `harmful_features.csv`. v4 silently lumped these with REVIEW. |
| **Feature family aggregation** | `feature_family_summary.csv` shows which prefix families (`D_`, `X_`, `W_`, `WQ_`, `M_`, `Comb_`, `regime_`, etc.) carry weight, how many members are alive vs dead, total gain%, mean perm drop, mean &#124;IC&#124;. Easy to spot a whole family that isn't pulling its weight. |
| **Forward-time generalization** | Stage 4 trains on the first 60% of train dates vs dates 60–80%, compares feature ranks. `rank_drift < 0.25` → generalizes. v4's price-bucket test was a weak proxy for generalization. |
| **Regime-specialist tagging** | `regime_specialist_features.csv` lists features that are strong in 1–2 regimes and dead in the others. Strict mode preserves them in KEEP. |
| **Year-over-year rank stability** | Replaces v4's `1 - std/mean` formula (which broke for low-mean features). v5 uses pairwise rank-distance: 1 = identical rank in every year, 0 = rank uncorrelated. |
| **`regime_features.json`** | Production hand-off file. Contains: `drop_list`, `keep_list_global`, `per_regime` (per-regime keep lists), thresholds, base AUCs. New_model.py can read this directly to build per-regime feature lists. |

## Output files

| File | What's in it |
|---|---|
| `feature_pruning_recommendation.csv` | Main file. One row per feature with KEEP / REVIEW / DROP / DROP_HARMFUL + reason + per-regime AUC drops + per-regime IC. |
| `regime_features.json` | Machine-readable per-regime keep lists for production. |
| `harmful_features.csv` | Subset: features that hurt the model. Drop these first. |
| `regime_specialist_features.csv` | Subset: features alive in 1–2 regimes only. Preserve. |
| `feature_family_summary.csv` | Per-family aggregates (D_, X_, W_, WQ_, etc.). |
| `feature_importance_global.csv` | Gain + split importance from the production model (or diagnostic if prod model unloadable). |
| `feature_ic_by_year.csv` | IC pivot: feature × year. |
| `regime_ic_summary.csv` | IC per regime. |
| `feature_stability_by_year.csv` | Per-year importance + rank_stability column. |
| `feature_generalization.csv` | rank_early vs rank_late + drift score. |
| `feature_correlation_matrix.csv` | Spearman corr (on TRAIN slice only). |
| `feature_redundant_pairs.csv` | Pairs with &#124;r&#124; ≥ 0.85, plus a keep/drop suggestion based on &#124;IC&#124;. |
| `feature_permutation_importance.csv` | Global perm: mean + std across shuffles. |
| `regime_permutation_importance.csv` | Per-regime perm: mean + std across shuffles. |
| `regime_consistency.csv` | Per-feature consistency score, best regime, specialist/dead/harmful flags. |
| `feature_diagnostics_full.xlsx` | All of the above as sheets. |

## Expected behavior change

The reported global AUC will likely drop by **1–3 bps** vs v4. That is not a regression — it's the embargo + time-based split removing the cross-stock leakage that v4 was implicitly exploiting. The numbers v5 produces are honest.

Expect a meaningful number of features to move from KEEP→REVIEW and from REVIEW→DROP, *especially* features whose v4 importance was stock-specific (e.g., features that were strong on the 80% of symbols in the train slice but weak on the 20% in the test slice).

Conversely, some features that v4 marked DROP may move to KEEP under v5 if their signal is genuinely temporal (regime-specialist features get caught by Stage 7 + REGIME_EXCEPTIONAL=0.0030).

## Config knobs

All in the top of the file:

```python
TEST_FRAC          = 0.20      # fraction of calendar dates held out
EMBARGO_DAYS       = 6         # horizon (5) + 1
PERM_SHUFFLES      = 5         # multi-shuffle averaging
CV_FOLDS           = 5         # purged k-fold (used for permutation if you wire it in)
PERM_DROP_KEEP     = 0.0010    # AUC drop above this counts as useful
PERM_DROP_HARMFUL  = -0.0005   # below this = feature hurts
REGIME_EXCEPTIONAL = 0.0030    # AUC drop in any single regime
IC_THRESHOLD       = 0.02
CORR_THRESHOLD     = 0.85
```

## How to wire into New_model.py

After running v5, in `New_model.py:run_pipeline` you can read the per-regime list:

```python
import json
regime_feats = json.loads((Path(out_dir) / "feature_diagnostics" / "regime_features.json").read_text())

# Use per-regime feature lists when training each regime ensemble
for r in REGIMES:
    feats_for_r = regime_feats["per_regime"].get(r, regime_feats["keep_list_global"])
    # ... fit_regime_ensembles(panel, feats_for_r, ...)
```

This requires a small change in `fit_regime_ensembles` to accept a per-regime feature list. That change is **not** in this PR — v5 just produces the JSON and leaves the wiring to a follow-up.

## Run

```bash
python feature_imp_v5.py --base-dir "C:\Users\karanvsi\Desktop\Kite Connect\v3_2_output_full"
```

Or set `DEFAULT_BASE_DIR` in the file and run with no args.
