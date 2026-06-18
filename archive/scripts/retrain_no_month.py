"""
Retrain the GradientBoosting model WITHOUT the 'month' and 'is_volatile_month'
features, then compare predictions and signal distributions for 2024-2025.

Training period: 2014 - 2023 (extended from original 2014-2022)
Test period: 2024 - 2025

Drops 'month' and 'is_volatile_month' from the 37 features → 35 features.
Uses the exact same hyperparameters as the original model.
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
from datetime import date
from collections import Counter

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_ROOT)

from shared.feature_compute import compute_all_features, load_dhan_options_features

# ─── Config ───
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
FEATURES_TO_DROP = ['month', 'is_volatile_month']

# ─── Load data ───
print("Loading data...")
merged_path = os.path.join(BACKEND_ROOT, "data", "merged_daily.parquet")
merged_daily = pd.read_parquet(merged_path)
print(f"Merged daily: {len(merged_daily)} rows, {merged_daily.index[0].date()} to {merged_daily.index[-1].date()}")

dhan_raw = os.path.join(BACKEND_ROOT, "data", "dhan_raw_options.parquet")
dhan_skew = os.path.join(BACKEND_ROOT, "data", "daily_iv_skew_params.parquet")
dhan_features = load_dhan_options_features(dhan_raw, dhan_skew)

# ─── Compute full feature matrix ───
print("Computing features (this takes ~1 min)...")
df, feature_cols = compute_all_features(merged_daily, dhan_features,
                                         data_start="2014-01-01", data_end="2026-12-31")
print(f"Feature matrix: {len(df)} rows, {len(feature_cols)} features")
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

# ─── Load original model ───
original_model = joblib.load(os.path.join(BACKEND_ROOT, "ml", "downside_model.pkl"))
original_feature_names = joblib.load(os.path.join(BACKEND_ROOT, "ml", "feature_names.pkl"))
print(f"\nOriginal model: {type(original_model).__name__}, {len(original_feature_names)} features")
print(f"Hyperparams: n_estimators={original_model.n_estimators}, "
      f"lr={original_model.learning_rate}, max_depth={original_model.max_depth}, "
      f"subsample={original_model.subsample}, max_features={original_model.max_features}")

# ─── Prepare datasets ───
# Training: up to end of 2022
train_mask = df.index <= TRAIN_END
train_df = df[train_mask].copy()

# Test: 2024 onwards (gap year 2023 is intentional — same as original)
test_mask = df.index >= TEST_START
test_df = df[test_mask].copy()

print(f"\nTraining set: {len(train_df)} rows ({train_df.index[0].date()} to {train_df.index[-1].date()})")
print(f"Test set:     {len(test_df)} rows ({test_df.index[0].date()} to {test_df.index[-1].date()})")

target_col = 'max_drawdown_30d'

# Features for new model (drop month features)
new_feature_cols = [f for f in feature_cols if f not in FEATURES_TO_DROP]
print(f"\nOriginal features: {len(feature_cols)}")
print(f"New features:      {len(new_feature_cols)} (dropped: {FEATURES_TO_DROP})")

# ─── Train new model ───
from sklearn.ensemble import GradientBoostingRegressor

print("\n" + "="*80)
print("TRAINING NEW MODEL (without month features)")
print("="*80)

X_train = train_df[new_feature_cols].values
y_train = train_df[target_col].values

# Remove NaN targets
valid = ~np.isnan(y_train)
X_train = X_train[valid]
y_train = y_train[valid]
print(f"Training samples (after NaN removal): {len(X_train)}")

new_model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    max_features=0.8,
    min_samples_leaf=50,
    random_state=42,
)

print("Training... (this takes ~30 seconds)")
new_model.fit(X_train, y_train)
print("Training complete!")

# ─── Feature importance comparison ───
print("\n" + "="*80)
print("FEATURE IMPORTANCE COMPARISON")
print("="*80)

orig_imp = dict(zip(original_feature_names, original_model.feature_importances_))
new_imp = dict(zip(new_feature_cols, new_model.feature_importances_))

print(f"\n{'Feature':35s}  {'Original':>10s}  {'New':>10s}  {'Change':>10s}")
print("-" * 70)
orig_total = sum(original_model.feature_importances_)
new_total = sum(new_model.feature_importances_)

# Rank by new importance
all_features = set(original_feature_names) | set(new_feature_cols)
for feat in sorted(all_features, key=lambda f: new_imp.get(f, 0), reverse=True):
    o = orig_imp.get(feat, 0) / orig_total * 100
    n = new_imp.get(feat, 0) / new_total * 100
    diff = n - o
    marker = ""
    if feat in FEATURES_TO_DROP:
        marker = " [DROPPED]"
    elif abs(diff) > 2:
        marker = " ***"
    print(f"  {feat:33s}  {o:>9.1f}%  {n:>9.1f}%  {diff:>+9.1f}%{marker}")

# ─── Generate predictions ───
print("\n" + "="*80)
print("PREDICTIONS: ORIGINAL vs NEW MODEL (2024-2025)")
print("="*80)

X_test_orig = test_df[feature_cols].values  # original 37 features
X_test_new = test_df[new_feature_cols].values  # new 35 features

# Handle NaN: fill with 0 for Dhan features that might be missing
X_test_orig = np.nan_to_num(X_test_orig, nan=0.0)
X_test_new = np.nan_to_num(X_test_new, nan=0.0)

preds_orig = original_model.predict(X_test_orig)
preds_new = new_model.predict(X_test_new)

test_df = test_df.copy()
test_df['pred_orig'] = preds_orig
test_df['pred_new'] = preds_new
test_df['pred_diff'] = preds_new - preds_orig
test_df['actual_month'] = test_df.index.month

# ─── Training set accuracy comparison ───
print("\n--- Training Set Performance ---")
X_train_orig_full = train_df[feature_cols].values
X_train_orig_full = np.nan_to_num(X_train_orig_full, nan=0.0)
X_train_new_full = train_df[new_feature_cols].values
X_train_new_full = np.nan_to_num(X_train_new_full, nan=0.0)
y_train_full = train_df[target_col].values

valid_train = ~np.isnan(y_train_full)
pred_train_orig = original_model.predict(X_train_orig_full[valid_train])
pred_train_new = new_model.predict(X_train_new_full[valid_train])
y_train_valid = y_train_full[valid_train]

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

for label, preds in [("Original", pred_train_orig), ("New (no month)", pred_train_new)]:
    rmse = np.sqrt(mean_squared_error(y_train_valid, preds))
    mae = mean_absolute_error(y_train_valid, preds)
    r2 = r2_score(y_train_valid, preds)
    print(f"  {label:20s}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

# ─── Test set accuracy ───
print("\n--- Test Set Performance (2024-2025) ---")
y_test = test_df[target_col].values
valid_test = ~np.isnan(y_test)

if valid_test.sum() > 0:
    for label, preds in [("Original", preds_orig[valid_test]), ("New (no month)", preds_new[valid_test])]:
        rmse = np.sqrt(mean_squared_error(y_test[valid_test], preds))
        mae = mean_absolute_error(y_test[valid_test], preds)
        r2 = r2_score(y_test[valid_test], preds)
        print(f"  {label:20s}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

# ─── Monthly prediction comparison ───
print(f"\n--- Monthly Average Predictions ---")
print(f"{'Month':>6s}  {'Original':>10s}  {'New Model':>10s}  {'Shift':>8s}  {'Actual DD':>10s}")
print("-" * 55)
for m in range(1, 13):
    mask_m = test_df['actual_month'] == m
    if mask_m.sum() == 0:
        continue
    orig_avg = test_df.loc[mask_m, 'pred_orig'].mean()
    new_avg = test_df.loc[mask_m, 'pred_new'].mean()
    shift = new_avg - orig_avg
    actual = test_df.loc[mask_m, target_col].dropna().mean()
    actual_str = f"{actual*100:.2f}%" if not np.isnan(actual) else "N/A"
    arrow = "  <<<" if shift > 0.005 else ""
    print(f"  {m:4d}  {orig_avg*100:>9.2f}%  {new_avg*100:>9.2f}%  {shift*100:>+7.2f}%  {actual_str:>10s}{arrow}")

# ─── Signal mapping functions ───
def map_signal_sharp(pred_dd):
    if pred_dd > -0.038: return "bull_full", 1.0
    elif pred_dd > -0.050: return "bull_half", 1.0
    elif pred_dd > -0.065: return "iron_condor", 1.0
    else: return "no_trade", 0.0

def map_signal_graduated(pred_dd):
    if pred_dd > -0.01: return "bull_full", 1.0
    elif pred_dd > -0.02:
        frac = (pred_dd - (-0.02)) / ((-0.01) - (-0.02))
        return "bull_full", 0.7 + 0.3 * frac
    elif pred_dd > -0.03:
        frac = (pred_dd - (-0.03)) / ((-0.02) - (-0.03))
        if frac > 0.5: return "bull_half", 0.5 + 0.5 * frac
        else: return "iron_condor", 0.5 + 0.5 * (1 - frac)
    elif pred_dd > -0.04:
        frac = (pred_dd - (-0.04)) / ((-0.03) - (-0.04))
        return "iron_condor", max(0.25, 0.5 * frac)
    else: return "no_trade", 0.0

def map_signal_graduated_gentle(pred_dd):
    hw, floor = 0.0025, 0.80
    t = {"bf": -0.038, "bh": -0.050, "ic": -0.065}
    def _lerp(v, lo, hi, olo, ohi):
        return olo + max(0, min(1, (v-lo)/(hi-lo))) * (ohi-olo) if hi!=lo else (olo+ohi)/2
    if pred_dd > t["bf"]+hw: return "bull_full", 1.0
    if pred_dd > t["bf"]-hw: return "bull_full", _lerp(pred_dd, t["bf"]-hw, t["bf"]+hw, floor, 1.0)
    if pred_dd > t["bh"]+hw: return "bull_half", 1.0
    if pred_dd > t["bh"]-hw: return "bull_half", _lerp(pred_dd, t["bh"]-hw, t["bh"]+hw, floor, 1.0)
    if pred_dd > t["ic"]+hw: return "iron_condor", 1.0
    if pred_dd > t["ic"]-hw: return "iron_condor", _lerp(pred_dd, t["ic"]-hw, t["ic"]+hw, floor, 1.0)
    return "no_trade", 0.0

VERSIONS = {
    "v5.4.2": map_signal_sharp,
    "v5.4.3": map_signal_graduated,
    "v5.4.4": map_signal_graduated_gentle,
}

# ─── Signal distribution comparison ───
for version, mapper in VERSIONS.items():
    print(f"\n{'='*80}")
    print(f"VERSION: {version}")
    print(f"{'='*80}")

    sigs_orig = [mapper(p)[0] for p in preds_orig]
    sigs_new = [mapper(p)[0] for p in preds_new]

    c_orig = Counter(sigs_orig)
    c_new = Counter(sigs_new)

    print(f"\n--- Signal Distribution ---")
    print(f"{'Signal':>15s}  {'Original':>10s}  {'New Model':>10s}  {'Diff':>8s}")
    print("-" * 48)
    for sig in ["bull_full", "bull_half", "iron_condor", "no_trade"]:
        o, n = c_orig.get(sig, 0), c_new.get(sig, 0)
        print(f"  {sig:>13s}  {o:>10d}  {n:>10d}  {n-o:>+8d}")

    trade_orig = len(test_df) - c_orig.get("no_trade", 0)
    trade_new = len(test_df) - c_new.get("no_trade", 0)
    print(f"\n  Trade days:  {trade_orig}  →  {trade_new}  ({trade_new - trade_orig:+d})")

    # Monthly no-trade breakdown
    test_df[f'sig_orig_{version}'] = sigs_orig
    test_df[f'sig_new_{version}'] = sigs_new

    print(f"\n--- Monthly Trade Rate ---")
    print(f"{'Month':>6s}  {'Orig trade%':>12s}  {'New trade%':>12s}  {'Diff':>8s}")
    print("-" * 45)
    for m in range(1, 13):
        mask_m = test_df['actual_month'] == m
        if mask_m.sum() == 0:
            continue
        total_m = mask_m.sum()
        orig_trades = (test_df.loc[mask_m, f'sig_orig_{version}'] != 'no_trade').sum()
        new_trades = (test_df.loc[mask_m, f'sig_new_{version}'] != 'no_trade').sum()
        o_pct = orig_trades / total_m * 100
        n_pct = new_trades / total_m * 100
        arrow = "  <<<" if n_pct - o_pct > 10 else ""
        print(f"  {m:4d}  {o_pct:>11.0f}%  {n_pct:>11.0f}%  {n_pct-o_pct:>+7.0f}%{arrow}")

# ─── February deep dive ───
print(f"\n{'='*80}")
print(f"FEBRUARY DEEP DIVE (NEW MODEL)")
print(f"{'='*80}")
feb_mask = test_df['actual_month'] == 2
feb = test_df[feb_mask]
print(f"\nFeb trading days: {len(feb)}")
print(f"Original pred range: {feb['pred_orig'].min()*100:.2f}% to {feb['pred_orig'].max()*100:.2f}%")
print(f"New pred range:      {feb['pred_new'].min()*100:.2f}% to {feb['pred_new'].max()*100:.2f}%")
print(f"Average shift: {feb['pred_diff'].mean()*100:+.2f}%")

actual_dd = feb[target_col].dropna()
if len(actual_dd) > 0:
    print(f"Actual drawdowns:    mean={actual_dd.mean()*100:.2f}%, worst={actual_dd.min()*100:.2f}%")

for version in VERSIONS:
    orig_nt = (feb[f'sig_orig_{version}'] == 'no_trade').sum()
    new_nt = (feb[f'sig_new_{version}'] == 'no_trade').sum()
    print(f"\n  {version}:")
    print(f"    Original: {orig_nt}/{len(feb)} no-trade ({orig_nt/len(feb)*100:.0f}%)")
    print(f"    New:      {new_nt}/{len(feb)} no-trade ({new_nt/len(feb)*100:.0f}%)")

# ─── Year-month view for v5.4.2 ───
print(f"\n{'='*80}")
print(f"MONTH-BY-MONTH: v5.4.2 (Original vs New Model)")
print(f"{'='*80}")
print(f"{'Year-Month':>10s}  {'Orig Pred':>10s}  {'New Pred':>10s}  {'Shift':>8s}  {'Orig Tr':>8s}  {'New Tr':>8s}  {'Actual DD':>10s}")
print("-" * 75)

test_df['ym'] = test_df.index.to_period('M')
for ym in sorted(test_df['ym'].unique()):
    mask_ym = test_df['ym'] == ym
    sub = test_df[mask_ym]
    o_avg = sub['pred_orig'].mean()
    n_avg = sub['pred_new'].mean()
    shift = n_avg - o_avg
    total = len(sub)
    o_tr = (sub['sig_orig_v5.4.2'] != 'no_trade').sum()
    n_tr = (sub['sig_new_v5.4.2'] != 'no_trade').sum()
    actual = sub[target_col].dropna().mean()
    actual_str = f"{actual*100:.2f}%" if not np.isnan(actual) else "N/A"
    print(f"  {str(ym):>8s}  {o_avg*100:>9.2f}%  {n_avg*100:>9.2f}%  {shift*100:>+7.2f}%  {o_tr:>3d}/{total:<3d}  {n_tr:>3d}/{total:<3d}  {actual_str:>10s}")

# ─── Save new model ───
output_dir = os.path.join(BACKEND_ROOT, "ml")
new_model_path = os.path.join(output_dir, "downside_model_no_month.pkl")
new_features_path = os.path.join(output_dir, "feature_names_no_month.pkl")

joblib.dump(new_model, new_model_path)
joblib.dump(new_feature_cols, new_features_path)
print(f"\n{'='*80}")
print(f"NEW MODEL SAVED")
print(f"  Model:    {new_model_path}")
print(f"  Features: {new_features_path} ({len(new_feature_cols)} features)")
print(f"{'='*80}")
