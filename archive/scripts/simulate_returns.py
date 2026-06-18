"""
Simulate simplified P&L for new vs old model on 2024-2025 test period.

Uses actual max_drawdown_30d outcomes to estimate trade results:
- bull_full/bull_half credit spreads: profit if drawdown < spread width, loss otherwise
- iron_condor: profit if drawdown stays within range, loss if breached
- no_trade: 0 P&L

This is a signal-level simulation (not full option pricing), but gives a
directional comparison of old vs new model performance.
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_ROOT)

from shared.feature_compute import compute_all_features, load_dhan_options_features

TEST_START = "2024-01-01"

# ─── Load data ───
print("Loading data and models...")
merged_path = os.path.join(BACKEND_ROOT, "data", "merged_daily.parquet")
merged_daily = pd.read_parquet(merged_path)

dhan_raw = os.path.join(BACKEND_ROOT, "data", "dhan_raw_options.parquet")
dhan_skew = os.path.join(BACKEND_ROOT, "data", "daily_iv_skew_params.parquet")
dhan_features = load_dhan_options_features(dhan_raw, dhan_skew)

df, feature_cols = compute_all_features(merged_daily, dhan_features,
                                         data_start="2014-01-01", data_end="2026-12-31")

# Load both models
new_model = joblib.load(os.path.join(BACKEND_ROOT, "ml", "downside_model_no_month.pkl"))
new_features = joblib.load(os.path.join(BACKEND_ROOT, "ml", "feature_names_no_month.pkl"))
old_model = joblib.load(os.path.join(BACKEND_ROOT, "ml", "downside_model_OLD_with_month.pkl"))
old_features = joblib.load(os.path.join(BACKEND_ROOT, "ml", "feature_names_OLD_with_month.pkl"))

# Filter test period
test_df = df[df.index >= TEST_START].copy()
test_df = test_df[~test_df['max_drawdown_30d'].isna()].copy()
print(f"Test period: {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} days)")

# Ensure month/is_volatile_month columns exist for old model
# (compute_all_features still computes them in the DataFrame even though
# they're no longer in BASE_FEATURE_COLS)
if 'month' not in test_df.columns:
    test_df['month'] = test_df.index.month
if 'is_volatile_month' not in test_df.columns:
    test_df['is_volatile_month'] = test_df['month'].isin([9, 10]).astype(int)

# Generate predictions
test_df['pred_new'] = new_model.predict(test_df[new_features].values)
test_df['pred_old'] = old_model.predict(test_df[old_features].values)
test_df['actual_dd'] = test_df['max_drawdown_30d']

# ─── Signal mapping functions ───
def signal_sharp(pred, t1, t2, t3):
    if pred > t1: return "bull_full", 1.0
    elif pred > t2: return "bull_half", 1.0
    elif pred > t3: return "iron_condor", 1.0
    else: return "no_trade", 0.0

def signal_graduated(pred, t1, t2, t3, floor=0.50, hw_pct=0.50):
    p = pred * 100
    t1p, t2p, t3p = t1*100, t2*100, t3*100
    if p > t1p + hw_pct: return "bull_full", 1.0
    elif p > t1p - hw_pct:
        progress = (t1p + hw_pct - p) / (2 * hw_pct)
        return "bull_full", round(1.0 - progress * (1.0 - floor), 3)
    elif p > t2p + hw_pct: return "bull_half", floor
    elif p > t2p - hw_pct: return "iron_condor", floor
    elif p > t3p + hw_pct: return "iron_condor", floor
    elif p > t3p - hw_pct:
        progress = (t3p + hw_pct - p) / (2 * hw_pct)
        mult = floor * (1.0 - progress)
        if mult > 0.1: return "iron_condor", round(mult, 3)
        return "no_trade", 0.0
    else: return "no_trade", 0.0

def signal_graduated_gentle(pred, t1, t2, t3):
    return signal_graduated(pred, t1, t2, t3, floor=0.80, hw_pct=0.25)

# ─── P&L simulation parameters ───
# Simplified credit spread economics:
#   - Bull put spread: sell 3% OTM put, buy 5.5% OTM put
#     Credit received ≈ 1.0-1.5% of notional (simplified to fixed amount per lot)
#     Max loss = spread width - credit = ~1.5% of notional
#   - Iron condor: credit from both sides
#     Credit ≈ 1.5-2.0%, max loss ≈ 1.0% per side
#
# Win condition: actual drawdown stays above the sold strike (3% OTM)
# Loss condition: drawdown breaches the sold strike

INITIAL_CAPITAL = 2_500_000
LOT_SIZE = 25
MARGIN_PER_LOT = 13_750

# Simplified per-trade economics (as fraction of margin deployed)
CREDIT_PCT = {
    "bull_full": 0.035,     # ~3.5% of margin as credit (full position)
    "bull_half": 0.035,     # same credit rate, but half position size
    "iron_condor": 0.045,   # IC collects more premium (both sides)
}

# Stop loss: position closed at 3x credit (net loss = 2x credit)
STOP_LOSS_MULT = 3.0

# Win rate model based on actual drawdown vs OTM distance
# Bull spread wins if actual drawdown > -3% (sold put OTM distance)
# IC wins if actual drawdown > -3% AND no upside spike > call side
BULL_LOSS_THRESHOLD = -0.03     # sold put is 3% OTM
IC_LOSS_THRESHOLD = -0.03       # put side same

# Profit target: close at 50% of max credit (early exit)
PROFIT_TARGET = 0.50

# Position sizing
POSITION_SIZE_PCT = {
    "v5.4.2": 0.20,
    "v5.4.3": 0.15,
    "v5.4.4": 0.20,
}
IC_POSITION_SIZE_PCT = {
    "v5.4.2": 0.20,
    "v5.4.3": 0.15,
    "v5.4.4": 0.20,
}


def simulate_pnl(predictions, actuals, signal_fn, signal_kwargs, t1, t2, t3,
                  version, position_size_pct, ic_position_size_pct):
    """Simulate P&L for a series of daily signals."""
    capital = INITIAL_CAPITAL
    total_pnl = 0.0
    trades = 0
    wins = 0
    losses = 0
    trade_pnl_list = []
    monthly_pnl = defaultdict(float)
    monthly_trades = defaultdict(int)

    # Simplified: one trade per signal day, held to outcome
    # In reality there's entry gap, concurrent limits, etc. — this is directional only.
    last_entry = None

    for i, (pred, actual_dd, date) in enumerate(zip(predictions, actuals, predictions.index)):
        sig, mult = signal_fn(pred, t1, t2, t3, **signal_kwargs)

        if sig == "no_trade":
            continue

        # Entry gap: skip if too close to last trade
        if last_entry is not None and (date - last_entry).days < 2:
            continue

        last_entry = date
        month_key = f"{date.year}-{date.month:02d}"

        # Position sizing
        if sig == "iron_condor":
            size_pct = ic_position_size_pct
        else:
            size_pct = position_size_pct

        deployed = capital * size_pct * mult
        lots = max(1, int(deployed / MARGIN_PER_LOT))
        margin_used = lots * MARGIN_PER_LOT

        # Credit received
        credit_rate = CREDIT_PCT.get(sig, CREDIT_PCT.get("bull_full"))
        credit = margin_used * credit_rate

        # Determine outcome based on actual drawdown
        if sig in ("bull_full", "bull_half"):
            threshold = BULL_LOSS_THRESHOLD
        else:  # iron_condor
            threshold = IC_LOSS_THRESHOLD

        if actual_dd > threshold:
            # WIN: drawdown stayed above our sold strike
            # Collect profit target (50% of credit for early exit)
            pnl = credit * PROFIT_TARGET
            wins += 1
        else:
            # LOSS: drawdown breached sold strike
            # Severity determines loss magnitude
            breach_severity = abs(actual_dd - threshold) / abs(threshold)

            if breach_severity < 0.5:
                # Minor breach: stopped out at ~2x credit loss
                pnl = -credit * (STOP_LOSS_MULT - 1)
            else:
                # Major breach: full stop loss hit
                pnl = -credit * STOP_LOSS_MULT
            losses += 1

        total_pnl += pnl
        capital += pnl
        trades += 1
        trade_pnl_list.append(pnl)
        monthly_pnl[month_key] += pnl
        monthly_trades[month_key] += 1

    return {
        "total_pnl": total_pnl,
        "return_pct": total_pnl / INITIAL_CAPITAL * 100,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / max(trades, 1) * 100,
        "final_capital": capital,
        "avg_trade_pnl": np.mean(trade_pnl_list) if trade_pnl_list else 0,
        "max_drawdown": min(np.minimum.accumulate(np.cumsum(trade_pnl_list))) if trade_pnl_list else 0,
        "monthly_pnl": dict(monthly_pnl),
        "monthly_trades": dict(monthly_trades),
    }


# ─── Run simulations ───
print("\n" + "=" * 90)
print("RETURNS SIMULATION: NEW MODEL vs OLD MODEL (2024-2025)")
print("=" * 90)

# Thresholds
NEW_T1, NEW_T2, NEW_T3 = -0.020, -0.047, -0.076
OLD_T1, OLD_T2, OLD_T3 = -0.038, -0.050, -0.065

versions = {
    "v5.4.2": {"fn": signal_sharp, "kwargs": {}},
    "v5.4.3": {"fn": signal_graduated, "kwargs": {"floor": 0.50, "hw_pct": 0.50}},
    "v5.4.4": {"fn": signal_graduated_gentle, "kwargs": {}},
}

for ver, cfg in versions.items():
    print(f"\n{'─' * 90}")
    print(f"  {ver}")
    print(f"{'─' * 90}")

    pos_pct = POSITION_SIZE_PCT[ver]
    ic_pct = IC_POSITION_SIZE_PCT[ver]

    # New model + new thresholds
    new_result = simulate_pnl(
        test_df['pred_new'], test_df['actual_dd'], cfg["fn"], cfg["kwargs"],
        NEW_T1, NEW_T2, NEW_T3, ver, pos_pct, ic_pct
    )

    # Old model + old thresholds
    old_result = simulate_pnl(
        test_df['pred_old'], test_df['actual_dd'], cfg["fn"], cfg["kwargs"],
        OLD_T1, OLD_T2, OLD_T3, ver, pos_pct, ic_pct
    )

    # Old model + new thresholds (to isolate threshold vs model effect)
    hybrid_result = simulate_pnl(
        test_df['pred_old'], test_df['actual_dd'], cfg["fn"], cfg["kwargs"],
        NEW_T1, NEW_T2, NEW_T3, ver, pos_pct, ic_pct
    )

    print(f"\n  {'Metric':<25} {'New Model+Thresh':>18} {'Old Model+Thresh':>18} {'Old Model+NewThr':>18}")
    print(f"  {'-'*80}")

    for label, n, o, h in [
        ("Total P&L", f"₹{new_result['total_pnl']:,.0f}", f"₹{old_result['total_pnl']:,.0f}", f"₹{hybrid_result['total_pnl']:,.0f}"),
        ("Return %", f"{new_result['return_pct']:+.2f}%", f"{old_result['return_pct']:+.2f}%", f"{hybrid_result['return_pct']:+.2f}%"),
        ("Trades", f"{new_result['trades']}", f"{old_result['trades']}", f"{hybrid_result['trades']}"),
        ("Wins / Losses", f"{new_result['wins']} / {new_result['losses']}", f"{old_result['wins']} / {old_result['losses']}", f"{hybrid_result['wins']} / {hybrid_result['losses']}"),
        ("Win Rate", f"{new_result['win_rate']:.1f}%", f"{old_result['win_rate']:.1f}%", f"{hybrid_result['win_rate']:.1f}%"),
        ("Avg Trade P&L", f"₹{new_result['avg_trade_pnl']:,.0f}", f"₹{old_result['avg_trade_pnl']:,.0f}", f"₹{hybrid_result['avg_trade_pnl']:,.0f}"),
        ("Max Drawdown", f"₹{new_result['max_drawdown']:,.0f}", f"₹{old_result['max_drawdown']:,.0f}", f"₹{hybrid_result['max_drawdown']:,.0f}"),
        ("Final Capital", f"₹{new_result['final_capital']:,.0f}", f"₹{old_result['final_capital']:,.0f}", f"₹{hybrid_result['final_capital']:,.0f}"),
    ]:
        print(f"  {label:<25} {n:>18} {o:>18} {h:>18}")

    # Monthly breakdown
    print(f"\n  Monthly P&L (New Model):")
    print(f"  {'Month':<10} {'P&L':>12} {'Trades':>8} {'Avg/Trade':>12}")
    print(f"  {'-'*45}")
    for month_key in sorted(new_result['monthly_pnl'].keys()):
        pnl = new_result['monthly_pnl'][month_key]
        tr = new_result['monthly_trades'][month_key]
        avg = pnl / tr if tr > 0 else 0
        print(f"  {month_key:<10} ₹{pnl:>11,.0f} {tr:>8} ₹{avg:>11,.0f}")

    total_new = sum(new_result['monthly_pnl'].values())
    total_trades = sum(new_result['monthly_trades'].values())
    print(f"  {'TOTAL':<10} ₹{total_new:>11,.0f} {total_trades:>8} ₹{total_new/max(total_trades,1):>11,.0f}")

    # Monthly comparison
    print(f"\n  Monthly P&L comparison (New vs Old):")
    print(f"  {'Month':<10} {'New Model':>12} {'Old Model':>12} {'Delta':>12}")
    print(f"  {'-'*50}")
    all_months = sorted(set(list(new_result['monthly_pnl'].keys()) + list(old_result['monthly_pnl'].keys())))
    for mk in all_months:
        np_ = new_result['monthly_pnl'].get(mk, 0)
        op = old_result['monthly_pnl'].get(mk, 0)
        delta = np_ - op
        print(f"  {mk:<10} ₹{np_:>11,.0f} ₹{op:>11,.0f} ₹{delta:>+11,.0f}")


# ─── Summary comparison ───
print(f"\n{'=' * 90}")
print("SUMMARY")
print(f"{'=' * 90}")
print(f"\n  {'Version':<10} {'New Return%':>14} {'Old Return%':>14} {'Improvement':>14}")
print(f"  {'-'*55}")
for ver, cfg in versions.items():
    pos_pct = POSITION_SIZE_PCT[ver]
    ic_pct = IC_POSITION_SIZE_PCT[ver]

    new_r = simulate_pnl(test_df['pred_new'], test_df['actual_dd'], cfg["fn"], cfg["kwargs"],
                          NEW_T1, NEW_T2, NEW_T3, ver, pos_pct, ic_pct)
    old_r = simulate_pnl(test_df['pred_old'], test_df['actual_dd'], cfg["fn"], cfg["kwargs"],
                          OLD_T1, OLD_T2, OLD_T3, ver, pos_pct, ic_pct)
    delta = new_r['return_pct'] - old_r['return_pct']
    print(f"  {ver:<10} {new_r['return_pct']:>+13.2f}% {old_r['return_pct']:>+13.2f}% {delta:>+13.2f}%")

print(f"\n  Note: Simplified simulation — assumes one trade per signal day with 2-day gap,")
print(f"  fixed credit rates, and binary win/loss based on actual max_drawdown_30d vs 3% OTM.")
print(f"  Real returns depend on option pricing, IV, DTE, and actual exit timing.")
