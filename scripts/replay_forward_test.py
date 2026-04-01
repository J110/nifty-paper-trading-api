"""
Replay forward test trades with backtest-aligned exit logic.
Compares OLD vs NEW exit behavior using daily close prices.

Usage:
    cd nifty-paper-trading-api
    python scripts/replay_forward_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta
from core.option_pricer import compute_spread_value
from config import RISK_FREE_RATE, NIFTY_LOT_SIZE, VERSION_CONFIGS

# ── Forward test daily close prices (from Dhan/chart API) + real VIX (from /api/indicators) ──
DAILY_PRICES = {
    date(2026, 2, 13): {"spot": 25471.1, "vix": 11.73},
    date(2026, 2, 16): {"spot": 25682.75, "vix": 12.25},
    date(2026, 2, 17): {"spot": 25725.4, "vix": 13.33},
    date(2026, 2, 18): {"spot": 25819.35, "vix": 12.67},
    date(2026, 2, 19): {"spot": 25454.35, "vix": 13.21},
    date(2026, 2, 20): {"spot": 25571.25, "vix": 13.46},
    date(2026, 2, 23): {"spot": 25713.0, "vix": 14.17},
    date(2026, 2, 24): {"spot": 25424.65, "vix": 14.17},
    date(2026, 2, 25): {"spot": 25482.5, "vix": 14.15},
    date(2026, 2, 26): {"spot": 25496.55, "vix": 13.49},
    date(2026, 2, 27): {"spot": 25178.65, "vix": 13.70},
    date(2026, 3, 2): {"spot": 24865.7, "vix": 13.70},
    date(2026, 3, 4): {"spot": 24480.5, "vix": 17.13},
    date(2026, 3, 5): {"spot": 24765.9, "vix": 21.14},
    date(2026, 3, 6): {"spot": 24450.45, "vix": 17.86},
    date(2026, 3, 9): {"spot": 24028.05, "vix": 23.36},
    date(2026, 3, 10): {"spot": 24261.6, "vix": 23.36},
    date(2026, 3, 11): {"spot": 23866.85, "vix": 18.91},
    date(2026, 3, 12): {"spot": 23639.15, "vix": 21.06},
    date(2026, 3, 13): {"spot": 23151.1, "vix": 21.52},
    date(2026, 3, 16): {"spot": 23408.8, "vix": 21.76},
    date(2026, 3, 17): {"spot": 23581.15, "vix": 19.79},
    date(2026, 3, 18): {"spot": 23777.8, "vix": 19.79},
    date(2026, 3, 19): {"spot": 23002.15, "vix": 18.72},
    date(2026, 3, 20): {"spot": 23114.5, "vix": 22.80},
    date(2026, 3, 23): {"spot": 22512.65, "vix": 22.81},
    date(2026, 3, 25): {"spot": 23306.45, "vix": 24.74},
    date(2026, 3, 26): {"spot": 23306.45, "vix": 24.64},  # entry day (from prediction/Dhan)
    date(2026, 3, 27): {"spot": 22819.6, "vix": 24.64},
    date(2026, 3, 30): {"spot": 22331.4, "vix": 26.80},
    date(2026, 3, 31): {"spot": 22331.4, "vix": 27.89},
    date(2026, 4, 1): {"spot": 22868.0, "vix": 27.89},
}

# ── Forward test losing trades (from API) ──
LOSING_TRADES = [
    {
        "trade_id": "v542-2026-03-11-bull_half",
        "version": "v5.4.2",
        "entry_date": date(2026, 3, 11),
        "entry_spot": 24285.90,
        "trade_type": "bull_put",
        "sell_strike": 23550.0,
        "buy_strike": 22950.0,
        "ic_call_sell": None,
        "ic_call_buy": None,
        "expiry": date(2026, 3, 12),
        "credit_received": 0.16,  # per share (v5.4.3 shows 0.16)
        "num_lots": 36,
        "capital_deployed": 495000.0,
        "actual_exit_date": date(2026, 3, 12),
        "actual_exit_reason": "expiry",
        "actual_realized_pnl": -110086.04,
    },
    {
        "trade_id": "v542-2026-03-18-bull_full",
        "version": "v5.4.2",
        "entry_date": date(2026, 3, 18),
        "entry_spot": 23721.60,
        "trade_type": "bull_put",
        "sell_strike": 23000.0,
        "buy_strike": 22400.0,
        "ic_call_sell": None,
        "ic_call_buy": None,
        "expiry": date(2026, 3, 19),
        "credit_received": 0.18,
        "num_lots": 36,
        "capital_deployed": 495000.0,
        "actual_exit_date": date(2026, 3, 19),
        "actual_exit_reason": "expiry",
        "actual_realized_pnl": -21852.38,
    },
    {
        "trade_id": "v542-2026-03-26-bull_half",
        "version": "v5.4.2",
        "entry_date": date(2026, 3, 26),
        "entry_spot": 23306.45,
        "trade_type": "bull_put",
        "sell_strike": 22600.0,
        "buy_strike": 22000.0,
        "ic_call_sell": None,
        "ic_call_buy": None,
        "expiry": date(2026, 4, 2),
        "credit_received": 78.54,
        "num_lots": 36,
        "capital_deployed": 495000.0,
        "actual_exit_date": date(2026, 4, 1),
        "actual_exit_reason": "expiry",
        "actual_realized_pnl": -238404.91,
    },
]


def check_exit_old(trade, spot, vix, trade_date, cfg):
    """OLD exit logic: expiry first, >= operator, flat trailing stop."""
    dte = (trade["expiry"] - trade_date).days
    T = max(dte / 365.0, 1 / 365.0)
    sigma = vix / 100.0 if vix > 0 else 0.15

    cv = compute_spread_value(
        trade["trade_type"], spot,
        trade["sell_strike"], trade["buy_strike"],
        trade.get("ic_call_sell"), trade.get("ic_call_buy"),
        T, RISK_FREE_RATE, sigma,
    )
    pnl_pct = (trade["credit_received"] - cv) / trade["credit_received"] if trade["credit_received"] > 0 else 0

    # 1. Expiry FIRST
    if dte <= cfg.get("MIN_DTE_EXIT", 1):
        return "expiry", cv, pnl_pct

    # 2. Profit target
    days_held = (trade_date - trade["entry_date"]).days
    if days_held <= 3:
        pt = cfg.get("PROFIT_TARGET_EARLY", 0.50)
    elif days_held <= 7:
        pt = cfg.get("PROFIT_TARGET_MID", 0.65)
    else:
        pt = cfg.get("PROFIT_TARGET_LATE", 0.80)
    if pnl_pct >= pt:
        return "profit_target", cv, pnl_pct

    # 3. Trailing stop (old: gated by activation, flat subtraction)
    ta = cfg.get("TRAILING_STOP_ACTIVATE", 0.40)
    tl = cfg.get("TRAILING_STOP_LEVEL", 0.10)
    if pnl_pct >= ta:
        peak = trade.get("peak_pnl_pct", 0.0)
        if peak > 0 and pnl_pct < peak - tl:
            return "trailing_stop", cv, pnl_pct
        trade["peak_pnl_pct"] = max(peak, pnl_pct)

    # 4. Stop loss (old: >= operator)
    sl_mult = cfg.get("STOP_LOSS_MULTIPLIER", 3.0)
    cd = cfg.get("STOP_LOSS_CONFIRM_DAYS", 2)
    if cv >= trade["credit_received"] * sl_mult + trade["credit_received"]:
        last_breach = trade.get("sl_last_breach_date")
        if last_breach != trade_date:
            trade["sl_breach_days"] = trade.get("sl_breach_days", 0) + 1
            trade["sl_last_breach_date"] = trade_date
        if trade["sl_breach_days"] >= cd:  # OLD: >=
            return "stop_loss", cv, pnl_pct
    else:
        trade["sl_breach_days"] = 0
        trade["sl_last_breach_date"] = None

    return None, cv, pnl_pct


def check_exit_new(trade, spot, vix, trade_date, cfg, is_eod=False):
    """NEW exit logic: backtest-aligned. Expiry EOD-only, > operator, proportional trailing."""
    dte = (trade["expiry"] - trade_date).days
    T = max(dte / 365.0, 1 / 365.0)
    sigma = vix / 100.0 if vix > 0 else 0.15

    cv = compute_spread_value(
        trade["trade_type"], spot,
        trade["sell_strike"], trade["buy_strike"],
        trade.get("ic_call_sell"), trade.get("ic_call_buy"),
        T, RISK_FREE_RATE, sigma,
    )
    pnl_pct = (trade["credit_received"] - cv) / trade["credit_received"] if trade["credit_received"] > 0 else 0

    # Always update peak unconditionally
    trade["peak_pnl_pct"] = max(trade.get("peak_pnl_pct", 0.0), pnl_pct)

    # 1. Profit target
    days_held = (trade_date - trade["entry_date"]).days
    if days_held <= 3:
        pt = cfg.get("PROFIT_TARGET_EARLY", 0.50)
    elif days_held <= 7:
        pt = cfg.get("PROFIT_TARGET_MID", 0.65)
    else:
        pt = cfg.get("PROFIT_TARGET_LATE", 0.80)
    if pnl_pct >= pt:
        return "profit_target", cv, pnl_pct

    # 2. Trailing stop (persistent activation, proportional retracement, pnl > 0)
    ta = cfg.get("TRAILING_STOP_ACTIVATE", 0.40)
    tl = cfg.get("TRAILING_STOP_LEVEL", 0.10)
    if pnl_pct >= ta:
        trade["trailing_active"] = True
    if trade.get("trailing_active"):
        peak = trade.get("peak_pnl_pct", 0.0)
        if peak > 0:
            retrace = peak - pnl_pct
            if retrace > peak * tl and pnl_pct > 0:
                return "trailing_stop", cv, pnl_pct

    # 3. Stop loss (> operator, checked BEFORE expiry)
    sl_mult = cfg.get("STOP_LOSS_MULTIPLIER", 3.0)
    cd = cfg.get("STOP_LOSS_CONFIRM_DAYS", 2)
    if cv >= trade["credit_received"] * sl_mult + trade["credit_received"]:
        last_breach = trade.get("sl_last_breach_date")
        if last_breach != trade_date:
            trade["sl_breach_days"] = trade.get("sl_breach_days", 0) + 1
            trade["sl_last_breach_date"] = trade_date
        if trade["sl_breach_days"] > cd:  # strict >
            return "stop_loss", cv, pnl_pct
    else:
        trade["sl_breach_days"] = 0
        trade["sl_last_breach_date"] = None

    # 4. VIX spike
    vix_threshold = cfg.get("VIX_SPIKE_EXIT")
    entry_vix = trade.get("entry_vix", 15.0)
    if vix_threshold and vix > vix_threshold and entry_vix and vix > entry_vix * 1.3:
        return "vix_spike", cv, pnl_pct

    # 5. Expiry — EOD only (matches backtest daily-close evaluation)
    if is_eod and dte <= cfg.get("MIN_DTE_EXIT", 1):
        return "expiry", cv, pnl_pct

    return None, cv, pnl_pct


def replay_trade(trade_template, check_fn, label):
    """Replay a single trade through daily prices using given exit function."""
    trade = dict(trade_template)
    trade["peak_pnl_pct"] = 0.0
    trade["trailing_active"] = False
    trade["sl_breach_days"] = 0
    trade["sl_last_breach_date"] = None
    trade["entry_vix"] = DAILY_PRICES.get(trade["entry_date"], {}).get("vix", 15.0)

    cfg = VERSION_CONFIGS["v5.4.2"]
    all_dates = sorted(DAILY_PRICES.keys())

    print(f"\n{'='*80}")
    print(f"  {label}: {trade['trade_id']}")
    print(f"  Entry: {trade['entry_date']}, Expiry: {trade['expiry']}")
    print(f"  Sell: {trade['sell_strike']}, Buy: {trade['buy_strike']}, Credit: {trade['credit_received']:.2f}/share")
    print(f"{'='*80}")

    for d in all_dates:
        if d < trade["entry_date"]:
            continue
        if d > trade["expiry"]:
            break

        dp = DAILY_PRICES[d]
        spot = dp["spot"]
        vix = dp["vix"]

        # Daily replay = EOD check (is_eod=True for new logic)
        if check_fn == check_exit_new:
            exit_reason, cv, pnl_pct = check_fn(trade, spot, vix, d, cfg, is_eod=True)
        else:
            exit_reason, cv, pnl_pct = check_fn(trade, spot, vix, d, cfg)

        dte = (trade["expiry"] - d).days
        sl_mult = cfg.get("STOP_LOSS_MULTIPLIER", 3.0)
        sl_threshold = trade["credit_received"] * (1 + sl_mult)
        breach = "BREACH" if cv >= sl_threshold else ""
        breach_days = trade.get("sl_breach_days", 0)

        status = f"EXIT: {exit_reason}" if exit_reason else "open"
        print(f"  {d} | spot={spot:>10.2f} | DTE={dte} | spread_val={cv:>8.2f} | "
              f"pnl%={pnl_pct:>+8.2%} | SL_thresh={sl_threshold:>8.2f} {breach:>6s} | "
              f"breach_days={breach_days} | peak={trade.get('peak_pnl_pct', 0):>+.2%} | {status}")

        if exit_reason:
            pnl_per_unit = trade["credit_received"] - cv
            realized_pnl = pnl_per_unit * trade["num_lots"] * NIFTY_LOT_SIZE
            rpnl_pct = realized_pnl / trade["capital_deployed"] * 100
            print(f"\n  >>> Closed as '{exit_reason}' on {d}")
            print(f"  >>> Realized PnL: Rs. {realized_pnl:,.2f} ({rpnl_pct:+.2f}% of capital)")
            print(f"  >>> Actual was: '{trade_template['actual_exit_reason']}' on {trade_template['actual_exit_date']}")
            print(f"  >>> Actual PnL:  Rs. {trade_template['actual_realized_pnl']:,.2f}")
            return {
                "trade_id": trade["trade_id"],
                "new_exit_date": d,
                "new_exit_reason": exit_reason,
                "new_realized_pnl": realized_pnl,
                "old_exit_date": trade_template["actual_exit_date"],
                "old_exit_reason": trade_template["actual_exit_reason"],
                "old_realized_pnl": trade_template["actual_realized_pnl"],
            }

    print("  >>> Trade still open at end of data!")
    return None


def main():
    print("=" * 80)
    print("FORWARD TEST REPLAY: OLD vs NEW EXIT LOGIC (v5.4.2)")
    print("=" * 80)

    for trade in LOSING_TRADES:
        print("\n\n" + "#" * 80)
        print(f"# TRADE: {trade['trade_id']}")
        print("#" * 80)

        old_result = replay_trade(trade, check_exit_old, "OLD LOGIC")
        new_result = replay_trade(trade, check_exit_new, "NEW LOGIC (backtest-aligned)")

        if old_result and new_result:
            diff = new_result["new_realized_pnl"] - old_result["new_realized_pnl"]
            print(f"\n  *** DIFFERENCE: Rs. {diff:+,.2f} ***")
            if new_result["new_exit_reason"] != old_result["new_exit_reason"]:
                print(f"  *** EXIT REASON CHANGED: {old_result['new_exit_reason']} -> {new_result['new_exit_reason']} ***")
            if new_result["new_exit_date"] != old_result["new_exit_date"]:
                print(f"  *** EXIT DATE CHANGED: {old_result['new_exit_date']} -> {new_result['new_exit_date']} ***")

    print("\n\nNOTE: VIX data is unavailable (defaulting to 15.0). VIX spike exit cannot be evaluated.")
    print("NOTE: This uses daily close prices only. Intraday 5-min checks may produce different results.")


if __name__ == "__main__":
    main()
