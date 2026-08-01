"""
Intraday exit-replay harness for the forward test.

Re-evaluates each version's exits at INTRADAY resolution (15-min price
snapshots) to test whether intraday stops change the drawdown picture vs the
daily-close recompute. Pure functions — no DB. The caller (the
/api/debug/intraday-replay endpoint, or a unit test) supplies predictions +
the intraday price path.

Why this exists: the daily-close recompute (main.py) can only fire stops on the
3:30pm close, but live all 3 versions check stop-loss intraday every 5 min
(v5.4.3 runs FULL checks intraday; v5.4.2/v5.4.4 run loss-only intraday with
profit-target/trailing at EOD + a 12:30 midday check for v5.4.2). So the daily
sim systematically under-fires stops and over-attributes losses to `expiry`.

breach_mode brackets the live answer:
  'close'   — one EOD mark/day from the merged-daily close → REPRODUCES the
              daily-close recompute (a correctness check on this engine).
  'spot'    — 15-min point samples (discrete checks, coarser than live 5-min →
              tends to UNDER-fire).
  'extreme' — 15-min interval low/high (continuous monitoring → tends to
              OVER-fire).
The true live result sits between 'spot' and 'extreme'.

Entry/exit ORDERING matches the recompute exactly (per day: check exits on
prior-day opens, THEN open new trades), so the ONLY variable that changes
between modes is the exit-check price path.
"""

from datetime import date, time

from config import (
    VERSION_CONFIGS, INITIAL_CAPITAL, HISTORICAL_LOT_SIZE, RISK_FREE_RATE,
    MARGIN_PER_LOT_BULL, MARGIN_PER_LOT_IC, BULL_OTM_SELL, BULL_OTM_BUY,
    MIN_TOTAL_CREDIT, MIN_ENTRY_DTE,
)
from core.signal_mapper import map_signal
from core.option_pricer import (
    select_strikes, price_bull_put_spread, price_iron_condor,
    compute_spread_value, get_next_weekly_expiry,
)

EOD_TIME = time(15, 25)


# ---------------------------------------------------------------------------
# Pricing helper
# ---------------------------------------------------------------------------

def _value(trade, spot, vix, day):
    """Black-Scholes value of the spread (cost to close) at *spot*/*vix* on *day*."""
    dte = (trade["expiry"] - day).days
    T = max(dte / 365.0, 1 / 365.0)
    sigma = vix / 100.0 if vix and vix > 0 else 0.15
    return compute_spread_value(
        trade["trade_type"], spot,
        trade["sell_strike"], trade["buy_strike"],
        trade.get("ic_call_sell"), trade.get("ic_call_buy"),
        T, RISK_FREE_RATE, sigma,
    )


# ---------------------------------------------------------------------------
# Exit check — a faithful port of trade_manager._check_single_exit, driven by
# one intraday mark. Returns (exit_reason, realize_value) or (None, None).
# ---------------------------------------------------------------------------

def _check_exit(trade, day, spot, low, high, vix, cfg, loss_only, is_eod, breach_mode):
    dte = (trade["expiry"] - day).days

    val_spot = _value(trade, spot, vix, day)
    if breach_mode == "extreme" and low is not None and high is not None:
        # Worst single-spot value over the interval: low hurts the put side,
        # high hurts the IC call side. max() picks the adverse extreme.
        val_breach = max(_value(trade, low, vix, day), _value(trade, high, vix, day))
    else:
        val_breach = val_spot

    credit = trade["credit"]
    pnl_pct = (credit - val_spot) / credit if credit > 0 else 0.0

    # Peak tracking — always (matches _check_single_exit).
    trade["peak_pnl_pct"] = max(trade.get("peak_pnl_pct", 0.0), pnl_pct)

    # 1. Profit target — only when not loss_only (EOD for v5.4.2/4; every mark for v5.4.3)
    if not loss_only:
        days_held = (day - trade["entry_date"]).days
        if days_held <= 3:
            pt = cfg.get("PROFIT_TARGET_EARLY", 0.50)
        elif days_held <= 7:
            pt = cfg.get("PROFIT_TARGET_MID", 0.65)
        else:
            pt = cfg.get("PROFIT_TARGET_LATE", 0.80)
        if pnl_pct >= pt:
            return "profit_target", val_spot

    # 2. Trailing stop — activation persistent; exit only when not loss_only
    if pnl_pct >= cfg.get("TRAILING_STOP_ACTIVATE", 0.40):
        trade["trailing_stop_active"] = True
    if not loss_only and trade.get("trailing_stop_active"):
        peak = trade.get("peak_pnl_pct", 0.0)
        if peak > 0:
            retrace = peak - pnl_pct
            if retrace > peak * cfg.get("TRAILING_STOP_LEVEL", 0.10) and pnl_pct > 0:
                return "trailing_stop", val_spot

    # 3. Stop loss — always checked; per-CALENDAR-DAY breach with N-day confirm
    sl_mult = cfg.get("IC_STOP_LOSS_MULTIPLIER", 3.0) if trade["trade_type"] == "iron_condor" \
        else cfg.get("STOP_LOSS_MULTIPLIER", 3.0)
    arming_pct = cfg.get("STOP_LOSS_ARMING_PCT")
    if arming_pct is not None and trade.get("peak_pnl_pct", 0.0) < arming_pct:
        sl_mult = cfg.get("STOP_LOSS_ARMING_MULT", 3.0)
    confirm_days = cfg.get("IC_STOP_LOSS_CONFIRM_DAYS", 2) if trade["trade_type"] == "iron_condor" \
        else cfg.get("STOP_LOSS_CONFIRM_DAYS", 2)

    if val_breach >= credit * sl_mult + credit:
        if trade.get("stop_loss_last_breach_date") != day:
            trade["stop_loss_breach_days"] = trade.get("stop_loss_breach_days", 0) + 1
            trade["stop_loss_last_breach_date"] = day
        if trade["stop_loss_breach_days"] > confirm_days:
            return "stop_loss", val_breach
    else:
        trade["stop_loss_breach_days"] = 0
        trade["stop_loss_last_breach_date"] = None

    # 4. VIX spike — always
    vix_thr = cfg.get("VIX_SPIKE_EXIT")
    if vix_thr and vix and trade.get("entry_vix") and vix > vix_thr and vix > trade["entry_vix"] * 1.3:
        return "vix_spike", val_spot

    # 5. Expiry — EOD only
    if is_eod and dte <= cfg.get("MIN_DTE_EXIT", 1):
        return "expiry", val_spot

    return None, None


# ---------------------------------------------------------------------------
# Per-version intraday cadence: does this mark run PT/trailing (full check)?
# ---------------------------------------------------------------------------

def _full_check(version, mtime, is_eod):
    if version == "v5.4.3":
        return True                       # full checks every mark
    if is_eod:
        return True                       # v5.4.2/4 run PT/trailing at EOD
    if version == "v5.4.2" and mtime.hour == 12 and 28 <= mtime.minute <= 32:
        return True                       # v5.4.2 12:30 midday PT/trailing
    return False                          # otherwise loss-only intraday


# ---------------------------------------------------------------------------
# Day marks: build the ordered list of (time, spot, low, high, vix, is_eod).
# Intraday snapshots (is_eod=False) first, then a final EOD mark from the
# merged-daily close (is_eod=True) — so EOD/expiry valuation is identical across
# modes and only the intraday checks differ.
# ---------------------------------------------------------------------------

def _day_marks(day, intraday_by_day, daily_close, mode):
    cl = daily_close[day]
    eod_mark = (EOD_TIME, cl["spot"], cl["spot"], cl["spot"], cl.get("vix"), True)
    if mode == "close":
        return [eod_mark]
    snaps = intraday_by_day.get(day)
    if not snaps:
        return [eod_mark]                 # fallback day → daily close only
    marks = [(t, s, lo, hi, vx, False) for (t, s, lo, hi, vx) in sorted(snaps)]
    marks.append(eod_mark)
    return marks


# ---------------------------------------------------------------------------
# Open a new trade — faithful port of the recompute open logic incl. the
# DTE/credit entry gate. Returns a trade dict or None.
# ---------------------------------------------------------------------------

def _try_open(version, cfg, day, pred, open_trades, last_entry_date):
    signal = map_signal(pred["predicted_drawdown"], cfg)
    tt = signal["trade_type"]
    if signal["signal"] == "no_trade" or not tt:
        return None

    open_count = sum(1 for t in open_trades if t["trade_type"] == tt)
    max_c = cfg.get("IC_MAX_CONCURRENT", 2) if tt == "iron_condor" else cfg.get("MAX_CONCURRENT_POSITIONS", 3)
    if open_count >= max_c:
        return None

    min_gap = cfg.get("MIN_ENTRY_GAP_DAYS", 2)
    if last_entry_date and (day - last_entry_date).days < min_gap:
        return None

    entry_spot = pred["spot"]
    vix = pred["vix"] if pred.get("vix") else 15.0
    strikes = select_strikes(entry_spot, tt, {
        "BULL_OTM_SELL": BULL_OTM_SELL, "BULL_OTM_BUY": BULL_OTM_BUY,
        "IC_PUT_OTM_SELL": cfg.get("IC_PUT_OTM_SELL", 0.03),
        "IC_PUT_OTM_BUY": cfg.get("IC_PUT_OTM_BUY", 0.055),
        "IC_CALL_OTM_SELL": cfg.get("IC_CALL_OTM_SELL", 0.04),
        "IC_CALL_OTM_BUY": cfg.get("IC_CALL_OTM_BUY", 0.065),
    })
    expiry = get_next_weekly_expiry(day)
    T = max((expiry - day).days / 365.0, 1 / 365.0)
    sigma = vix / 100.0 if vix > 0 else 0.15
    if tt == "bull_put":
        credit = price_bull_put_spread(entry_spot, strikes["sell_strike"], strikes["buy_strike"],
                                       T, RISK_FREE_RATE, sigma, apply_slippage=True)
    elif tt == "iron_condor":
        credit = price_iron_condor(entry_spot, strikes["sell_strike"], strikes["buy_strike"],
                                   strikes["ic_call_sell"], strikes["ic_call_buy"],
                                   T, RISK_FREE_RATE, sigma, apply_slippage=True)
    else:
        return None

    pos_pct = cfg.get("IC_POSITION_SIZE_PCT", 0.15) if tt == "iron_condor" else cfg.get("POSITION_SIZE_PCT", 0.20)
    margin = MARGIN_PER_LOT_IC if tt == "iron_condor" else MARGIN_PER_LOT_BULL
    num_lots = max(1, int(INITIAL_CAPITAL * pos_pct * signal["size_mult"] / margin))
    total_credit = credit * num_lots * HISTORICAL_LOT_SIZE

    dte_at_entry = (expiry - day).days
    if total_credit < MIN_TOTAL_CREDIT or dte_at_entry < MIN_ENTRY_DTE:
        return None  # entry quality gate (matches live open_trade)

    return {
        "trade_id": f"{version.replace('.', '')}-{day.isoformat()}-{signal['signal']}",
        "version": version, "trade_type": tt, "signal": signal["signal"],
        "entry_date": day, "expiry": expiry,
        "sell_strike": strikes["sell_strike"], "buy_strike": strikes["buy_strike"],
        "ic_call_sell": strikes.get("ic_call_sell"), "ic_call_buy": strikes.get("ic_call_buy"),
        "credit": credit, "num_lots": num_lots, "capital_deployed": num_lots * margin,
        "entry_vix": vix, "peak_pnl_pct": 0.0, "trailing_stop_active": False,
        "stop_loss_breach_days": 0, "stop_loss_last_breach_date": None,
    }


# ---------------------------------------------------------------------------
# Replay one version over the window for one breach_mode.
# ---------------------------------------------------------------------------

def replay(version, predictions, intraday_by_day, daily_close, breach_mode="spot"):
    """
    predictions:     {date: {"predicted_drawdown": float, "spot": float, "vix": float}}
    intraday_by_day: {date: [(time, spot, low, high, vix), ...]}
    daily_close:     {date: {"spot": float, "vix": float}}
    Returns a summary dict for (version, breach_mode).
    """
    cfg = VERSION_CONFIGS[version]
    open_trades, closed = [], []
    last_entry_date = None
    cum_pnl = 0.0
    equity = []  # list of (date, cumulative_return_pct) using realized + EOD unrealized

    for day in sorted(daily_close.keys()):
        # --- EXIT prior-day opens across the day's marks ---
        for (mtime, spot, low, high, vix, is_eod) in _day_marks(day, intraday_by_day, daily_close, breach_mode):
            survivors = []
            for t in open_trades:
                loss_only = not _full_check(version, mtime, is_eod)
                reason, val = _check_exit(t, day, spot, low, high, vix, cfg, loss_only, is_eod, breach_mode)
                if reason:
                    pnl = (t["credit"] - val) * t["num_lots"] * HISTORICAL_LOT_SIZE
                    t.update(exit_date=day, exit_reason=reason, realized_pnl=pnl)
                    cum_pnl += pnl
                    closed.append(t)
                else:
                    survivors.append(t)
            open_trades = survivors

        # --- OPEN new trade (after exits — matches recompute ordering) ---
        pred = predictions.get(day)
        if pred:
            t = _try_open(version, cfg, day, pred, open_trades, last_entry_date)
            if t:
                open_trades.append(t)
                last_entry_date = day

        # --- EOD equity (realized + unrealized MTM at the close) ---
        cl = daily_close[day]
        unreal = sum((t["credit"] - _value(t, cl["spot"], cl.get("vix"), day)) * t["num_lots"] * HISTORICAL_LOT_SIZE
                     for t in open_trades)
        equity.append((day, (cum_pnl + unreal) / INITIAL_CAPITAL * 100.0))

    # --- summary ---
    from collections import defaultdict
    by_reason = defaultdict(lambda: [0, 0.0])
    wins = 0
    for t in closed:
        by_reason[t["exit_reason"]][0] += 1
        by_reason[t["exit_reason"]][1] += t["realized_pnl"]
        if t["realized_pnl"] > 0:
            wins += 1

    curve = [r for (_, r) in equity]
    peak = float("-inf")
    max_dd = 0.0
    for r in curve:
        peak = max(peak, r)
        max_dd = min(max_dd, r - peak)

    realized = sum(t["realized_pnl"] for t in closed)
    return {
        "version": version,
        "breach_mode": breach_mode,
        "final_return_pct": round(realized / INITIAL_CAPITAL * 100.0, 2),
        "min_cum_return_pct": round(min(curve), 2) if curve else 0.0,
        "max_drawdown_pct": round(max_dd, 2),
        "closed_trades": len(closed),
        "open_at_end": len(open_trades),
        "win_rate": round(100.0 * wins / len(closed), 1) if closed else 0.0,
        "exit_reasons": {k: {"count": v[0], "pnl": round(v[1], 0)} for k, v in sorted(by_reason.items())},
    }
