"""
Real-priced backtest from NSE F&O Bhavcopy.

Re-runs each version over the forward-test window using REAL daily option prices
instead of synthetic Black-Scholes:
  entry credit -> OPEN (≈ the 9:20 entry)
  daily mark / exit value -> CLOSE
  stop-loss breach -> intraday extremes (short HIGH - long LOW)
  v5.4.3 profit-target/trailing -> intraday best (short LOW - long HIGH)

Drives entries from the stored predictions (pulled to /tmp/preds_by_date.json).
Strikes/expiries are snapped to what actually listed so coverage is maximal.

Run:  python3 scripts/bhavcopy_backtest.py
"""
import sys, os, zipfile, csv, io, json
from datetime import date, datetime
from collections import defaultdict

# Archived reference tool — hardcoded repo root so it runs from any location.
REPO = "/Users/anmolmohan/Music/Nifty/nifty-paper-trading-api"
sys.path.insert(0, REPO)
from config import (
    VERSION_CONFIGS, INITIAL_CAPITAL, NIFTY_LOT_SIZE,
    MARGIN_PER_LOT_BULL, MARGIN_PER_LOT_IC,
    BULL_OTM_SELL, BULL_OTM_BUY, MIN_TOTAL_CREDIT, MIN_ENTRY_DTE,
)
from core.signal_mapper import map_signal
from core.option_pricer import select_strikes, get_next_weekly_expiry

ZIPS = [f"/Users/anmolmohan/Downloads/{z}.zip" for z in (
    "2026-01-18-2026-02-17", "2026-02-18-2026-03-17", "2026-03-18-2026-04-17",
    "2026-04-18-2026-05-17", "2026-05-18-2026-06-17",
)]
WINDOW_START = date(2026, 1, 18)
WINDOW_END = date(2026, 6, 17)
O, H, L, C = 0, 1, 2, 3


# ─── load Bhavcopy ───────────────────────────────────────────────────────────
def load_bhavcopy():
    px = {}                       # (d, exp, strike, opt) -> (open, high, low, close)
    exps = defaultdict(set)       # d -> {expiries}
    strikes = defaultdict(set)    # (d, exp) -> {strikes}
    for z in ZIPS:
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if not name.endswith("_NSEFO.csv"):
                    continue
                try:
                    d = datetime.strptime(os.path.basename(name)[:8], "%Y%m%d").date()
                except ValueError:
                    continue
                with zf.open(name) as fh:
                    rdr = csv.reader(io.TextIOWrapper(fh, "utf-8", newline=""))
                    next(rdr, None)
                    for r in rdr:
                        if len(r) < 10 or r[0] != "OPTIDX" or r[1] != "NIFTY":
                            continue
                        try:
                            exp = datetime.strptime(r[3], "%Y-%m-%d").date()
                            k = float(r[4]); opt = r[5]
                            vals = (float(r[6]), float(r[7]), float(r[8]), float(r[9]))
                        except (ValueError, IndexError):
                            continue
                        px[(d, exp, k, opt)] = vals
                        exps[d].add(exp)
                        strikes[(d, exp)].add(k)
    return px, exps, strikes


print("Loading Bhavcopy ...")
PX, EXPS, STRIKES = load_bhavcopy()
print(f"  {len(PX):,} NIFTY option rows; {len(EXPS)} trading days")


def snap_expiry(d, target):
    fut = [e for e in EXPS.get(d, ()) if e >= d]
    return min(fut, key=lambda e: abs((e - target).days)) if fut else target


def snap_strike(d, exp, k):
    av = STRIKES.get((d, exp))
    if not av:
        return k
    return k if k in av else min(av, key=lambda s: abs(s - k))


def _leg(d, exp, k, opt, field):
    v = PX.get((d, exp, float(k), opt))
    return v[field] if v else None


def trade_legs(tt, st):
    if tt == "bull_put":
        return [("PE", "short", st["sell_strike"]), ("PE", "long", st["buy_strike"])]
    if tt == "iron_condor":
        return [("PE", "short", st["sell_strike"]), ("PE", "long", st["buy_strike"]),
                ("CE", "short", st["ic_call_sell"]), ("CE", "long", st["ic_call_buy"])]
    return None


def spread_value(d, exp, tt, st, mode):
    """Net spread value (credit if entry, cost-to-close if exit). modes:
    open/close = single field; worst = short HIGH - long LOW; best = short LOW - long HIGH."""
    legs = trade_legs(tt, st)
    if not legs:
        return None
    tot = 0.0
    for opt, side, k in legs:
        if mode in ("open", "close"):
            p = _leg(d, exp, k, opt, O if mode == "open" else C)
        elif mode == "worst":
            p = _leg(d, exp, k, opt, H if side == "short" else L)
        else:  # best
            p = _leg(d, exp, k, opt, L if side == "short" else H)
        if p is None:
            return None
        tot += p if side == "short" else -p
    return max(tot, 0.0)


# ─── per-trade exit logic (real prices, version cadence) ─────────────────────
def check_exit(t, d, vix, cfg, version, use_best):
    dte = (t["expiry"] - d).days
    close_v = spread_value(d, t["expiry"], t["trade_type"], t["strikes"], "close")
    if close_v is None:
        return None  # can't price today — hold
    worst_v = spread_value(d, t["expiry"], t["trade_type"], t["strikes"], "worst") or close_v
    best_v = spread_value(d, t["expiry"], t["trade_type"], t["strikes"], "best") or close_v
    credit = t["credit"]
    # use_best: take profit at the day's best (v5.4.3 intraday cadence); else at close
    fav = best_v if use_best else close_v
    pnl_fav = (credit - fav) / credit if credit > 0 else 0
    t["peak"] = max(t.get("peak", 0.0), pnl_fav)

    held = (d - t["entry_date"]).days
    pt = (cfg.get("PROFIT_TARGET_EARLY", 0.5) if held <= 3
          else cfg.get("PROFIT_TARGET_MID", 0.65) if held <= 7
          else cfg.get("PROFIT_TARGET_LATE", 0.8))
    if pnl_fav >= pt:
        return "profit_target", fav

    if pnl_fav >= cfg.get("TRAILING_STOP_ACTIVATE", 0.4):
        t["trail"] = True
    if t.get("trail"):
        pk = t["peak"]
        if pk > 0 and (pk - pnl_fav) > pk * cfg.get("TRAILING_STOP_LEVEL", 0.1) and pnl_fav > 0:
            return "trailing_stop", fav

    sl = (cfg.get("IC_STOP_LOSS_MULTIPLIER", 3.0) if t["trade_type"] == "iron_condor"
          else cfg.get("STOP_LOSS_MULTIPLIER", 3.0))
    arm = cfg.get("STOP_LOSS_ARMING_PCT")
    if arm is not None and t.get("peak", 0.0) < arm:
        sl = cfg.get("STOP_LOSS_ARMING_MULT", 3.0)
    cd = (cfg.get("IC_STOP_LOSS_CONFIRM_DAYS", 2) if t["trade_type"] == "iron_condor"
          else cfg.get("STOP_LOSS_CONFIRM_DAYS", 2))
    if worst_v >= credit * sl + credit:
        if t.get("last_breach") != d:
            t["breach"] = t.get("breach", 0) + 1
            t["last_breach"] = d
        if t["breach"] > cd:
            return "stop_loss", worst_v
    else:
        t["breach"] = 0
        t["last_breach"] = None

    thr = cfg.get("VIX_SPIKE_EXIT")
    if thr and vix and t.get("entry_vix") and vix > thr and vix > t["entry_vix"] * 1.3:
        return "vix_spike", close_v

    if dte <= cfg.get("MIN_DTE_EXIT", 1):
        return "expiry", close_v
    return None


def try_open(version, cfg, d, pred, open_trades, last_entry):
    sig = map_signal(pred["predicted_drawdown"], cfg)
    tt = sig["trade_type"]
    if sig["signal"] == "no_trade" or tt not in ("bull_put", "iron_condor"):
        return None
    max_c = cfg.get("IC_MAX_CONCURRENT", 2) if tt == "iron_condor" else cfg.get("MAX_CONCURRENT_POSITIONS", 3)
    if sum(1 for t in open_trades if t["trade_type"] == tt) >= max_c:
        return None
    if last_entry and (d - last_entry).days < cfg.get("MIN_ENTRY_GAP_DAYS", 2):
        return None
    spot = pred["spot"]
    st = select_strikes(spot, tt, {
        "BULL_OTM_SELL": BULL_OTM_SELL, "BULL_OTM_BUY": BULL_OTM_BUY,
        "IC_PUT_OTM_SELL": cfg.get("IC_PUT_OTM_SELL", 0.03), "IC_PUT_OTM_BUY": cfg.get("IC_PUT_OTM_BUY", 0.055),
        "IC_CALL_OTM_SELL": cfg.get("IC_CALL_OTM_SELL", 0.04), "IC_CALL_OTM_BUY": cfg.get("IC_CALL_OTM_BUY", 0.065),
    })
    expiry = snap_expiry(d, get_next_weekly_expiry(d))
    st = {k: (snap_strike(d, expiry, v) if v else v) for k, v in st.items()}
    credit = spread_value(d, expiry, tt, st, "open")
    if credit is None or credit <= 0:
        return "skip"
    pos_pct = cfg.get("IC_POSITION_SIZE_PCT", 0.15) if tt == "iron_condor" else cfg.get("POSITION_SIZE_PCT", 0.20)
    margin = MARGIN_PER_LOT_IC if tt == "iron_condor" else MARGIN_PER_LOT_BULL
    lots = max(1, int(INITIAL_CAPITAL * pos_pct * sig["size_mult"] / margin))
    total_credit = credit * lots * NIFTY_LOT_SIZE
    dte = (expiry - d).days
    if total_credit < MIN_TOTAL_CREDIT or dte < MIN_ENTRY_DTE:
        return None
    return {"trade_type": tt, "signal": sig["signal"], "entry_date": d, "expiry": expiry,
            "strikes": st, "credit": credit, "num_lots": lots,
            "capital_deployed": lots * margin, "entry_vix": pred.get("vix"),
            "peak": 0.0, "trail": False, "breach": 0, "last_breach": None}


def run_version(version, preds, dates, use_best):
    cfg = VERSION_CONFIGS[version]
    open_trades, closed = [], []
    last_entry, cum = None, 0.0
    equity, skipped = [], 0
    for d in dates:
        vix = preds.get(d.isoformat(), {}).get("vix")
        surv = []
        for t in open_trades:
            r = check_exit(t, d, vix, cfg, version, use_best)
            if r:
                reason, val = r
                pnl = (t["credit"] - val) * t["num_lots"] * NIFTY_LOT_SIZE
                t.update(exit_date=d, exit_reason=reason, realized_pnl=pnl)
                cum += pnl
                closed.append(t)
            else:
                surv.append(t)
        open_trades = surv
        p = preds.get(d.isoformat())
        if p:
            t = try_open(version, cfg, d, p, open_trades, last_entry)
            if t == "skip":
                skipped += 1
            elif t:
                open_trades.append(t)
                last_entry = d
        unreal = 0.0
        for t in open_trades:
            v = spread_value(d, t["expiry"], t["trade_type"], t["strikes"], "close")
            if v is not None:
                unreal += (t["credit"] - v) * t["num_lots"] * NIFTY_LOT_SIZE
        equity.append((cum + unreal) / INITIAL_CAPITAL * 100)

    by = defaultdict(lambda: [0, 0.0])
    wins = 0
    for t in closed:
        by[t["exit_reason"]][0] += 1
        by[t["exit_reason"]][1] += t["realized_pnl"]
        if t["realized_pnl"] > 0:
            wins += 1
    peak, mdd = float("-inf"), 0.0
    for r in equity:
        peak = max(peak, r)
        mdd = min(mdd, r - peak)
    return {
        "return_pct": round(sum(t["realized_pnl"] for t in closed) / INITIAL_CAPITAL * 100, 2),
        "max_dd": round(mdd, 2),
        "min_cum": round(min(equity), 2) if equity else 0,
        "trades": len(closed),
        "open_end": len(open_trades),
        "win": round(100 * wins / len(closed), 1) if closed else 0,
        "skipped_entries": skipped,
        "exits": {k: f"{v[0]}/₹{v[1]:,.0f}" for k, v in sorted(by.items(), key=lambda x: x[1][1])},
    }


# ─── run ─────────────────────────────────────────────────────────────────────
preds = json.load(open("/tmp/preds_by_date.json"))
dates = sorted(d for d in EXPS if WINDOW_START <= d <= WINDOW_END)
print(f"Re-pricing {WINDOW_START} → {WINDOW_END}  ({len(dates)} trading days)\n")

BS = {"v5.4.2": 8.25, "v5.4.3": 18.34, "v5.4.4": 6.45}

print("=== CONSERVATIVE: all versions take profit at CLOSE (fair, parameters-only) ===")
print(f"{'version':8} {'REAL ret':>9} {'BS ret':>8} {'maxDD':>8} {'win%':>6} {'trades':>7}  exit mix")
for v in ("v5.4.2", "v5.4.3", "v5.4.4"):
    r = run_version(v, preds, dates, use_best=False)
    print(f"{v:8} {r['return_pct']:>8}% {BS[v]:>7}% {r['max_dd']:>7}% {r['win']:>5}% {r['trades']:>7}  {r['exits']}")

print("\n=== CADENCE: v5.4.3 takes profit intraday (its real live edge; others at close) ===")
print(f"{'version':8} {'REAL ret':>9} {'BS ret':>8} {'maxDD':>8} {'win%':>6} {'trades':>7}  exit mix")
for v in ("v5.4.2", "v5.4.3", "v5.4.4"):
    r = run_version(v, preds, dates, use_best=(v == "v5.4.3"))
    print(f"{v:8} {r['return_pct']:>8}% {BS[v]:>7}% {r['max_dd']:>7}% {r['win']:>5}% {r['trades']:>7}  {r['exits']}")
