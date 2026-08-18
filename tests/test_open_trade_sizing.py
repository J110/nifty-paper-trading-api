"""End-to-end test of open_trade sizing: lot_size pinning, compounding, margin cap.

Run:  python3 tests/test_open_trade_sizing.py
Needs: sqlalchemy, aiosqlite, greenlet, numpy, scipy, python-dotenv.
Uses in-memory SQLite, so it never touches the real database.


Covers: lot_size pinning, compounding off current equity, and the 85% margin cap
(full size -> downsize -> skip). Uses in-memory SQLite via aiosqlite.
"""
import asyncio, sys, os, logging
logging.basicConfig(level=logging.INFO, format="      [log] %(message)s")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from db.models import Base, Trade
from core.trade_manager import TradeManager
from core.signal_mapper import map_signal
from core.option_pricer import select_strikes
from config import (VERSION_CONFIGS, INITIAL_CAPITAL, NIFTY_LOT_SIZE,
                    MAX_MARGIN_UTILISATION, MARGIN_PER_LOT_IC, COMPOUND_SIZING)

VERSION, SPOT, VIX = "v5.4.4", 24265.0, 12.07
# Neutralise the entry-quality gate: at 3 DTE / VIX 12 the BS credit is ~Rs558,
# below MIN_TOTAL_CREDIT. That gate is correct but unrelated to the sizing logic
# under test, so relax it here.
import core.trade_manager as TM
TM.MIN_TOTAL_CREDIT = 1
OK = True

def check(label, got, want, tol=0):
    global OK
    good = abs(got - want) <= tol if isinstance(want, (int, float)) else got == want
    OK &= good
    print(f"   {'PASS' if good else 'FAIL'}  {label}: got {got!r}, expected {want!r}"
          + (f" (+/-{tol})" if tol else ""))

async def fresh():
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as c:
        await c.run_sync(Base.metadata.create_all)
    return eng, async_sessionmaker(eng, expire_on_commit=False)

def ic_signal():
    # predicted drawdown in the iron-condor band (-5.0% .. -6.5%)
    s = map_signal(-0.0594, VERSION_CONFIGS[VERSION])
    assert s["trade_type"] == "iron_condor", s
    s["size_mult"] = 1.0
    return s

async def main():
    tm = TradeManager()
    cfg = VERSION_CONFIGS[VERSION]
    # Entry-gap rule is unrelated to sizing and would block same-day seeded tests.
    cfg["MIN_ENTRY_GAP_DAYS"] = 0
    print(f"config: LOT={NIFTY_LOT_SIZE} cap={MAX_MARGIN_UTILISATION:.0%} "
          f"compound={COMPOUND_SIZING} IC_pct={cfg['IC_POSITION_SIZE_PCT']}\n")

    # ---- 1. first trade on a clean book -----------------------------------
    print("1. first trade, empty book")
    eng, Session = await fresh()
    async with Session() as db:
        t1 = await tm.open_trade(ic_signal(), VERSION, SPOT, VIX, db, dhan_client=None)
        assert t1, "no trade opened"
        expected_lots = int(INITIAL_CAPITAL * cfg["IC_POSITION_SIZE_PCT"] / MARGIN_PER_LOT_IC)
        check("num_lots", t1["num_lots"], expected_lots)
        row = (await db.execute(__import__("sqlalchemy").select(Trade))).scalars().first()
        check("lot_size pinned on row", row.lot_size, NIFTY_LOT_SIZE)
        used = await tm._open_margin_used(db, VERSION, SPOT)
        pct = 100 * used / INITIAL_CAPITAL
        print(f"        -> {t1['num_lots']} lots = {t1['num_lots']*NIFTY_LOT_SIZE} units, "
              f"real margin Rs{used:,.0f} ({pct:.1f}% of equity)")
        check("within cap", used <= INITIAL_CAPITAL * MAX_MARGIN_UTILISATION, True)

    # ---- 2 & 3. margin cap: seed existing open positions ------------------
    # (open_trade dedupes on version+signal+date, so a second call the same day
    #  returns the existing trade_id — seed rows directly to test the cap instead.)
    from datetime import date, datetime, timezone
    def seed(db, tid, lots):
        db.add(Trade(trade_id=tid, version=VERSION, date=date(2026,8,1),
                     signal_type="iron_condor", trade_type="iron_condor",
                     entry_mode="normal", entry_date=date(2026,8,1),
                     entry_time=datetime.now(timezone.utc), entry_spot=SPOT,
                     expiry=date(2026,8,4),
                     sell_strike=23550, buy_strike=22950,
                     ic_call_sell=25250, ic_call_buy=25850,
                     num_lots=lots, lot_size=65, credit_received=4.95,
                     total_credit=4.95*lots*65, status="open",
                     position_size_pct=0.5766, capital_deployed=lots*MARGIN_PER_LOT_IC))

    print("\n2. one full-size position already open -> cap must downsize")
    eng, Session = await fresh()
    async with Session() as db:
        seed(db, "seeded-1", 14)
        await db.commit()
        used0 = await tm._open_margin_used(db, VERSION, SPOT)
        print(f"        pre-existing margin Rs{used0:,.0f} ({100*used0/INITIAL_CAPITAL:.1f}%)")
        t2 = await tm.open_trade(ic_signal(), VERSION, SPOT, VIX, db, dhan_client=None)
        assert isinstance(t2, dict), f"expected a trade dict, got {t2!r}"
        check("downsized below full size", t2["num_lots"] < expected_lots, True)
        used = await tm._open_margin_used(db, VERSION, SPOT)
        print(f"        -> {t2['num_lots']} lots, total Rs{used:,.0f} "
              f"({100*used/INITIAL_CAPITAL:.1f}% of equity)")
        from config import SIZING_CAPITAL_OVERRIDE as _OV
        _base = _OV or INITIAL_CAPITAL
        check("total within 85% cap (of the real-capital base)",
              used <= _base * MAX_MARGIN_UTILISATION, True)

    print("\n3. cap already consumed -> trade must be refused")
    eng, Session = await fresh()
    async with Session() as db:
        seed(db, "seeded-1", 14); seed(db, "seeded-2", 6)
        await db.commit()
        used0 = await tm._open_margin_used(db, VERSION, SPOT)
        print(f"        pre-existing margin Rs{used0:,.0f} ({100*used0/INITIAL_CAPITAL:.1f}%)")
        t3 = await tm.open_trade(ic_signal(), VERSION, SPOT, VIX, db, dhan_client=None)
        check("returns None", t3, None)

    # ---- 4. compounding: profit should increase size -----------------------
    print("\n4. compounding (seed +Rs10L realised profit)")
    eng2, Session2 = await fresh()
    async with Session2() as db:
        from datetime import date, datetime, timezone
        db.add(Trade(trade_id="seed", version=VERSION, date=date(2026,1,1),
                     signal_type="iron_condor", trade_type="iron_condor",
                     entry_mode="normal", entry_date=date(2026,1,1),
                     entry_time=datetime.now(timezone.utc), entry_spot=SPOT,
                     expiry=date(2026,1,6), sell_strike=1, buy_strike=1,
                     num_lots=1, lot_size=25, credit_received=1.0, total_credit=1.0,
                     status="closed", realized_pnl=1_000_000.0,
                     position_size_pct=0.2, capital_deployed=1.0))
        await db.commit()
        eq = await tm._get_current_capital(db, VERSION)
        check("equity reflects profit", round(eq), INITIAL_CAPITAL + 1_000_000)
        t4 = await tm.open_trade(ic_signal(), VERSION, SPOT, VIX, db, dhan_client=None)
        assert t4, "no trade opened"
        import config as _cfgmod
        if _cfgmod.SIZING_CAPITAL_OVERRIDE:
            # THE FIX: Rs10L of paper profit must not grow the real position.
            print(f"        -> paper equity Rs{eq:,.0f}, but override pins sizing to "
                  f"Rs{_cfgmod.SIZING_CAPITAL_OVERRIDE:,} -> {t4['num_lots']} lots")
            check("paper profit does NOT inflate the real size",
                  t4["num_lots"] <= expected_lots, True)
            # and compounding must still work when the override is removed
            import core.trade_manager as _tm
            _saved = _tm.SIZING_CAPITAL_OVERRIDE
            _tm.SIZING_CAPITAL_OVERRIDE = None
            try:
                eng3, Session3 = await fresh()
                async with Session3() as db3:
                    db3.add(Trade(trade_id="seed3", version=VERSION, date=date(2026,1,1),
                                  signal_type="iron_condor", trade_type="iron_condor",
                                  entry_mode="normal", entry_date=date(2026,1,1),
                                  entry_time=datetime.now(timezone.utc), entry_spot=SPOT,
                                  expiry=date(2026,1,6), sell_strike=1, buy_strike=1,
                                  num_lots=1, lot_size=25, credit_received=1.0,
                                  total_credit=1.0, status="closed",
                                  realized_pnl=1_000_000.0, position_size_pct=0.2,
                                  capital_deployed=1.0))
                    await db3.commit()
                    t4b = await tm.open_trade(ic_signal(), VERSION, SPOT, VIX, db3,
                                              dhan_client=None)
                    check("compounding still works when override is None",
                          t4b["num_lots"] > expected_lots, True)
            finally:
                _tm.SIZING_CAPITAL_OVERRIDE = _saved
        else:
            print(f"        -> equity Rs{eq:,.0f} gives {t4['num_lots']} lots "
                  f"(vs {expected_lots} at starting capital)")
            check("compounded size is larger", t4["num_lots"] > expected_lots, True)

    # ---- 5. bear_call respects the margin cap -----------------------------
    # _real_margin_per_unit used to return 0.0 for bear_call (its legs live in
    # the ic_call_* keys, not sell_strike/buy_strike), and open_trade guards on
    # `if per_unit > 0` — so every bear_call entry silently skipped the cap.
    import copy
    bc_cfg = copy.deepcopy(VERSION_CONFIGS[VERSION])
    bc_cfg["BEAR_CALL_ENABLED"] = True
    bc_sig = map_signal(-0.075, bc_cfg)
    check("no_trade band maps to bear_call when enabled",
          bc_sig["trade_type"], "bear_call")

    bc_strikes = select_strikes(SPOT, "bear_call", bc_cfg)
    per_unit = TradeManager._real_margin_per_unit("bear_call", bc_strikes, 5.0, SPOT)
    check("bear_call margin/unit is non-zero (cap will apply)", per_unit > 0, True)

    # same one-sided shape as a bull_put, so the two must agree
    bp_strikes = select_strikes(SPOT, "bull_put", bc_cfg)
    bp_unit = TradeManager._real_margin_per_unit("bull_put", bp_strikes, 5.0, SPOT)
    check("bear_call margin matches bull_put (one short leg)",
          round(per_unit), round(bp_unit))
    ic_unit = TradeManager._real_margin_per_unit(
        "iron_condor", select_strikes(SPOT, "iron_condor", bc_cfg), 5.0, SPOT)
    check("iron_condor margin is higher (two short legs)", ic_unit > per_unit, True)
    print(f"        -> per unit: bull_put Rs{bp_unit:,.0f}  "
          f"bear_call Rs{per_unit:,.0f}  iron_condor Rs{ic_unit:,.0f}")

    # ---- 6. BULL_HALF_AS_IC swaps the band's structure ---------------------
    swap_cfg = copy.deepcopy(VERSION_CONFIGS[VERSION])
    check("bull_half is a put spread by default",
          map_signal(-0.044, swap_cfg)["trade_type"], "bull_put")
    swap_cfg["BULL_HALF_AS_IC"] = True
    check("bull_half becomes an iron condor when enabled",
          map_signal(-0.044, swap_cfg)["trade_type"], "iron_condor")
    check("bull_full is untouched by the swap",
          map_signal(-0.020, swap_cfg)["trade_type"], "bull_put")

    # ---- 7. live v5.4.2 has both flags ON (turned on 2026-08-12) -----------
    # v5.4.2 is the version traded with real money, so these assert the shipped
    # behaviour, not a default. Flipping either flag off should fail here.
    # NOTE: VERSION above is v5.4.4 (the sizing tests predate the version switch).
    # The traded version is v5.4.2, so these checks name it explicitly.
    live = VERSION_CONFIGS["v5.4.2"]
    check("BEAR_CALL_ENABLED is live", live.get("BEAR_CALL_ENABLED", False), True)
    check("BULL_HALF_AS_IC is live", live.get("BULL_HALF_AS_IC", False), True)
    check("severe band now sells calls instead of sitting out",
          map_signal(-0.075, live)["trade_type"], "bear_call")
    check("mild-worry band now trades as a condor",
          map_signal(-0.044, live)["trade_type"], "iron_condor")
    check("calm band still a put spread",
          map_signal(-0.020, live)["trade_type"], "bull_put")
    check("bear_call strikes are 3.0%/5.5%, not the IC 4.0%/6.5%",
          (live["BEAR_CALL_OTM_SELL"], live["BEAR_CALL_OTM_BUY"]), (0.030, 0.055))

    # the other two versions must be untouched
    for v in ("v5.4.3", "v5.4.4"):
        check(f"{v} unchanged (bear off)",
              VERSION_CONFIGS[v].get("BEAR_CALL_ENABLED", False), False)
        check(f"{v} unchanged (condor swap off)",
              VERSION_CONFIGS[v].get("BULL_HALF_AS_IC", False), False)

    # ---- 8. SIZING_CAPITAL_OVERRIDE sizes off the REAL account ------------
    # Compounding sized off _get_current_capital(), which sums P&L from a trades
    # table dominated by PAPER trades since Mar 2024. On 2026-08-18 that implied
    # Rs37.47L of equity (Rs25L start + Rs12.47L paper profit) and the signal email
    # said 21 lots against a real Rs25.12L account. Correct answer was 14.
    from config import SIZING_CAPITAL_OVERRIDE, MARGIN_PER_LOT_BULL
    v542 = VERSION_CONFIGS["v5.4.2"]
    pct = v542["POSITION_SIZE_PCT"]
    check("override is set to the real balance", SIZING_CAPITAL_OVERRIDE, 2_512_000)
    real_lots = int(SIZING_CAPITAL_OVERRIDE * pct / MARGIN_PER_LOT_BULL)
    paper_lots = int((2_500_000 + 1_247_433) * pct / MARGIN_PER_LOT_BULL)
    check("real capital gives the size actually placed", real_lots, 14)
    check("paper equity would have given the emailed size", paper_lots, 21)
    print(f"        -> real Rs{SIZING_CAPITAL_OVERRIDE:,} = {real_lots} lots  |  "
          f"paper Rs{2_500_000+1_247_433:,} = {paper_lots} lots")

    # and the margin cap must key off the same base, not the paper curve
    check("margin cap headroom uses real capital",
          round(SIZING_CAPITAL_OVERRIDE * MAX_MARGIN_UTILISATION), 2_135_200)

    print("\n" + ("ALL CHECKS PASSED" if OK else "SOME CHECKS FAILED"))
    return 0 if OK else 1

sys.exit(asyncio.run(main()))
