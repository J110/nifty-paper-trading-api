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
        check("total within 85% cap", used <= INITIAL_CAPITAL * MAX_MARGIN_UTILISATION, True)

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
        print(f"        -> equity Rs{eq:,.0f} gives {t4['num_lots']} lots "
              f"(vs {expected_lots} at starting capital)")
        check("compounded size is larger", t4["num_lots"] > expected_lots, True)

    print("\n" + ("ALL CHECKS PASSED" if OK else "SOME CHECKS FAILED"))
    return 0 if OK else 1

sys.exit(asyncio.run(main()))
