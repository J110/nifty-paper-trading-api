"""
Records Nifty spot and option spread prices at specific intervals
after each trade signal, to answer:
"If I executed 10 min / 1 hr / 3 hr / 6 hr / 12 hr later,
 how would my entry price differ?"
"""

import logging
from datetime import datetime, timedelta, time, date
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import DelayPrice, Trade, PriceSnapshot
from core.option_pricer import compute_spread_value
from config import RISK_FREE_RATE, NIFTY_LOT_SIZE

logger = logging.getLogger(__name__)

# Market hours IST
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

DELAY_INTERVALS = [
    ("10min", timedelta(minutes=10)),
    ("1hr", timedelta(hours=1)),
    ("3hr", timedelta(hours=3)),
    ("6hr", timedelta(hours=6)),
    ("12hr", timedelta(hours=12)),
]


def is_market_hours(dt: datetime = None) -> bool:
    """Check if given time (or now) is during market hours."""
    if dt is None:
        dt = datetime.now()
    t = dt.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE and dt.weekday() < 5


def adjust_to_market_hours(target_time: datetime) -> datetime:
    """
    If target_time is outside market hours, adjust to next market open
    plus the remaining time offset.
    """
    t = target_time.time()

    if target_time.weekday() >= 5:  # Weekend
        # Move to Monday
        days_ahead = 7 - target_time.weekday()
        target_time = target_time.replace(
            hour=9, minute=15, second=0
        ) + timedelta(days=days_ahead)
        return target_time

    if t > MARKET_CLOSE:
        # Move to next trading day
        next_day = target_time + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day.replace(hour=9, minute=15, second=0)

    if t < MARKET_OPEN:
        return target_time.replace(hour=9, minute=15, second=0)

    return target_time


class PriceTracker:
    """Records prices at delay intervals after trade signals."""

    def get_scheduled_times(self, signal_time: datetime) -> list:
        """
        Calculate when to record delay prices, adjusting for market hours.
        Returns list of (label, target_datetime) tuples.
        """
        scheduled = []
        for label, delta in DELAY_INTERVALS:
            target = signal_time + delta
            target = adjust_to_market_hours(target)
            scheduled.append((label, target))
        return scheduled

    async def record_initial_price(
        self, trade_id: str, version: str,
        signal_time: datetime, spot: float,
        spread_price: float, db: AsyncSession
    ):
        """Record the initial price at signal time (T+0)."""
        delay_record = DelayPrice(
            trade_id=trade_id,
            version=version,
            signal_date=signal_time.date(),
            signal_time=signal_time,
            price_at_signal=spot,
            spread_price_at_signal=spread_price,
        )
        db.add(delay_record)
        await db.commit()
        logger.info(f"Recorded initial delay price for {trade_id}")

    async def record_delay_price(
        self, trade_id: str, delay_label: str,
        spot: float, dhan_client, db: AsyncSession
    ):
        """Record spot and spread prices at a delay point."""
        # Get the trade for strike info
        result = await db.execute(
            select(Trade).where(Trade.trade_id == trade_id)
        )
        trade = result.scalar_one_or_none()
        if not trade:
            logger.warning(f"Trade {trade_id} not found for delay recording")
            return

        # Compute current spread value
        dte = (trade.expiry - date.today()).days
        T = max(dte / 365.0, 1 / 365.0)
        try:
            vix = await dhan_client.get_india_vix()
            sigma = vix / 100.0
        except Exception:
            sigma = 0.15

        spread_value = compute_spread_value(
            trade.trade_type, spot,
            trade.sell_strike, trade.buy_strike,
            trade.ic_call_sell, trade.ic_call_buy,
            T, RISK_FREE_RATE, sigma
        )

        # PnL delta = what you'd get entering NOW vs at signal time
        pnl_delta = (spread_value - trade.credit_received) * \
                    trade.num_lots * NIFTY_LOT_SIZE

        # Update delay record
        result = await db.execute(
            select(DelayPrice).where(
                DelayPrice.trade_id == trade_id
            )
        )
        delay_record = result.scalar_one_or_none()
        if not delay_record:
            logger.warning(f"No delay record for {trade_id}")
            return

        # Map label to column
        col_map = {
            "10min": ("price_at_10min", "spread_price_at_10min", "pnl_delta_10min"),
            "1hr": ("price_at_1hr", "spread_price_at_1hr", "pnl_delta_1hr"),
            "3hr": ("price_at_3hr", "spread_price_at_3hr", "pnl_delta_3hr"),
            "6hr": ("price_at_6hr", "spread_price_at_6hr", "pnl_delta_6hr"),
            "12hr": ("price_at_12hr", "spread_price_at_12hr", "pnl_delta_12hr"),
        }

        if delay_label in col_map:
            price_col, spread_col, pnl_col = col_map[delay_label]
            setattr(delay_record, price_col, spot)
            setattr(delay_record, spread_col, spread_value)
            setattr(delay_record, pnl_col, pnl_delta)

        await db.commit()
        logger.info(
            f"Recorded {delay_label} delay price for {trade_id}: "
            f"spot={spot}, spread={spread_value:.2f}, pnl_delta={pnl_delta:.0f}"
        )

    async def record_price_snapshot(
        self, spot: float, vix: Optional[float],
        high: Optional[float], low: Optional[float],
        db: AsyncSession
    ):
        """Record a periodic price snapshot (every 5 min)."""
        now = datetime.now()
        snapshot = PriceSnapshot(
            timestamp=now,
            nifty_spot=spot,
            nifty_high=high,
            nifty_low=low,
            vix=vix,
        )
        try:
            db.add(snapshot)
            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.debug(f"Snapshot insert failed (likely duplicate): {e}")

    async def compute_delay_analysis(
        self, version: str, db: AsyncSession,
        data_mode: str = "combined",
        forward_test_start: Optional[date] = None,
    ) -> dict:
        """
        Aggregate delay impact across all trades for a version.
        data_mode: 'backtest', 'forwardtest', or 'combined'.
        """
        stmt = select(DelayPrice).where(DelayPrice.version == version)

        # Apply data mode filter using signal_date
        if data_mode != "combined" and forward_test_start is not None:
            if data_mode == "backtest":
                stmt = stmt.where(DelayPrice.signal_date < forward_test_start)
            elif data_mode == "forwardtest":
                stmt = stmt.where(DelayPrice.signal_date >= forward_test_start)

        result = await db.execute(stmt)
        records = result.scalars().all()

        if not records:
            return {
                "immediate": {"total_pnl": 0, "avg_entry_credit": 0},
                "10min": {"total_pnl": 0, "avg_entry_credit": 0, "pnl_delta": 0},
                "1hr": {"total_pnl": 0, "avg_entry_credit": 0, "pnl_delta": 0},
                "3hr": {"total_pnl": 0, "avg_entry_credit": 0, "pnl_delta": 0},
                "6hr": {"total_pnl": 0, "avg_entry_credit": 0, "pnl_delta": 0},
                "12hr": {"total_pnl": 0, "avg_entry_credit": 0, "pnl_delta": 0},
                "best_delay": "immediate",
                "per_trade": [],
            }

        # Aggregate
        analysis = {
            "immediate": {"total_pnl": 0, "avg_entry_credit": 0, "count": 0},
        }
        for label in ["10min", "1hr", "3hr", "6hr", "12hr"]:
            analysis[label] = {
                "total_pnl": 0, "avg_entry_credit": 0,
                "pnl_delta": 0, "count": 0
            }

        per_trade = []
        for r in records:
            analysis["immediate"]["count"] += 1
            analysis["immediate"]["avg_entry_credit"] += r.spread_price_at_signal or 0

            trade_data = {
                "trade_id": r.trade_id,
                "signal_date": r.signal_date.isoformat() if r.signal_date else None,
                "price_at_signal": r.price_at_signal,
                "spread_at_signal": r.spread_price_at_signal,
            }

            for label, price_attr, spread_attr, pnl_attr in [
                ("10min", "price_at_10min", "spread_price_at_10min", "pnl_delta_10min"),
                ("1hr", "price_at_1hr", "spread_price_at_1hr", "pnl_delta_1hr"),
                ("3hr", "price_at_3hr", "spread_price_at_3hr", "pnl_delta_3hr"),
                ("6hr", "price_at_6hr", "spread_price_at_6hr", "pnl_delta_6hr"),
                ("12hr", "price_at_12hr", "spread_price_at_12hr", "pnl_delta_12hr"),
            ]:
                pnl_val = getattr(r, pnl_attr, None)
                if pnl_val is not None:
                    analysis[label]["pnl_delta"] += pnl_val
                    analysis[label]["count"] += 1
                trade_data[f"price_{label}"] = getattr(r, price_attr, None)
                trade_data[f"spread_{label}"] = getattr(r, spread_attr, None)
                trade_data[f"pnl_delta_{label}"] = pnl_val

            per_trade.append(trade_data)

        # Compute averages
        n = analysis["immediate"]["count"]
        if n > 0:
            analysis["immediate"]["avg_entry_credit"] /= n

        # Find best delay
        best_delay = "immediate"
        best_delta = 0
        for label in ["10min", "1hr", "3hr", "6hr", "12hr"]:
            delta = analysis[label]["pnl_delta"]
            if delta > best_delta:
                best_delta = delta
                best_delay = label
            # Clean up internal count
            del analysis[label]["count"]

        del analysis["immediate"]["count"]

        result_data = {
            **analysis,
            "best_delay": best_delay,
            "per_trade": per_trade,
        }
        return result_data
