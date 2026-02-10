"""API routes for chart data."""

import logging
from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, desc, func, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db.models import PriceSnapshot, DailyPnl, Prediction
from core.timezone import today_ist
from config import ACTIVE_VERSIONS, INITIAL_CAPITAL
from core.dhan_client import DhanClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Shared Dhan client — lazily initialised on first chart request
_dhan_client: DhanClient | None = None


async def _get_dhan() -> DhanClient:
    """Return a started DhanClient singleton."""
    global _dhan_client
    if _dhan_client is None:
        _dhan_client = DhanClient()
    await _dhan_client.start()
    return _dhan_client


def _epoch_to_iso_date(ts) -> str:
    """Convert a Dhan epoch timestamp (seconds) to ISO date string (IST)."""
    try:
        from core.timezone import IST
        return datetime.fromtimestamp(int(ts), tz=IST).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return str(ts)


@router.get("/chart-data/nifty")
async def get_nifty_chart(
    period: str = Query("1y", regex="^(1d|5d|1w|1m|3m|6m|1y|2y|3y|5y)$"),
    interval: str = Query("1d", regex="^(5m|1d)$"),
    db: AsyncSession = Depends(get_db),
):
    """
    Nifty price data for charting.

    Priority order:
      1. price_snapshots table (intraday or daily aggregation)
      2. Dhan historical API (daily OHLC — works for all periods)
      3. Prediction nifty_spot fallback (limited to paper-trading period)
    """
    today = today_ist()

    period_days = {
        "1d": 1, "5d": 5, "1w": 7, "1m": 30, "3m": 90,
        "6m": 180, "1y": 365, "2y": 730, "3y": 1095, "5y": 1825,
    }
    days = period_days.get(period, 365)
    start_date = today - timedelta(days=days)

    # ---- Intraday 5-min from DB snapshots ----
    if period == "1d" and interval == "5m":
        # Try DB snapshots first
        result = await db.execute(
            select(PriceSnapshot).where(
                PriceSnapshot.timestamp >= start_date
            ).order_by(PriceSnapshot.timestamp)
        )
        snapshots = result.scalars().all()

        if snapshots:
            candles = []
            for s in snapshots:
                candles.append({
                    "timestamp": s.timestamp.isoformat(),
                    "open": s.nifty_spot,
                    "high": s.nifty_high or s.nifty_spot,
                    "low": s.nifty_low or s.nifty_spot,
                    "close": s.nifty_spot,
                    "volume": 0,
                })
            return candles

        # Fallback: Dhan intraday
        try:
            dhan = await _get_dhan()
            candles_raw = await dhan.get_intraday_ohlc(interval="5")
            if candles_raw:
                return [
                    {
                        "timestamp": _epoch_to_iso_date(c["timestamp"]),
                        "open": c["open"],
                        "high": c["high"],
                        "low": c["low"],
                        "close": c["close"],
                        "volume": c.get("volume", 0),
                    }
                    for c in candles_raw
                ]
        except Exception:
            logger.exception("Dhan intraday fallback failed")

        return []

    # ---- Daily candles ----
    else:
        # 1. Try price_snapshots (aggregated to daily)
        result = await db.execute(
            select(PriceSnapshot).where(
                PriceSnapshot.timestamp >= start_date
            ).order_by(PriceSnapshot.timestamp)
        )
        snapshots = result.scalars().all()

        if snapshots:
            from collections import defaultdict
            daily = defaultdict(lambda: {"open": None, "high": 0, "low": 999999, "close": 0})
            for s in snapshots:
                d = s.timestamp.date().isoformat()
                if daily[d]["open"] is None:
                    daily[d]["open"] = s.nifty_spot
                daily[d]["high"] = max(daily[d]["high"], s.nifty_high or s.nifty_spot)
                daily[d]["low"] = min(daily[d]["low"], s.nifty_low or s.nifty_spot)
                daily[d]["close"] = s.nifty_spot

            candles = []
            for d in sorted(daily.keys()):
                data = daily[d]
                if data["open"] is not None:
                    candles.append({
                        "timestamp": d,
                        "open": data["open"],
                        "high": data["high"],
                        "low": data["low"],
                        "close": data["close"],
                        "volume": 0,
                    })
            return candles

        # 2. Dhan historical daily OHLC (primary fallback — real candle data)
        try:
            dhan = await _get_dhan()
            candles_raw = await dhan.get_historical_ohlc(days=days)
            if candles_raw:
                candles = []
                for c in candles_raw:
                    candles.append({
                        "timestamp": _epoch_to_iso_date(c["timestamp"]),
                        "open": c["open"],
                        "high": c["high"],
                        "low": c["low"],
                        "close": c["close"],
                        "volume": c.get("volume", 0),
                    })
                logger.info(
                    "Served %d Dhan candles for period=%s (%d days)",
                    len(candles), period, days,
                )
                return candles
        except Exception:
            logger.exception("Dhan historical fallback failed for period=%s", period)

        # 3. Last resort: prediction nifty_spot (paper-trading period only)
        result = await db.execute(
            select(Prediction).where(
                Prediction.date >= start_date,
                Prediction.nifty_spot.isnot(None),
                Prediction.version == ACTIVE_VERSIONS[0],
            ).order_by(Prediction.date)
        )
        predictions = result.scalars().all()
        candles = []
        for p in predictions:
            if p.nifty_spot and p.nifty_spot > 0:
                candles.append({
                    "timestamp": p.date.isoformat(),
                    "open": p.nifty_spot,
                    "high": p.nifty_spot,
                    "low": p.nifty_spot,
                    "close": p.nifty_spot,
                    "volume": 0,
                })
        return candles


@router.get("/chart-data/equity/{version}")
async def get_equity_curve(
    version: str,
    db: AsyncSession = Depends(get_db),
):
    """Paper trading equity curve for a version."""
    if version not in ACTIVE_VERSIONS:
        return {"error": f"Unknown version: {version}"}

    result = await db.execute(
        select(DailyPnl).where(
            DailyPnl.version == version
        ).order_by(DailyPnl.date)
    )
    daily_pnls = result.scalars().all()

    equity_points = []
    for dp in daily_pnls:
        equity_points.append({
            "date": dp.date.isoformat(),
            "capital": dp.ending_capital,
            "daily_pnl": dp.daily_pnl,
            "cumulative_pnl": dp.cumulative_pnl,
            "cumulative_return_pct": dp.cumulative_return_pct,
            "open_positions": dp.open_positions,
        })

    return {
        "version": version,
        "starting_capital": INITIAL_CAPITAL,
        "equity_curve": equity_points,
    }
