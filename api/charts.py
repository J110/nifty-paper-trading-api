"""API routes for chart data."""

from datetime import date, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, desc, func, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db.models import PriceSnapshot, DailyPnl, Prediction
from config import ACTIVE_VERSIONS, INITIAL_CAPITAL

router = APIRouter(prefix="/api")


@router.get("/chart-data/nifty")
async def get_nifty_chart(
    period: str = Query("1y", regex="^(1d|1w|1m|3m|6m|1y)$"),
    interval: str = Query("1d", regex="^(5m|1d)$"),
    db: AsyncSession = Depends(get_db),
):
    """
    Nifty price data for charting.
    Tries price_snapshots first; falls back to daily prediction spot prices.
    """
    today = date.today()

    period_days = {
        "1d": 1, "1w": 7, "1m": 30, "3m": 90, "6m": 180, "1y": 365
    }
    days = period_days.get(period, 365)
    start_date = today - timedelta(days=days)

    if period == "1d" and interval == "5m":
        # Intraday from snapshots
        result = await db.execute(
            select(PriceSnapshot).where(
                PriceSnapshot.timestamp >= start_date
            ).order_by(PriceSnapshot.timestamp)
        )
        snapshots = result.scalars().all()

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

    else:
        # Try daily candles from price_snapshots
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

        # Fallback: use nifty_spot from predictions (one per day)
        result = await db.execute(
            select(Prediction).where(
                Prediction.date >= start_date,
                Prediction.nifty_spot.isnot(None),
                Prediction.version == ACTIVE_VERSIONS[0],  # One per day
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
