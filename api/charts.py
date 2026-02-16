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

            # Only use snapshots if they cover enough of the requested period.
            # Otherwise fall through to Dhan historical for full history.
            min_days_needed = max(days // 3, 5)
            if len(daily) >= min_days_needed:
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
            else:
                logger.info(
                    "Only %d snapshot days for %d-day period, falling through to Dhan",
                    len(daily), days,
                )

        # 2. Dhan historical daily OHLC (primary fallback — real candle data)
        try:
            dhan = await _get_dhan()
            candles_raw = await dhan.get_historical_ohlc(days=days)
            if candles_raw:
                candles = []
                seen_dates = set()
                for c in candles_raw:
                    d = _epoch_to_iso_date(c["timestamp"])
                    seen_dates.add(d)
                    candles.append({
                        "timestamp": d,
                        "open": c["open"],
                        "high": c["high"],
                        "low": c["low"],
                        "close": c["close"],
                        "volume": c.get("volume", 0),
                    })

                # Append today's data from price_snapshots if Dhan
                # doesn't include it yet (historical API lags intraday)
                today_str = today.isoformat()
                if today_str not in seen_dates and snapshots:
                    from collections import defaultdict
                    today_snaps = [
                        s for s in snapshots
                        if s.timestamp.date() == today
                    ]
                    if today_snaps:
                        t_open = today_snaps[0].nifty_spot
                        t_high = max(
                            s.nifty_high or s.nifty_spot for s in today_snaps
                        )
                        t_low = min(
                            s.nifty_low or s.nifty_spot for s in today_snaps
                        )
                        t_close = today_snaps[-1].nifty_spot
                        candles.append({
                            "timestamp": today_str,
                            "open": t_open,
                            "high": t_high,
                            "low": t_low,
                            "close": t_close,
                            "volume": 0,
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


@router.get("/chart-data/drawdown-comparison")
async def get_drawdown_comparison(
    period: str = Query("6m", regex="^(3m|6m|1y|all)$"),
    db: AsyncSession = Depends(get_db),
):
    """
    Predicted vs actual 30-day max drawdown for model accuracy comparison.

    For each prediction date, returns:
      - predicted_drawdown_pct: what the model predicted
      - actual_drawdown_pct: (min close in next 30 trading days - close) / close
      - is_partial: True if fewer than 21 trading days of forward data available
    """
    today = today_ist()

    period_days = {"3m": 90, "6m": 180, "1y": 365, "all": 1825}
    days = period_days.get(period, 180)
    start_date = today - timedelta(days=days)

    # 1. Fetch predictions (one per date, all versions have same predicted_drawdown)
    result = await db.execute(
        select(
            Prediction.date,
            Prediction.predicted_drawdown,
            Prediction.nifty_spot,
        ).where(
            Prediction.date >= start_date,
            Prediction.version == ACTIVE_VERSIONS[0],
            Prediction.predicted_drawdown.isnot(None),
        ).order_by(Prediction.date)
    )
    predictions = result.all()

    if not predictions:
        return {"data": [], "count": 0, "partial_from": None}

    # 2. Fetch Nifty daily closes — need extra 45 calendar days for forward window
    daily_closes = {}
    try:
        dhan = await _get_dhan()
        candles_raw = await dhan.get_historical_ohlc(days=days + 60)
        if candles_raw:
            for c in candles_raw:
                d = _epoch_to_iso_date(c["timestamp"])
                daily_closes[d] = c["close"]
    except Exception:
        logger.exception("Dhan fetch failed for drawdown comparison")

    # Fallback: use Prediction.nifty_spot if Dhan unavailable
    if not daily_closes:
        for p in predictions:
            if p.nifty_spot and p.nifty_spot > 0:
                daily_closes[p.date.isoformat()] = p.nifty_spot

    # Build sorted list of (date_str, close) for forward lookups
    sorted_closes = sorted(daily_closes.items())
    close_dates = [d for d, _ in sorted_closes]
    close_values = [v for _, v in sorted_closes]

    # 3. Compute actual drawdown for each prediction date
    import bisect

    data = []
    partial_from = None
    min_complete_days = 21  # ~30 calendar days ≈ 21 trading days

    for pred_date, pred_dd, nifty_spot in predictions:
        pred_date_str = pred_date.isoformat()
        pred_dd_pct = round(pred_dd * 100, 2)

        # Find this date's close (prefer Dhan, fall back to nifty_spot)
        today_close = daily_closes.get(pred_date_str, nifty_spot)
        if not today_close or today_close <= 0:
            continue

        # Find next 30 trading-day closes after pred_date
        idx = bisect.bisect_right(close_dates, pred_date_str)
        future_closes = close_values[idx:idx + 30]

        if len(future_closes) >= min_complete_days:
            min_future = min(future_closes)
            actual_dd = (min_future - today_close) / today_close
            actual_dd_pct = round(min(actual_dd * 100, 0.0), 2)  # cap at 0
            is_partial = len(future_closes) < 30
        elif len(future_closes) >= 5:
            # Need at least 5 trading days for a meaningful partial
            min_future = min(future_closes)
            actual_dd = (min_future - today_close) / today_close
            actual_dd_pct = round(min(actual_dd * 100, 0.0), 2)  # cap at 0
            is_partial = True
        else:
            # Too few forward days — don't show actual
            actual_dd_pct = None
            is_partial = True

        if is_partial and partial_from is None:
            partial_from = pred_date_str

        data.append({
            "date": pred_date_str,
            "predicted_drawdown_pct": pred_dd_pct,
            "actual_drawdown_pct": actual_dd_pct,
            "nifty_close": round(today_close, 2),
            "is_partial": is_partial,
        })

    return {
        "data": data,
        "count": len(data),
        "partial_from": partial_from,
    }


FORWARD_TEST_START = date(2026, 2, 13)


@router.get("/chart-data/equity/{version}")
async def get_equity_curve(
    version: str,
    data_mode: str = Query("combined", regex="^(backtest|forwardtest|combined)$"),
    db: AsyncSession = Depends(get_db),
):
    """Paper trading equity curve for a version, filterable by data mode."""
    if version not in ACTIVE_VERSIONS:
        return {"error": f"Unknown version: {version}"}

    stmt = select(DailyPnl).where(DailyPnl.version == version)

    if data_mode == "backtest":
        stmt = stmt.where(DailyPnl.date < FORWARD_TEST_START)
    elif data_mode == "forwardtest":
        stmt = stmt.where(DailyPnl.date >= FORWARD_TEST_START)

    stmt = stmt.order_by(DailyPnl.date)
    result = await db.execute(stmt)
    daily_pnls = result.scalars().all()

    # For non-combined modes, rebase the equity curve from INITIAL_CAPITAL
    equity_points = []
    cumulative_pnl = 0.0
    for dp in daily_pnls:
        cumulative_pnl += dp.daily_pnl
        capital = INITIAL_CAPITAL + cumulative_pnl
        equity_points.append({
            "date": dp.date.isoformat(),
            "capital": round(capital, 2),
            "daily_pnl": dp.daily_pnl,
            "cumulative_pnl": round(cumulative_pnl, 2),
            "cumulative_return_pct": round(cumulative_pnl / INITIAL_CAPITAL * 100, 2),
            "open_positions": dp.open_positions,
        })

    return {
        "version": version,
        "data_mode": data_mode,
        "starting_capital": INITIAL_CAPITAL,
        "equity_curve": equity_points,
    }
