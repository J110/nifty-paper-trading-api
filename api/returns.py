"""API routes for indicators and additional endpoints."""

from datetime import date, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db.models import DailyFeature
from core.feature_engine import format_indicators_for_display

router = APIRouter(prefix="/api")


@router.get("/indicators")
async def get_indicators(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """Return historical indicator values for charting."""
    start_date = date.today() - timedelta(days=days)

    result = await db.execute(
        select(DailyFeature).where(
            DailyFeature.date >= start_date
        ).order_by(DailyFeature.date)
    )
    features_list = result.scalars().all()

    history = []
    for f in features_list:
        entry = {
            "date": f.date.isoformat(),
            "vix": f.vix,
            "vix_20d_avg": f.vix_20d_avg,
            "nifty_20d_return": f.nifty_20d_return,
            "nifty_50d_return": f.nifty_50d_return,
            "iv_skew": f.iv_skew,
            "fii_net": f.fii_net,
            "dii_net": f.dii_net,
            "put_call_ratio": f.put_call_ratio,
            "rsi_14": f.rsi_14,
            "adx_14": f.adx_14,
        }
        history.append(entry)

    # Latest indicators formatted for display
    latest = None
    if features_list:
        last = features_list[-1]
        if last.features:
            latest = format_indicators_for_display(last.features)

    return {
        "latest": latest,
        "history": history,
        "count": len(history),
    }
