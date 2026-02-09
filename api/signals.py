"""API routes for market signals and predictions."""

from datetime import date, datetime
from fastapi import APIRouter, Depends
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db.models import Prediction, DailyFeature
from core.signal_mapper import map_signal, get_classification_breakdown
from core.feature_engine import format_indicators_for_display
from config import VERSION_CONFIGS, ACTIVE_VERSIONS

router = APIRouter(prefix="/api")


@router.get("/signals/current")
async def get_current_signals(db: AsyncSession = Depends(get_db)):
    """
    Returns current model prediction and classification breakdown
    for all active versions.
    """
    today = date.today()

    # Get latest prediction (any version — same model, same prediction)
    result = await db.execute(
        select(Prediction).where(
            Prediction.date == today
        ).order_by(desc(Prediction.timestamp)).limit(1)
    )
    prediction = result.scalar_one_or_none()

    # Get latest features
    feat_result = await db.execute(
        select(DailyFeature).where(
            DailyFeature.date == today
        )
    )
    daily_feature = feat_result.scalar_one_or_none()

    if not prediction:
        # No prediction today yet — return placeholder
        return {
            "timestamp": datetime.now().isoformat(),
            "nifty_spot": None,
            "predicted_drawdown": None,
            "classification": None,
            "version_signals": {},
            "indicators": [],
            "status": "no_prediction_today",
        }

    # Build classification breakdown
    classification = get_classification_breakdown(prediction.predicted_drawdown)

    # Build version-specific signals
    version_signals = {}
    for version in ACTIVE_VERSIONS:
        cfg = VERSION_CONFIGS[version]
        signal = map_signal(prediction.predicted_drawdown, cfg)
        version_signals[version] = {
            "signal": signal["signal"],
            "trade_type": signal["trade_type"],
            "size_mult": signal["size_mult"],
            "label": cfg["label"],
            "color": cfg["color"],
            "position_size_pct": cfg["POSITION_SIZE_PCT"],
        }

    # Build indicators
    indicators = []
    if daily_feature and daily_feature.features:
        indicators = format_indicators_for_display(daily_feature.features)

    return {
        "timestamp": prediction.timestamp.isoformat() if prediction.timestamp else None,
        "nifty_spot": prediction.nifty_spot,
        "vix": prediction.vix,
        "predicted_drawdown": prediction.predicted_drawdown,
        "predicted_drawdown_pct": round(prediction.predicted_drawdown * 100, 2),
        "classification": classification,
        "version_signals": version_signals,
        "indicators": indicators,
        "confidence_score": prediction.confidence_score,
        "status": "active",
    }


@router.get("/signals/history")
async def get_signal_history(
    days: int = 30,
    db: AsyncSession = Depends(get_db)
):
    """Past predictions with actual outcomes."""
    result = await db.execute(
        select(Prediction).order_by(
            desc(Prediction.date)
        ).limit(days)
    )
    predictions = result.scalars().all()

    history = []
    for p in predictions:
        classification = get_classification_breakdown(p.predicted_drawdown)
        history.append({
            "date": p.date.isoformat(),
            "predicted_drawdown": p.predicted_drawdown,
            "predicted_drawdown_pct": round(p.predicted_drawdown * 100, 2),
            "signal_type": p.signal_type,
            "version": p.version,
            "nifty_spot": p.nifty_spot,
            "vix": p.vix,
            "current_zone": classification["current_zone"],
            "confidence_score": p.confidence_score,
        })

    return {"history": history, "count": len(history)}
