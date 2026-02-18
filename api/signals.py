"""API routes for market signals and predictions."""

import logging
from collections import defaultdict
from datetime import date, datetime

from fastapi import APIRouter, Depends
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db.models import Prediction, DailyFeature, Trade, PriceSnapshot, DailyPnl
from core.timezone import now_ist, today_ist, IST
from core.signal_mapper import map_signal, get_classification_breakdown
from core.feature_engine import format_indicators_for_display
from core.model_runner import get_model_runner
from core.prediction_reasoning import generate_prediction_reasons
from config import VERSION_CONFIGS, ACTIVE_VERSIONS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/signals/current")
async def get_current_signals(db: AsyncSession = Depends(get_db)):
    """
    Returns current model prediction and classification breakdown
    for all active versions.
    """
    today = today_ist()

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
        # No prediction today — try most recent prediction
        fallback_result = await db.execute(
            select(Prediction).order_by(desc(Prediction.date)).limit(1)
        )
        prediction = fallback_result.scalar_one_or_none()
        if prediction:
            # Also get the matching daily features
            feat_result2 = await db.execute(
                select(DailyFeature).where(DailyFeature.date == prediction.date)
            )
            daily_feature = feat_result2.scalar_one_or_none()
        else:
            return {
                "timestamp": now_ist().isoformat(),
                "nifty_spot": None,
                "predicted_drawdown": None,
                "classification": None,
                "version_signals": {},
                "indicators": [],
                "status": "no_prediction_today",
            }

    # Build version-specific signals
    version_signals = {}
    for version in ACTIVE_VERSIONS:
        cfg = VERSION_CONFIGS[version]
        signal = map_signal(prediction.predicted_drawdown, cfg)
        ver_signal = {
            "signal": signal["signal"],
            "trade_type": signal["trade_type"],
            "size_mult": signal["size_mult"],
            "label": cfg["label"],
            "color": cfg["color"],
            "position_size_pct": cfg["POSITION_SIZE_PCT"],
        }
        # Add bear debit info for v6.2+
        if signal.get("bear_tier", 0) > 0:
            ver_signal["bear_tier"] = signal["bear_tier"]
        version_signals[version] = ver_signal

    # Build classification breakdown (use v6.2 for 7-zone if available)
    classification_version = "v6.2" if "v6.2" in ACTIVE_VERSIONS else None
    classification = get_classification_breakdown(
        prediction.predicted_drawdown, version=classification_version
    )

    # Build indicators
    indicators = []
    if daily_feature and daily_feature.features:
        indicators = format_indicators_for_display(daily_feature.features)

    # Build prediction reasoning (top features driving the prediction)
    prediction_reasons = []
    prediction_summary = ""
    if daily_feature and daily_feature.features:
        try:
            mr = get_model_runner()
            importances = mr.get_feature_importances()
            reasoning = generate_prediction_reasons(
                daily_feature.features, importances, top_n=7,
                predicted_drawdown=prediction.predicted_drawdown,
            )
            prediction_reasons = reasoning["reasons"]
            prediction_summary = reasoning["summary"]
        except Exception as e:
            logger.warning(f"Failed to generate prediction reasons: {e}")

    # Use latest price snapshot for current Nifty spot & VIX.
    # prediction.nifty_spot is from 9:20 AM and goes stale.
    # Always fetch the most recent snapshot (even across days — e.g. on
    # mornings before today's prediction runs, show yesterday's close).
    live_spot = prediction.nifty_spot  # fallback
    # VIX in the header should match the model's input (from features),
    # not the latest snapshot — avoids confusing mismatch with the
    # India VIX indicator which shows the model's feature value.
    model_vix = None
    if daily_feature and daily_feature.features:
        model_vix = daily_feature.features.get("india_vix")
    live_vix = model_vix if model_vix else prediction.vix

    snap_result = await db.execute(
        select(PriceSnapshot)
        .order_by(desc(PriceSnapshot.timestamp))
        .limit(1)
    )
    latest_snap = snap_result.scalar_one_or_none()
    if latest_snap:
        if latest_snap.nifty_spot:
            live_spot = latest_snap.nifty_spot

    return {
        "timestamp": prediction.timestamp.isoformat() if prediction.timestamp else None,
        "nifty_spot": live_spot,
        "vix": live_vix,
        "predicted_drawdown": prediction.predicted_drawdown,
        "predicted_drawdown_pct": round(prediction.predicted_drawdown * 100, 2),
        "classification": classification,
        "version_signals": version_signals,
        "indicators": indicators,
        "prediction_reasons": prediction_reasons,
        "prediction_summary": prediction_summary,
        "confidence_score": prediction.confidence_score,
        "status": "active" if prediction.date == today else "latest_available",
        "prediction_date": prediction.date.isoformat(),
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
        classification = get_classification_breakdown(p.predicted_drawdown, version=p.version)
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


@router.get("/activity/today")
async def get_today_activity(db: AsyncSession = Depends(get_db)):
    """
    Today's trading activity log: predictions, trade entries/exits,
    and pipeline health status.
    """
    today = today_ist()

    # 1. Predictions for today (all versions)
    pred_result = await db.execute(
        select(Prediction).where(Prediction.date == today).order_by(Prediction.timestamp)
    )
    predictions = pred_result.scalars().all()

    # 2. Trades opened today
    opened_result = await db.execute(
        select(Trade).where(Trade.entry_date == today).order_by(Trade.entry_time)
    )
    opened_trades = opened_result.scalars().all()

    # 3. Trades closed today
    closed_result = await db.execute(
        select(Trade).where(Trade.exit_date == today).order_by(Trade.exit_time)
    )
    closed_trades = closed_result.scalars().all()

    # 4. Price snapshot count for today
    snap_result = await db.execute(
        select(func.count()).select_from(PriceSnapshot).where(
            func.date(PriceSnapshot.timestamp) == today
        )
    )
    snapshot_count = snap_result.scalar() or 0

    # 5. EOD processed? Check if DailyPnl exists for today
    eod_result = await db.execute(
        select(func.count()).select_from(DailyPnl).where(DailyPnl.date == today)
    )
    eod_count = eod_result.scalar() or 0

    # 6. Exit check done? Any trade updated after 15:00 today with exit_date = today
    exits_checked = any(
        t.exit_time and t.exit_time.astimezone(IST).hour >= 15
        for t in closed_trades
    )
    # Also check if any open trade was updated after 15:00 (exit check ran but no exits)
    if not exits_checked and predictions:
        now = now_ist()
        exits_checked = now.hour >= 15 and now.minute >= 30

    # --- Build pipeline status ---
    prediction_time = None
    if predictions:
        ts = predictions[0].timestamp
        if ts:
            prediction_time = ts.astimezone(IST).strftime("%H:%M:%S")

    pipeline_status = {
        "prediction_generated": len(predictions) > 0,
        "prediction_time": prediction_time,
        "trades_entered": len(opened_trades),
        "exits_checked": exits_checked,
        "eod_processed": eod_count > 0,
        "price_snapshots": snapshot_count,
    }

    # --- Build events timeline ---
    events = []

    # Prediction event (group all versions into one event)
    if predictions:
        pred = predictions[0]
        pred_pct = round(pred.predicted_drawdown * 100, 2)

        # Group signals by type across versions
        signal_groups = defaultdict(list)
        for p in predictions:
            signal_groups[p.signal_type].append(p.version)

        signal_parts = []
        for sig_type, versions in signal_groups.items():
            label = sig_type.replace("_", " ").title()
            ver_str = ", ".join(versions)
            signal_parts.append(f"{label} ({ver_str})")

        events.append({
            "time": pred.timestamp.astimezone(IST).strftime("%H:%M:%S") if pred.timestamp else "",
            "type": "prediction",
            "version": None,
            "message": f"Model predicted {pred_pct}% drawdown",
            "detail": " | ".join(signal_parts),
        })

    # No-trade event (if prediction exists but no trades opened)
    if predictions and not opened_trades:
        all_no_trade = all(p.signal_type == "no_trade" for p in predictions)
        if all_no_trade:
            events.append({
                "time": predictions[0].timestamp.astimezone(IST).strftime("%H:%M:%S") if predictions[0].timestamp else "",
                "type": "no_trade",
                "version": None,
                "message": "No trades today — all versions in No Trade zone",
                "detail": "",
            })

    # Trade opened events
    for t in opened_trades:
        trade_label = t.trade_type.replace("_", " ").title()

        # Build strike detail
        detail_parts = []
        if t.trade_type == "iron_condor":
            if t.sell_strike and t.buy_strike:
                detail_parts.append(f"Put {int(t.sell_strike)}/{int(t.buy_strike)}")
            if t.ic_call_sell and t.ic_call_buy:
                detail_parts.append(f"Call {int(t.ic_call_sell)}/{int(t.ic_call_buy)}")
        elif t.sell_strike and t.buy_strike:
            detail_parts.append(f"Strikes {int(t.sell_strike)}/{int(t.buy_strike)}")

        detail_parts.append(f"Credit: {t.credit_received:.2f}/lot")
        if t.expiry:
            detail_parts.append(f"Expiry: {t.expiry.strftime('%d %b')}")

        events.append({
            "time": t.entry_time.astimezone(IST).strftime("%H:%M:%S") if t.entry_time else "",
            "type": "trade_opened",
            "version": t.version,
            "message": f"Opened {trade_label}",
            "detail": " | ".join(detail_parts),
        })

    # Trade closed events
    for t in closed_trades:
        trade_label = t.trade_type.replace("_", " ").title()
        exit_label = (t.exit_reason or "closed").replace("_", " ").title()

        detail_parts = []
        if t.realized_pnl is not None:
            pnl_sign = "+" if t.realized_pnl >= 0 else ""
            detail_parts.append(f"PnL: {pnl_sign}{t.realized_pnl:.0f}")
            if t.current_pnl_pct:
                detail_parts.append(f"{pnl_sign}{t.current_pnl_pct:.1f}%")
        if t.entry_date and t.exit_date:
            held = (t.exit_date - t.entry_date).days
            detail_parts.append(f"Held {held} days")

        events.append({
            "time": t.exit_time.astimezone(IST).strftime("%H:%M:%S") if t.exit_time else "",
            "type": "trade_closed",
            "version": t.version,
            "message": f"Closed {trade_label} — {exit_label}",
            "detail": " | ".join(detail_parts),
        })

    # Sort events by time
    events.sort(key=lambda e: e["time"])

    return {
        "date": today.isoformat(),
        "pipeline_status": pipeline_status,
        "events": events,
    }
