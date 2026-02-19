"""
Nifty Paper Trading API — FastAPI Application

Serves predictions, trades, and analytics for 3 model versions (v5.4.2, v5.4.3, v5.4.4).
Runs scheduled jobs for daily predictions and trade management.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.database import init_db
from api.signals import router as signals_router
from api.trades import router as trades_router
from api.charts import router as charts_router
from api.returns import router as returns_router
from scheduler.jobs import setup_scheduler, check_and_recover_missed_prediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Scheduler instance (global for cleanup)
scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global scheduler

    # Startup
    logger.info("Starting Nifty Paper Trading API...")

    # Initialize database tables
    await init_db()
    logger.info("Database initialized")

    # Start scheduler
    scheduler = setup_scheduler()
    scheduler.start()
    logger.info("APScheduler started")

    # Check if we missed today's prediction (server restart recovery)
    # Delay recovery by 30s so the server binds to port first
    # (Render free tier times out if port isn't open quickly)
    import asyncio

    async def _delayed_recovery():
        await asyncio.sleep(30)
        await check_and_recover_missed_prediction()

    asyncio.create_task(_delayed_recovery())

    yield

    # Shutdown
    if scheduler:
        scheduler.shutdown()
        logger.info("APScheduler stopped")
    logger.info("Nifty Paper Trading API stopped")


app = FastAPI(
    title="Nifty Paper Trading API",
    description="Paper trading system for Nifty options with 3 model versions",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Flutter web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(signals_router)
app.include_router(trades_router)
app.include_router(charts_router)
app.include_router(returns_router)


@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    """Health check endpoint for Render + UptimeRobot pinger."""
    from sqlalchemy import select, func
    from db.database import async_session_factory
    from db.models import Prediction
    from core.timezone import now_ist, today_ist

    # Quick DB check: is today's prediction present?
    today_predictions = 0
    last_prediction_date = None
    db_healthy = False
    db_error = None
    try:
        async with async_session_factory() as db:
            result = await db.execute(
                select(func.count()).select_from(Prediction).where(
                    Prediction.date == today_ist()
                )
            )
            today_predictions = result.scalar() or 0

            result2 = await db.execute(
                select(func.max(Prediction.date))
            )
            last_prediction_date = result2.scalar()
            db_healthy = True
    except Exception as e:
        db_error = str(e)
        logger.error(f"Health check DB query failed: {e}")

    scheduler_running = scheduler is not None and scheduler.running
    overall_healthy = db_healthy and scheduler_running

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "service": "nifty-paper-trading-api",
        "version": "1.0.0",
        "db_healthy": db_healthy,
        "db_error": db_error,
        "scheduler_running": scheduler_running,
        "today_predictions": today_predictions,
        "last_prediction_date": last_prediction_date.isoformat() if last_prediction_date else None,
        "server_time_ist": now_ist().isoformat(),
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Nifty Paper Trading API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "signals": "/api/signals/current",
            "trades": "/api/trades/{version}",
            "delay_analysis": "/api/trades/{version}/delay-analysis",
            "returns": "/api/trades/{version}/returns",
            "chart_nifty": "/api/chart-data/nifty",
            "chart_equity": "/api/chart-data/equity/{version}",
            "indicators": "/api/indicators",
            "trigger": "/api/trigger-prediction (POST)",
            "renew_token": "/api/renew-token (POST)",
            "update_data": "/api/update-data (POST)",
        },
    }


@app.post("/api/trigger-prediction")
async def trigger_prediction():
    """
    Manually trigger the daily prediction pipeline.
    Runs synchronously so errors are returned in the response.
    """
    from scheduler.jobs import generate_daily_predictions
    from db.database import async_session_factory
    from db.models import Prediction
    from core.timezone import today_ist, now_ist
    from sqlalchemy import select, func
    import traceback

    logger.info("Manual prediction trigger received")

    # Count predictions before
    before_count = 0
    try:
        async with async_session_factory() as db:
            result = await db.execute(
                select(func.count()).select_from(Prediction).where(
                    Prediction.date == today_ist()
                )
            )
            before_count = result.scalar() or 0
    except Exception:
        pass

    error_info = None
    try:
        await generate_daily_predictions()
    except Exception as e:
        logger.error(f"Manual trigger failed: {e}", exc_info=True)
        error_info = {"error": str(e), "traceback": traceback.format_exc()}

    # Count predictions after
    after_count = 0
    try:
        async with async_session_factory() as db:
            result = await db.execute(
                select(func.count()).select_from(Prediction).where(
                    Prediction.date == today_ist()
                )
            )
            after_count = result.scalar() or 0
    except Exception:
        pass

    result = {
        "status": "error" if error_info else ("success" if after_count > before_count else "warning"),
        "message": f"Pipeline completed. Predictions for {today_ist()}: {before_count} -> {after_count}",
        "today_date": str(today_ist()),
        "server_time": str(now_ist()),
        "predictions_before": before_count,
        "predictions_after": after_count,
    }
    if error_info:
        result.update(error_info)
    return result


@app.post("/api/debug/run-pipeline")
async def debug_run_pipeline():
    """Run the exact pipeline code inline with full error capture at every step."""
    import traceback
    from db.database import async_session_factory
    from db.models import Prediction, DailyFeature
    from core.timezone import today_ist, now_ist
    from core.feature_engine import build_live_features
    from core.model_runner import ModelRunner
    from core.signal_mapper import map_signal
    from core.dhan_client import DhanClient
    from config import DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH, ACTIVE_VERSIONS, VERSION_CONFIGS
    from sqlalchemy import delete
    import pandas as pd

    steps = {}
    try:
        dc = DhanClient()
        spot = await dc.get_nifty_ltp()
        vix_live = await dc.get_india_vix()
        steps["1_dhan"] = {"spot": spot, "vix": vix_live}

        features = await build_live_features(dc, pd.DataFrame(), None)
        steps["2_features"] = {"count": len(features)}

        if spot is None:
            spot = features.get("nifty_close", 0)
        if vix_live is None:
            vix_live = features.get("india_vix", 0)
        steps["3_fallback"] = {"spot": float(spot), "vix": float(vix_live)}

        mr = ModelRunner(DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH)
        prediction_value = mr.predict(features)
        steps["4_prediction"] = float(prediction_value)

        async with async_session_factory() as db:
            try:
                await db.execute(delete(DailyFeature).where(DailyFeature.date == today_ist()))
                df = DailyFeature(date=today_ist(), features=features, vix=features.get("vix"), rsi_14=features.get("rsi_14"))
                db.add(df)
                await db.flush()
                steps["5a_features_flush"] = "OK"

                await db.execute(delete(Prediction).where(Prediction.date == today_ist()))
                for version in ACTIVE_VERSIONS:
                    cfg = VERSION_CONFIGS[version]
                    signal = map_signal(prediction_value, cfg)
                    pred = Prediction(date=today_ist(), timestamp=now_ist(), predicted_drawdown=prediction_value,
                        signal_type=signal["signal"], version=version, nifty_spot=spot, vix=vix_live,
                        confidence_score=0.25, graduated_mult=signal["size_mult"], features=features)
                    db.add(pred)

                await db.flush()
                steps["5b_predictions_flush"] = "OK"

                await db.commit()
                steps["5c_commit"] = "OK"

            except Exception as e:
                await db.rollback()
                steps["5_db_error"] = str(e)

        steps["status"] = "success"
    except Exception as e:
        steps["fatal_error"] = f"{e}"
        steps["status"] = "failed"

    steps["today"] = str(today_ist())
    return steps


@app.get("/api/debug/pipeline-test")
async def debug_pipeline_test():
    """Test each step of the prediction pipeline independently."""
    import traceback
    import os
    results = {}

    # Step 1: Check parquet file
    try:
        from core.feature_engine import MERGED_DAILY_PATH, DHAN_RAW_PATH, DHAN_SKEW_PATH
        import pandas as pd
        results["parquet_exists"] = os.path.exists(MERGED_DAILY_PATH)
        results["parquet_path"] = MERGED_DAILY_PATH
        if results["parquet_exists"]:
            df = pd.read_parquet(MERGED_DAILY_PATH)
            results["parquet_rows"] = len(df)
            results["parquet_last_date"] = str(df.index[-1].date())
        results["dhan_raw_exists"] = os.path.exists(DHAN_RAW_PATH)
        results["dhan_skew_exists"] = os.path.exists(DHAN_SKEW_PATH)
    except Exception as e:
        results["parquet_error"] = f"{e}\n{traceback.format_exc()}"

    # Step 2: Check Dhan API
    try:
        from core.dhan_client import DhanClient
        dc = DhanClient()
        spot = await dc.get_nifty_ltp()
        vix = await dc.get_india_vix()
        results["dhan_spot"] = spot
        results["dhan_vix"] = vix
    except Exception as e:
        results["dhan_error"] = f"{e}\n{traceback.format_exc()}"

    # Step 3: Build features
    try:
        from core.feature_engine import build_live_features
        features = await build_live_features(dc, None, None)
        results["features_count"] = len(features) if features else 0
        results["features_sample"] = {k: features[k] for k in list(features.keys())[:5]} if features else {}
    except Exception as e:
        results["features_error"] = f"{e}\n{traceback.format_exc()}"

    # Step 4: Model prediction
    try:
        from core.model_runner import ModelRunner
        from config import DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH
        mr = ModelRunner(DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH)
        if 'features_error' not in results:
            pred = mr.predict(features)
            results["prediction"] = float(pred)
            results["prediction_pct"] = f"{pred*100:.2f}%"
        else:
            results["model_skipped"] = "Features failed"
    except Exception as e:
        results["model_error"] = f"{e}\n{traceback.format_exc()}"

    # Step 5: Check today_ist
    from core.timezone import today_ist, now_ist
    results["today_ist"] = str(today_ist())
    results["now_ist"] = str(now_ist())

    return results


@app.post("/api/renew-token")
async def renew_token_endpoint():
    """
    Manually trigger Dhan token renewal.
    Extends the current token for another 24 hours.
    Token must still be active (not expired) for this to work.
    """
    from scheduler.jobs import dhan_client
    from core.timezone import now_ist
    import traceback

    logger.info("Manual token renewal triggered")
    try:
        result = await dhan_client.renew_token()
        return {
            "status": "renewed",
            "expiry": result.get("expiryTime"),
            "client_id": result.get("dhanClientId"),
            "server_time": str(now_ist()),
        }
    except Exception as e:
        logger.error(f"Manual token renewal failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "server_time": str(now_ist()),
        }


@app.post("/api/update-data")
async def update_data_endpoint():
    """
    Manually trigger data update (yfinance → merged_daily.parquet).
    Useful for ensuring latest market data is available before predictions.
    Runs automatically at 9:05 AM IST, but can be triggered manually.
    """
    from core.data_updater import update_merged_daily
    from core.timezone import now_ist
    import traceback

    logger.info("Manual data update triggered")
    try:
        result = await update_merged_daily()
        result["server_time"] = str(now_ist())
        return result
    except Exception as e:
        logger.error(f"Manual data update failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "server_time": str(now_ist()),
        }


@app.get("/api/debug/yahoo-test")
async def debug_yahoo_test():
    """Diagnostic: test Yahoo Finance direct API download."""
    import traceback
    from core.timezone import now_ist
    from datetime import timedelta
    from core.data_updater import _download_yahoo

    results = {"method": "direct_yahoo_api", "server_time": str(now_ist())}
    try:
        start = (now_ist() - timedelta(days=7)).strftime("%Y-%m-%d")
        end = (now_ist() + timedelta(days=1)).strftime("%Y-%m-%d")
        results["download_range"] = f"{start} to {end}"

        df = _download_yahoo("^NSEI", "Nifty 50", start, end)
        if df is not None and not df.empty:
            results["rows"] = len(df)
            results["dates"] = [str(d.date()) for d in df.index]
            results["last_close"] = float(df["Close"].iloc[-1])
        else:
            results["error"] = "Yahoo API returned no data"
    except Exception as e:
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()

    return results


@app.post("/api/backfill")
async def run_backfill_endpoint():
    """
    Run historical backfill from Jan 2026 to today.
    Populates predictions, trades, and daily PnL for all versions.
    Takes ~60-120 seconds. Run once after initial deployment.
    Runs synchronously so errors are returned in the response.
    """
    from backfill import run_backfill
    from db.database import async_session_factory

    logger.info("Backfill endpoint triggered — running synchronously")
    try:
        async with async_session_factory() as db:
            result = await run_backfill(db)
            logger.info(f"Backfill result: {result}")
            return {"status": "backfill_complete", "result": result}
    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
        import traceback
        return {
            "status": "backfill_failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


@app.get("/api/debug/db-counts")
async def db_counts():
    """Check how many rows are in each table — useful for debugging backfill."""
    from sqlalchemy import text
    from db.database import async_session_factory

    async with async_session_factory() as db:
        tables = ["predictions", "trades", "daily_pnl", "daily_features"]
        counts = {}
        for table in tables:
            try:
                result = await db.execute(text(f"SELECT COUNT(*) FROM {table}"))
                counts[table] = result.scalar()
            except Exception as e:
                counts[table] = f"error: {e}"
        return counts


@app.post("/api/debug/test-email")
async def debug_test_email():
    """Send a test email to verify Resend API key and email delivery."""
    import os
    import httpx

    api_key = os.environ.get("RESEND_API_KEY", "")
    notify_email = os.environ.get("NOTIFY_EMAIL", "anmol@turings.xyz")
    from_email = "Nifty Trading Bot <onboarding@resend.dev>"

    if not api_key:
        return {"status": "error", "message": "RESEND_API_KEY not set in environment"}

    # Show config (mask API key)
    config_info = {
        "api_key_set": bool(api_key),
        "api_key_prefix": api_key[:10] + "..." if len(api_key) > 10 else "too_short",
        "api_key_length": len(api_key),
        "notify_email": notify_email,
        "from_email": from_email,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": from_email,
                    "to": [notify_email],
                    "subject": "Test Email from Nifty Trading Bot",
                    "html": "<h2>Email delivery is working!</h2><p>This is a test email from your Nifty Trading Bot.</p>",
                },
                timeout=10,
            )

            return {
                "status": "sent" if response.status_code in (200, 202) else "failed",
                "resend_status_code": response.status_code,
                "resend_response": response.json() if response.status_code < 500 else response.text,
                "config": config_info,
            }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "config": config_info,
        }


@app.post("/api/debug/recompute-date")
async def debug_recompute_date(target_date: str):
    """
    Recompute predictions for a specific past date using current parquet data.
    Useful for fixing stale predictions caused by data pipeline failures.
    Does NOT touch trades — only updates Prediction and DailyFeature rows.
    """
    import traceback
    from datetime import date as date_cls, datetime
    from db.database import async_session_factory
    from db.models import Prediction, DailyFeature
    from core.model_runner import ModelRunner
    from core.signal_mapper import map_signal, get_classification_breakdown
    from core.timezone import IST
    from config import (
        DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH,
        ACTIVE_VERSIONS, VERSION_CONFIGS,
    )
    from sqlalchemy import delete
    import pandas as pd

    try:
        target = date_cls.fromisoformat(target_date)
    except ValueError:
        return {"status": "error", "message": f"Invalid date format: {target_date}. Use YYYY-MM-DD."}

    try:
        # Load model
        mr = ModelRunner(DOWNSIDE_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH)

        # Load merged_daily (server's current data)
        merged = pd.read_parquet("/app/data/merged_daily.parquet")
        logger.info(f"Recompute: merged_daily has {len(merged)} rows, last={merged.index[-1].date()}")

        if pd.Timestamp(target) > merged.index[-1]:
            return {"status": "error", "message": f"Target date {target} is beyond parquet data (last: {merged.index[-1].date()})"}

        # Compute features using the PREVIOUS trading day's data.
        # This matches how the live system works: at 9:20 AM on date X,
        # only data through X-1 (yesterday's close) is available.
        from shared.feature_compute import compute_features_for_date, load_dhan_options_features

        target_ts = pd.Timestamp(target)
        dates_before = merged.index[merged.index < target_ts]
        if len(dates_before) == 0:
            return {"status": "error", "message": f"No data before {target} to compute features from"}
        prev_trading_day = dates_before[-1]
        logger.info(f"Recompute {target}: using data through {prev_trading_day.date()} (previous trading day)")

        dhan = None
        try:
            dhan = load_dhan_options_features(
                "/app/data/dhan_raw_options.parquet",
                "/app/data/daily_iv_skew_params.parquet",
            )
        except Exception as e:
            logger.warning(f"Recompute: Dhan features unavailable: {e}")

        features = compute_features_for_date(merged, prev_trading_day, dhan)
        if features is None:
            return {"status": "error", "message": f"Could not compute features for {target} (missing data?)"}

        # Run model prediction
        prediction_value = mr.predict(features)
        logger.info(f"Recompute {target}: prediction={prediction_value*100:.4f}%")

        # Store in DB (replace old predictions + features for this date)
        async with async_session_factory() as db:
            await db.execute(delete(Prediction).where(Prediction.date == target))
            await db.execute(delete(DailyFeature).where(DailyFeature.date == target))

            ts = datetime(target.year, target.month, target.day, 9, 20, 0, tzinfo=IST)

            version_signals = {}
            for version in ACTIVE_VERSIONS:
                cfg = VERSION_CONFIGS[version]
                signal = map_signal(prediction_value, cfg)

                breakdown = get_classification_breakdown(prediction_value, version=version)
                active_zone = next(
                    (z for z in breakdown["zones"] if z.get("confidence") == "active"),
                    None,
                )
                confidence = active_zone["distance_to_boundary"] if active_zone else 0

                pred = Prediction(
                    date=target,
                    timestamp=ts,
                    predicted_drawdown=prediction_value,
                    signal_type=signal["signal"],
                    version=version,
                    nifty_spot=features.get("nifty_close", 0),
                    vix=features.get("india_vix", 0),
                    confidence_score=confidence,
                    graduated_mult=signal["size_mult"],
                    features=features,
                )
                db.add(pred)
                version_signals[version] = signal["signal"]

            daily_feature = DailyFeature(
                date=target,
                features=features,
                vix=features.get("india_vix"),
                vix_20d_avg=features.get("vix_20d_avg"),
                nifty_20d_return=features.get("nifty_20d_return"),
                nifty_50d_return=features.get("nifty_50d_return"),
                iv_skew=features.get("iv_skew"),
                fii_net=features.get("fii_net_5d"),
                dii_net=features.get("dii_net_5d"),
                put_call_ratio=features.get("put_call_ratio"),
                rsi_14=features.get("rsi_14"),
                adx_14=features.get("adx_14"),
            )
            db.add(daily_feature)
            await db.commit()

        return {
            "status": "recomputed",
            "date": target_date,
            "data_through": str(prev_trading_day.date()),
            "prediction": prediction_value,
            "prediction_pct": f"{prediction_value*100:.2f}%",
            "versions": version_signals,
            "features_count": len(features),
            "nifty_close": features.get("nifty_close"),
        }

    except Exception as e:
        logger.error(f"Recompute failed for {target_date}: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }


@app.post("/api/debug/delete-trade")
async def debug_delete_trade(trade_id: str):
    """
    Delete a specific trade by trade_id.
    Use this to remove trades that were opened on stale/incorrect data.
    Returns the deleted trade details for confirmation.
    """
    from db.database import async_session_factory
    from db.models import Trade, DelayPrice
    from sqlalchemy import select, delete

    try:
        async with async_session_factory() as db:
            # First, fetch the trade to show what's being deleted
            result = await db.execute(
                select(Trade).where(Trade.trade_id == trade_id)
            )
            trade = result.scalar_one_or_none()

            if not trade:
                return {"status": "not_found", "message": f"No trade found with id: {trade_id}"}

            trade_info = {
                "trade_id": trade.trade_id,
                "version": trade.version,
                "entry_date": str(trade.entry_date),
                "trade_type": trade.trade_type,
                "status": trade.status,
                "entry_spot": trade.entry_spot,
                "predicted_drawdown": trade.predicted_drawdown,
                "realized_pnl": trade.realized_pnl,
            }

            # Delete related rows first (foreign key constraints)
            await db.execute(delete(DelayPrice).where(DelayPrice.trade_id == trade_id))
            # Delete the trade
            await db.execute(delete(Trade).where(Trade.trade_id == trade_id))
            await db.commit()

            return {
                "status": "deleted",
                "deleted_trade": trade_info,
            }

    except Exception as e:
        logger.error(f"Delete trade failed for {trade_id}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
