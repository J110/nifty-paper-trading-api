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
    import asyncio
    asyncio.create_task(check_and_recover_missed_prediction())

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
    except Exception:
        pass  # Don't fail health check if DB is slow

    scheduler_running = scheduler is not None and scheduler.running

    return {
        "status": "healthy",
        "service": "nifty-paper-trading-api",
        "version": "1.0.0",
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
