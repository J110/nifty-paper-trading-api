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
        },
    }


@app.post("/api/trigger-prediction")
async def trigger_prediction():
    """
    Manually trigger the daily prediction pipeline.
    Use this when the scheduler missed (e.g., Render free tier was asleep at 9:20 AM).
    """
    from scheduler.jobs import generate_daily_predictions
    import asyncio

    logger.info("Manual prediction trigger received")
    # Run in background so the API responds immediately
    asyncio.create_task(generate_daily_predictions())
    return {
        "status": "triggered",
        "message": "Prediction pipeline started. Check /api/signals/current in ~30 seconds.",
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
