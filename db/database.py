"""
Async database engine, session factory, and helpers for FastAPI.
Uses Supabase PostgreSQL (free tier) with SSL.
"""

import ssl
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from typing import AsyncGenerator

from config import DATABASE_URL

# Supabase requires SSL — create SSL context for asyncpg
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    connect_args={"ssl": ssl_context},
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # handles connection drops gracefully
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Alias used by scheduler jobs (context manager)
async_session_factory = async_session


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async DB session."""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables and ensure new columns exist.

    create_all only creates tables that don't exist — it does NOT add
    columns to existing tables.  We run lightweight ALTER TABLE statements
    for any columns added after the initial schema so that deploys don't
    silently break queries/inserts.
    """
    from db.models import Base  # local import to avoid circular deps
    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Columns added after initial schema — safe to re-run (IF NOT EXISTS)
        migrations = [
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS stop_loss_last_breach_date DATE",
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS trailing_stop_active BOOLEAN DEFAULT FALSE",
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS entry_vix FLOAT",
        ]
        for sql in migrations:
            await conn.execute(text(sql))
