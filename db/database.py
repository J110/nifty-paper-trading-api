"""
Async database engine, session factory, and helpers for FastAPI.
Uses Neon PostgreSQL (serverless, free tier) with SSL.
"""

import ssl
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from typing import AsyncGenerator

from config import DATABASE_URL

# Neon requires SSL — create SSL context for asyncpg
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
    pool_pre_ping=True,  # handles Neon cold starts gracefully
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
    """Create all tables defined on the ORM Base."""
    from db.models import Base  # local import to avoid circular deps

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
