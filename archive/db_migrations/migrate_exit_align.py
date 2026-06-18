"""
One-time migration: add trailing_stop_active and entry_vix columns
to the trades table. Required for backtest-aligned exit logic.

Usage:
    DATABASE_URL="postgresql+asyncpg://..." python -m db.migrate_exit_align
"""

import asyncio
import ssl
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from config import DATABASE_URL

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

COLUMNS = [
    ("trailing_stop_active", "BOOLEAN DEFAULT FALSE"),
    ("entry_vix", "FLOAT"),
]


async def migrate():
    engine = create_async_engine(
        DATABASE_URL,
        connect_args={"ssl": ssl_context},
    )

    async with engine.begin() as conn:
        for col_name, col_type in COLUMNS:
            try:
                await conn.execute(
                    text(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
                )
                print(f"  Added column: {col_name}")
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    print(f"  Column {col_name} already exists, skipping")
                else:
                    print(f"  Error adding {col_name}: {e}")

    await engine.dispose()
    print("Migration complete.")


if __name__ == "__main__":
    asyncio.run(migrate())
