"""
One-time migration: add bear debit columns to the trades table.
Run this once after deploying v6.2 code.

Usage:
    python -m db.migrate_v62
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
    ("is_bear_debit", "BOOLEAN DEFAULT FALSE"),
    ("bear_tier", "INTEGER DEFAULT 0"),
    ("entry_debit", "FLOAT"),
    ("predicted_drawdown", "FLOAT"),
    ("max_profit", "FLOAT"),
    ("max_loss_amount", "FLOAT"),
    ("bear_trail_high", "FLOAT DEFAULT 0.0"),
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
