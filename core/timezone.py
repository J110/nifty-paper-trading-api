"""
IST timezone utilities.

All trading logic is in Indian Standard Time (IST = UTC+5:30).
Render servers run in UTC, so we must NEVER use bare date.today()
or datetime.now() — always use the IST-aware versions below.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def now_ist() -> datetime:
    """Current datetime in IST (timezone-aware)."""
    return datetime.now(IST)


def today_ist() -> date:
    """Current date in IST."""
    return datetime.now(IST).date()
