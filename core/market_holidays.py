"""
NSE market holidays — used by the staleness guard to count trading days
instead of calendar days, preventing false alerts on long weekends.

Update this list each year from the NSE circular:
https://www.nseindia.com/regulations/stock-exchange-holidays
"""

import numpy as np
from datetime import date

# NSE holidays for 2026 (only weekday closures — Sat/Sun handled by numpy)
# Source: https://www.calendarlabs.com/nse-market-holidays-2026/
NSE_HOLIDAYS_2026 = [
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 3),    # Holi
    date(2026, 3, 26),   # Ram Navami
    date(2026, 3, 31),   # Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 28),   # Bakri Id / Eid ul-Adha
    date(2026, 6, 26),   # Muharram
    date(2026, 9, 14),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 10),  # Diwali (Balipratipada)
    date(2026, 11, 24),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
]

_ALL_HOLIDAYS = sorted(NSE_HOLIDAYS_2026)
_NP_HOLIDAYS = np.array([np.datetime64(d) for d in _ALL_HOLIDAYS])


def trading_days_between(start: date, end: date) -> int:
    """Count trading days between two dates (exclusive of start, inclusive of end).

    Uses numpy busday_count which excludes weekends, plus our NSE holiday list.
    """
    return int(np.busday_count(
        np.datetime64(start),
        np.datetime64(end),
        holidays=_NP_HOLIDAYS,
    ))
