"""IST calendar helpers. NSE holidays are not published as free machine-readable data,
so weekends are skipped and holidays are reported as 'possible holiday' warnings."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
OPEN, CLOSE = time(9, 15), time(15, 30)


def now_ist() -> datetime:
    return datetime.now(IST)


def expected_last_session(now: datetime | None = None, holidays: set[str] = frozenset()) -> date:
    """Most recent weekday session whose daily bar should exist by `now`."""
    now = now or now_ist()
    d = now.date() if now.time() >= CLOSE else now.date() - timedelta(days=1)
    while d.weekday() >= 5 or d.isoformat() in holidays:
        d -= timedelta(days=1)
    return d


def is_market_open(now: datetime | None = None) -> bool:
    now = now or now_ist()
    return now.weekday() < 5 and OPEN <= now.time() < CLOSE
