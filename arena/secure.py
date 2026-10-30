"""Keep secrets out of logs, files and the journal.

API keys and broker tokens live only in environment variables loaded from
`.env`. This module makes sure they never leak from there: log output is
redacted, and `.env` and the journal are readable by their owner only.
"""
from __future__ import annotations

import logging
import os
import re
import stat
from pathlib import Path

SECRET_ENV_KEYS = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
    "UPSTOX_API_KEY", "UPSTOX_API_SECRET", "UPSTOX_ACCESS_TOKEN",
    "TELEGRAM_BOT_TOKEN",
)

_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}")
_ANTHROPIC_KEY = re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}")
_KEY_VALUE = re.compile(
    r"(?i)(\b(?:access_token|api_key|api_secret|client_secret|password|token|code)[\"']?\s*[=:]\s*[\"']?)"
    r"[^\s&\"',}]{6,}"
)


def redact(text: str) -> str:
    """Replace known secret values and secret-looking tokens with [REDACTED]."""
    for k in SECRET_ENV_KEYS:
        v = os.environ.get(k, "")
        if len(v) >= 8:
            text = text.replace(v, "[REDACTED]")
    text = _BEARER.sub(r"\1[REDACTED]", text)
    text = _ANTHROPIC_KEY.sub("[REDACTED]", text)
    return _KEY_VALUE.sub(r"\1[REDACTED]", text)


class RedactingFormatter(logging.Formatter):
    """Redacts the fully formatted record, including exception tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def owner_only(path: str | Path) -> bool:
    """chmod 600 on POSIX. Returns True if the permissions had to be tightened."""
    p = Path(path)
    if os.name != "posix" or not p.exists():
        return False
    if stat.S_IMODE(p.stat().st_mode) & 0o077:
        try:
            os.chmod(p, 0o600)
        except OSError:
            return False
        return True
    return False
