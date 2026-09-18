"""Structured logging setup shared by every service entrypoint."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from libs.config import settings

_RESERVED_RECORD_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON with any ``extra`` fields merged in."""

    def __init__(self, service: str) -> None:
        """Initialise the formatter.

        Args:
            service: Logical service name stamped on every record.
        """
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        """Render one record as a JSON line."""
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "service": self._service,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_KEYS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(service: str, *, level: str | None = None) -> logging.Logger:
    """Install the JSON handler on the root logger and return a service logger.

    Args:
        service: Logical service name, e.g. ``"market_analyst"``.
        level: Log level override; defaults to the configured level.

    Returns:
        A logger named after the service.
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter(service))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel((level or settings.log_level).upper())

    logging.getLogger("aiokafka").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    return logging.getLogger(service)


__all__ = ["JsonFormatter", "configure_logging"]
