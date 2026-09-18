"""Shared conversion helpers for Hyperliquid payloads.

The exchange sends numbers as strings and timestamps as epoch milliseconds.
These helpers centralise that conversion so every collector fails the same way
on a malformed field: with a :class:`PayloadError` naming the field.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any


class PayloadError(ValueError):
    """Raised when an exchange payload cannot be converted into an event."""


def to_decimal(value: Any, field: str) -> Decimal:
    """Convert an exchange string or number into a ``Decimal``.

    Args:
        value: Raw value from the payload.
        field: Field name, used in the error message.

    Returns:
        The parsed decimal.

    Raises:
        PayloadError: If the value is missing or not numeric.
    """
    if value is None:
        raise PayloadError(f"Missing numeric field {field!r}.")
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError) as exc:
        raise PayloadError(f"Field {field!r} is not numeric: {value!r}") from exc


def to_datetime(value: Any, field: str) -> datetime:
    """Convert exchange epoch milliseconds into an aware UTC datetime.

    Args:
        value: Epoch milliseconds.
        field: Field name, used in the error message.

    Returns:
        The parsed timestamp.

    Raises:
        PayloadError: If the value is missing or not an epoch-ms timestamp.
    """
    if value is None:
        raise PayloadError(f"Missing timestamp field {field!r}.")
    try:
        return datetime.fromtimestamp(int(value) / 1000.0, tz=UTC)
    except (TypeError, ValueError, OSError, OverflowError) as exc:
        raise PayloadError(f"Field {field!r} is not an epoch-ms timestamp: {value!r}") from exc


__all__ = ["PayloadError", "to_datetime", "to_decimal"]
