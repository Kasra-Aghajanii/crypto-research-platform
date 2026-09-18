"""Event (de)serialisation helpers for the Kafka transport.

Events travel as UTF-8 JSON produced by Pydantic, so the wire format is exactly
the model schema and stays readable in ``rpk topic consume`` output.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from libs.schemas.base import BaseEvent


class EventDecodeError(RuntimeError):
    """Raised when a Kafka payload cannot be decoded into the expected event."""


def serialize(event: BaseEvent) -> bytes:
    """Serialise an event to UTF-8 JSON bytes.

    Computed fields are excluded.  They are derived from the stored fields, so
    they carry no information on the wire -- and because every event model sets
    ``extra="forbid"``, leaving them in makes the payload fail to validate on
    the way back in.  Emitting them would silently poison every topic carrying
    an event with a computed property.

    Args:
        event: The event to serialise.

    Returns:
        The encoded payload.
    """
    computed = set(type(event).model_computed_fields)
    return event.model_dump_json(exclude=computed).encode("utf-8")


def deserialize[EventT: BaseEvent](payload: bytes, model: type[EventT]) -> EventT:
    """Decode a Kafka payload into a concrete event model.

    Args:
        payload: Raw message bytes.
        model: Expected event class.

    Returns:
        The validated event instance.

    Raises:
        EventDecodeError: If the payload is not valid JSON for ``model``.
    """
    try:
        return model.model_validate_json(payload)
    except ValidationError as exc:
        raise EventDecodeError(f"Payload is not a valid {model.__name__}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EventDecodeError(f"Payload is not valid JSON: {exc}") from exc


def encode_key(key: str | None) -> bytes | None:
    """Encode a partition key, or return ``None`` for round-robin partitioning."""
    return key.encode("utf-8") if key is not None else None


__all__ = ["EventDecodeError", "deserialize", "encode_key", "serialize"]
