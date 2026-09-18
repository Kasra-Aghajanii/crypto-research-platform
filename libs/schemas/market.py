"""Market data events produced by the Hyperliquid ingestion services.

Prices and sizes are carried as ``Decimal`` so that no precision is lost between
the exchange wire format and storage.  Indicator maths converts to ``float``
explicitly at the point of use.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator

from libs.schemas.base import BaseEvent, EventType


class Side(StrEnum):
    """Aggressor side of a trade, or side of an order."""

    BUY = "buy"
    SELL = "sell"


class Candle(BaseEvent):
    """A single OHLCV candle for one symbol and interval.

    Attributes:
        symbol: Hyperliquid coin, e.g. ``"BTC"`` (no slash, no USDT suffix).
        interval: Candle interval, e.g. ``"1m"``.
        open_time: Start of the candle window (UTC).
        close_time: End of the candle window (UTC).
        is_closed: ``True`` once the exchange has finalised the candle.
    """

    event_type: Literal[EventType.CANDLE] = EventType.CANDLE

    symbol: str = Field(description="Hyperliquid perp coin symbol.")
    interval: str = Field(description="Candle interval identifier, e.g. 1m/5m/15m/1h.")
    open_time: datetime = Field(description="Candle window start (UTC).")
    close_time: datetime = Field(description="Candle window end (UTC).")
    open: Decimal = Field(description="Open price.")
    high: Decimal = Field(description="High price.")
    low: Decimal = Field(description="Low price.")
    close: Decimal = Field(description="Close price.")
    volume: Decimal = Field(default=Decimal(0), ge=0, description="Base-asset volume.")
    trade_count: int = Field(default=0, ge=0, description="Number of trades in the window.")
    is_closed: bool = Field(default=False, description="Whether the candle is final.")

    @field_validator("symbol")
    @classmethod
    def _normalise_symbol(cls, value: str) -> str:
        """Uppercase the symbol and reject centralised-exchange style pairs."""
        symbol = value.strip().upper()
        if "/" in symbol or symbol.endswith("USDT"):
            raise ValueError(
                f"Invalid Hyperliquid symbol {value!r}: expected a bare coin such as 'BTC'."
            )
        return symbol

    @property
    def typical_price(self) -> Decimal:
        """Return the (high + low + close) / 3 typical price."""
        return (self.high + self.low + self.close) / Decimal(3)


class TradeTick(BaseEvent):
    """A single executed trade printed on the Hyperliquid tape."""

    event_type: Literal[EventType.TRADE_TICK] = EventType.TRADE_TICK

    symbol: str = Field(description="Hyperliquid perp coin symbol.")
    price: Decimal = Field(gt=0, description="Execution price.")
    size: Decimal = Field(gt=0, description="Executed size in base asset.")
    side: Side = Field(description="Aggressor side.")
    trade_id: str | None = Field(default=None, description="Exchange trade identifier, if any.")


class FundingRate(BaseEvent):
    """One funding-rate observation for a perpetual market.

    Hyperliquid settles funding hourly.  ``premium`` is the mark-to-index
    premium the rate is derived from, and is the more direct measure of
    positioning pressure of the two.
    """

    event_type: Literal[EventType.FUNDING_RATE] = EventType.FUNDING_RATE

    symbol: str = Field(description="Hyperliquid perp coin symbol.")
    funding_rate: Decimal = Field(description="Funding rate for the interval, as a fraction.")
    premium: Decimal | None = Field(
        default=None, description="Mark-to-index premium the rate derives from."
    )


class PerpMetrics(BaseEvent):
    """Point-in-time contract state for one perpetual market.

    Sourced from ``metaAndAssetCtxs``.  Open interest is the field Phase 5 set
    out to record; the rest arrives in the same payload at no extra cost and is
    worth keeping -- ``premium`` and ``oracle_price`` together give the live
    perp-versus-index basis, which is the input to any carry analysis.

    Hyperliquid publishes no history for any of this, so a row that is not
    recorded now is gone permanently.
    """

    event_type: Literal[EventType.PERP_METRICS] = EventType.PERP_METRICS

    symbol: str = Field(description="Hyperliquid perp coin symbol.")
    open_interest: Decimal = Field(ge=0, description="Open interest in base asset.")
    mark_price: Decimal = Field(gt=0, description="Mark price.")
    oracle_price: Decimal | None = Field(default=None, description="Oracle (index) price.")
    mid_price: Decimal | None = Field(default=None, description="Mid price.")
    funding_rate: Decimal | None = Field(default=None, description="Current hourly funding.")
    premium: Decimal | None = Field(default=None, description="Mark-to-index premium.")
    day_notional_volume: Decimal | None = Field(
        default=None, ge=0, description="Rolling 24h notional volume in USD."
    )
    day_base_volume: Decimal | None = Field(
        default=None, ge=0, description="Rolling 24h volume in base asset."
    )

    @property
    def open_interest_notional(self) -> Decimal:
        """Return open interest valued at the mark price."""
        return self.open_interest * self.mark_price

    @property
    def basis_bps(self) -> Decimal | None:
        """Return the perp premium over the index, in basis points."""
        if self.oracle_price is None or self.oracle_price <= 0:
            return None
        return (self.mark_price - self.oracle_price) / self.oracle_price * Decimal(10_000)


class BookLevel(BaseEvent):
    """One price level of the order book.

    Inherits the immutable envelope so a level can never be edited in place; it
    is only ever carried inside an :class:`OrderBookSnapshot`.
    """

    event_type: Literal[EventType.ORDER_BOOK] = EventType.ORDER_BOOK

    price: Decimal = Field(gt=0, description="Level price.")
    size: Decimal = Field(ge=0, description="Resting size at this level.")
    order_count: int = Field(default=0, ge=0, description="Resting orders at this level.")


class OrderBookSnapshot(BaseEvent):
    """Top-of-book snapshot with a bounded number of levels per side.

    Bids are ordered best (highest) first; asks best (lowest) first.
    """

    event_type: Literal[EventType.ORDER_BOOK] = EventType.ORDER_BOOK

    symbol: str = Field(description="Hyperliquid perp coin symbol.")
    bids: tuple[BookLevel, ...] = Field(default=(), description="Bid levels, best first.")
    asks: tuple[BookLevel, ...] = Field(default=(), description="Ask levels, best first.")
    sequence: int | None = Field(default=None, description="Exchange sequence number, if provided.")

    @property
    def best_bid(self) -> Decimal | None:
        """Return the best bid price, or ``None`` for an empty book side."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        """Return the best ask price, or ``None`` for an empty book side."""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Decimal | None:
        """Return the mid price, or ``None`` if either side is empty."""
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / Decimal(2)

    @property
    def spread_bps(self) -> Decimal | None:
        """Return the bid/ask spread in basis points, or ``None`` if unavailable."""
        bid, ask, mid = self.best_bid, self.best_ask, self.mid_price
        if bid is None or ask is None or mid is None or mid == 0:
            return None
        return (ask - bid) / mid * Decimal(10_000)

    def imbalance(self, depth: int = 5) -> float:
        """Return order-book imbalance in ``[-1, 1]`` over the top ``depth`` levels.

        Positive values mean resting bid size dominates (buy-side pressure).

        Args:
            depth: Number of levels per side to include.

        Returns:
            Normalised imbalance, or ``0.0`` when the book is empty.
        """
        bid_size = sum(float(level.size) for level in self.bids[:depth])
        ask_size = sum(float(level.size) for level in self.asks[:depth])
        total = bid_size + ask_size
        if total <= 0.0:
            return 0.0
        return (bid_size - ask_size) / total


__all__ = [
    "BookLevel",
    "Candle",
    "FundingRate",
    "OrderBookSnapshot",
    "PerpMetrics",
    "Side",
    "TradeTick",
]
