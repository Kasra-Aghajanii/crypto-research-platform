"""AgentContext and the buffer that pre-loads it.

Architecture rule (``CLAUDE.md``): **AgentContext is pre-loaded -- agents never
self-fetch.**  An agent's ``analyze`` method receives a fully populated,
read-only context and has no client for the exchange, the database or the cache.

In this Phase 2 vertical slice the orchestration role is played by
:class:`MarketContextBuilder`, which lives in the agent runtime (not in agent
code): it consumes the market data topics, keeps bounded rolling buffers, and
hands a finished context to the agent whenever a candle closes.  When the full
Orchestrator lands, only the builder is replaced -- the agent-facing contract
does not change.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from libs.schemas.base import utc_now
from libs.schemas.market import Candle, OrderBookSnapshot, TradeTick
from libs.schemas.portfolio import PortfolioSnapshot

logger = logging.getLogger(__name__)


class AgentContext(BaseModel):
    """Everything an agent is allowed to see when forming a view.

    Attributes:
        symbol: The symbol under analysis.
        as_of: Instant the context was assembled.
        candles: Closed candles per interval, oldest first.
        order_book: Most recent order book snapshot, if any.
        recent_trades: Most recent trade prints, oldest first.
        portfolio: Latest portfolio snapshot, if the agent is allowed to see it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=False)

    symbol: str = Field(description="Symbol under analysis.")
    as_of: datetime = Field(default_factory=utc_now, description="Context assembly time (UTC).")
    candles: Mapping[str, tuple[Candle, ...]] = Field(
        default_factory=dict, description="Closed candles per interval, oldest first."
    )
    order_book: OrderBookSnapshot | None = Field(
        default=None, description="Latest order book snapshot."
    )
    recent_trades: tuple[TradeTick, ...] = Field(
        default=(), description="Recent trade prints, oldest first."
    )
    portfolio: PortfolioSnapshot | None = Field(
        default=None, description="Latest portfolio snapshot."
    )
    metadata: Mapping[str, str] = Field(
        default_factory=dict, description="Free-form orchestration metadata."
    )

    @property
    def intervals(self) -> tuple[str, ...]:
        """Return the intervals present in the context."""
        return tuple(self.candles)

    def candles_for(self, interval: str) -> tuple[Candle, ...]:
        """Return closed candles for ``interval``, oldest first (empty if absent)."""
        return self.candles.get(interval, ())

    def latest_close(self, interval: str) -> Decimal | None:
        """Return the most recent close price on ``interval``, if available."""
        series = self.candles_for(interval)
        return series[-1].close if series else None

    @property
    def reference_price(self) -> Decimal | None:
        """Return the best available current price.

        Prefers the order book mid, falls back to the last trade, then to the
        most recent close on any interval.
        """
        if self.order_book is not None:
            mid = self.order_book.mid_price
            if mid is not None:
                return mid
        if self.recent_trades:
            return self.recent_trades[-1].price
        for series in self.candles.values():
            if series:
                return series[-1].close
        return None

    def has_warmup(self, interval: str, required: int) -> bool:
        """Return whether ``interval`` holds at least ``required`` closed candles."""
        return len(self.candles_for(interval)) >= required


class MarketContextBuilder:
    """Maintains bounded rolling market state and produces :class:`AgentContext`.

    The builder is deliberately *outside* agent code: it is the component that
    performs the fetching, so agents can stay pure functions of their context.

    Args:
        symbols: Symbols to track.
        intervals: Candle intervals to buffer.
        max_candles: Candles retained per symbol and interval.
        max_trades: Trade prints retained per symbol.
    """

    def __init__(
        self,
        *,
        symbols: Sequence[str],
        intervals: Sequence[str],
        max_candles: int,
        max_trades: int = 200,
    ) -> None:
        """Initialise empty buffers for every tracked symbol and interval."""
        self._intervals = tuple(intervals)
        self._max_candles = max_candles
        self._max_trades = max_trades
        self._candles: dict[str, dict[str, deque[Candle]]] = {
            symbol: {interval: deque(maxlen=max_candles) for interval in self._intervals}
            for symbol in symbols
        }
        self._books: dict[str, OrderBookSnapshot] = {}
        self._trades: dict[str, deque[TradeTick]] = {
            symbol: deque(maxlen=max_trades) for symbol in symbols
        }
        self._portfolio: PortfolioSnapshot | None = None

    @property
    def intervals(self) -> tuple[str, ...]:
        """Return the buffered intervals."""
        return self._intervals

    def _ensure_symbol(self, symbol: str) -> None:
        """Create buffers for a symbol seen for the first time."""
        if symbol not in self._candles:
            self._candles[symbol] = {
                interval: deque(maxlen=self._max_candles) for interval in self._intervals
            }
            self._trades[symbol] = deque(maxlen=self._max_trades)

    def add_candle(self, candle: Candle) -> bool:
        """Buffer a candle.

        Only closed candles are retained: an agent must never form a view from a
        candle that can still change.  A repeated ``open_time`` replaces the
        stored candle rather than appending a duplicate.

        Args:
            candle: The candle event.

        Returns:
            ``True`` if a closed candle was stored (i.e. analysis should run).
        """
        if not candle.is_closed:
            return False
        if candle.interval not in self._intervals:
            return False
        self._ensure_symbol(candle.symbol)
        buffer = self._candles[candle.symbol][candle.interval]
        if buffer and buffer[-1].open_time == candle.open_time:
            buffer[-1] = candle
            return True
        if buffer and buffer[-1].open_time > candle.open_time:
            logger.debug(
                "Dropping out-of-order candle",
                extra={"symbol": candle.symbol, "interval": candle.interval},
            )
            return False
        buffer.append(candle)
        return True

    def add_order_book(self, book: OrderBookSnapshot) -> None:
        """Store the latest order book snapshot for a symbol."""
        self._ensure_symbol(book.symbol)
        self._books[book.symbol] = book

    def add_trade(self, trade: TradeTick) -> None:
        """Append a trade print to the rolling tape buffer."""
        self._ensure_symbol(trade.symbol)
        self._trades[trade.symbol].append(trade)

    def set_portfolio(self, snapshot: PortfolioSnapshot) -> None:
        """Store the latest portfolio snapshot shared with agents."""
        self._portfolio = snapshot

    def build(self, symbol: str, *, include_portfolio: bool = False) -> AgentContext:
        """Assemble an immutable context for ``symbol``.

        Args:
            symbol: Symbol to build the context for.
            include_portfolio: Whether to expose portfolio state to the agent.

        Returns:
            A fully populated, read-only :class:`AgentContext`.
        """
        self._ensure_symbol(symbol)
        candles = {
            interval: tuple(buffer) for interval, buffer in self._candles[symbol].items() if buffer
        }
        return AgentContext(
            symbol=symbol,
            candles=candles,
            order_book=self._books.get(symbol),
            recent_trades=tuple(self._trades[symbol]),
            portfolio=self._portfolio if include_portfolio else None,
        )


__all__ = ["AgentContext", "MarketContextBuilder"]
