"""Portfolio state events published by the portfolio manager."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import Field

from libs.schemas.base import BaseEvent, EventType
from libs.schemas.signals import Direction


class Position(BaseEvent):
    """An open perp position.

    ``size`` is always positive; ``direction`` carries the sign.  Realised PnL
    accumulates on the position until it is fully closed and removed.
    """

    event_type: Literal[EventType.PORTFOLIO_SNAPSHOT] = EventType.PORTFOLIO_SNAPSHOT

    symbol: str = Field(description="Hyperliquid perp coin symbol.")
    direction: Direction = Field(description="Long or short exposure.")
    size: Decimal = Field(gt=0, description="Absolute position size in base asset.")
    entry_price: Decimal = Field(gt=0, description="Volume-weighted average entry price.")
    mark_price: Decimal = Field(gt=0, description="Latest mark price used for valuation.")
    realized_pnl: Decimal = Field(default=Decimal(0), description="Realised PnL booked so far.")
    fees_paid: Decimal = Field(default=Decimal(0), ge=0, description="Cumulative fees in USD.")
    opened_at: datetime = Field(description="When the position was first opened.")
    updated_at: datetime = Field(description="When the position was last touched.")
    stop_price: Decimal | None = Field(default=None, description="Protective stop, if set.")
    take_profit_price: Decimal | None = Field(default=None, description="Take-profit, if set.")
    trailing_stop_pct: float | None = Field(
        default=None, description="Trailing stop distance in percent, if enabled."
    )
    opening_correlation_id: UUID | None = Field(
        default=None, description="Correlation id of the chain that opened the position."
    )
    opening_decision_id: UUID | None = Field(
        default=None, description="Decision that opened the position."
    )

    @property
    def notional(self) -> Decimal:
        """Return the position notional at the mark price."""
        return self.size * self.mark_price

    @property
    def unrealized_pnl(self) -> Decimal:
        """Return mark-to-market PnL for the open size."""
        return (self.mark_price - self.entry_price) * self.size * Decimal(self.direction.sign)

    @property
    def unrealized_pnl_pct(self) -> float:
        """Return unrealised PnL as a percentage of the entry notional."""
        cost = self.entry_price * self.size
        if cost == 0:
            return 0.0
        return float(self.unrealized_pnl / cost) * 100.0


class PortfolioSnapshot(BaseEvent):
    """Point-in-time view of the paper (or live) portfolio."""

    event_type: Literal[EventType.PORTFOLIO_SNAPSHOT] = EventType.PORTFOLIO_SNAPSHOT

    equity: Decimal = Field(description="Cash plus unrealised PnL.")
    cash: Decimal = Field(description="Realised cash balance after fees.")
    starting_equity: Decimal = Field(gt=0, description="Equity the account started with.")
    day_start_equity: Decimal = Field(gt=0, description="Equity at the start of the UTC day.")
    realized_pnl: Decimal = Field(default=Decimal(0), description="Lifetime realised PnL.")
    unrealized_pnl: Decimal = Field(default=Decimal(0), description="Current unrealised PnL.")
    fees_paid: Decimal = Field(default=Decimal(0), ge=0, description="Lifetime fees in USD.")
    gross_notional: Decimal = Field(
        default=Decimal(0), ge=0, description="Sum of absolute position notionals."
    )
    positions: tuple[Position, ...] = Field(default=(), description="Currently open positions.")
    trade_count: int = Field(default=0, ge=0, description="Number of fills processed.")
    win_count: int = Field(default=0, ge=0, description="Closed trades with positive PnL.")
    loss_count: int = Field(default=0, ge=0, description="Closed trades with negative PnL.")
    is_paper: bool = Field(default=True, description="Whether this is a simulated portfolio.")

    @property
    def open_position_count(self) -> int:
        """Return the number of open positions."""
        return len(self.positions)

    @property
    def leverage(self) -> float:
        """Return gross notional divided by equity."""
        if self.equity <= 0:
            return 0.0
        return float(self.gross_notional / self.equity)

    @property
    def day_pnl_pct(self) -> float:
        """Return today's PnL as a percentage of the day's starting equity."""
        if self.day_start_equity <= 0:
            return 0.0
        return float((self.equity - self.day_start_equity) / self.day_start_equity) * 100.0

    @property
    def total_return_pct(self) -> float:
        """Return lifetime return as a percentage of starting equity."""
        if self.starting_equity <= 0:
            return 0.0
        return float((self.equity - self.starting_equity) / self.starting_equity) * 100.0

    @property
    def win_rate(self) -> float:
        """Return the fraction of closed trades that were profitable."""
        closed = self.win_count + self.loss_count
        if closed == 0:
            return 0.0
        return self.win_count / closed

    def position_for(self, symbol: str) -> Position | None:
        """Return the open position for ``symbol``, if one exists."""
        for position in self.positions:
            if position.symbol == symbol:
                return position
        return None


__all__ = ["PortfolioSnapshot", "Position"]
