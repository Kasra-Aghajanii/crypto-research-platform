"""Replay engine -- runs the live agents over historical candles.

Phase 3, item 4 (part two).  The point of this engine is to answer a question
Phase 2 could not: do the indicator weights hold up on real market data, or were
they only ever tuned against synthetic series?

It is deliberately *not* a second implementation of the strategy: it drives the
real :class:`~services.agents.market_analyst.agent.MarketAnalystAgent`, over a
context window bounded exactly as the live
:class:`~services.agents.common.context.MarketContextBuilder` bounds it, so a
result here says something about the code that would actually trade.

It applies the decision engine's confidence floor but *not* its cooldown, which
is measured in wall-clock seconds and has no meaning in a replay.  Risk-based
position sizing is also skipped: every trade is one unit, so hit rate and
calibration are not distorted by how large a position the sizer would have
taken.

Fill model
----------
Entries fill at the close of the signalling candle plus a configured cost in
basis points (fee + slippage), which stands in for the order book the replay does
not have.  Exits are evaluated against each subsequent candle:

* the **stop is checked before the take-profit** on every bar. A single candle
  whose range spans both levels is ambiguous, and assuming the profitable one is
  how a backtest lies to you;
* the trailing stop ratchets off the candle extreme, matching
  :class:`~services.execution.position_monitor.ProtectedPosition`;
* an unresolved position at the end of the series is closed at the final close
  and reported separately, so open-trade luck cannot inflate the hit rate.

Look-ahead
----------
Only candles strictly *before* the decision point are ever passed to the analyst.
The engine walks the series forward and rebuilds the context from a slice, so an
indicator cannot see a bar that had not printed yet.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Final

from libs.config import Settings, settings
from libs.schemas.learning import ExitReason
from libs.schemas.market import Candle
from libs.schemas.signals import AgentSignal, Direction
from services.agents.common.context import AgentContext
from services.agents.market_analyst.agent import MarketAnalystAgent
from services.decision_engine.engine import DecisionEngine

logger = logging.getLogger(__name__)

_HUNDRED: Final[Decimal] = Decimal(100)
_BPS: Final[Decimal] = Decimal(10_000)


@dataclass(frozen=True, slots=True)
class ReplayTrade:
    """One completed round trip produced by the replay."""

    symbol: str
    direction: Direction
    confidence: float
    entry_time: datetime
    entry_price: Decimal
    exit_time: datetime
    exit_price: Decimal
    exit_reason: ExitReason
    size: Decimal
    gross_pnl: Decimal
    costs: Decimal
    bars_held: int
    forced_close: bool = False

    @property
    def net_pnl(self) -> Decimal:
        """Return PnL after entry and exit costs."""
        return self.gross_pnl - self.costs

    @property
    def was_correct(self) -> bool:
        """Return whether the directional call made money after costs."""
        return self.net_pnl > 0

    @property
    def return_pct(self) -> float:
        """Return net PnL as a percentage of the entry notional."""
        notional = self.entry_price * self.size
        if notional == 0:
            return 0.0
        return float(self.net_pnl / notional) * 100.0


@dataclass(slots=True)
class _OpenTrade:
    """Internal state for a position the replay is holding."""

    symbol: str
    direction: Direction
    confidence: float
    entry_time: datetime
    entry_price: Decimal
    size: Decimal
    stop_price: Decimal | None
    take_profit_price: Decimal | None
    trailing_stop_pct: float | None
    extreme_price: Decimal
    entry_cost: Decimal
    bars: int = 0

    def observe(self, candle: Candle) -> None:
        """Update the high/low-water mark from a candle."""
        if self.direction is Direction.LONG:
            self.extreme_price = max(self.extreme_price, candle.high)
        else:
            self.extreme_price = min(self.extreme_price, candle.low)

    @property
    def trailing_stop(self) -> Decimal | None:
        """Return the trailing stop derived from the extreme, if enabled."""
        if self.trailing_stop_pct is None:
            return None
        distance = Decimal(str(self.trailing_stop_pct)) / _HUNDRED
        if self.direction is Direction.LONG:
            return self.extreme_price * (Decimal(1) - distance)
        return self.extreme_price * (Decimal(1) + distance)

    @property
    def effective_stop(self) -> Decimal | None:
        """Return the tighter of the fixed and trailing stops."""
        fixed, trailing = self.stop_price, self.trailing_stop
        if fixed is None:
            return trailing
        if trailing is None:
            return fixed
        return max(fixed, trailing) if self.direction is Direction.LONG else min(fixed, trailing)


@dataclass(slots=True)
class ReplayResult:
    """Everything the replay produced, ready to be reported on."""

    symbol: str
    interval: str
    trades: list[ReplayTrade] = field(default_factory=list)
    signals_evaluated: int = 0
    signals_actionable: int = 0
    bars_replayed: int = 0
    first_bar: datetime | None = None
    last_bar: datetime | None = None
    confidences: list[float] = field(default_factory=list)

    @property
    def closed_trades(self) -> list[ReplayTrade]:
        """Return trades that resolved on their own, excluding forced closes."""
        return [trade for trade in self.trades if not trade.forced_close]

    @property
    def hit_rate(self) -> float:
        """Return the fraction of trades that made money after costs."""
        if not self.trades:
            return 0.0
        return sum(1 for trade in self.trades if trade.was_correct) / len(self.trades)

    @property
    def net_pnl(self) -> Decimal:
        """Return total net PnL across every trade."""
        return sum((trade.net_pnl for trade in self.trades), Decimal(0))

    @property
    def gross_pnl(self) -> Decimal:
        """Return total PnL before trading costs."""
        return sum((trade.gross_pnl for trade in self.trades), Decimal(0))

    @property
    def gross_hit_rate(self) -> float:
        """Return the fraction of trades profitable *before* costs.

        Comparing this with :attr:`hit_rate` separates two very different
        failures: no directional edge at all, versus an edge too small to pay
        for the spread and fees.
        """
        if not self.trades:
            return 0.0
        return sum(1 for trade in self.trades if trade.gross_pnl > 0) / len(self.trades)

    @property
    def total_costs(self) -> Decimal:
        """Return total modelled trading costs."""
        return sum((trade.costs for trade in self.trades), Decimal(0))

    @property
    def profit_factor(self) -> float:
        """Return gross profit divided by gross loss.

        Returns:
            The ratio, or ``inf`` when there were no losing trades.
        """
        wins = sum((t.net_pnl for t in self.trades if t.net_pnl > 0), Decimal(0))
        losses = -sum((t.net_pnl for t in self.trades if t.net_pnl < 0), Decimal(0))
        if losses == 0:
            return float("inf") if wins > 0 else 0.0
        return float(wins / losses)

    @property
    def average_bars_held(self) -> float:
        """Return the mean holding period in bars."""
        if not self.trades:
            return 0.0
        return sum(trade.bars_held for trade in self.trades) / len(self.trades)

    def exit_breakdown(self) -> dict[str, int]:
        """Return a count of trades per exit reason."""
        counts: dict[str, int] = {}
        for trade in self.trades:
            counts[trade.exit_reason.value] = counts.get(trade.exit_reason.value, 0) + 1
        return dict(sorted(counts.items()))


class ReplayEngine:
    """Walks historical candles through the live analysis and risk components.

    Args:
        config: Settings override, mainly for tests.
        cost_bps: Round-trip cost per side in basis points (fee plus slippage).
        warmup_bars: Bars fed to the analyst before the first decision point.
        position_size: Notional-neutral unit size used for every trade, so hit
            rate and PnL are not distorted by position sizing.
        window_bars: Bars visible to the analyst at each decision point,
            matching the live context buffer.
        max_bars_held: Force-close a trade that has not resolved within this many
            bars. ``None`` lets trades run to the end of the series.
    """

    def __init__(
        self,
        *,
        config: Settings | None = None,
        cost_bps: float | None = None,
        warmup_bars: int | None = None,
        position_size: Decimal = Decimal(1),
        max_bars_held: int | None = None,
    ) -> None:
        """Build the engine and the live components it drives."""
        self.settings = config or settings
        self.analyst = MarketAnalystAgent(config=self.settings)
        self.engine = DecisionEngine(config=self.settings)
        paper = self.settings.paper
        self.cost_bps = (
            cost_bps if cost_bps is not None else paper.taker_fee_bps + paper.slippage_bps
        )
        self.warmup_bars = warmup_bars or self.settings.analyst.warmup_candles
        self.window_bars = self.settings.analyst.warmup_candles
        self.position_size = position_size
        self.max_bars_held = max_bars_held

    def _cost_for(self, price: Decimal) -> Decimal:
        """Return the modelled cost of transacting ``position_size`` at ``price``."""
        return price * self.position_size * Decimal(str(self.cost_bps)) / _BPS

    async def _signal_at(
        self, symbol: str, history: Mapping[str, tuple[Candle, ...]]
    ) -> AgentSignal | None:
        """Ask the analyst for a view given only the bars available so far."""
        context = AgentContext(symbol=symbol, candles=dict(history))
        return await self.analyst.analyze(context)

    @staticmethod
    def _exit_on(candle: Candle, trade: _OpenTrade) -> tuple[ExitReason, Decimal] | None:
        """Return the exit this candle triggers, if any.

        The stop is evaluated first: when one bar's range covers both the stop
        and the take-profit, the losing outcome is assumed.

        Args:
            candle: The bar being evaluated.
            trade: The open trade.

        Returns:
            A ``(reason, exit_price)`` pair, or ``None``.
        """
        stop = trade.effective_stop
        if stop is not None:
            hit = candle.low <= stop if trade.direction is Direction.LONG else candle.high >= stop
            if hit:
                trailing = trade.trailing_stop
                is_trailing = trailing is not None and (
                    trade.stop_price is None
                    or (
                        trailing > trade.stop_price
                        if trade.direction is Direction.LONG
                        else trailing < trade.stop_price
                    )
                )
                return (ExitReason.TRAILING_STOP if is_trailing else ExitReason.STOP_LOSS, stop)

        target = trade.take_profit_price
        if target is not None:
            hit = (
                candle.high >= target if trade.direction is Direction.LONG else candle.low <= target
            )
            if hit:
                return ExitReason.TAKE_PROFIT, target
        return None

    def _close(
        self,
        trade: _OpenTrade,
        *,
        exit_price: Decimal,
        exit_time: datetime,
        reason: ExitReason,
        forced: bool = False,
    ) -> ReplayTrade:
        """Convert an open trade into a completed :class:`ReplayTrade`."""
        sign = Decimal(trade.direction.sign)
        gross = (exit_price - trade.entry_price) * trade.size * sign
        costs = trade.entry_cost + self._cost_for(exit_price)
        return ReplayTrade(
            symbol=trade.symbol,
            direction=trade.direction,
            confidence=trade.confidence,
            entry_time=trade.entry_time,
            entry_price=trade.entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=reason,
            size=trade.size,
            gross_pnl=gross,
            costs=costs,
            bars_held=trade.bars,
            forced_close=forced,
        )

    async def run(
        self,
        symbol: str,
        candles_by_interval: Mapping[str, Sequence[Candle]],
        *,
        driving_interval: str,
    ) -> ReplayResult:
        """Replay one symbol over its history.

        Args:
            symbol: Symbol being replayed.
            candles_by_interval: Candle series per interval, oldest first.
            driving_interval: The interval whose closes drive decisions.

        Returns:
            The replay result.

        Raises:
            ValueError: If the driving interval is missing or too short.
        """
        driver = tuple(candles_by_interval.get(driving_interval, ()))
        if len(driver) <= self.warmup_bars:
            raise ValueError(
                f"Need more than {self.warmup_bars} candles on {driving_interval}; "
                f"got {len(driver)}."
            )

        others = {
            interval: tuple(series)
            for interval, series in candles_by_interval.items()
            if interval != driving_interval
        }

        result = ReplayResult(symbol=symbol, interval=driving_interval)
        result.first_bar = driver[self.warmup_bars].open_time
        result.last_bar = driver[-1].open_time

        open_trade: _OpenTrade | None = None

        for index in range(self.warmup_bars, len(driver)):
            bar = driver[index]
            result.bars_replayed += 1

            # ---- manage an open trade against this bar first ----------
            if open_trade is not None:
                open_trade.bars += 1
                open_trade.observe(bar)
                exit_hit = self._exit_on(bar, open_trade)
                if exit_hit is not None:
                    reason, price = exit_hit
                    result.trades.append(
                        self._close(
                            open_trade, exit_price=price, exit_time=bar.close_time, reason=reason
                        )
                    )
                    open_trade = None
                elif self.max_bars_held is not None and open_trade.bars >= self.max_bars_held:
                    result.trades.append(
                        self._close(
                            open_trade,
                            exit_price=bar.close,
                            exit_time=bar.close_time,
                            reason=ExitReason.MANUAL,
                            forced=True,
                        )
                    )
                    open_trade = None

            if open_trade is not None:
                continue

            # ---- ask for a view using only bars up to and including now ----
            # The window is bounded to `warmup_bars`, matching the live
            # MarketContextBuilder's buffer exactly. Passing the whole series
            # instead would both diverge from production and make the replay
            # quadratic in the length of the history.
            lower = max(0, index + 1 - self.window_bars)
            history: dict[str, tuple[Candle, ...]] = {
                driving_interval: driver[lower : index + 1]
            }
            for interval, series in others.items():
                visible = tuple(c for c in series if c.close_time <= bar.close_time)
                if visible:
                    history[interval] = visible[-self.window_bars :]

            signal = await self._signal_at(symbol, history)
            if signal is None:
                continue

            result.signals_evaluated += 1
            result.confidences.append(signal.confidence)

            if (
                signal.direction is Direction.FLAT
                or signal.confidence < self.settings.decision.min_confidence
            ):
                continue

            result.signals_actionable += 1
            entry_price = bar.close
            open_trade = _OpenTrade(
                symbol=symbol,
                direction=signal.direction,
                confidence=signal.confidence,
                entry_time=bar.close_time,
                entry_price=entry_price,
                size=self.position_size,
                stop_price=signal.suggested_stop,
                take_profit_price=signal.suggested_take_profit,
                trailing_stop_pct=self.settings.risk.trailing_stop_pct,
                extreme_price=entry_price,
                entry_cost=self._cost_for(entry_price),
            )

        if open_trade is not None:
            final = driver[-1]
            result.trades.append(
                self._close(
                    open_trade,
                    exit_price=final.close,
                    exit_time=final.close_time,
                    reason=ExitReason.MANUAL,
                    forced=True,
                )
            )

        logger.info(
            "Replay complete",
            extra={
                "symbol": symbol,
                "interval": driving_interval,
                "bars": result.bars_replayed,
                "trades": len(result.trades),
                "hit_rate": round(result.hit_rate, 4),
            },
        )
        return result


__all__ = ["ReplayEngine", "ReplayResult", "ReplayTrade"]
