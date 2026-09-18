"""Risk Manager Agent.

Phase 2, item 5.  The mandatory gate between a decision and an order: nothing
reaches the execution layer without an approving ``RiskVerdict``.

The agent consumes ``TradeDecision`` events and the portfolio snapshots that
tell it what is currently at risk, then either

* **approves** -- publishing a ``RiskVerdict`` *and* the sized ``OrderIntent``
  the execution layer will fill, or
* **vetoes** -- publishing a ``RiskVerdict`` carrying every limit that was
  breached, and no order.

Sizing
------
Size is derived from risk, not from account size: the position is sized so that
being stopped out costs ``risk_per_trade_pct`` of equity.  The stop comes from
the decision when the analyst supplied one, otherwise from the configured
default stop distance.  The result is then clamped by the per-position notional
cap, by remaining leverage headroom, and by remaining equity.

Veto rules (all evaluated, so the verdict reports every breach at once):

* confidence below the floor;
* the daily loss limit has been hit (kill switch -- no new entries);
* the maximum number of open positions is already reached;
* the position would exceed the per-position notional cap after clamping to zero;
* portfolio leverage would exceed the ceiling;
* an opposing position is already open for the symbol (Phase 2 does not flip
  positions in one step -- the close must be decided explicitly);
* no usable reference price;
* the computed size is below the minimum order notional.

Run with::

    python -m services.agents.risk_manager.agent
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from decimal import ROUND_DOWN, Decimal
from typing import Final
from uuid import uuid4

from libs.config import RiskSettings, Settings
from libs.kafka_client import Topics
from libs.logging_config import configure_logging
from libs.schemas.base import BaseEvent
from libs.schemas.market import Side
from libs.schemas.portfolio import PortfolioSnapshot
from libs.schemas.signals import Direction
from libs.schemas.trading import (
    DecisionAction,
    OrderIntent,
    OrderType,
    RiskVerdict,
    TradeDecision,
    VetoReason,
)
from services.agents.common.base_agent import BaseAgent, Publication

logger = logging.getLogger(__name__)

SERVICE_NAME: Final[str] = "risk_manager"

_SIZE_QUANTUM: Final[Decimal] = Decimal("0.00000001")
"""Size rounding increment. Rounds down so a limit is never exceeded."""


class RiskManagerAgent(BaseAgent):
    """Sizes approved trades and vetoes everything that breaches a limit.

    Args:
        config: Settings override, mainly for tests.
    """

    name = SERVICE_NAME
    version = "1.0.0"

    def __init__(self, *, config: Settings | None = None) -> None:
        """Initialise the agent with a bootstrap portfolio view."""
        super().__init__(config=config)
        self.limits: RiskSettings = self.settings.risk
        self._portfolio: PortfolioSnapshot | None = None
        self.approvals = 0
        self.vetoes = 0

    @property
    def input_topics(self) -> Mapping[str, type[BaseEvent]]:
        """Consume decisions to gate, and portfolio snapshots for state."""
        return {
            Topics.TRADE_DECISIONS: TradeDecision,
            Topics.PORTFOLIO_SNAPSHOTS: PortfolioSnapshot,
        }

    async def handle(self, topic: str, event: BaseEvent) -> Sequence[Publication]:
        """Update portfolio state, or gate one decision.

        Args:
            topic: Topic the event arrived on.
            event: The decoded event.

        Returns:
            The verdict, plus an order intent when the decision is approved.
        """
        if isinstance(event, PortfolioSnapshot):
            self._portfolio = event
            return ()
        if not isinstance(event, TradeDecision):
            return ()

        verdict, order = self.evaluate(event)
        publications: list[Publication] = [
            (Topics.RISK_VERDICTS, verdict.with_correlation(event.correlation_id))
        ]
        if order is not None:
            publications.append(
                (Topics.ORDER_INTENTS, order.with_correlation(event.correlation_id))
            )

        if verdict.approved:
            self.approvals += 1
            logger.info(
                "Approved decision",
                extra={
                    "symbol": verdict.symbol,
                    "size": str(verdict.approved_size),
                    "notional": str(verdict.approved_notional),
                    "risk_amount": str(verdict.risk_amount),
                },
            )
        else:
            self.vetoes += 1
            logger.warning(
                "Vetoed decision",
                extra={
                    "symbol": verdict.symbol,
                    "reasons": [reason.value for reason in verdict.veto_reasons],
                },
            )
        return publications

    def evaluate(self, decision: TradeDecision) -> tuple[RiskVerdict, OrderIntent | None]:
        """Apply every risk rule to a decision.

        Args:
            decision: The decision to gate.

        Returns:
            A ``(verdict, order_intent)`` tuple; the order is ``None`` on veto.
        """
        if decision.action is DecisionAction.HOLD:
            return self._veto(decision, [VetoReason.LOW_CONFIDENCE], "Decision is a hold."), None

        reasons: list[VetoReason] = []
        limits = self.limits
        portfolio = self._portfolio

        equity = (
            portfolio.equity
            if portfolio is not None
            else Decimal(str(self.settings.paper.starting_equity))
        )
        if equity <= 0:
            return self._veto(decision, [VetoReason.INSUFFICIENT_EQUITY], "Equity is zero."), None

        price = decision.reference_price
        if price is None or price <= 0:
            return self._veto(decision, [VetoReason.NO_MARKET_DATA], "No reference price."), None

        if decision.weighted_confidence < limits.min_confidence:
            reasons.append(VetoReason.LOW_CONFIDENCE)

        if portfolio is not None and portfolio.day_pnl_pct <= -limits.daily_loss_limit_pct:
            reasons.append(VetoReason.DAILY_LOSS_LIMIT)

        direction = decision.action.direction
        existing = portfolio.position_for(decision.symbol) if portfolio is not None else None
        if existing is not None and existing.direction is not direction:
            reasons.append(VetoReason.DUPLICATE_EXPOSURE)

        open_positions = portfolio.open_position_count if portfolio is not None else 0
        if existing is None and open_positions >= limits.max_open_positions:
            reasons.append(VetoReason.MAX_OPEN_POSITIONS)

        stop_price = self._resolve_stop(decision, price, direction)
        stop_distance = abs(price - stop_price)
        if stop_distance <= 0:
            return (
                self._veto(decision, [VetoReason.NO_MARKET_DATA], "Stop distance is zero."),
                None,
            )

        risk_budget = equity * Decimal(str(limits.risk_per_trade_pct)) / Decimal(100)
        size = risk_budget / stop_distance

        notional_cap = Decimal(str(limits.max_position_notional))
        if size * price > notional_cap:
            size = notional_cap / price

        gross = portfolio.gross_notional if portfolio is not None else Decimal(0)
        leverage_headroom = equity * Decimal(str(limits.max_portfolio_leverage)) - gross
        if leverage_headroom <= 0:
            reasons.append(VetoReason.MAX_LEVERAGE)
        elif size * price > leverage_headroom:
            size = leverage_headroom / price

        size = size.quantize(_SIZE_QUANTUM, rounding=ROUND_DOWN)
        notional = (size * price).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

        if size <= 0 or notional < Decimal(str(limits.min_order_notional)):
            reasons.append(VetoReason.SIZE_BELOW_MINIMUM)

        if reasons:
            return self._veto(decision, reasons, self._veto_rationale(reasons), equity=equity), None

        verdict = RiskVerdict(
            source=self.name,
            decision_id=decision.decision_id,
            symbol=decision.symbol,
            approved=True,
            approved_size=size,
            approved_notional=notional,
            stop_price=stop_price,
            take_profit_price=decision.suggested_take_profit,
            risk_amount=(size * stop_distance).quantize(Decimal("0.01"), rounding=ROUND_DOWN),
            equity_at_decision=equity,
            rationale=(
                f"Approved {direction.value} {size} {decision.symbol} "
                f"(~${notional}) risking ${size * stop_distance:.2f} to a stop at {stop_price}."
            ),
        )
        order = OrderIntent(
            source=self.name,
            order_id=uuid4(),
            decision_id=decision.decision_id,
            symbol=decision.symbol,
            side=Side.BUY if direction is Direction.LONG else Side.SELL,
            size=size,
            order_type=OrderType.MARKET,
            stop_price=stop_price,
            take_profit_price=decision.suggested_take_profit,
            trailing_stop_pct=self.limits.trailing_stop_pct,
            is_paper=self.settings.is_paper,
        )
        return verdict, order

    def _resolve_stop(
        self, decision: TradeDecision, price: Decimal, direction: Direction
    ) -> Decimal:
        """Return the stop price to size against.

        Uses the analyst's stop when it sits on the correct side of the entry;
        otherwise falls back to the configured default distance.

        Args:
            decision: The decision being sized.
            price: Reference entry price.
            direction: Intended exposure.

        Returns:
            A stop price strictly on the losing side of the entry.
        """
        suggested = decision.suggested_stop
        if suggested is not None and suggested > 0:
            correct_side = (direction is Direction.LONG and suggested < price) or (
                direction is Direction.SHORT and suggested > price
            )
            distance_pct = abs(price - suggested) / price * Decimal(100)
            min_distance = Decimal(str(self.limits.min_stop_distance_pct))
            if correct_side and distance_pct >= min_distance:
                return suggested
            logger.warning(
                "Rejecting unusable stop; falling back to the default distance",
                extra={
                    "symbol": decision.symbol,
                    "direction": direction.value,
                    "suggested_stop": str(suggested),
                    "price": str(price),
                    "distance_pct": float(distance_pct),
                    "reason": "wrong_side" if not correct_side else "too_tight",
                },
            )
        offset = price * Decimal(str(self.limits.default_stop_distance_pct)) / Decimal(100)
        return price - offset if direction is Direction.LONG else price + offset

    def _veto(
        self,
        decision: TradeDecision,
        reasons: Sequence[VetoReason],
        rationale: str,
        *,
        equity: Decimal = Decimal(0),
    ) -> RiskVerdict:
        """Build a rejecting verdict.

        Args:
            decision: The rejected decision.
            reasons: Every limit that was breached.
            rationale: Human-readable explanation.
            equity: Equity used during evaluation, when known.

        Returns:
            The veto verdict.
        """
        return RiskVerdict(
            source=self.name,
            decision_id=decision.decision_id,
            symbol=decision.symbol,
            approved=False,
            veto_reasons=tuple(reasons),
            equity_at_decision=equity,
            rationale=rationale,
        )

    @staticmethod
    def _veto_rationale(reasons: Sequence[VetoReason]) -> str:
        """Render a veto reason list as one sentence."""
        return "Vetoed: " + ", ".join(reason.value.replace("_", " ") for reason in reasons) + "."

    async def on_stop(self) -> None:
        """Log an approval/veto summary on shutdown."""
        logger.info(
            "Risk manager summary",
            extra={"approvals": self.approvals, "vetoes": self.vetoes},
        )


async def main() -> None:
    """Service entrypoint for ``python -m services.agents.risk_manager.agent``."""
    configure_logging(SERVICE_NAME)
    await RiskManagerAgent.main()


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    asyncio.run(main())


__all__ = ["RiskManagerAgent"]
