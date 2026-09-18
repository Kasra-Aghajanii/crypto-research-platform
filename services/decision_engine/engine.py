"""Decision Engine -- single-agent passthrough mode.

Phase 2, item 4.  The engine sits between the analysis agents and the risk
manager: it consumes ``AgentSignal`` events and emits ``TradeDecision`` events.

Passthrough mode
----------------
Only signals from the configured passthrough agent (the Market Analyst) produce
decisions, and a decision is a direct translation of one signal.  No ensemble
maths runs yet -- but the decision still carries ``weighted_confidence``, the
contributing signal ids and the per-agent trust weights, so the ensemble mode
that replaces it in a later phase changes only :meth:`DecisionEngine._decide`
and nothing downstream.

Guards applied before a decision is emitted:

* signals from other agents are ignored (recorded, not acted on);
* expired signals are dropped -- a stale view must not trade;
* neutral/degraded signals (``confidence == 0``) never open a position;
* confidence below the configured floor produces no decision;
* a per-symbol cooldown prevents the same view firing repeatedly on every close;
* a repeat of the direction already decided is suppressed until it flips or the
  cooldown lapses.

Run with::

    python -m services.decision_engine.engine
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final
from uuid import uuid4

from libs.config import Settings
from libs.kafka_client import Topics
from libs.logging_config import configure_logging
from libs.schemas.base import BaseEvent, utc_now
from libs.schemas.learning import AgentOutcome
from libs.schemas.signals import AgentSignal, Direction
from libs.schemas.trading import DecisionAction, TradeDecision
from services.agents.common.base_agent import BaseAgent, Publication
from services.decision_engine.trust import TrustRegistry

logger = logging.getLogger(__name__)

SERVICE_NAME: Final[str] = "decision_engine"


@dataclass(slots=True)
class SymbolState:
    """Per-symbol decision history used for cooldown and repeat suppression.

    Attributes:
        last_direction: Direction of the most recent decision.
        last_decision_at: Monotonic timestamp of the most recent decision.
    """

    last_direction: Direction = Direction.FLAT
    last_decision_at: float = 0.0


class DecisionEngine(BaseAgent):
    """Turns agent signals into trade decisions.

    Args:
        config: Settings override, mainly for tests.
        trust: Trust registry override, mainly for tests.
    """

    name = SERVICE_NAME
    version = "1.0.0"

    def __init__(
        self, *, config: Settings | None = None, trust: TrustRegistry | None = None
    ) -> None:
        """Initialise the engine and its trust registry."""
        super().__init__(config=config)
        self.params = self.settings.decision
        learning = self.settings.learning
        self.trust = trust or TrustRegistry(
            default_weight=self.params.default_trust_weight,
            min_samples=learning.min_samples_for_weight,
            min_weight=learning.min_weight,
            max_weight=learning.max_weight,
            window=learning.score_window,
        )
        self._state: dict[str, SymbolState] = {}
        self.decisions_emitted = 0
        self.signals_seen = 0
        self.outcomes_seen = 0

    @property
    def input_topics(self) -> Mapping[str, type[BaseEvent]]:
        """Consume agent signals, and the scored outcomes that adapt trust."""
        return {
            Topics.AGENT_SIGNALS: AgentSignal,
            Topics.AGENT_OUTCOMES: AgentOutcome,
        }

    async def handle(self, topic: str, event: BaseEvent) -> Sequence[Publication]:
        """Translate one signal into zero or one decision.

        Args:
            topic: Topic the event arrived on.
            event: The inbound event, expected to be an ``AgentSignal``.

        Returns:
            A single ``(topic, TradeDecision)`` pair, or nothing.
        """
        if isinstance(event, AgentOutcome):
            self._record_outcome(event)
            return ()
        if not isinstance(event, AgentSignal):
            return ()
        self.signals_seen += 1

        decision = self._decide(event)
        if decision is None:
            return ()

        self.decisions_emitted += 1
        state = self._state.setdefault(event.symbol, SymbolState())
        state.last_direction = decision.action.direction
        state.last_decision_at = time.monotonic()

        logger.info(
            "Emitted decision",
            extra={
                "symbol": decision.symbol,
                "action": decision.action.value,
                "confidence": decision.confidence,
                "weighted_confidence": decision.weighted_confidence,
                "agent": event.agent_name,
            },
        )
        return ((Topics.TRADE_DECISIONS, decision.with_correlation(event.correlation_id)),)

    def _record_outcome(self, outcome: AgentOutcome) -> None:
        """Feed a scored outcome into the trust registry.

        This is the live half of the adaptive-weight loop: the outcome recorder
        scores a closed trade, and the next decision from that agent is weighted
        by its updated track record.

        Args:
            outcome: The scored forecast.
        """
        self.outcomes_seen += 1
        self.trust.record_outcome(
            outcome.agent_name,
            confidence=outcome.confidence,
            was_correct=outcome.was_correct,
        )
        logger.info(
            "Updated agent trust",
            extra={
                "agent": outcome.agent_name,
                "brier": outcome.brier_score,
                "weight": round(self.trust.weight_for(outcome.agent_name), 4),
            },
        )

    def _decide(self, signal: AgentSignal) -> TradeDecision | None:
        """Apply the passthrough rules to one signal.

        Args:
            signal: The inbound agent signal.

        Returns:
            The decision to publish, or ``None`` if the signal is not actionable.
        """
        if signal.agent_name != self.params.passthrough_agent:
            logger.debug(
                "Ignoring signal from non-passthrough agent",
                extra={"agent": signal.agent_name, "mode": "passthrough"},
            )
            return None

        if signal.degraded or signal.confidence <= 0.0:
            logger.debug(
                "Neutral or degraded signal; no decision",
                extra={"symbol": signal.symbol, "error": signal.error},
            )
            return None

        if signal.is_expired(utc_now()):
            logger.warning(
                "Dropping expired signal",
                extra={"symbol": signal.symbol, "valid_until": str(signal.valid_until)},
            )
            return None

        if signal.direction is Direction.FLAT:
            return None

        if signal.confidence < self.params.min_confidence:
            logger.debug(
                "Confidence below floor",
                extra={
                    "symbol": signal.symbol,
                    "confidence": signal.confidence,
                    "floor": self.params.min_confidence,
                },
            )
            return None

        state = self._state.setdefault(signal.symbol, SymbolState())
        elapsed = time.monotonic() - state.last_decision_at
        within_cooldown = state.last_decision_at > 0.0 and elapsed < self.params.cooldown_s
        if within_cooldown and state.last_direction is signal.direction:
            logger.debug(
                "Suppressing repeat decision within cooldown",
                extra={"symbol": signal.symbol, "elapsed_s": round(elapsed, 2)},
            )
            return None

        weight = self.trust.weight_for(signal.agent_name)
        weighted_confidence = min(1.0, signal.confidence * weight)

        return TradeDecision(
            source=self.name,
            decision_id=uuid4(),
            symbol=signal.symbol,
            action=DecisionAction.from_direction(signal.direction),
            confidence=signal.confidence,
            weighted_confidence=round(weighted_confidence, 4),
            reference_price=signal.reference_price,
            suggested_stop=signal.suggested_stop,
            suggested_take_profit=signal.suggested_take_profit,
            rationale=signal.rationale,
            contributing_signals=(signal.event_id,),
            agent_weights={signal.agent_name: round(weight, 4)},
            mode="passthrough",
        )

    async def on_stop(self) -> None:
        """Log a summary of the session on shutdown."""
        logger.info(
            "Decision engine summary",
            extra={
                "signals_seen": self.signals_seen,
                "outcomes_seen": self.outcomes_seen,
                "decisions_emitted": self.decisions_emitted,
                "trust": dict(self.trust.stats()),
            },
        )


async def main() -> None:
    """Service entrypoint for ``python -m services.decision_engine.engine``."""
    configure_logging(SERVICE_NAME)
    await DecisionEngine.main()


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    asyncio.run(main())


__all__ = ["DecisionEngine", "SymbolState"]
