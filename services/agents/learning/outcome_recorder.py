"""Outcome Recorder -- attributes realised PnL back to the signals that caused it.

Phase 3, item 3.  This is the service that closes the learning loop.  Until it
runs, ``TrustRegistry`` has no outcomes to learn from and every agent keeps the
default weight forever.

The join
-------
Attribution hangs on ``correlation_id``.  One id is stamped on the ``AgentSignal``
and carried unchanged through decision, verdict, order and fill, and the
portfolio manager copies it onto the position as ``opening_correlation_id``.
When the position closes, ``PositionClosed`` carries it back -- so the recorder
can look up exactly which signals were responsible, even weeks later.

The recorder keeps a bounded in-memory index of recent signals and decisions and
falls back to the database for anything older, so a trade held longer than the
in-memory window is still attributed.

Scoring
-------
For each contributing signal:

* ``was_correct`` -- did the trade make money after fees? A directional call
  that produced a loss is wrong regardless of how the exit was reached.
* ``brier = (confidence - outcome)^2`` -- squared error of the forecast, where
  outcome is 1.0 for correct and 0.0 for incorrect. Lower is better; 0.25 is
  the score of a permanent coin-flip.

Neutral/degraded signals (``confidence == 0``) express no view.  Scoring them
would feed a stream of near-free 0.0s into the mean and flatter a broken agent,
so they are skipped unless ``LEARNING_ATTRIBUTE_DEGRADED_SIGNALS`` is on.

Output goes two ways: a row in ``agent_performance`` (the durable ledger) and an
``AgentOutcome`` event on ``agents.outcomes``, which the decision engine consumes
to adapt trust weights while running.

Run with::

    python -m services.agents.learning.outcome_recorder
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Final
from uuid import UUID

from libs.config import Settings, settings
from libs.kafka_client import Topics
from libs.logging_config import configure_logging
from libs.persistence import PerformanceRepository, SignalRepository, try_connect
from libs.persistence.database import PostgresDatabase
from libs.schemas.base import BaseEvent
from libs.schemas.learning import AgentOutcome, PositionClosed
from libs.schemas.signals import AgentSignal
from libs.schemas.trading import TradeDecision
from services.agents.common.base_agent import BaseAgent, Publication
from services.decision_engine.trust import brier_score

logger = logging.getLogger(__name__)

SERVICE_NAME: Final[str] = "outcome_recorder"

_INDEX_LIMIT: Final[int] = 5_000
"""Recent signals/decisions held in memory before falling back to the database."""


class OutcomeRecorder(BaseAgent):
    """Scores agent forecasts against the trades they produced.

    Args:
        config: Settings override, mainly for tests.
        signal_repository: Signal repository override.
        performance_repository: Performance repository override.
    """

    name = SERVICE_NAME
    version = "1.0.0"

    def __init__(
        self,
        *,
        config: Settings | None = None,
        signal_repository: SignalRepository | None = None,
        performance_repository: PerformanceRepository | None = None,
    ) -> None:
        """Initialise the recorder with empty indexes."""
        super().__init__(config=config)
        self.params = (config or settings).learning
        self._signals: OrderedDict[UUID, AgentSignal] = OrderedDict()
        self._signals_by_correlation: OrderedDict[UUID, list[UUID]] = OrderedDict()
        self._decisions: OrderedDict[UUID, TradeDecision] = OrderedDict()
        self._signal_repo = signal_repository
        self._performance_repo = performance_repository
        self._database: PostgresDatabase | None = None
        self.outcomes_recorded = 0
        self.closures_seen = 0
        self.unattributed = 0

    @property
    def input_topics(self) -> Mapping[str, type[BaseEvent]]:
        """Consume signals and decisions to index, and closures to score."""
        return {
            Topics.AGENT_SIGNALS: AgentSignal,
            Topics.TRADE_DECISIONS: TradeDecision,
            Topics.POSITION_CLOSURES: PositionClosed,
        }

    async def on_start(self) -> None:
        """Connect persistence, if enabled."""
        if self._signal_repo is None and self.settings.persistence_enabled:
            self._database = await try_connect()
            if self._database is not None:
                self._signal_repo = SignalRepository(self._database)
                self._performance_repo = PerformanceRepository(self._database)
        if self._signal_repo is None:
            logger.warning(
                "Outcome recorder running without persistence; attribution is limited to "
                "signals still held in memory"
            )

    async def on_stop(self) -> None:
        """Close the database and log a summary."""
        if self._database is not None:
            await self._database.close()
            self._database = None
        logger.info(
            "Outcome recorder summary",
            extra={
                "closures_seen": self.closures_seen,
                "outcomes_recorded": self.outcomes_recorded,
                "unattributed_closures": self.unattributed,
            },
        )

    async def handle(self, topic: str, event: BaseEvent) -> Sequence[Publication]:
        """Index signals and decisions, and score closures.

        Args:
            topic: Topic the event arrived on.
            event: The decoded event.

        Returns:
            One ``AgentOutcome`` per contributing signal of a closed position.
        """
        if isinstance(event, AgentSignal):
            await self._index_signal(event)
            return ()
        if isinstance(event, TradeDecision):
            await self._index_decision(event)
            return ()
        if isinstance(event, PositionClosed):
            return await self.attribute(event)
        return ()

    async def _index_signal(self, signal: AgentSignal) -> None:
        """Remember a signal and persist it for later attribution."""
        self._signals[signal.event_id] = signal
        self._signals_by_correlation.setdefault(signal.correlation_id, []).append(signal.event_id)
        self._trim(self._signals)
        self._trim(self._signals_by_correlation)
        if self._signal_repo is not None:
            await self._signal_repo.record_signal(signal)

    async def _index_decision(self, decision: TradeDecision) -> None:
        """Remember a decision and persist it for later attribution."""
        self._decisions[decision.correlation_id] = decision
        self._trim(self._decisions)
        if self._signal_repo is not None:
            await self._signal_repo.record_decision(decision)

    @staticmethod
    def _trim[V](index: OrderedDict[UUID, V]) -> None:
        """Evict the oldest entries once an index exceeds its bound."""
        while len(index) > _INDEX_LIMIT:
            index.popitem(last=False)

    async def _signals_for(self, closure: PositionClosed) -> tuple[AgentSignal, ...]:
        """Find the signals responsible for a closed position.

        Tries the decision's explicit ``contributing_signals`` first, then the
        in-memory correlation index, then the database.

        Args:
            closure: The closed position.

        Returns:
            The contributing signals, empty when none can be found.
        """
        correlation_id = closure.opening_correlation_id
        if correlation_id is None:
            return ()

        decision = self._decisions.get(correlation_id)
        if decision is not None and decision.contributing_signals:
            found = [
                self._signals[signal_id]
                for signal_id in decision.contributing_signals
                if signal_id in self._signals
            ]
            if found:
                return tuple(found)

        signal_ids = self._signals_by_correlation.get(correlation_id)
        if signal_ids:
            found = [self._signals[sid] for sid in signal_ids if sid in self._signals]
            if found:
                return tuple(found)

        if self._signal_repo is not None:
            return await self._signal_repo.signals_for_correlation(correlation_id)
        return ()

    async def attribute(self, closure: PositionClosed) -> Sequence[Publication]:
        """Score every signal responsible for one closed position.

        Args:
            closure: The closed position carrying the realised outcome.

        Returns:
            One ``(topic, AgentOutcome)`` publication per scored signal.
        """
        self.closures_seen += 1
        signals = await self._signals_for(closure)
        if not signals:
            self.unattributed += 1
            logger.warning(
                "Closure could not be attributed to any signal",
                extra={
                    "symbol": closure.symbol,
                    "correlation_id": str(closure.opening_correlation_id),
                    "net_pnl": str(closure.net_pnl),
                },
            )
            return ()

        was_correct = closure.was_profitable
        publications: list[Publication] = []

        for signal in signals:
            if signal.degraded and not self.params.attribute_degraded_signals:
                continue
            if signal.confidence <= 0.0 and not self.params.attribute_degraded_signals:
                continue

            outcome = AgentOutcome(
                source=self.name,
                correlation_id=closure.correlation_id,
                agent_name=signal.agent_name,
                agent_version=signal.agent_version,
                signal_id=signal.event_id,
                symbol=closure.symbol,
                direction=signal.direction,
                confidence=signal.confidence,
                realized_pnl=closure.net_pnl,
                was_correct=was_correct,
                brier_score=round(brier_score(signal.confidence, was_correct), 6),
                exit_reason=closure.exit_reason,
                closed_at=closure.closed_at,
                holding_period_s=closure.holding_period_s,
            )
            if self._performance_repo is not None:
                await self._performance_repo.record_outcome(
                    outcome, closure.opening_correlation_id
                )
            self.outcomes_recorded += 1
            publications.append((Topics.AGENT_OUTCOMES, outcome))

            logger.info(
                "Scored agent forecast",
                extra={
                    "agent": outcome.agent_name,
                    "symbol": outcome.symbol,
                    "confidence": outcome.confidence,
                    "was_correct": outcome.was_correct,
                    "brier": outcome.brier_score,
                    "net_pnl": str(outcome.realized_pnl),
                    "exit_reason": outcome.exit_reason.value,
                },
            )
        return publications


async def main() -> None:
    """Service entrypoint for ``python -m services.agents.learning.outcome_recorder``."""
    configure_logging(SERVICE_NAME)
    await OutcomeRecorder.main()


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    asyncio.run(main())


__all__ = ["OutcomeRecorder"]
