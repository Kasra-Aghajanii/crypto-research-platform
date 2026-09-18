"""Brier-score based agent trust weighting.

Architecture rule (``CLAUDE.md``): **Brier score drives adaptive agent trust
weights.**  This module holds the scoring maths and the in-memory registry that
the decision engine consults.

The Brier score for a probabilistic forecast is ``(p - o)^2`` where ``p`` is the
forecast probability and ``o`` is the realised outcome (1 or 0).  Lower is
better: 0.0 is perfect, 0.25 is the score of a permanent 50/50 guess, and 1.0 is
maximally wrong.  A trust weight is derived by comparing an agent's mean Brier
score against that 0.25 baseline, so an agent that is merely guessing converges
to a weight of 1.0 and a consistently sharp agent earns more influence.

Phase 2 runs in passthrough mode with one agent, so every weight is the
configured default until outcomes start being recorded.  Wiring outcomes in --
the job of the Phase 3 attribution service -- requires no change here beyond
calling :meth:`TrustRegistry.record_outcome`.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def brier_score(forecast: float, outcome: bool) -> float:
    """Return the Brier score of a single probabilistic forecast.

    Args:
        forecast: Forecast probability in ``[0, 1]``.
        outcome: Whether the forecast event actually happened.

    Returns:
        The squared error in ``[0, 1]``; lower is better.

    Raises:
        ValueError: If ``forecast`` is outside ``[0, 1]``.
    """
    if not 0.0 <= forecast <= 1.0:
        raise ValueError(f"Forecast must be in [0, 1], got {forecast}.")
    return (forecast - (1.0 if outcome else 0.0)) ** 2


@dataclass(slots=True)
class AgentTrust:
    """Rolling Brier statistics for one agent.

    Attributes:
        agent_name: The agent these statistics describe.
        window: Most recent Brier scores, bounded by the registry window.
    """

    agent_name: str
    window: deque[float] = field(default_factory=lambda: deque(maxlen=200))

    @property
    def sample_count(self) -> int:
        """Return how many scored outcomes are in the window."""
        return len(self.window)

    @property
    def mean_brier(self) -> float | None:
        """Return the mean Brier score, or ``None`` before any outcome."""
        if not self.window:
            return None
        return sum(self.window) / len(self.window)


class TrustRegistry:
    """Tracks agent forecast accuracy and converts it into trust weights.

    Args:
        default_weight: Weight used before an agent has enough scored outcomes.
        min_samples: Scored outcomes required before a weight adapts.
        min_weight: Lower bound on an adapted weight.
        max_weight: Upper bound on an adapted weight.
        window: Number of recent outcomes retained per agent.
    """

    def __init__(
        self,
        *,
        default_weight: float = 1.0,
        min_samples: int = 30,
        min_weight: float = 0.25,
        max_weight: float = 2.0,
        window: int = 200,
    ) -> None:
        """Initialise an empty registry."""
        self._default_weight = default_weight
        self._min_samples = min_samples
        self._min_weight = min_weight
        self._max_weight = max_weight
        self._window = window
        self._agents: dict[str, AgentTrust] = defaultdict(lambda: AgentTrust(agent_name="unknown"))
        self._agents.clear()

    def record_outcome(self, agent_name: str, *, confidence: float, was_correct: bool) -> float:
        """Record one resolved forecast and return its Brier score.

        Args:
            agent_name: Agent that produced the forecast.
            confidence: The confidence it published, in ``[0, 1]``.
            was_correct: Whether the directional call turned out right.

        Returns:
            The Brier score for this single forecast.
        """
        score = brier_score(confidence, was_correct)
        trust = self._agents.get(agent_name)
        if trust is None:
            trust = AgentTrust(agent_name=agent_name, window=deque(maxlen=self._window))
            self._agents[agent_name] = trust
        trust.window.append(score)
        logger.debug(
            "Recorded forecast outcome",
            extra={
                "agent": agent_name,
                "brier": round(score, 4),
                "mean_brier": trust.mean_brier,
                "samples": trust.sample_count,
            },
        )
        return score

    def weight_for(self, agent_name: str) -> float:
        """Return the current trust weight for an agent.

        The weight scales linearly with how far the agent's mean Brier score
        beats the 0.25 coin-flip baseline, clamped to the configured bounds.

        Args:
            agent_name: Agent to look up.

        Returns:
            The trust weight; the default until ``min_samples`` is reached.
        """
        trust = self._agents.get(agent_name)
        if trust is None or trust.sample_count < self._min_samples:
            return self._default_weight
        mean = trust.mean_brier
        if mean is None:
            return self._default_weight
        baseline = 0.25
        weight = self._default_weight * (1.0 + (baseline - mean) / baseline)
        return max(self._min_weight, min(self._max_weight, weight))

    def stats(self) -> dict[str, tuple[int, float | None, float]]:
        """Return ``{agent: (samples, mean_brier, weight)}`` for observability."""
        return {
            name: (trust.sample_count, trust.mean_brier, self.weight_for(name))
            for name, trust in self._agents.items()
        }


__all__ = ["AgentTrust", "TrustRegistry", "brier_score"]
