"""Forward-horizon signal study.

Measures one signal against one fixed forward horizon: at every bar where the
signal fires, record the return over the next ``H`` bars in the direction it
called, and nothing else.

Why not the trading replay engine
---------------------------------
:mod:`services.backtest.replay` simulates a strategy -- entries, stops,
take-profits, forced closes.  Running a signal through it measures the signal
*and the exit policy together*, and the exit policy dominates: Phase 3's 34% net
hit rate was partly a statement about where the stops sat.  A fixed horizon has
no exit policy to confound the measurement, which is the whole point of asking
"does this indicator predict the next N bars".

What is shared with the replay engine is the discipline that makes a result
trustworthy: the same candle source, the same bounded context window (a signal
never sees more history than the live agent would), the same cost model, and the
same no-look-ahead construction -- the window handed to a signal ends at the
decision bar, so a future bar is not merely unused but absent.

Two statistical hazards, handled structurally
---------------------------------------------
**Overlapping observations.**  A signal firing on consecutive bars with a 100-bar
horizon produces observations sharing 99 of their 100 bars.  Those are not
independent draws, and treating them as such inflates significance enormously.
Every p-value here is computed on a *non-overlapping* subsample: take an
observation, then skip ``H`` bars before taking another.  Both counts are
reported so the difference is visible.

**Cost asymmetry.**  Hit rate is reported gross *and* net.  A signal can predict
direction genuinely and still lose money once the round trip is paid for; those
are different findings and are kept separate.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from libs.config import Settings, settings
from libs.schemas.market import Candle, FundingRate
from libs.schemas.signals import Direction
from services.research.signals import SignalSpec, SignalWindow
from services.research.statistics import TestResult, binomial_test, one_sample_t_test

logger = logging.getLogger(__name__)

_BPS: Final[float] = 10_000.0


@dataclass(frozen=True, slots=True)
class Observation:
    """One signal firing and the forward return that resolved it.

    Attributes:
        gross_return: Signed return over the horizon, before costs.
        net_return: Signed return after the modelled round-trip cost.
        independent: Whether this observation is in the non-overlapping subsample.
    """

    symbol: str
    index: int
    timestamp: datetime
    direction: Direction
    probability: float
    conviction: float
    value: float
    entry_price: float
    exit_price: float
    gross_return: float
    net_return: float
    independent: bool = False

    @property
    def gross_win(self) -> bool:
        """Return whether the direction call was right before costs."""
        return self.gross_return > 0

    @property
    def net_win(self) -> bool:
        """Return whether the trade made money after costs."""
        return self.net_return > 0


@dataclass(slots=True)
class StudyResult:
    """Everything one (signal, horizon) study produced.

    All p-values are computed on the independent subsample only.
    """

    signal: str
    horizon: int
    symbols: tuple[str, ...]
    cost_bps: float
    bars_scanned: int = 0
    observations: list[Observation] = field(default_factory=list)
    first_bar: datetime | None = None
    last_bar: datetime | None = None
    note: str = ""

    # ---- sample sizes -------------------------------------------------

    @property
    def sample_size(self) -> int:
        """Return the number of signal firings, including overlaps."""
        return len(self.observations)

    @property
    def independent(self) -> list[Observation]:
        """Return the non-overlapping subsample used for every p-value."""
        return [o for o in self.observations if o.independent]

    @property
    def independent_size(self) -> int:
        """Return the number of non-overlapping observations."""
        return len(self.independent)

    @property
    def firing_rate(self) -> float:
        """Return the fraction of scanned bars on which the signal fired."""
        if self.bars_scanned == 0:
            return 0.0
        return self.sample_size / self.bars_scanned

    # ---- accuracy -----------------------------------------------------

    @property
    def gross_hit_rate(self) -> float:
        """Return the fraction of independent calls right before costs."""
        sample = self.independent
        if not sample:
            return 0.0
        return sum(1 for o in sample if o.gross_win) / len(sample)

    @property
    def net_hit_rate(self) -> float:
        """Return the fraction of independent calls profitable after costs."""
        sample = self.independent
        if not sample:
            return 0.0
        return sum(1 for o in sample if o.net_win) / len(sample)

    @property
    def brier(self) -> float:
        """Return the Brier score of the forecast probabilities.

        Scored against the *gross* outcome: this measures directional skill, not
        whether the edge survived the fee. ``0.25`` is what a permanent 50/50
        forecast scores.
        """
        sample = self.independent
        if not sample:
            return 0.0
        return sum((o.probability - (1.0 if o.gross_win else 0.0)) ** 2 for o in sample) / len(
            sample
        )

    @property
    def brier_skill_score(self) -> float:
        """Return improvement over the 0.25 coin-flip baseline.

        Positive means better than a coin flip; ``1.0`` would be perfect.
        """
        return 1.0 - (self.brier / 0.25) if self.independent else 0.0

    # ---- returns ------------------------------------------------------

    @property
    def gross_returns(self) -> list[float]:
        """Return independent gross returns as fractions."""
        return [o.gross_return for o in self.independent]

    @property
    def net_returns(self) -> list[float]:
        """Return independent net returns as fractions."""
        return [o.net_return for o in self.independent]

    @property
    def mean_gross_bps(self) -> float:
        """Return the mean gross return per observation, in basis points."""
        values = self.gross_returns
        return (sum(values) / len(values)) * _BPS if values else 0.0

    @property
    def mean_net_bps(self) -> float:
        """Return the mean net return per observation, in basis points."""
        values = self.net_returns
        return (sum(values) / len(values)) * _BPS if values else 0.0

    @property
    def total_gross_pct(self) -> float:
        """Return summed gross return across independent observations, in percent."""
        return sum(self.gross_returns) * 100.0

    @property
    def total_net_pct(self) -> float:
        """Return summed net return across independent observations, in percent."""
        return sum(self.net_returns) * 100.0

    # ---- significance -------------------------------------------------

    @property
    def hit_rate_test(self) -> TestResult:
        """Return the two-sided binomial test of gross hit rate against 50%."""
        sample = self.independent
        wins = sum(1 for o in sample if o.gross_win)
        return binomial_test(wins, len(sample), 0.5)

    @property
    def net_return_test(self) -> TestResult:
        """Return the two-sided t-test of mean net return against zero."""
        return one_sample_t_test(self.net_returns, 0.0)

    @property
    def gross_return_test(self) -> TestResult:
        """Return the two-sided t-test of mean gross return against zero."""
        return one_sample_t_test(self.gross_returns, 0.0)

    def summary(self) -> dict[str, float | int | str]:
        """Return a flat summary suitable for a results table."""
        return {
            "signal": self.signal,
            "horizon": self.horizon,
            "n_all": self.sample_size,
            "n_independent": self.independent_size,
            "firing_rate": round(self.firing_rate, 4),
            "gross_hit_rate": round(self.gross_hit_rate, 4),
            "net_hit_rate": round(self.net_hit_rate, 4),
            "brier": round(self.brier, 4),
            "brier_skill": round(self.brier_skill_score, 4),
            "mean_gross_bps": round(self.mean_gross_bps, 2),
            "mean_net_bps": round(self.mean_net_bps, 2),
            "total_net_pct": round(self.total_net_pct, 2),
            "p_hit_rate": round(self.hit_rate_test.p_value, 6),
            "p_net_return": round(self.net_return_test.p_value, 6),
        }


class ForwardHorizonStudy:
    """Runs one signal over history at one forward horizon.

    Args:
        horizon: Bars ahead the forward return is measured over.
        config: Settings override, mainly for tests.
        cost_bps: Cost per side in basis points. Defaults to the configured
            Hyperliquid taker fee, so the round trip is twice this.
        window_bars: Bars visible to the signal, matching the live agent buffer.
    """

    def __init__(
        self,
        *,
        horizon: int,
        config: Settings | None = None,
        cost_bps: float | None = None,
        window_bars: int | None = None,
    ) -> None:
        """Configure the study.

        Raises:
            ValueError: If the horizon is not positive.
        """
        if horizon < 1:
            raise ValueError(f"Horizon must be >= 1 bar, got {horizon}.")
        self.settings = config or settings
        self.horizon = horizon
        self.cost_bps = (
            cost_bps if cost_bps is not None else self.settings.paper.taker_fee_bps
        )
        self.window_bars = window_bars or self.settings.analyst.warmup_candles

    @property
    def round_trip_cost(self) -> float:
        """Return the modelled round-trip cost as a fraction of notional."""
        return 2.0 * self.cost_bps / _BPS

    def run(
        self,
        spec: SignalSpec,
        symbol: str,
        candles: Sequence[Candle],
        *,
        funding: Sequence[FundingRate] = (),
        open_interest: Sequence[float] = (),
    ) -> StudyResult:
        """Evaluate one signal over one symbol's history.

        Args:
            spec: The signal under test.
            symbol: Symbol being studied.
            candles: Closed candles, oldest first.
            funding: Funding history, oldest first, if the signal needs it.
            open_interest: Open-interest series aligned to ``candles``.

        Returns:
            The study result, with observations marked for independence.
        """
        result = StudyResult(
            signal=spec.name,
            horizon=self.horizon,
            symbols=(symbol,),
            cost_bps=self.cost_bps,
        )
        start = max(spec.min_bars, 1) - 1
        last = len(candles) - self.horizon
        if candles:
            result.first_bar = candles[0].open_time
            result.last_bar = candles[-1].open_time
        if last <= start:
            result.note = (
                f"insufficient history: {len(candles)} bars, need "
                f"> {start + self.horizon}"
            )
            return result

        funding_cursor = 0
        cost = self.round_trip_cost

        for index in range(start, last):
            bar = candles[index]
            result.bars_scanned += 1

            # Advance the funding pointer to everything settled by this bar.
            while (
                funding_cursor < len(funding)
                and funding[funding_cursor].occurred_at <= bar.close_time
            ):
                funding_cursor += 1

            lower = max(0, index + 1 - self.window_bars)
            window = SignalWindow(
                candles=tuple(candles[lower : index + 1]),
                funding=tuple(funding[:funding_cursor][-self.window_bars :]),
                open_interest=tuple(open_interest[: index + 1][-self.window_bars :]),
            )

            reading = spec.evaluate(window)
            if reading is None or reading.direction is Direction.FLAT:
                continue

            entry = float(bar.close)
            exit_price = float(candles[index + self.horizon].close)
            if entry <= 0:
                continue

            change = (exit_price - entry) / entry
            gross = change * float(reading.direction.sign)
            result.observations.append(
                Observation(
                    symbol=symbol,
                    index=index,
                    timestamp=bar.close_time,
                    direction=reading.direction,
                    probability=reading.probability,
                    conviction=reading.conviction,
                    value=reading.value,
                    entry_price=entry,
                    exit_price=exit_price,
                    gross_return=gross,
                    net_return=gross - cost,
                )
            )

        self._mark_independent(result)
        logger.info(
            "Study complete",
            extra={
                "signal": spec.name,
                "symbol": symbol,
                "horizon": self.horizon,
                "n_all": result.sample_size,
                "n_independent": result.independent_size,
            },
        )
        return result

    def _mark_independent(self, result: StudyResult) -> None:
        """Mark a maximal non-overlapping subsample of the observations.

        Greedy left-to-right: take an observation, then skip every observation
        whose forward window overlaps it.  This is what makes the p-values
        defensible -- overlapping windows share most of their price path and are
        nowhere near independent draws.

        Args:
            result: The study result to annotate in place.
        """
        next_allowed = -1
        marked: list[Observation] = []
        for observation in result.observations:
            if observation.index >= next_allowed:
                marked.append(
                    Observation(
                        symbol=observation.symbol,
                        index=observation.index,
                        timestamp=observation.timestamp,
                        direction=observation.direction,
                        probability=observation.probability,
                        conviction=observation.conviction,
                        value=observation.value,
                        entry_price=observation.entry_price,
                        exit_price=observation.exit_price,
                        gross_return=observation.gross_return,
                        net_return=observation.net_return,
                        independent=True,
                    )
                )
                next_allowed = observation.index + self.horizon
            else:
                marked.append(observation)
        result.observations = marked


def pool(results: Sequence[StudyResult]) -> StudyResult:
    """Combine per-symbol results into one pooled result.

    Pooling correlated symbols (BTC, ETH and SOL move together) understates the
    true variance, so a pooled p-value is optimistic. Per-symbol results are
    reported alongside for exactly that reason.

    Args:
        results: Per-symbol results for the same signal and horizon.

    Returns:
        A pooled result.

    Raises:
        ValueError: If the results are empty or disagree on signal/horizon.
    """
    if not results:
        raise ValueError("Cannot pool an empty result set.")
    signals = {r.signal for r in results}
    horizons = {r.horizon for r in results}
    if len(signals) != 1 or len(horizons) != 1:
        raise ValueError(f"Cannot pool across signals {signals} or horizons {horizons}.")

    pooled = StudyResult(
        signal=results[0].signal,
        horizon=results[0].horizon,
        symbols=tuple(sorted({s for r in results for s in r.symbols})),
        cost_bps=results[0].cost_bps,
        bars_scanned=sum(r.bars_scanned for r in results),
        first_bar=min((r.first_bar for r in results if r.first_bar), default=None),
        last_bar=max((r.last_bar for r in results if r.last_bar), default=None),
    )
    for result in results:
        pooled.observations.extend(result.observations)
    notes = sorted({r.note for r in results if r.note})
    pooled.note = "; ".join(notes)
    return pooled


__all__ = ["ForwardHorizonStudy", "Observation", "StudyResult", "pool"]
