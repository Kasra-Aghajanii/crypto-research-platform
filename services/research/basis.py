"""Funding carry: a structural, non-forecasting trade.

Phase 5, item 3.  Everything measured so far tried to *predict* price.  This does
not.  The cash-and-carry is mechanical: hold spot, short the perpetual against
it, and collect funding while the position is delta neutral.  It makes money
from the contract's design rather than from being right about direction.

The position and its PnL
------------------------
Short perp + long spot, equal notional.  Over a holding period the return on
notional is::

    PnL = sum(funding over the period)      <- the short receives it when positive
        - (premium_exit - premium_entry)    <- basis convergence, both legs together
        - costs                              <- round trip on *both* legs

The basis term follows from holding both legs: the perp leg loses what the spot
leg gains except for the change in the gap between them, and ``premium`` is
exactly that gap expressed as a fraction of the index.

Hyperliquid publishes ``fundingRate`` and ``premium`` together in
``fundingHistory``, so both terms are measurable from one series without needing
spot price history.

What this does not model
------------------------
* **Execution of the spot leg.**  The analysis assumes both legs can be entered
  and exited at the quoted price.  Liquidity is finite and the spot books are
  much thinner than the perp books.
* **Margin and liquidation.**  A short perp needs collateral, and an adverse move
  can force a close even when the combined position is flat.  That risk is real
  and unpriced here.
* **Funding regime changes.**  A rate that has been positive for months can go
  negative, at which point the short pays.

Statistical treatment matches the rest of the research harness: non-overlapping
holding windows only, and multiple-comparison correction across the grid.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from libs.schemas.market import FundingRate
from services.research.statistics import TestResult, binomial_test, one_sample_t_test

logger = logging.getLogger(__name__)

_BPS: Final[float] = 10_000.0
HOURS_PER_YEAR: Final[float] = 24.0 * 365.0


@dataclass(frozen=True, slots=True)
class CarryTrade:
    """One simulated delta-neutral holding period.

    Attributes:
        funding_collected: Summed funding over the period, as a fraction.
        basis_change: ``premium_exit - premium_entry``; a cost when positive.
        gross_return: Funding minus basis change, before costs.
        net_return: Gross return after the round trip on both legs.
    """

    symbol: str
    index: int
    entry_at: datetime
    exit_at: datetime
    hours: int
    funding_collected: float
    basis_change: float
    gross_return: float
    net_return: float
    independent: bool = False

    @property
    def gross_win(self) -> bool:
        """Return whether the trade was profitable before costs."""
        return self.gross_return > 0

    @property
    def net_win(self) -> bool:
        """Return whether the trade was profitable after costs."""
        return self.net_return > 0


@dataclass(slots=True)
class CarryResult:
    """Outcome of one (symbol, holding period) carry study."""

    symbol: str
    hours: int
    cost_bps: float
    trades: list[CarryTrade] = field(default_factory=list)
    first_at: datetime | None = None
    last_at: datetime | None = None
    note: str = ""

    @property
    def independent(self) -> list[CarryTrade]:
        """Return the non-overlapping subsample used for every p-value."""
        return [trade for trade in self.trades if trade.independent]

    @property
    def sample_size(self) -> int:
        """Return the number of non-overlapping trades."""
        return len(self.independent)

    @property
    def net_win_rate(self) -> float:
        """Return the fraction of independent trades profitable after costs."""
        sample = self.independent
        if not sample:
            return 0.0
        return sum(1 for trade in sample if trade.net_win) / len(sample)

    @property
    def gross_win_rate(self) -> float:
        """Return the fraction profitable before costs."""
        sample = self.independent
        if not sample:
            return 0.0
        return sum(1 for trade in sample if trade.gross_win) / len(sample)

    @property
    def net_returns(self) -> list[float]:
        """Return independent net returns as fractions."""
        return [trade.net_return for trade in self.independent]

    @property
    def mean_net_bps(self) -> float:
        """Return the mean net return per trade, in basis points."""
        values = self.net_returns
        return (sum(values) / len(values)) * _BPS if values else 0.0

    @property
    def mean_gross_bps(self) -> float:
        """Return the mean gross return per trade, in basis points."""
        sample = self.independent
        if not sample:
            return 0.0
        return sum(t.gross_return for t in sample) / len(sample) * _BPS

    @property
    def mean_funding_bps(self) -> float:
        """Return the mean funding collected per trade, in basis points."""
        sample = self.independent
        if not sample:
            return 0.0
        return sum(t.funding_collected for t in sample) / len(sample) * _BPS

    @property
    def mean_basis_cost_bps(self) -> float:
        """Return the mean basis-change drag per trade, in basis points."""
        sample = self.independent
        if not sample:
            return 0.0
        return sum(t.basis_change for t in sample) / len(sample) * _BPS

    @property
    def annualized_net_pct(self) -> float:
        """Return the mean net return scaled to an annual rate, in percent.

        Assumes the position is rolled continuously at the same holding period.
        """
        if not self.independent or self.hours <= 0:
            return 0.0
        per_hour = (sum(self.net_returns) / len(self.net_returns)) / self.hours
        return per_hour * HOURS_PER_YEAR * 100.0

    @property
    def win_rate_test(self) -> TestResult:
        """Return the binomial test of the net win rate against 50%."""
        sample = self.independent
        wins = sum(1 for trade in sample if trade.net_win)
        return binomial_test(wins, len(sample), 0.5)

    @property
    def net_return_test(self) -> TestResult:
        """Return the t-test of mean net return against zero."""
        return one_sample_t_test(self.net_returns, 0.0)


@dataclass(frozen=True, slots=True)
class FundingProfile:
    """Descriptive statistics of a funding series."""

    symbol: str
    observations: int
    first_at: datetime | None
    last_at: datetime | None
    mean_hourly: float
    positive_fraction: float
    pinned_fraction: float
    annualized_pct: float
    mean_premium_bps: float

    @property
    def days(self) -> float:
        """Return the span of the series in days."""
        if self.first_at is None or self.last_at is None:
            return 0.0
        return (self.last_at - self.first_at).total_seconds() / 86_400.0


def profile_funding(
    symbol: str, funding: Sequence[FundingRate], *, base_rate: float = 0.0000125
) -> FundingProfile:
    """Summarise a funding series.

    Args:
        symbol: Symbol the series belongs to.
        funding: Funding observations, oldest first.
        base_rate: Hyperliquid's floor rate (0.01% per 8h = 0.00125% per hour).
            Funding pins here whenever the premium term is small, which turns out
            to be most of the time.

    Returns:
        The descriptive profile.
    """
    if not funding:
        return FundingProfile(
            symbol=symbol,
            observations=0,
            first_at=None,
            last_at=None,
            mean_hourly=0.0,
            positive_fraction=0.0,
            pinned_fraction=0.0,
            annualized_pct=0.0,
            mean_premium_bps=0.0,
        )
    rates = [float(point.funding_rate) for point in funding]
    premiums = [float(point.premium) for point in funding if point.premium is not None]
    mean_hourly = sum(rates) / len(rates)
    pinned = sum(1 for rate in rates if abs(rate - base_rate) < base_rate * 1e-6)
    return FundingProfile(
        symbol=symbol,
        observations=len(rates),
        first_at=funding[0].occurred_at,
        last_at=funding[-1].occurred_at,
        mean_hourly=mean_hourly,
        positive_fraction=sum(1 for rate in rates if rate > 0) / len(rates),
        pinned_fraction=pinned / len(rates),
        annualized_pct=mean_hourly * HOURS_PER_YEAR * 100.0,
        mean_premium_bps=(sum(premiums) / len(premiums) * _BPS) if premiums else 0.0,
    )


class CarryStudy:
    """Simulates rolling delta-neutral carry positions over funding history.

    Args:
        hours: Holding period in hours (funding settles hourly).
        cost_bps: Total round-trip cost across **both** legs, in basis points.
    """

    def __init__(self, *, hours: int, cost_bps: float) -> None:
        """Configure the study.

        Raises:
            ValueError: If the holding period is not positive.
        """
        if hours < 1:
            raise ValueError(f"Holding period must be >= 1 hour, got {hours}.")
        self.hours = hours
        self.cost_bps = cost_bps

    @property
    def cost(self) -> float:
        """Return the round-trip cost as a fraction of notional."""
        return self.cost_bps / _BPS

    def run(self, symbol: str, funding: Sequence[FundingRate]) -> CarryResult:
        """Simulate every entry point over the series.

        Args:
            symbol: Symbol being studied.
            funding: Funding observations, oldest first.

        Returns:
            The carry result, with non-overlapping trades marked.
        """
        result = CarryResult(symbol=symbol, hours=self.hours, cost_bps=self.cost_bps)
        if len(funding) <= self.hours:
            result.note = f"insufficient funding history: {len(funding)} points"
            return result

        result.first_at = funding[0].occurred_at
        result.last_at = funding[-1].occurred_at

        # Prefix sums make every holding window O(1) rather than O(hours).
        cumulative = [0.0]
        for point in funding:
            cumulative.append(cumulative[-1] + float(point.funding_rate))

        next_allowed = -1
        for index in range(len(funding) - self.hours):
            entry, exit_point = funding[index], funding[index + self.hours]
            collected = cumulative[index + self.hours] - cumulative[index]

            entry_premium = float(entry.premium) if entry.premium is not None else 0.0
            exit_premium = float(exit_point.premium) if exit_point.premium is not None else 0.0
            basis_change = exit_premium - entry_premium

            gross = collected - basis_change
            independent = index >= next_allowed
            if independent:
                next_allowed = index + self.hours

            result.trades.append(
                CarryTrade(
                    symbol=symbol,
                    index=index,
                    entry_at=entry.occurred_at,
                    exit_at=exit_point.occurred_at,
                    hours=self.hours,
                    funding_collected=collected,
                    basis_change=basis_change,
                    gross_return=gross,
                    net_return=gross - self.cost,
                    independent=independent,
                )
            )
        return result


def pool_carry(results: Sequence[CarryResult]) -> CarryResult:
    """Pool per-symbol carry results.

    Args:
        results: Results for the same holding period.

    Returns:
        A pooled result.

    Raises:
        ValueError: If the results are empty or disagree on holding period.
    """
    if not results:
        raise ValueError("Cannot pool an empty result set.")
    periods = {result.hours for result in results}
    if len(periods) != 1:
        raise ValueError(f"Cannot pool across holding periods {periods}.")

    pooled = CarryResult(
        symbol="+".join(sorted({r.symbol for r in results})),
        hours=results[0].hours,
        cost_bps=results[0].cost_bps,
        first_at=min((r.first_at for r in results if r.first_at), default=None),
        last_at=max((r.last_at for r in results if r.last_at), default=None),
    )
    for result in results:
        pooled.trades.extend(result.trades)
    return pooled


__all__ = [
    "HOURS_PER_YEAR",
    "CarryResult",
    "CarryStudy",
    "CarryTrade",
    "FundingProfile",
    "pool_carry",
    "profile_funding",
]
