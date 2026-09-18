"""Long-horizon signal research on daily Hyperliquid candles.

Phase 6.  Phases 4 and 5 found no tradeable edge at 1h horizons, where the
round-trip cost (9 bps) exceeds the entire standard deviation of a one-bar move
(5.1 bps).  This tests the regime where that arithmetic reverses: at 3 to 90
days, a 9 bps round trip is a rounding error against the size of the move.

    python -m scripts.research.long_horizon
    python -m scripts.research.long_horizon --horizons 3 7 14 30 --min-bars 900
    python -m scripts.research.long_horizon --include-synthetic --csv out.csv

The report is deliberately ordered: **data inventory first, statistical power
second, results last**.  At 90-day horizons even six years of daily history
yields tens of independent observations, so which cells can be concluded at all
is settled before any p-value is shown.

Two data facts drive everything and are reported up front:

* Hyperliquid backfills daily candles from an index source for dates before the
  venue traded.  Those bars carry zero volume and zero trades.  They are real
  prices but not Hyperliquid market data, and no funding exists for them.  Only
  traded bars are used unless ``--include-synthetic`` is passed.
* Crypto majors are highly correlated, so pooling symbols multiplies the row
  count far faster than the information.  Effective sample size is reported
  alongside the raw count and is what the conclusions rest on.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from libs.config import settings
from libs.logging_config import configure_logging
from libs.schemas.market import Candle, FundingRate
from services.research.data import DEFAULT_CACHE_DIR, load_candles, load_funding
from services.research.long_horizon import (
    LONG_HORIZON_SIGNALS,
    MIN_TESTABLE_SAMPLE,
    Benchmark,
    ExcessStats,
    compute_benchmark,
    correlation_adjusted_tests,
    cross_sectional_momentum,
    effective_sample_size,
    excess_statistics,
    mean_pairwise_correlation,
    traded_bars,
)
from services.research.signals import DataRequirement, SignalSpec
from services.research.statistics import (
    benjamini_hochberg,
    binomial_test,
    bonferroni,
    one_sample_t_test,
)
from services.research.study import ForwardHorizonStudy, StudyResult, pool

logger = logging.getLogger("long_horizon")

DEFAULT_HORIZONS: tuple[int, ...] = (3, 7, 14, 30, 60, 90)
DEFAULT_SYMBOLS: tuple[str, ...] = (
    "BTC", "ETH", "SOL", "DOGE", "BNB", "LTC", "ATOM", "LINK", "CRV", "XRP",
    "AVAX", "BCH", "AAVE", "TRX", "UNI", "DOT", "ADA", "NEAR", "FIL", "ETC",
)
"""Perps with the longest daily history, from the universe survey."""


@dataclass(slots=True)
class Inventory:
    """What history is actually available for one symbol."""

    symbol: str
    total_bars: int
    traded_bars: int
    first_traded: datetime | None
    last_bar: datetime | None
    funding_points: int

    @property
    def synthetic_bars(self) -> int:
        """Return backfilled bars carrying no volume."""
        return self.total_bars - self.traded_bars


@dataclass(slots=True)
class Cell:
    """One (signal, horizon) result and its testability."""

    signal: str
    horizon: int
    result: StudyResult
    per_symbol_n: float
    symbols: int
    correlation: float
    excess: ExcessStats
    hit_p: float = 1.0
    excess_p: float = 1.0
    hit_significant: bool = False
    excess_significant: bool = False

    @property
    def effective_n(self) -> float:
        """Return the correlation-adjusted independent sample size."""
        return effective_sample_size(self.per_symbol_n, self.symbols, self.correlation)

    @property
    def testable(self) -> bool:
        """Return whether the effective sample supports any conclusion."""
        return self.effective_n >= MIN_TESTABLE_SAMPLE


def daily_funding(points: tuple[FundingRate, ...]) -> tuple[FundingRate, ...]:
    """Aggregate hourly funding into one observation per UTC day.

    Args:
        points: Hourly funding observations, oldest first.

    Returns:
        Daily observations stamped at the end of each day.
    """
    totals: dict[datetime, float] = defaultdict(float)
    symbol = points[0].symbol if points else ""
    for point in points:
        day = point.occurred_at.replace(hour=0, minute=0, second=0, microsecond=0)
        totals[day] += float(point.funding_rate)
    return tuple(
        FundingRate(
            source="daily_aggregate",
            symbol=symbol,
            funding_rate=Decimal(str(total)),
            occurred_at=day + timedelta(hours=23, minutes=59),
        )
        for day, total in sorted(totals.items())
    )


async def gather_data(
    symbols: list[str],
    *,
    days: int,
    cache_dir: Path,
    refresh: bool,
    include_synthetic: bool,
    need_funding: bool,
) -> tuple[dict[str, tuple[Candle, ...]], dict[str, tuple[FundingRate, ...]], list[Inventory]]:
    """Load daily candles and funding, and inventory what came back.

    Args:
        symbols: Symbols to load.
        days: History window to request.
        cache_dir: Cache directory.
        refresh: Refetch instead of using the cache.
        include_synthetic: Keep pre-launch backfilled bars.
        need_funding: Whether to load funding history.

    Returns:
        Candles per symbol, daily funding per symbol, and the inventory.
    """
    candles: dict[str, tuple[Candle, ...]] = {}
    funding: dict[str, tuple[FundingRate, ...]] = {}
    inventory: list[Inventory] = []

    for symbol in symbols:
        raw = await load_candles(symbol, "1d", days=days, cache_dir=cache_dir, refresh=refresh)
        traded = traded_bars(raw)
        chosen = raw if include_synthetic else traded

        daily: tuple[FundingRate, ...] = ()
        if need_funding:
            hourly = await load_funding(symbol, days=days, cache_dir=cache_dir, refresh=refresh)
            daily = daily_funding(hourly)

        if chosen:
            candles[symbol] = chosen
            funding[symbol] = daily
        inventory.append(
            Inventory(
                symbol=symbol,
                total_bars=len(raw),
                traded_bars=len(traded),
                first_traded=traded[0].open_time if traded else None,
                last_bar=raw[-1].open_time if raw else None,
                funding_points=len(daily),
            )
        )
    return candles, funding, inventory


def render_inventory(inventory: list[Inventory], *, include_synthetic: bool) -> str:
    """Render the data inventory, before any testing."""
    lines = ["=" * 92, "DATA INVENTORY  (reported before any testing)", "=" * 92]
    lines.append(
        f"  {'symbol':<8}{'total':>8}{'traded':>8}{'synthetic':>11}"
        f"{'first traded':>15}{'last bar':>13}{'funding d':>11}"
    )
    for item in sorted(inventory, key=lambda i: -i.traded_bars):
        first = item.first_traded.strftime("%Y-%m-%d") if item.first_traded else "--"
        last = item.last_bar.strftime("%Y-%m-%d") if item.last_bar else "--"
        lines.append(
            f"  {item.symbol:<8}{item.total_bars:>8}{item.traded_bars:>8}"
            f"{item.synthetic_bars:>11}{first:>15}{last:>13}{item.funding_points:>11}"
        )
    usable = [i for i in inventory if i.traded_bars > 0]
    if usable:
        lines.append("")
        lines.append(
            f"  symbols with traded history : {len(usable)} of {len(inventory)}"
        )
        lines.append(
            f"  median traded bars          : "
            f"{sorted(i.traded_bars for i in usable)[len(usable) // 2]}"
        )
        lines.append(
            f"  longest traded history      : {max(i.traded_bars for i in usable)} bars"
        )
    lines.append("")
    lines.append("  Hyperliquid backfills daily candles from an index source for dates before")
    lines.append("  the venue traded; those bars carry zero volume and zero trades. They are")
    lines.append("  real prices but not Hyperliquid market data, and carry no funding.")
    lines.append(
        f"  Using: {'ALL bars including backfilled' if include_synthetic else 'TRADED bars only'}"
    )
    lines.append("=" * 92)
    return "\n".join(lines)


def render_power(
    horizons: list[int],
    *,
    bars_per_symbol: float,
    symbols: int,
    correlation: float,
) -> str:
    """Render the power inventory: which cells can be concluded at all."""
    lines = [
        "=" * 92,
        "STATISTICAL POWER  (which cells are testable, before any results)",
        "=" * 92,
    ]
    lines.append(f"  usable symbols               : {symbols}")
    lines.append(f"  median traded bars / symbol  : {bars_per_symbol:.0f}")
    lines.append(f"  mean pairwise correlation    : {correlation:.3f}")
    lines.append("")
    lines.append(
        "  Pooling correlated symbols multiplies rows far faster than information."
    )
    lines.append("  n_effective = (k * n) / (1 + (k - 1) * rho)")
    lines.append("")
    lines.append(
        f"  {'horizon':>9}{'n/symbol':>11}{'pooled raw':>13}{'n_effective':>14}{'verdict':>16}"
    )
    for horizon in horizons:
        per_symbol = bars_per_symbol / horizon
        pooled = per_symbol * symbols
        effective = effective_sample_size(per_symbol, symbols, correlation)
        verdict = "testable" if effective >= MIN_TESTABLE_SAMPLE else "UNTESTABLE"
        lines.append(
            f"  {str(horizon) + 'd':>9}{per_symbol:>11.1f}{pooled:>13.0f}"
            f"{effective:>14.1f}{verdict:>16}"
        )
    lines.append("")
    lines.append(
        f"  A cell with fewer than {MIN_TESTABLE_SAMPLE} effective observations supports"
    )
    lines.append("  neither a positive nor a negative conclusion, so no p-value is reported")
    lines.append("  for it. That is a statement about the data, not about the signal.")
    lines.append("=" * 92)
    return "\n".join(lines)


def _apply_corrections(cells: list[Cell]) -> None:
    """Apply FDR correction across testable cells only.

    Underpowered cells are excluded from the family rather than corrected: they
    cannot be concluded either way, and including them would inflate the number
    of tests and penalise the cells that *can* be concluded.
    """
    testable = [cell for cell in cells if cell.testable and cell.result.independent_size > 0]
    if not testable:
        return
    hit_flags = benjamini_hochberg([c.hit_p for c in testable])
    excess_flags = benjamini_hochberg([c.excess_p for c in testable])
    for cell, hit, excess in zip(testable, hit_flags, excess_flags, strict=True):
        cell.hit_significant = hit
        cell.excess_significant = excess


def render_benchmarks(
    benchmarks: dict[int, dict[str, Benchmark]], horizons: list[int]
) -> str:
    """Render what buy-and-hold earned, which every edge is measured against."""
    lines = [
        "=" * 92,
        "BENCHMARK  (what doing nothing clever earned over the same windows)",
        "=" * 92,
    ]
    lines.append(f"  {'horizon':>9}{'mean return':>15}{'up-rate':>11}   per-symbol range")
    for horizon in horizons:
        row = benchmarks[horizon]
        if not row:
            continue
        means = [b.mean_return * 10_000 for b in row.values()]
        ups = [b.up_rate for b in row.values()]
        lines.append(
            f"  {str(horizon) + 'd':>9}{sum(means) / len(means):>+14.0f}b"
            f"{sum(ups) / len(ups) * 100:>10.1f}%"
            f"   {min(means):+.0f} to {max(means):+.0f} bps"
        )
    lines.append("")
    lines.append("  The sample covers a period in which crypto rose substantially. A signal")
    lines.append("  that is simply long most of the time collects this drift without knowing")
    lines.append("  anything, so it is subtracted from every result that follows.")
    lines.append("=" * 92)
    return "\n".join(lines)


def render_results(cells: list[Cell]) -> str:
    """Render the per-cell results table, measured against buy-and-hold."""
    lines = [
        "=" * 92,
        "RESULTS  (all edges measured against buy-and-hold, not against zero)",
        "=" * 92,
    ]
    lines.append(
        f"  {'signal':<17}{'h':>4}{'n_ind':>7}{'n_eff':>7}{'long%':>7}"
        f"{'hit%':>7}{'null%':>7}{'bench':>9}{'excess':>9}{'net':>9}"
        f"{'p_hit':>8}{'p_exc':>8}  edge"
    )
    for cell in cells:
        result, excess = cell.result, cell.excess
        if result.independent_size == 0:
            lines.append(
                f"  {cell.signal:<17}{cell.horizon:>4}{0:>7}{'--':>7}{'--':>7}"
                f"{'--':>7}{'--':>7}{'--':>9}{'--':>9}{'--':>9}{'--':>8}{'--':>8}  no firings"
            )
            continue
        base = (
            f"  {cell.signal:<17}{cell.horizon:>4}{result.independent_size:>7}"
            f"{cell.effective_n:>7.0f}{excess.long_fraction * 100:>7.0f}"
            f"{excess.hit_rate * 100:>7.1f}{excess.null_probability * 100:>7.1f}"
            f"{excess.mean_benchmark_bps:>+9.0f}{excess.mean_excess_bps:>+9.0f}"
            f"{result.mean_net_bps:>+9.0f}"
        )
        if not cell.testable:
            lines.append(base + f"{'--':>8}{'--':>8}  UNTESTABLE")
            continue
        marks = []
        if cell.hit_significant:
            marks.append("HIT")
        if cell.excess_significant:
            marks.append("EXCESS")
        if not marks and (cell.hit_p < 0.05 or cell.excess_p < 0.05):
            marks.append("raw-only")
        lines.append(
            base
            + f"{cell.hit_p:>8.4f}{cell.excess_p:>8.4f}  {', '.join(marks)}"
        )
    lines.append("")
    lines.append("  long%   = share of firings that were long (exposes directional bias)")
    lines.append("  hit%    = win rate;  null% = win rate expected with NO skill, given how")
    lines.append("            often the signal went long and how often the market rose")
    lines.append("  bench   = what buy-and-hold earned over the same windows, in bps")
    lines.append("  excess  = return in excess of that benchmark -- THIS is the edge")
    lines.append("  net     = raw return after costs, NOT benchmark adjusted")
    lines.append("")
    lines.append("  A long-biased signal in a rising market earns the drift whether or not it")
    lines.append("  knows anything. Testing gross return against zero would score that as a")
    lines.append("  win. Every test here is against the benchmark instead.")
    return "\n".join(lines)


def render_cross_sectional(
    results: dict[int, dict[str, dict[str, float]]], cost_bps: float
) -> str:
    """Render the cross-sectional momentum results."""
    lines = [
        "=" * 92,
        "CROSS-SECTIONAL MOMENTUM  (long top / short bottom, dollar neutral)",
        "=" * 92,
    ]
    lines.append(
        "  Not a directional bet on crypto: it bets that relative ranking persists,"
    )
    lines.append("  which survives the correlation that cripples pooling elsewhere.")
    lines.append("")
    lines.append(
        f"  {'lookback':>10}{'horizon':>9}{'n_ind':>8}{'win%':>8}"
        f"{'gross bps':>12}{'net bps':>11}{'p_gross':>10}  verdict"
    )
    for horizon in sorted(results):
        for key, row in sorted(results[horizon].items()):
            n = int(row["n"])
            verdict = "testable" if n >= MIN_TESTABLE_SAMPLE else "UNTESTABLE"
            p_text = f"{row['p_gross']:>10.4f}" if n >= MIN_TESTABLE_SAMPLE else f"{'--':>10}"
            lines.append(
                f"  {key:>10}{horizon:>9}{n:>8}{row['win_rate'] * 100:>8.1f}"
                f"{row['gross_bps']:>+12.1f}{row['net_bps']:>+11.1f}{p_text}  {verdict}"
            )
    lines.append("")
    lines.append(f"  Costs: {cost_bps} bps per side, charged on both legs of the spread.")
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    """Execute the long-horizon study."""
    signals = args.signals or list(LONG_HORIZON_SIGNALS)
    need_funding = any(
        DataRequirement.FUNDING in LONG_HORIZON_SIGNALS[name].requires for name in signals
    )

    candles, funding, inventory = await gather_data(
        args.symbols,
        days=args.days,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
        include_synthetic=args.include_synthetic,
        need_funding=need_funding,
    )

    print(render_inventory(inventory, include_synthetic=args.include_synthetic))
    print()

    usable = {s: c for s, c in candles.items() if len(c) >= args.min_bars}
    if not usable:
        print("NO SYMBOL HAS ENOUGH TRADED HISTORY. Nothing can be tested.")
        return 2

    dropped = sorted(set(candles) - set(usable))
    if dropped:
        print(f"  Dropped for < {args.min_bars} traded bars: {', '.join(dropped)}\n")

    # Daily returns drive the correlation estimate that the power table needs.
    returns = {
        symbol: [
            float(series[i].close) / float(series[i - 1].close) - 1.0
            for i in range(1, len(series))
            if series[i - 1].close > 0
        ]
        for symbol, series in usable.items()
    }
    correlation = mean_pairwise_correlation(returns)
    counts = sorted(len(series) for series in usable.values())
    median_bars = float(counts[len(counts) // 2])

    print(
        render_power(
            args.horizons,
            bars_per_symbol=median_bars,
            symbols=len(usable),
            correlation=correlation,
        )
    )
    print()

    # ---- the matrix --------------------------------------------------
    # Benchmarks first: what buy-and-hold earned over the same windows. Every
    # edge below is measured against these, never against zero.
    benchmarks: dict[int, dict[str, Benchmark]] = {
        horizon: {
            symbol: compute_benchmark(symbol, [float(c.close) for c in series], horizon)
            for symbol, series in usable.items()
        }
        for horizon in args.horizons
    }
    print(render_benchmarks(benchmarks, args.horizons))
    print()

    cells: list[Cell] = []
    for name in signals:
        spec: SignalSpec = LONG_HORIZON_SIGNALS[name]
        for horizon in args.horizons:
            study = ForwardHorizonStudy(
                horizon=horizon, cost_bps=args.cost_bps, window_bars=args.window_bars
            )
            per_symbol = [
                study.run(spec, symbol, series, funding=funding.get(symbol, ()))
                for symbol, series in usable.items()
            ]
            pooled = pool(per_symbol)
            cells.append(
                Cell(
                    signal=name,
                    horizon=horizon,
                    result=pooled,
                    per_symbol_n=median_bars / horizon,
                    symbols=len(usable),
                    correlation=correlation,
                    excess=excess_statistics(pooled.independent, benchmarks[horizon]),
                )
            )
            cell = cells[-1]
            cell.hit_p, cell.excess_p = correlation_adjusted_tests(
                [
                    o.gross_return
                    - float(o.direction.sign)
                    * benchmarks[horizon][o.symbol].mean_return
                    for o in pooled.independent
                    if o.symbol in benchmarks[horizon]
                ],
                wins=cell.excess.wins,
                null_probability=cell.excess.null_probability,
                effective_n=cell.effective_n,
            )
    _apply_corrections(cells)
    print(render_results(cells))
    print()

    # ---- cross-sectional ---------------------------------------------
    cross: dict[int, dict[str, dict[str, float]]] = {}
    if len(usable) >= args.min_cross_symbols:
        for horizon in args.horizons:
            cross[horizon] = {}
            for lookback in args.cross_lookbacks:
                observations = cross_sectional_momentum(
                    usable,
                    lookback=lookback,
                    horizon=horizon,
                    cost_bps=args.cost_bps,
                    min_symbols=args.min_cross_symbols,
                )
                independent = [o for o in observations if o.independent]
                if not independent:
                    continue
                gross = [o.gross_return for o in independent]
                net = [o.net_return for o in independent]
                cross[horizon][f"{lookback}d"] = {
                    "n": float(len(independent)),
                    "win_rate": sum(1 for g in gross if g > 0) / len(gross),
                    "gross_bps": sum(gross) / len(gross) * 10_000,
                    "net_bps": sum(net) / len(net) * 10_000,
                    "p_gross": one_sample_t_test(gross, 0.0).p_value,
                    "p_hit": binomial_test(
                        sum(1 for g in gross if g > 0), len(gross), 0.5
                    ).p_value,
                }
        print(render_cross_sectional(cross, args.cost_bps))
        print()
    else:
        print(
            f"Cross-sectional momentum skipped: needs >= {args.min_cross_symbols} symbols, "
            f"have {len(usable)}.\n"
        )

    print(render_verdict(cells, cross, correlation=correlation, symbols=len(usable)))

    if args.csv:
        write_csv(args.csv, cells)
        print(f"\nWrote {args.csv}")
    return 0


def render_verdict(
    cells: list[Cell],
    cross: dict[int, dict[str, dict[str, float]]],
    *,
    correlation: float,
    symbols: int,
) -> str:
    """Render the closing verdict."""
    with_data = [c for c in cells if c.result.independent_size > 0]
    testable = [c for c in with_data if c.testable]
    untestable = [c for c in with_data if not c.testable]
    edge = [c for c in testable if c.hit_significant or c.excess_significant]
    tradeable = [c for c in edge if c.excess.mean_excess_bps > 0]

    lines = ["=" * 92, "VERDICT", "=" * 92]
    lines.append(f"  cells with any firings        : {len(with_data)}")
    lines.append(f"  testable (n_eff >= {MIN_TESTABLE_SAMPLE})        : {len(testable)}")
    lines.append(f"  UNTESTABLE (too few obs)      : {len(untestable)}")
    lines.append(f"  edge surviving FDR            : {len(edge)}")
    lines.append(f"  ...and profitable after costs : {len(tradeable)}")
    lines.append("")

    if tradeable:
        lines.append("  POSITIVE EXCESS OVER BUY-AND-HOLD, SURVIVING CORRECTION:")
        for cell in sorted(tradeable, key=lambda c: -c.excess.mean_excess_bps):
            lines.append(
                f"    {cell.signal} @ {cell.horizon}d: excess "
                f"{cell.excess.mean_excess_bps:+.1f} bps over a benchmark of "
                f"{cell.excess.mean_benchmark_bps:+.1f} bps, hit {cell.excess.hit_rate:.1%} "
                f"vs {cell.excess.null_probability:.1%} expected, n_eff={cell.effective_n:.0f}, "
                f"p_excess={cell.excess_p:.4f}"
            )
    elif edge:
        lines.append("  Survives correction but with NEGATIVE excess over buy-and-hold:")
        for cell in edge:
            lines.append(
                f"    {cell.signal} @ {cell.horizon}d: excess "
                f"{cell.excess.mean_excess_bps:+.1f} bps, benchmark "
                f"{cell.excess.mean_benchmark_bps:+.1f} bps"
            )
    else:
        lines.append("  NO SIGNAL SURVIVES CORRECTION on the testable cells.")

    if untestable:
        lines.append("")
        lines.append(
            f"  {len(untestable)} cells could not be tested at all. That is a limit of the"
        )
        lines.append("  available history, not a finding about those signals:")
        horizons = sorted({c.horizon for c in untestable})
        lines.append(f"    affected horizons: {', '.join(f'{h}d' for h in horizons)}")

    bonferroni_flags = bonferroni([c.excess_p for c in testable])
    survivors = sum(bonferroni_flags)
    lines.append("")
    lines.append(f"  Under Bonferroni: {survivors} cell(s) show excess over buy-and-hold.")
    lines.append("")
    lines.append(
        f"  Pooling caveat: mean pairwise correlation across {symbols} symbols is "
        f"{correlation:.3f}."
    )
    lines.append(
        "  Every p-value above is computed on n_eff, not on the raw pooled count, so"
    )
    lines.append("  the correlation is priced into the numbers rather than only the prose.")
    lines.append("=" * 92)
    return "\n".join(lines)


def write_csv(path: Path, cells: list[Cell]) -> None:
    """Write the matrix to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "signal", "horizon_days", "n_independent", "n_effective", "testable",
                "gross_hit_rate", "brier", "mean_gross_bps", "mean_net_bps",
                "p_hit", "p_excess", "mean_excess_bps", "hit_fdr", "excess_fdr",
            ]
        )
        for cell in cells:
            result = cell.result
            writer.writerow(
                [
                    cell.signal, cell.horizon, result.independent_size,
                    f"{cell.effective_n:.2f}", cell.testable,
                    f"{result.gross_hit_rate:.6f}", f"{result.brier:.6f}",
                    f"{result.mean_gross_bps:.4f}", f"{result.mean_net_bps:.4f}",
                    f"{cell.hit_p:.8f}",
                    f"{cell.excess_p:.8f}",
                    f"{cell.excess.mean_excess_bps:.4f}",
                    cell.hit_significant, cell.excess_significant,
                ]
            )


def cli() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description="Long-horizon signal research on daily candles.")
    parser.add_argument("--signals", nargs="+", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=2500, help="History window to request.")
    parser.add_argument(
        "--min-bars", type=int, default=400, help="Traded bars required to include a symbol."
    )
    parser.add_argument(
        "--window-bars", type=int, default=400, help="Bars visible to a signal at each decision."
    )
    parser.add_argument(
        "--cost-bps", type=float, default=None, help="Cost per side; defaults to the taker fee."
    )
    parser.add_argument("--cross-lookbacks", nargs="+", type=int, default=[30, 90, 180])
    parser.add_argument("--min-cross-symbols", type=int, default=6)
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Include pre-launch backfilled bars that carry no volume.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    if args.cost_bps is None:
        args.cost_bps = settings.paper.taker_fee_bps

    configure_logging("long_horizon", level="WARNING")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    cli()
