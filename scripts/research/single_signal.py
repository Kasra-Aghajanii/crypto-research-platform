"""Measure ONE signal at ONE forward horizon on real Hyperliquid history.

Phase 4, item 1.  Answers a narrow question honestly rather than a broad one
vaguely: does this one indicator, read this one way, predict the next N bars?

    python -m scripts.research.single_signal --signal rsi --horizon 20
    python -m scripts.research.single_signal --signal macd_histogram --horizon 5 \
        --symbols BTC ETH SOL --interval 1h --days 240
    python -m scripts.research.single_signal --list

Reported for each run:

* gross hit rate against the 50% baseline, with an exact binomial p-value
* Brier score against the 0.25 coin-flip baseline, plus the skill score
* gross PnL before costs, and net PnL after Hyperliquid taker fees both ways
* sample size, both raw and after removing overlapping forward windows

**Every p-value uses the non-overlapping subsample.** With a 100-bar horizon,
consecutive firings share 99 of their 100 bars; counting them as independent
draws is how a noise signal comes out "significant".
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from libs.config import settings
from libs.logging_config import configure_logging
from services.research.data import DEFAULT_CACHE_DIR, load_candles, load_funding
from services.research.signals import (
    SIGNALS,
    DataRequirement,
    SignalSpec,
    describe,
    get_signal,
)
from services.research.study import ForwardHorizonStudy, StudyResult, pool

logger = logging.getLogger("single_signal")


async def study_signal(
    spec: SignalSpec,
    *,
    symbols: list[str],
    interval: str,
    days: int,
    horizon: int,
    cost_bps: float | None,
    cache_dir: Path,
    refresh: bool,
) -> tuple[StudyResult, list[StudyResult]]:
    """Run one signal at one horizon across symbols.

    Args:
        spec: The signal under test.
        symbols: Symbols to study.
        interval: Candle interval.
        days: History window.
        horizon: Forward horizon in bars.
        cost_bps: Cost per side; defaults to the configured taker fee.
        cache_dir: Candle/funding cache directory.
        refresh: Refetch instead of using the cache.

    Returns:
        The pooled result and the per-symbol results.
    """
    study = ForwardHorizonStudy(horizon=horizon, cost_bps=cost_bps)
    needs_funding = DataRequirement.FUNDING in spec.requires

    per_symbol: list[StudyResult] = []
    for symbol in symbols:
        candles = await load_candles(
            symbol, interval, days=days, cache_dir=cache_dir, refresh=refresh
        )
        funding = (
            await load_funding(symbol, days=days, cache_dir=cache_dir, refresh=refresh)
            if needs_funding
            else ()
        )
        if needs_funding and not funding:
            logger.warning("No funding history for %s; signal will stay flat", symbol)
        per_symbol.append(study.run(spec, symbol, candles, funding=funding))

    return pool(per_symbol), per_symbol


def render(
    spec: SignalSpec,
    pooled: StudyResult,
    per_symbol: list[StudyResult],
    *,
    interval: str,
    days: int,
) -> str:
    """Render the full report for one signal and horizon."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"SIGNAL STUDY  {spec.name}  @  {pooled.horizon}-bar horizon")
    lines.append("=" * 78)
    lines.append(f"  description   : {spec.description}")
    lines.append(f"  convention    : {spec.convention}")
    span = ""
    if pooled.first_bar and pooled.last_bar:
        span = f"  [{pooled.first_bar:%Y-%m-%d} to {pooled.last_bar:%Y-%m-%d}]"
    lines.append(
        f"  symbols       : {', '.join(pooled.symbols)}  ({interval}, {days}d requested){span}"
    )
    lines.append(
        f"  cost model    : {pooled.cost_bps} bps/side "
        f"= {pooled.cost_bps * 2:.1f} bps round trip"
    )
    lines.append("")

    if pooled.independent_size == 0:
        lines.append("  NO USABLE OBSERVATIONS")
        if pooled.note:
            lines.append(f"  reason: {pooled.note}")
        lines.append("=" * 78)
        return "\n".join(lines)

    hit = pooled.hit_rate_test
    net = pooled.net_return_test
    gross = pooled.gross_return_test

    lines.append("  SAMPLE")
    lines.append(f"    bars scanned          : {pooled.bars_scanned}")
    lines.append(
        f"    firings               : {pooled.sample_size}"
        f"  ({pooled.firing_rate:.1%} of bars)"
    )
    lines.append(
        f"    independent (used)    : {pooled.independent_size}"
        f"  (non-overlapping, {pooled.horizon}-bar spacing)"
    )
    lines.append("")

    lines.append("  DIRECTIONAL SKILL")
    lines.append(
        f"    gross hit rate        : {pooled.gross_hit_rate:.2%}"
        f"   vs 50.00% baseline   ({pooled.gross_hit_rate - 0.5:+.2%})"
    )
    lines.append(
        f"    net hit rate          : {pooled.net_hit_rate:.2%}"
        f"   (after {pooled.cost_bps * 2:.1f} bps round trip)"
    )
    lines.append(
        f"    Brier score           : {pooled.brier:.4f}"
        f"   vs 0.2500 baseline   (skill {pooled.brier_skill_score:+.4f})"
    )
    lines.append("")

    lines.append("  RETURNS  (per independent observation)")
    lines.append(f"    mean gross            : {pooled.mean_gross_bps:+.2f} bps")
    lines.append(f"    mean net              : {pooled.mean_net_bps:+.2f} bps")
    lines.append(f"    total gross           : {pooled.total_gross_pct:+.2f}%")
    lines.append(f"    total net             : {pooled.total_net_pct:+.2f}%")
    lines.append("")

    lines.append("  SIGNIFICANCE  (two-sided, non-overlapping sample)")
    lines.append(
        f"    hit rate vs 50%       : p = {hit.p_value:.5f}"
        f"   n = {hit.sample_size}   {_verdict(hit.p_value)}"
    )
    lines.append(
        f"    gross return vs 0     : p = {gross.p_value:.5f}   t = {gross.statistic:+.3f}"
    )
    lines.append(
        f"    net return vs 0       : p = {net.p_value:.5f}   t = {net.statistic:+.3f}"
        f"   {_verdict(net.p_value)}"
    )
    lines.append("")

    if len(per_symbol) > 1:
        lines.append("  PER SYMBOL")
        for result in per_symbol:
            if result.independent_size == 0:
                lines.append(f"    {result.symbols[0]:5s} no usable observations")
                continue
            lines.append(
                f"    {result.symbols[0]:5s} n={result.independent_size:<5d}"
                f" hit={result.gross_hit_rate:.1%}"
                f" brier={result.brier:.4f}"
                f" net={result.mean_net_bps:+.2f}bps"
                f" p={result.hit_rate_test.p_value:.4f}"
            )
        lines.append("")
        lines.append(
            "    Note: symbols are correlated, so pooling them understates variance."
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def _verdict(p_value: float) -> str:
    """Return a short readable verdict for a p-value."""
    if p_value < 0.01:
        return "<- significant at 0.01"
    if p_value < 0.05:
        return "<- significant at 0.05 (uncorrected)"
    return "not significant"


async def run(args: argparse.Namespace) -> int:
    """Execute the requested study.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A process exit code.
    """
    spec = get_signal(args.signal)
    if not spec.is_backtestable:
        print("=" * 78)
        print(f"SIGNAL STUDY  {spec.name}  @  {args.horizon}-bar horizon")
        print("=" * 78)
        print(f"  convention    : {spec.convention}")
        print()
        print(f"  CANNOT BE BACKTESTED: no historical {', '.join(spec.missing_data)} data.")
        print("  Hyperliquid publishes candleSnapshot and fundingHistory as series;")
        print("  l2Book and metaAndAssetCtxs return only the current snapshot.")
        print("  Measuring this signal requires recording the series forward from now.")
        print("=" * 78)
        return 2

    pooled, per_symbol = await study_signal(
        spec,
        symbols=args.symbols,
        interval=args.interval,
        days=args.days,
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
    )
    print(render(spec, pooled, per_symbol, interval=args.interval, days=args.days))
    return 0


def cli() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(
        description="Measure one signal at one forward horizon on real history."
    )
    parser.add_argument("--signal", default=None, help=f"One of: {', '.join(SIGNALS)}")
    parser.add_argument("--horizon", type=int, default=20, help="Forward horizon in bars.")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--days", type=int, default=240)
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=None,
        help=f"Cost per side in bps (default: taker fee {settings.paper.taker_fee_bps}).",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--list", action="store_true", help="List signals and exit.")
    args = parser.parse_args()

    configure_logging("single_signal", level="WARNING")

    if args.list or args.signal is None:
        print("Available signals:\n")
        print(describe())
        print("\nHorizons are in bars of the chosen interval.")
        raise SystemExit(0 if args.list else 1)

    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    cli()
