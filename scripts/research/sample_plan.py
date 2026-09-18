"""How long the recorders must run before their data can answer anything.

Phase 5, item 1d.  The order book and open-interest recorders start with zero
history.  This computes, from the *measured* volatility of Hyperliquid forward
returns, how many independent observations are needed to detect an edge of a
given size -- and therefore how many days of recording that is.

    python -m scripts.research.sample_plan
    python -m scripts.research.sample_plan --interval 1h --symbols BTC ETH SOL

The return volatility is estimated from the cached candles used in Phase 4, so
these are empirical requirements rather than textbook illustrations.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from libs.config import settings
from libs.logging_config import configure_logging
from services.data_ingestion.market_data.hyperliquid_history import interval_seconds
from services.research.data import DEFAULT_CACHE_DIR, load_candles
from services.research.power import (
    days_to_collect,
    samples_for_hit_rate,
    samples_for_mean_return,
    standard_deviation_bps,
)

logger = logging.getLogger("sample_plan")

HIT_RATE_EFFECTS = (0.52, 0.53, 0.55, 0.60)
RETURN_EFFECTS_BPS = (5.0, 10.0, 20.0, 50.0)


async def measure_volatility(
    symbols: list[str], interval: str, horizons: list[int], *, cache_dir: Path, days: int
) -> dict[int, float]:
    """Measure the standard deviation of forward returns per horizon.

    Args:
        symbols: Symbols to pool.
        interval: Candle interval.
        horizons: Forward horizons in bars.
        cache_dir: Candle cache directory.
        days: History window.

    Returns:
        A mapping of horizon to return standard deviation in basis points.
    """
    series: dict[int, list[float]] = {horizon: [] for horizon in horizons}
    for symbol in symbols:
        candles = await load_candles(symbol, interval, days=days, cache_dir=cache_dir)
        closes = [float(candle.close) for candle in candles]
        for horizon in horizons:
            # Non-overlapping forward returns, matching how the study samples.
            for index in range(0, len(closes) - horizon, horizon):
                entry = closes[index]
                if entry > 0:
                    series[horizon].append((closes[index + horizon] - entry) / entry)
    return {horizon: standard_deviation_bps(values) for horizon, values in series.items()}


def render(
    volatility: dict[int, float],
    *,
    interval: str,
    symbol_count: int,
    round_trip_bps: float,
) -> str:
    """Render the sample-size and calendar-time plan.

    Args:
        volatility: Return standard deviation in bps, per horizon.
        interval: Candle interval the horizons are measured in.
        symbol_count: Symbols recorded in parallel.
        round_trip_bps: Round-trip cost, for the break-even reference.

    Returns:
        The printable report.
    """
    seconds = interval_seconds(interval)
    lines: list[str] = []

    lines.append("=" * 78)
    lines.append("TIME TO A USABLE SAMPLE")
    lines.append("=" * 78)
    lines.append(f"  interval        : {interval}  ({seconds / 3600:.2f}h per bar)")
    lines.append(f"  symbols         : {symbol_count} recorded in parallel")
    lines.append(f"  round trip cost : {round_trip_bps:.1f} bps")
    lines.append("  target          : alpha = 0.05 two-sided, power = 0.80")
    lines.append("")
    lines.append("  Independent observations are spaced one horizon apart, so a longer")
    lines.append("  horizon costs proportionally more calendar time per observation.")
    lines.append("")

    lines.append("-" * 78)
    lines.append("DETECTING A HIT RATE ABOVE 50%")
    lines.append("-" * 78)
    header = f"  {'true hit rate':<16}{'observations':>14}" + "".join(
        f"{'h=' + str(h):>12}" for h in sorted(volatility)
    )
    lines.append(header)
    lines.append(f"  {'':<16}{'needed':>14}" + "".join(f"{'days':>12}" for _ in volatility))
    for effect in HIT_RATE_EFFECTS:
        requirement = samples_for_hit_rate(effect)
        row = f"  {effect:<16.2%}{requirement.observations:>14,}"
        for horizon in sorted(volatility):
            days = days_to_collect(
                requirement.observations,
                horizon_bars=horizon,
                interval_seconds=seconds,
                symbols=symbol_count,
            )
            row += f"{days:>12,.0f}"
        lines.append(row)
    lines.append("")

    lines.append("-" * 78)
    lines.append("DETECTING A MEAN RETURN ABOVE ZERO")
    lines.append("-" * 78)
    lines.append("  Return volatility measured from real Hyperliquid history:")
    for horizon in sorted(volatility):
        lines.append(f"    h={horizon:<5} sigma = {volatility[horizon]:>10,.1f} bps")
    lines.append("")
    header = f"  {'edge (bps)':<16}" + "".join(f"{'h=' + str(h):>16}" for h in sorted(volatility))
    lines.append(header)
    lines.append(f"  {'':<16}" + "".join(f"{'obs / days':>16}" for _ in volatility))
    for effect in RETURN_EFFECTS_BPS:
        row = f"  {effect:<16.0f}"
        for horizon in sorted(volatility):
            sigma = volatility[horizon]
            if sigma <= 0:
                row += f"{'--':>16}"
                continue
            requirement = samples_for_mean_return(effect, sigma)
            days = days_to_collect(
                requirement.observations,
                horizon_bars=horizon,
                interval_seconds=seconds,
                symbols=symbol_count,
            )
            row += f"{requirement.observations:>9,} /{days:>5,.0f}"
        lines.append(row)
    lines.append("")
    lines.append(
        f"  An edge below {round_trip_bps:.1f} bps is not tradeable at all, so the rows"
    )
    lines.append("  below that threshold are shown only to bound the problem.")
    lines.append("=" * 78)
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    """Measure volatility and print the plan."""
    volatility = await measure_volatility(
        args.symbols,
        args.interval,
        args.horizons,
        cache_dir=args.cache_dir,
        days=args.days,
    )
    round_trip = 2.0 * settings.paper.taker_fee_bps
    print(
        render(
            volatility,
            interval=args.interval,
            symbol_count=len(args.symbols),
            round_trip_bps=round_trip,
        )
    )
    return 0


def cli() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description="Recording sample-size plan.")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 20, 100])
    parser.add_argument("--days", type=int, default=240)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    configure_logging("sample_plan", level="WARNING")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    cli()
