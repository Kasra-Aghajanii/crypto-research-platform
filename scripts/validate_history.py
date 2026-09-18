"""Historical validation: backfill real candles, replay, and report.

Phase 3, item 4.  Answers the question Phase 2 left open -- do the Market
Analyst's indicator weights hold up on real Hyperliquid data, or were they only
ever validated against synthetic series?

Run it::

    python -m scripts.validate_history --symbols BTC ETH --interval 1h --days 120

Data comes from the public REST endpoint and is cached on disk, so repeat runs
are offline and reproducible.  When TimescaleDB is reachable the candles are
also written there; when it is not, the run still completes from the cache --
validation should not be blocked on infrastructure.

The report covers what was asked for: hit rate, confidence calibration, and a
synthetic-versus-real comparison of the same replay.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Final

from libs.config import settings
from libs.logging_config import configure_logging
from libs.persistence import CandleRepository, try_connect
from libs.schemas.market import Candle
from services.backtest.calibration import CalibrationReport, analyze_calibration
from services.backtest.replay import ReplayEngine, ReplayResult
from services.data_ingestion.market_data.hyperliquid_history import (
    HyperliquidHistoryClient,
    cache_path,
    read_cache,
    write_cache,
)

logger = logging.getLogger("validate_history")

DEFAULT_CACHE_DIR: Final[Path] = Path("data/history")


async def obtain_candles(
    symbol: str,
    interval: str,
    *,
    days: int,
    cache_dir: Path,
    refresh: bool,
) -> tuple[Candle, ...]:
    """Load candles from cache, or fetch and cache them.

    Args:
        symbol: Coin to load.
        interval: Candle interval.
        days: How far back to fetch.
        cache_dir: Directory holding JSONL caches.
        refresh: Ignore any existing cache and refetch.

    Returns:
        Candles oldest first.
    """
    path = cache_path(cache_dir, symbol, interval)
    if path.exists() and not refresh:
        candles = read_cache(path)
        logger.info(
            "Loaded candles from cache",
            extra={"symbol": symbol, "interval": interval, "count": len(candles)},
        )
        return candles

    start = datetime.now(tz=UTC) - timedelta(days=days)
    async with HyperliquidHistoryClient() as client:
        candles = await client.fetch_range(symbol, interval, start=start)
    write_cache(path, candles)
    logger.info(
        "Fetched and cached candles",
        extra={"symbol": symbol, "interval": interval, "count": len(candles)},
    )
    return candles


async def persist_candles(candles: tuple[Candle, ...]) -> bool:
    """Write candles to TimescaleDB when it is reachable.

    Args:
        candles: Candles to persist.

    Returns:
        ``True`` if they were written.
    """
    if not candles or not settings.persistence_enabled:
        return False
    database = await try_connect()
    if database is None:
        return False
    try:
        written = await CandleRepository(database).upsert_many(candles)
        logger.info("Persisted candles to TimescaleDB", extra={"count": written})
        return True
    finally:
        await database.close()


def synthetic_series(
    count: int, *, start: float, drift: float, interval: str
) -> tuple[Candle, ...]:
    """Build the same kind of synthetic trend the Phase 2 tests used.

    Used only for the synthetic-versus-real comparison in the report.

    Args:
        count: Number of candles.
        start: Starting price.
        drift: Price change per bar.
        interval: Interval label to stamp.

    Returns:
        Synthetic candles, oldest first.
    """
    base = datetime(2026, 1, 1, tzinfo=UTC)
    candles: list[Candle] = []
    price = start
    for i in range(count):
        wobble = math.sin(i / 3.0) * abs(drift) * 0.8
        open_price = price
        close = price + drift + wobble
        high = max(open_price, close) + abs(drift)
        low = max(min(open_price, close) - abs(drift), 0.01)
        open_time = base + timedelta(hours=i)
        candles.append(
            Candle(
                source="synthetic",
                symbol="SYN",
                interval=interval,
                open_time=open_time,
                close_time=open_time + timedelta(hours=1),
                occurred_at=open_time,
                open=Decimal(str(round(open_price, 6))),
                high=Decimal(str(round(high, 6))),
                low=Decimal(str(round(low, 6))),
                close=Decimal(str(round(close, 6))),
                volume=Decimal("100"),
                trade_count=10,
                is_closed=True,
            )
        )
        price = close
    return tuple(candles)


def format_result(result: ReplayResult, calibration: CalibrationReport) -> str:
    """Render one replay result as a text block.

    Args:
        result: The replay outcome.
        calibration: Calibration of the trades it produced.

    Returns:
        A printable report section.
    """
    lines: list[str] = []
    span = ""
    if result.first_bar and result.last_bar:
        span = f"  {result.first_bar:%Y-%m-%d} to {result.last_bar:%Y-%m-%d}"
    lines.append(f"--- {result.symbol} {result.interval}{span} ---")
    lines.append(
        f"  bars replayed      : {result.bars_replayed}"
        f"   signals scored: {result.signals_evaluated}"
        f"   actionable: {result.signals_actionable}"
    )
    if not result.trades:
        lines.append("  no trades were taken")
        return "\n".join(lines)

    profit_factor = result.profit_factor
    pf_text = "inf" if profit_factor == float("inf") else f"{profit_factor:.2f}"
    lines.append(
        f"  trades             : {len(result.trades)}"
        f"   (forced closes: {sum(1 for t in result.trades if t.forced_close)})"
    )
    lines.append(
        f"  hit rate           : {result.hit_rate:.1%} net"
        f"   /  {result.gross_hit_rate:.1%} gross (before costs)"
    )
    lines.append(
        f"  PnL (1 unit)       : {result.net_pnl:.2f} net"
        f"   /  {result.gross_pnl:.2f} gross"
        f"   costs: {result.total_costs:.2f}"
    )
    lines.append(f"  profit factor      : {pf_text}")
    lines.append(f"  avg bars held      : {result.average_bars_held:.1f}")
    lines.append(f"  exits              : {result.exit_breakdown()}")
    lines.append(
        f"  calibration        : brier {calibration.brier:.4f}"
        f"   ECE {calibration.ece:.4f}"
        f"   mean conf {calibration.mean_confidence:.3f}"
        f"   hit {calibration.hit_rate:.3f}"
        f"   overconfidence {calibration.overconfidence:+.3f}"
    )
    for bucket in calibration.bins:
        lines.append(
            f"      conf {bucket.label}: n={bucket.count:<4d}"
            f" stated {bucket.mean_confidence:.3f}"
            f" actual {bucket.hit_rate:.3f}"
            f" gap {bucket.gap:+.3f}"
        )
    return "\n".join(lines)


def calibration_for(result: ReplayResult) -> CalibrationReport:
    """Build a calibration report from a replay's trades."""
    return analyze_calibration(
        [trade.confidence for trade in result.trades],
        [trade.was_correct for trade in result.trades],
    )


async def run(
    symbols: list[str],
    interval: str,
    days: int,
    cache_dir: Path,
    refresh: bool,
    max_bars_held: int | None,
) -> int:
    """Run the whole validation and print the report.

    Args:
        symbols: Symbols to validate.
        interval: Driving candle interval.
        days: History window in days.
        cache_dir: Cache directory.
        refresh: Refetch instead of using the cache.
        max_bars_held: Force-close trades after this many bars.

    Returns:
        A process exit code.
    """
    engine = ReplayEngine(max_bars_held=max_bars_held)
    sections: list[str] = []
    all_confidences: list[float] = []
    all_outcomes: list[bool] = []
    total_trades = 0

    for symbol in symbols:
        candles = await obtain_candles(
            symbol, interval, days=days, cache_dir=cache_dir, refresh=refresh
        )
        if len(candles) <= engine.warmup_bars:
            sections.append(
                f"--- {symbol} {interval} ---\n"
                f"  SKIPPED: only {len(candles)} candles, need > {engine.warmup_bars}"
            )
            continue
        await persist_candles(candles)

        result = await engine.run(symbol, {interval: candles}, driving_interval=interval)
        report = calibration_for(result)
        sections.append(format_result(result, report))
        total_trades += len(result.trades)
        all_confidences.extend(trade.confidence for trade in result.trades)
        all_outcomes.extend(trade.was_correct for trade in result.trades)

    # --- synthetic comparison, the baseline Phase 2 was tuned against ---
    synthetic = synthetic_series(400, start=1000.0, drift=1.5, interval=interval)
    synthetic_result = await engine.run("SYN", {interval: synthetic}, driving_interval=interval)
    sections.append(format_result(synthetic_result, calibration_for(synthetic_result)))

    print("\n" + "=" * 78)
    print("HISTORICAL VALIDATION -- Market Analyst on real Hyperliquid data")
    print("=" * 78)
    print(f"interval={interval}  days={days}  cost_bps={engine.cost_bps}")
    print(f"decision floor={settings.decision.min_confidence}")
    print(f"full_conviction_score={settings.analyst.full_conviction_score}")
    print("=" * 78)
    for section in sections:
        print(section)
        print()

    if all_confidences:
        overall = analyze_calibration(all_confidences, all_outcomes)
        print("=" * 78)
        print("AGGREGATE ACROSS REAL SYMBOLS")
        print("=" * 78)
        print(f"  trades           : {total_trades}")
        print(f"  hit rate         : {overall.hit_rate:.1%}")
        print(f"  mean confidence  : {overall.mean_confidence:.3f}")
        print(f"  overconfidence   : {overall.overconfidence:+.3f}")
        print(f"  brier            : {overall.brier:.4f}   (0.25 = coin flip)")
        print(f"  ECE              : {overall.ece:.4f}")
        for bucket in overall.bins:
            print(
                f"      conf {bucket.label}: n={bucket.count:<4d}"
                f" stated {bucket.mean_confidence:.3f}"
                f" actual {bucket.hit_rate:.3f}"
                f" gap {bucket.gap:+.3f}"
            )
    else:
        print("No trades were taken on real data; nothing to calibrate.")
    print()
    return 0


def cli() -> None:
    """Parse arguments and run the validation."""
    parser = argparse.ArgumentParser(description="Validate the analyst on real history.")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh", action="store_true", help="Refetch instead of using cache.")
    parser.add_argument(
        "--max-bars-held",
        type=int,
        default=None,
        help="Force-close trades after this many bars (default: run to resolution).",
    )
    args = parser.parse_args()

    configure_logging("validate_history", level="WARNING")
    raise SystemExit(
        asyncio.run(
            run(
                args.symbols,
                args.interval,
                args.days,
                args.cache_dir,
                args.refresh,
                args.max_bars_held,
            )
        )
    )


if __name__ == "__main__":
    cli()
