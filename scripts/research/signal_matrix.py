"""Signal x horizon results matrix.

Phase 4, items 2-4.  Runs every backtestable signal against every forward
horizon, one signal at a time with no blending, and prints a matrix with the
Brier score and net PnL in each cell.

    python -m scripts.research.signal_matrix
    python -m scripts.research.signal_matrix --horizons 1 5 20 100 --days 240
    python -m scripts.research.signal_matrix --csv results.csv

Reading the output
------------------
A cell is flagged only if it beats its baseline *after* correction for the size
of the search.  Scanning eight signals across four horizons is thirty-two
hypothesis tests; at alpha = 0.05 about one and a half will look significant on
pure noise.  Raw p-values are shown too, so the gap between "looked good" and
"survived correction" is visible rather than hidden.

Signals whose inputs Hyperliquid does not publish historically are listed
separately as NO DATA rather than silently omitted from the matrix.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from libs.logging_config import configure_logging
from services.research.data import DEFAULT_CACHE_DIR, load_candles, load_funding
from services.research.signals import (
    SIGNALS,
    DataRequirement,
    SignalSpec,
    get_signal,
    signal_names,
)
from services.research.statistics import benjamini_hochberg, bonferroni
from services.research.study import ForwardHorizonStudy, StudyResult, pool

logger = logging.getLogger("signal_matrix")

DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 20, 100)


@dataclass(slots=True)
class Cell:
    """One (signal, horizon) result plus its corrected significance."""

    signal: str
    horizon: int
    result: StudyResult
    hit_significant: bool = False
    gross_significant: bool = False
    net_significant: bool = False

    @property
    def usable(self) -> bool:
        """Return whether the cell has any independent observations."""
        return self.result.independent_size > 0


async def build_matrix(
    *,
    signals: list[str],
    horizons: list[int],
    symbols: list[str],
    interval: str,
    days: int,
    cost_bps: float | None,
    cache_dir: Path,
    refresh: bool,
) -> list[Cell]:
    """Run every signal against every horizon.

    Candles are loaded once per symbol and reused across the whole grid.

    Args:
        signals: Signal names to test.
        horizons: Forward horizons in bars.
        symbols: Symbols to pool over.
        interval: Candle interval.
        days: History window.
        cost_bps: Cost per side; defaults to the configured taker fee.
        cache_dir: Cache directory.
        refresh: Refetch instead of using the cache.

    Returns:
        Every cell, with FDR-corrected significance flags applied.
    """
    candles = {
        symbol: await load_candles(
            symbol, interval, days=days, cache_dir=cache_dir, refresh=refresh
        )
        for symbol in symbols
    }
    needs_funding = any(DataRequirement.FUNDING in get_signal(n).requires for n in signals)
    funding = {}
    if needs_funding:
        for symbol in symbols:
            funding[symbol] = await load_funding(
                symbol, days=days, cache_dir=cache_dir, refresh=refresh
            )

    cells: list[Cell] = []
    for name in signals:
        spec: SignalSpec = get_signal(name)
        for horizon in horizons:
            study = ForwardHorizonStudy(horizon=horizon, cost_bps=cost_bps)
            per_symbol = [
                study.run(spec, symbol, candles[symbol], funding=funding.get(symbol, ()))
                for symbol in symbols
            ]
            cells.append(Cell(signal=name, horizon=horizon, result=pool(per_symbol)))
            logger.info("Completed %s @ %d", name, horizon)

    _apply_corrections(cells)
    return cells


def _apply_corrections(cells: list[Cell]) -> None:
    """Apply Benjamini-Hochberg FDR control across the whole matrix.

    Three families are corrected separately, because they answer different
    questions:

    ``hit`` and ``gross``
        Does the signal carry information? These are the edge tests.
    ``net``
        Does it make money after fees? This one needs care: with a large sample
        and no real edge, mean net return converges on *minus the round-trip
        cost*, which is reliably non-zero. A significant negative net result is
        therefore usually detecting the fee, not the signal, and is only
        interesting alongside a significant gross result.

    Args:
        cells: Cells to annotate in place.
    """
    usable = [cell for cell in cells if cell.usable]
    if not usable:
        return
    hit_flags = benjamini_hochberg([c.result.hit_rate_test.p_value for c in usable])
    gross_flags = benjamini_hochberg([c.result.gross_return_test.p_value for c in usable])
    net_flags = benjamini_hochberg([c.result.net_return_test.p_value for c in usable])
    for cell, hit, gross, net in zip(usable, hit_flags, gross_flags, net_flags, strict=True):
        cell.hit_significant = hit
        cell.gross_significant = gross
        cell.net_significant = net


def render_matrix(cells: list[Cell], horizons: list[int]) -> str:
    """Render the Brier / net-PnL matrix.

    Args:
        cells: Completed cells.
        horizons: Horizons, in column order.

    Returns:
        The printable matrix.
    """
    by_key = {(cell.signal, cell.horizon): cell for cell in cells}
    signals = sorted({cell.signal for cell in cells}, key=lambda n: list(SIGNALS).index(n))

    width = 20
    header = f"{'signal':<{width}}" + "".join(f"{'h=' + str(h):>18}" for h in horizons)
    lines = [header, "-" * len(header)]

    for name in signals:
        brier_row = f"{name:<{width}}"
        pnl_row = f"{'  brier / net bps':<{width}}" if False else f"{'':<{width}}"
        for horizon in horizons:
            cell = by_key.get((name, horizon))
            if cell is None or not cell.usable:
                brier_row += f"{'--':>18}"
                pnl_row += f"{'':>18}"
                continue
            flag = "*" if (cell.hit_significant or cell.gross_significant) else " "
            brier_row += f"{cell.result.brier:>17.4f}{flag}"
            pnl_row += f"{cell.result.mean_net_bps:>+16.2f}b "
        lines.append(brier_row)
        lines.append(pnl_row)
    lines.append("")
    lines.append("  Top line per signal: Brier score (0.2500 = coin flip; lower is better).")
    lines.append("  Second line: mean net return per independent observation, in bps.")
    lines.append(
        "  * = edge (hit rate or gross return) survives Benjamini-Hochberg FDR"
    )
    lines.append(
        "      correction across the matrix. Net-return significance is NOT flagged"
    )
    lines.append(
        "      here: with a large sample it just detects the fee. See DETAIL."
    )
    return "\n".join(lines)


def render_detail(cells: list[Cell]) -> str:
    """Render the per-cell detail table with raw and corrected significance."""
    header = (
        f"{'signal':<20}{'h':>5}{'n_ind':>7}"
        f"{'hit%':>8}{'brier':>9}{'gross bps':>11}{'net bps':>10}"
        f"{'p_hit':>9}{'p_gross':>9}{'p_net':>9}  edge?"
    )
    lines = [header, "-" * len(header)]
    for cell in cells:
        result = cell.result
        if not cell.usable:
            lines.append(
                f"{cell.signal:<20}{cell.horizon:>5}{0:>7}"
                f"{'--':>8}{'--':>9}{'--':>11}{'--':>10}"
                f"{'--':>9}{'--':>9}{'--':>9}  no observations"
            )
            continue
        flags = []
        if cell.hit_significant:
            flags.append("HIT")
        if cell.gross_significant:
            flags.append("GROSS")
        if not flags and (
            result.hit_rate_test.p_value < 0.05
            or result.gross_return_test.p_value < 0.05
        ):
            flags.append("raw-only")
        lines.append(
            f"{cell.signal:<20}{cell.horizon:>5}"
            f"{result.independent_size:>7}"
            f"{result.gross_hit_rate * 100:>8.2f}{result.brier:>9.4f}"
            f"{result.mean_gross_bps:>+11.2f}{result.mean_net_bps:>+10.2f}"
            f"{result.hit_rate_test.p_value:>9.4f}"
            f"{result.gross_return_test.p_value:>9.4f}"
            f"{result.net_return_test.p_value:>9.4f}"
            f"  {', '.join(flags) if flags else ''}"
        )
    return "\n".join(lines)


def render_verdict(cells: list[Cell]) -> str:
    """Summarise which signals, if any, carry an edge that survives correction."""
    usable = [c for c in cells if c.usable]
    edge = [c for c in usable if c.hit_significant or c.gross_significant]
    raw_only = [
        c
        for c in usable
        if c not in edge
        and (
            c.result.hit_rate_test.p_value < 0.05
            or c.result.gross_return_test.p_value < 0.05
        )
    ]
    tradeable = [c for c in edge if c.result.mean_net_bps > 0]

    lines = ["=" * 78, "VERDICT", "=" * 78]
    lines.append(f"  cells tested                    : {len(usable)}")
    lines.append(
        f"  expected false positives        : {0.05 * len(usable):.1f} at raw p < 0.05"
    )
    lines.append(f"  raw p < 0.05 on an edge test    : {len(edge) + len(raw_only)}")
    lines.append(f"  survive FDR correction          : {len(edge)}")
    lines.append(f"  ...and are profitable after fees: {len(tradeable)}")
    lines.append("")

    if tradeable:
        lines.append("  TRADEABLE (edge survives correction AND beats costs):")
        for cell in sorted(tradeable, key=lambda c: -c.result.mean_net_bps):
            result = cell.result
            lines.append(
                f"    {cell.signal} @ h={cell.horizon}: "
                f"hit {result.gross_hit_rate:.1%}, gross {result.mean_gross_bps:+.2f} bps, "
                f"net {result.mean_net_bps:+.2f} bps, n={result.independent_size}"
            )
    else:
        lines.append("  NOTHING IS TRADEABLE.")
        lines.append(
            "  No signal has an edge that survives correction and also clears costs."
        )
    lines.append("")

    if edge:
        lines.append("  Edge survives correction, but does NOT clear costs:")
        for cell in sorted(edge, key=lambda c: c.result.hit_rate_test.p_value):
            if cell in tradeable:
                continue
            result = cell.result
            bias = "below" if result.gross_hit_rate < 0.5 else "above"
            lines.append(
                f"    {cell.signal} @ h={cell.horizon}: hit {result.gross_hit_rate:.1%} "
                f"({bias} 50%), gross {result.mean_gross_bps:+.2f} bps, "
                f"net {result.mean_net_bps:+.2f} bps, n={result.independent_size}, "
                f"p_hit={result.hit_rate_test.p_value:.4f}, "
                f"p_gross={result.gross_return_test.p_value:.4f}"
            )
        asymmetric = [
            c
            for c in edge
            if c not in tradeable
            and c.result.gross_hit_rate < 0.5
            and c.result.mean_gross_bps >= 0.0
        ]
        lines.append("")
        if asymmetric:
            lines.append(
                "    CAUTION -- do not simply invert these. The cells below are right"
            )
            lines.append(
                "    less than half the time yet still have a non-negative mean gross"
            )
            lines.append(
                "    return, which means their wins are larger than their losses."
            )
            lines.append(
                "    Inverting would raise the hit rate and make the expectancy worse:"
            )
            for cell in asymmetric:
                result = cell.result
                lines.append(
                    f"      {cell.signal} @ h={cell.horizon}: "
                    f"hit {result.gross_hit_rate:.1%} but gross "
                    f"{result.mean_gross_bps:+.2f} bps -> inverted gross would be "
                    f"{-result.mean_gross_bps:+.2f} bps"
                )
            lines.append("")
        lines.append(
            "    Where a low hit rate does come with a negative gross return, the"
        )
        lines.append(
            "    inverse reading may be worth testing -- but confirming it on this same"
        )
        lines.append("    data is a second look at the same evidence and needs fresh data.")

    if raw_only:
        lines.append("")
        lines.append("  Raw p < 0.05 but did NOT survive correction (most likely noise):")
        for cell in raw_only:
            lines.append(
                f"    {cell.signal} @ h={cell.horizon}: "
                f"p_hit={cell.result.hit_rate_test.p_value:.4f}, "
                f"p_gross={cell.result.gross_return_test.p_value:.4f}"
            )

    net_losers = [c for c in usable if c.net_significant and c.result.mean_net_bps < 0]
    lines.append("")
    lines.append(
        f"  {len(net_losers)} of {len(usable)} cells lose money significantly after fees."
    )
    lines.append(
        "  That is mostly the fee itself: with no edge, mean net return converges on"
    )
    lines.append(
        "  minus the round trip, and a large sample makes that reliably non-zero."
    )

    bonferroni_flags = bonferroni([c.result.gross_return_test.p_value for c in usable])
    survivors = [c for c, ok in zip(usable, bonferroni_flags, strict=True) if ok]
    lines.append("")
    lines.append(
        f"  Under the stricter Bonferroni correction: {len(survivors)} cell(s) show a"
    )
    lines.append("  gross edge.")
    lines.append("=" * 78)
    return "\n".join(lines)


def write_csv(path: Path, cells: list[Cell]) -> None:
    """Write the full matrix to CSV for further analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "signal",
                "horizon",
                "n_all",
                "n_independent",
                "gross_hit_rate",
                "net_hit_rate",
                "brier",
                "brier_skill",
                "mean_gross_bps",
                "mean_net_bps",
                "total_net_pct",
                "p_hit_rate",
                "p_net_return",
                "p_gross_return",
                "hit_fdr_significant",
                "gross_fdr_significant",
                "net_fdr_significant",
            ]
        )
        for cell in cells:
            result = cell.result
            writer.writerow(
                [
                    cell.signal,
                    cell.horizon,
                    result.sample_size,
                    result.independent_size,
                    f"{result.gross_hit_rate:.6f}",
                    f"{result.net_hit_rate:.6f}",
                    f"{result.brier:.6f}",
                    f"{result.brier_skill_score:.6f}",
                    f"{result.mean_gross_bps:.4f}",
                    f"{result.mean_net_bps:.4f}",
                    f"{result.total_net_pct:.4f}",
                    f"{result.hit_rate_test.p_value:.8f}",
                    f"{result.net_return_test.p_value:.8f}",
                    f"{result.gross_return_test.p_value:.8f}",
                    cell.hit_significant,
                    cell.gross_significant,
                    cell.net_significant,
                ]
            )


async def run(args: argparse.Namespace) -> int:
    """Build and print the matrix.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A process exit code.
    """
    signals = args.signals or list(signal_names())
    blocked = [name for name in SIGNALS if not get_signal(name).is_backtestable]

    cells = await build_matrix(
        signals=signals,
        horizons=args.horizons,
        symbols=args.symbols,
        interval=args.interval,
        days=args.days,
        cost_bps=args.cost_bps,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
    )

    first = cells[0].result if cells else None
    span = ""
    if first and first.first_bar and first.last_bar:
        span = f"  [{first.first_bar:%Y-%m-%d} to {first.last_bar:%Y-%m-%d}]"

    print("=" * 78)
    print("SIGNAL x HORIZON MATRIX -- one signal at a time, no blending")
    print("=" * 78)
    print(f"  symbols   : {', '.join(args.symbols)}  ({args.interval}){span}")
    print(f"  horizons  : {args.horizons} bars")
    cost = cells[0].result.cost_bps if cells else 0.0
    print(f"  costs     : {cost} bps/side ({cost * 2:.1f} bps round trip, Hyperliquid taker)")
    print("=" * 78)
    print()
    print(render_matrix(cells, args.horizons))
    print()
    print("=" * 78)
    print("DETAIL")
    print("=" * 78)
    print(render_detail(cells))
    print()
    print(render_verdict(cells))

    if blocked:
        print()
        print("NOT TESTED -- no historical data published by Hyperliquid:")
        for name in blocked:
            spec = get_signal(name)
            print(f"    {name:<22} needs {', '.join(spec.missing_data)}")
        print("    Both would have to be recorded forward from now to be measurable.")

    if args.csv:
        write_csv(args.csv, cells)
        print(f"\nWrote {args.csv}")
    return 0


def cli() -> None:
    """Parse arguments and run the matrix."""
    parser = argparse.ArgumentParser(description="Signal x horizon research matrix.")
    parser.add_argument("--signals", nargs="+", default=None, help="Defaults to all testable.")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--days", type=int, default=240)
    parser.add_argument("--cost-bps", type=float, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    configure_logging("signal_matrix", level="WARNING")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    cli()
