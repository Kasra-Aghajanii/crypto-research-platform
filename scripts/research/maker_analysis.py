"""Maker-side rerun of the Phase 4 matrix.

Phase 5, item 2.  **This is the same data as Phase 4 with different arithmetic.**
No new observations, no new signals, no new period.  Nothing here can turn a
signal that carries no information into one that does; the only question is
whether any cell that was close to break-even crosses it once the fee changes
sign.

Phase 4 used the Hyperliquid taker fee (4.5 bps per side, 9.0 bps round trip).
Resting an order earns the maker rate instead, which is materially cheaper and
can be negative for high-volume tiers.  The exact rate depends on 14-day volume
and staking tier, so this sweeps a range of scenarios rather than asserting one
number.

The catch, stated up front because it decides whether any of this is usable:
**a maker fill is not free.** A resting order fills when someone crosses the
spread into it, which is disproportionately when they know something you do not.
That adverse selection is a real cost and is *not* modelled here. A signal that
needs immediacy -- anything reacting to a move already underway -- cannot be
executed passively at all. Treat these numbers as an upper bound on maker
economics, not an estimate of them.

    python -m scripts.research.maker_analysis
    python -m scripts.research.maker_analysis --fees 1.5 0.0 -0.3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from libs.config import settings
from libs.logging_config import configure_logging
from scripts.research.signal_matrix import Cell, build_matrix
from services.research.data import DEFAULT_CACHE_DIR
from services.research.signals import signal_names

logger = logging.getLogger("maker_analysis")

DEFAULT_FEES: tuple[float, ...] = (4.5, 1.5, 0.0, -0.3)
"""Cost per side in bps: taker baseline, base maker, zero fee, maker rebate."""

FEE_LABELS: dict[float, str] = {
    4.5: "taker (Phase 4 baseline)",
    1.5: "maker, base tier",
    0.0: "zero fee (bound)",
    -0.3: "maker rebate (high tier)",
}


@dataclass(frozen=True, slots=True)
class Crossing:
    """A cell whose sign changes between two fee scenarios."""

    signal: str
    horizon: int
    gross_bps: float
    net_at_fee: float
    fee: float
    n_independent: int
    p_gross: float


def summarise(cells: list[Cell], fee: float) -> list[Crossing]:
    """Return the cells that are profitable at this fee level.

    Args:
        cells: Matrix cells computed at ``fee``.
        fee: Cost per side in bps.

    Returns:
        Profitable cells, richest first.
    """
    profitable = [
        Crossing(
            signal=cell.signal,
            horizon=cell.horizon,
            gross_bps=cell.result.mean_gross_bps,
            net_at_fee=cell.result.mean_net_bps,
            fee=fee,
            n_independent=cell.result.independent_size,
            p_gross=cell.result.gross_return_test.p_value,
        )
        for cell in cells
        if cell.usable and cell.result.mean_net_bps > 0
    ]
    return sorted(profitable, key=lambda c: -c.net_at_fee)


async def run(args: argparse.Namespace) -> int:
    """Rerun the matrix at each fee level and report break-even crossings."""
    signals = args.signals or list(signal_names())
    results: dict[float, list[Cell]] = {}

    for fee in args.fees:
        results[fee] = await build_matrix(
            signals=signals,
            horizons=args.horizons,
            symbols=args.symbols,
            interval=args.interval,
            days=args.days,
            cost_bps=fee,
            cache_dir=args.cache_dir,
            refresh=False,
        )
        logger.info("Completed fee level %s bps", fee)

    baseline = results[args.fees[0]]
    first = baseline[0].result if baseline else None
    span = ""
    if first and first.first_bar and first.last_bar:
        span = f"  [{first.first_bar:%Y-%m-%d} to {first.last_bar:%Y-%m-%d}]"

    print("=" * 78)
    print("MAKER-SIDE ANALYSIS -- same data as Phase 4, different fee arithmetic")
    print("=" * 78)
    print(f"  symbols  : {', '.join(args.symbols)}  ({args.interval}){span}")
    print(f"  horizons : {args.horizons} bars")
    print("  NOTE     : no new evidence. Identical observations, identical signals,")
    print("             identical period. Only the cost subtracted from each has changed.")
    print("=" * 78)
    print()

    # Gross return is fee-independent, so report it once.
    print("GROSS EDGE (identical at every fee level -- this is what does not change)")
    print("-" * 78)
    print(f"  {'signal':<22}{'h':>5}{'n_ind':>8}{'gross bps':>12}{'p_gross':>10}")
    for cell in baseline:
        if not cell.usable:
            continue
        print(
            f"  {cell.signal:<22}{cell.horizon:>5}{cell.result.independent_size:>8}"
            f"{cell.result.mean_gross_bps:>+12.2f}"
            f"{cell.result.gross_return_test.p_value:>10.4f}"
        )
    print()

    print("=" * 78)
    print("PROFITABLE CELLS BY FEE LEVEL")
    print("=" * 78)
    for fee in args.fees:
        label = FEE_LABELS.get(fee, "custom")
        profitable = summarise(results[fee], fee)
        round_trip = fee * 2
        print(f"\n  {fee:+.2f} bps/side ({round_trip:+.1f} round trip) -- {label}")
        if not profitable:
            print("      no cell is profitable")
            continue
        for crossing in profitable:
            marker = " *" if crossing.p_gross < 0.05 else "  "
            print(
                f"    {crossing.signal:<22} h={crossing.horizon:<4}"
                f" net {crossing.net_at_fee:>+7.2f} bps"
                f"  (gross {crossing.gross_bps:+.2f}, n={crossing.n_independent},"
                f" p_gross={crossing.p_gross:.4f}){marker}"
            )
    print()

    # Which cells cross break-even between the baseline and the cheapest fee?
    cheapest = args.fees[-1]
    base_profitable = {(c.signal, c.horizon) for c in summarise(baseline, args.fees[0])}
    cheap_profitable = {(c.signal, c.horizon) for c in summarise(results[cheapest], cheapest)}
    newly = cheap_profitable - base_profitable

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  profitable at {args.fees[0]:+.2f} bps/side : {len(base_profitable)}")
    print(f"  profitable at {cheapest:+.2f} bps/side : {len(cheap_profitable)}")
    print(f"  newly crossing break-even    : {len(newly)}")
    print()

    if not newly:
        print("  No cell crosses break-even on the maker side.")
        print("  The fee was not what stood between these signals and profitability.")
    else:
        print("  Cells that cross break-even only because the fee changed:")
        significant = 0
        for signal, horizon in sorted(newly):
            cell = next(
                c for c in results[cheapest] if c.signal == signal and c.horizon == horizon
            )
            p_gross = cell.result.gross_return_test.p_value
            if p_gross < 0.05:
                significant += 1
            print(
                f"    {signal:<22} h={horizon:<4} net {cell.result.mean_net_bps:>+7.2f} bps"
                f"  gross {cell.result.mean_gross_bps:+.2f}  p_gross={p_gross:.4f}"
                f"  n={cell.result.independent_size}"
            )
        print()
        print(
            f"  Of those, {significant} have a gross edge distinguishable from zero"
        )
        print("  (p_gross < 0.05, uncorrected). A cell that is profitable only because")
        print("  its cost fell, while its gross edge remains indistinguishable from")
        print("  noise, is not a finding -- it is a rounding error with a smaller")
        print("  subtraction applied.")

    print()
    print("  Reminder: adverse selection on passive fills is not modelled. These are")
    print("  upper bounds on maker economics, and any signal needing immediacy cannot")
    print("  be executed passively at all.")
    print("=" * 78)
    return 0


def cli() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description="Maker-side rerun of the Phase 4 matrix.")
    parser.add_argument("--signals", nargs="+", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 20, 100])
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--days", type=int, default=240)
    parser.add_argument(
        "--fees",
        nargs="+",
        type=float,
        default=list(DEFAULT_FEES),
        help=f"Cost per side in bps, baseline first (default: {list(DEFAULT_FEES)}). "
        f"Taker is {settings.paper.taker_fee_bps}.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    configure_logging("maker_analysis", level="WARNING")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    cli()
