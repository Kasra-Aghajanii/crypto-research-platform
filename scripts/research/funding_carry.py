"""Structural opportunity scan: funding carry (cash-and-carry).

Phase 5, item 3.  A non-forecasting trade: hold spot, short the perp against it,
collect funding.  It does not predict anything, so it is immune to the finding
that killed Phases 3 and 4.

    python -m scripts.research.funding_carry
    python -m scripts.research.funding_carry --hold 8 24 72 168 --cost-bps 18

Costs default to a round trip on **both** legs at the Hyperliquid taker fee, on
the pessimistic assumption that both are crossed rather than rested.  Use
``--cost-bps`` to explore maker execution.

Statistical treatment is the same as the rest of the harness: non-overlapping
holding windows only, and Benjamini-Hochberg correction across the whole
symbol x holding-period grid.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from libs.config import settings
from libs.logging_config import configure_logging
from services.research.basis import CarryResult, CarryStudy, pool_carry, profile_funding
from services.research.data import DEFAULT_CACHE_DIR, load_funding
from services.research.statistics import benjamini_hochberg, bonferroni

logger = logging.getLogger("funding_carry")

DEFAULT_HOLDS: tuple[int, ...] = (8, 24, 72, 168)
"""Holding periods in hours: a third of a day, a day, three days, a week."""


async def run(args: argparse.Namespace) -> int:
    """Profile funding and simulate the carry trade across the grid."""
    round_trip = (
        args.cost_bps
        if args.cost_bps is not None
        else 4.0 * settings.paper.taker_fee_bps  # two legs, in and out
    )

    funding = {
        symbol: await load_funding(
            symbol, days=args.days, cache_dir=args.cache_dir, refresh=args.refresh
        )
        for symbol in args.symbols
    }

    print("=" * 78)
    print("FUNDING CARRY -- structural, non-forecasting")
    print("=" * 78)
    print(f"  symbols   : {', '.join(args.symbols)}")
    print(f"  holds     : {args.hold} hours")
    print(f"  costs     : {round_trip:.1f} bps round trip across BOTH legs")
    print("  position  : long spot + short perp, delta neutral")
    print("=" * 78)
    print()

    # ---- descriptive profile -----------------------------------------
    print("FUNDING PROFILE")
    print("-" * 78)
    print(
        f"  {'symbol':<8}{'obs':>7}{'days':>7}{'mean/hr':>12}{'annualised':>12}"
        f"{'% positive':>12}{'% at floor':>12}{'mean prem':>11}"
    )
    for symbol in args.symbols:
        profile = profile_funding(symbol, funding[symbol])
        print(
            f"  {symbol:<8}{profile.observations:>7}{profile.days:>7.0f}"
            f"{profile.mean_hourly:>12.8f}{profile.annualized_pct:>11.2f}%"
            f"{profile.positive_fraction:>11.1%}{profile.pinned_fraction:>12.1%}"
            f"{profile.mean_premium_bps:>10.2f}b"
        )
    print()
    print("  'at floor' = funding sitting exactly on Hyperliquid's 0.01%/8h base rate,")
    print("  which is where it settles whenever the premium term is small.")
    print()

    # ---- carry simulation --------------------------------------------
    cells: list[CarryResult] = []
    per_symbol: dict[tuple[str, int], CarryResult] = {}
    for hours in args.hold:
        study = CarryStudy(hours=hours, cost_bps=round_trip)
        results = []
        for symbol in args.symbols:
            result = study.run(symbol, funding[symbol])
            per_symbol[(symbol, hours)] = result
            results.append(result)
        cells.append(pool_carry(results))

    usable = [cell for cell in cells if cell.sample_size > 0]
    win_flags = benjamini_hochberg([cell.win_rate_test.p_value for cell in usable])
    net_flags = benjamini_hochberg([cell.net_return_test.p_value for cell in usable])
    bonf_flags = bonferroni([cell.net_return_test.p_value for cell in usable])
    flags = {
        id(cell): (win, net, bonf)
        for cell, win, net, bonf in zip(usable, win_flags, net_flags, bonf_flags, strict=True)
    }

    print("CARRY RESULTS  (pooled across symbols, non-overlapping windows)")
    print("-" * 78)
    print(
        f"  {'hold':>6}{'n':>6}{'funding':>10}{'basis':>9}{'gross':>9}{'net':>9}"
        f"{'win%':>8}{'ann.net':>10}{'p_net':>9}  flags"
    )
    for cell in cells:
        if cell.sample_size == 0:
            print(f"  {cell.hours:>6}{0:>6}  {cell.note}")
            continue
        win_ok, net_ok, bonf_ok = flags[id(cell)]
        marks = []
        if net_ok:
            marks.append("NET-FDR")
        if bonf_ok:
            marks.append("BONF")
        if win_ok:
            marks.append("WIN-FDR")
        print(
            f"  {cell.hours:>6}{cell.sample_size:>6}"
            f"{cell.mean_funding_bps:>+10.2f}{cell.mean_basis_cost_bps:>+9.2f}"
            f"{cell.mean_gross_bps:>+9.2f}{cell.mean_net_bps:>+9.2f}"
            f"{cell.net_win_rate:>8.1%}{cell.annualized_net_pct:>9.1f}%"
            f"{cell.net_return_test.p_value:>9.4f}  {', '.join(marks)}"
        )
    print()
    print("  funding = collected over the hold; basis = premium change (a cost when")
    print("  positive); gross = funding - basis; net = gross - costs. All in bps.")
    print()

    print("PER SYMBOL")
    print("-" * 78)
    for hours in args.hold:
        for symbol in args.symbols:
            result = per_symbol[(symbol, hours)]
            if result.sample_size == 0:
                continue
            print(
                f"  {symbol:<6} h={hours:<5} n={result.sample_size:<5}"
                f" net={result.mean_net_bps:>+8.2f}bps"
                f" ann={result.annualized_net_pct:>7.1f}%"
                f" win={result.net_win_rate:>6.1%}"
                f" p={result.net_return_test.p_value:.4f}"
            )
    print()

    # ---- verdict ------------------------------------------------------
    profitable = [c for c in usable if c.mean_net_bps > 0]
    significant = [c for c in profitable if flags[id(c)][1]]

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  holding periods tested       : {len(usable)}")
    print(f"  profitable after costs       : {len(profitable)}")
    print(f"  ...and significant after FDR : {len(significant)}")
    print()

    if significant:
        best = max(significant, key=lambda c: c.annualized_net_pct)
        print("  STRUCTURAL EDGE FOUND (subject to the caveats below):")
        for cell in sorted(significant, key=lambda c: -c.annualized_net_pct):
            print(
                f"    hold {cell.hours}h: net {cell.mean_net_bps:+.2f} bps/trade, "
                f"{cell.annualized_net_pct:+.1f}% annualised, "
                f"win rate {cell.net_win_rate:.1%}, n={cell.sample_size}, "
                f"p={cell.net_return_test.p_value:.5f}"
            )
        print()
        print(f"  Best: {best.hours}h hold at {best.annualized_net_pct:+.1f}% annualised.")
    elif profitable:
        print("  Some holding periods are profitable but none survives correction.")
    else:
        print("  NOTHING IS PROFITABLE. Funding does not cover the round trip on both")
        print("  legs at any tested holding period.")

    print()
    print("  UNMODELLED RISKS -- these decide whether the above is real:")
    print("    * Spot leg execution. Hyperliquid spot books are far thinner than the")
    print("      perp books; size is limited and slippage is not modelled here.")
    print("    * Margin and liquidation. The short perp needs collateral, and an")
    print("      adverse move can force a close even when the pair is flat overall.")
    print("    * Funding regime change. A rate positive for months can invert; this")
    print("      measures one period, not a guarantee.")
    print("    * Correlated symbols. Pooling BTC/ETH/SOL understates variance.")
    print("=" * 78)
    return 0


def cli() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description="Funding carry structural scan.")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--hold", nargs="+", type=int, default=list(DEFAULT_HOLDS))
    parser.add_argument("--days", type=int, default=240)
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=None,
        help="Round-trip cost across both legs in bps "
        f"(default: 4 x taker {settings.paper.taker_fee_bps}).",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    configure_logging("funding_carry", level="WARNING")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    cli()
