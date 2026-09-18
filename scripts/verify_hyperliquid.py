"""Verify the collectors against the live Hyperliquid feed.

Connects to the public WebSocket, captures a few real frames per channel, and
runs them through the production parsers.  Read-only: it subscribes to public
market data and places no orders.

Run with::

    python -m scripts.verify_hyperliquid --seconds 25
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import websockets

from libs.config import settings
from services.data_ingestion.market_data.hyperliquid_collector import parse_candle, parse_trade
from services.data_ingestion.market_data.hyperliquid_orderbook import parse_order_book
from services.data_ingestion.market_data.parsing import PayloadError


async def capture(seconds: float, coin: str, interval: str) -> dict[str, list[Any]]:
    """Capture raw frames per channel for a fixed duration.

    Args:
        seconds: How long to listen.
        coin: Hyperliquid coin to subscribe to.
        interval: Candle interval to subscribe to.

    Returns:
        A mapping of channel name to captured payloads.
    """
    captured: dict[str, list[Any]] = {"candle": [], "trades": [], "l2Book": []}
    subs = [
        {"type": "candle", "coin": coin, "interval": interval},
        {"type": "trades", "coin": coin},
        {"type": "l2Book", "coin": coin},
    ]
    async with websockets.connect(settings.hyperliquid.ws_url, open_timeout=15) as ws:
        for sub in subs:
            await ws.send(json.dumps({"method": "subscribe", "subscription": sub}))
        deadline = asyncio.get_running_loop().time() + seconds
        while asyncio.get_running_loop().time() < deadline:
            remaining = deadline - asyncio.get_running_loop().time()
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(remaining, 0.1))
            except TimeoutError:
                break
            message = json.loads(raw)
            channel = message.get("channel")
            if channel in captured and len(captured[channel]) < 3:
                captured[channel].append(message.get("data"))
    return captured


def report(captured: dict[str, list[Any]]) -> int:
    """Run captured frames through the parsers and print the outcome.

    Args:
        captured: Frames grouped by channel.

    Returns:
        A process exit code: 0 when every channel parsed.
    """
    failures = 0

    for channel, payloads in captured.items():
        if not payloads:
            print(f"[WARN] {channel}: no frames received")
            continue
        sample = payloads[0]
        print(f"\n=== {channel} raw sample ===")
        print(json.dumps(sample, indent=2)[:700])
        try:
            if channel == "candle":
                candle = parse_candle(sample, is_closed=False)
                print(f"[OK] parsed candle {candle.symbol} {candle.interval} close={candle.close}")
            elif channel == "trades":
                entries = sample if isinstance(sample, list) else [sample]
                trade = parse_trade(entries[0])
                print(f"[OK] parsed trade {trade.symbol} {trade.side.value} @ {trade.price}")
            else:
                book = parse_order_book(sample, depth=settings.hyperliquid.orderbook_depth)
                print(
                    f"[OK] parsed book {book.symbol} bid={book.best_bid} ask={book.best_ask} "
                    f"spread_bps={book.spread_bps}"
                )
        except (PayloadError, ValueError, KeyError, IndexError, TypeError) as exc:
            failures += 1
            print(f"[FAIL] {channel}: {type(exc).__name__}: {exc}")

    print(f"\nResult: {'PASS' if failures == 0 else f'{failures} channel(s) FAILED'}")
    return 1 if failures else 0


def main() -> None:
    """Parse arguments, capture frames and report."""
    parser = argparse.ArgumentParser(description="Verify parsers against live Hyperliquid data.")
    parser.add_argument("--seconds", type=float, default=25.0, help="Listen duration.")
    parser.add_argument("--coin", default="BTC", help="Coin to subscribe to.")
    parser.add_argument("--interval", default="1m", help="Candle interval.")
    args = parser.parse_args()

    captured = asyncio.run(capture(args.seconds, args.coin, args.interval))
    raise SystemExit(report(captured))


if __name__ == "__main__":
    main()
