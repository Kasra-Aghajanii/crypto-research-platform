"""Tests for the Hyperliquid collectors: payload parsing and candle closing."""

from __future__ import annotations

from decimal import Decimal

import pytest

from libs.kafka_client import Topics
from libs.schemas.market import Candle, Side
from services.data_ingestion.market_data.hyperliquid_collector import (
    HyperliquidCollector,
    parse_candle,
    parse_trade,
)
from services.data_ingestion.market_data.hyperliquid_orderbook import (
    HyperliquidOrderBookCollector,
    parse_levels,
    parse_order_book,
)
from services.data_ingestion.market_data.parsing import PayloadError, to_datetime, to_decimal
from tests.conftest import FakeProducer

CANDLE_PAYLOAD = {
    "t": 1_700_000_000_000,
    "T": 1_700_000_059_999,
    "s": "BTC",
    "i": "1m",
    "o": "37000.5",
    "h": "37050.0",
    "l": "36990.25",
    "c": "37020.75",
    "v": "12.34",
    "n": 421,
}

BOOK_PAYLOAD = {
    "coin": "BTC",
    "time": 1_700_000_000_123,
    "levels": [
        [{"px": "36999.0", "sz": "1.5", "n": 3}, {"px": "36998.0", "sz": "2.0", "n": 2}],
        [{"px": "37001.0", "sz": "1.0", "n": 1}, {"px": "37002.0", "sz": "3.0", "n": 4}],
    ],
}


class TestParsingHelpers:
    """Conversion helpers used by both collectors."""

    def test_decimal_conversion_preserves_precision(self) -> None:
        """String prices convert without float rounding."""
        assert to_decimal("37000.123456789", "px") == Decimal("37000.123456789")

    def test_missing_field_raises(self) -> None:
        """A missing numeric field is a payload error, not a crash."""
        with pytest.raises(PayloadError):
            to_decimal(None, "px")

    def test_non_numeric_field_raises(self) -> None:
        """Junk in a numeric field is a payload error."""
        with pytest.raises(PayloadError):
            to_decimal("not-a-number", "px")

    def test_epoch_ms_converts_to_utc(self) -> None:
        """Timestamps arrive as epoch milliseconds and become aware UTC."""
        parsed = to_datetime(1_700_000_000_000, "t")
        assert parsed.tzinfo is not None
        assert parsed.year == 2023


class TestCandleParsing:
    """Candle payload translation."""

    def test_fields_map_correctly(self) -> None:
        """Every OHLCV field lands in the right place."""
        candle = parse_candle(CANDLE_PAYLOAD, is_closed=True)
        assert candle.symbol == "BTC"
        assert candle.interval == "1m"
        assert candle.open == Decimal("37000.5")
        assert candle.close == Decimal("37020.75")
        assert candle.volume == Decimal("12.34")
        assert candle.trade_count == 421
        assert candle.is_closed is True

    def test_missing_symbol_raises(self) -> None:
        """A payload without a symbol cannot become an event."""
        with pytest.raises(PayloadError):
            parse_candle({**CANDLE_PAYLOAD, "s": None}, is_closed=False)

    def test_symbol_rejects_centralised_exchange_format(self) -> None:
        """Binance-style pairs are rejected by the schema validator."""
        with pytest.raises(ValueError, match="Invalid Hyperliquid symbol"):
            parse_candle({**CANDLE_PAYLOAD, "s": "BTCUSDT"}, is_closed=False)

    def test_typical_price_is_computed(self) -> None:
        """The typical price is the mean of high, low and close."""
        candle = parse_candle(CANDLE_PAYLOAD, is_closed=True)
        expected = (candle.high + candle.low + candle.close) / Decimal(3)
        assert candle.typical_price == expected


class TestTradeParsing:
    """Trade payload translation."""

    def test_buy_side_code_maps_to_buy(self) -> None:
        """Hyperliquid encodes an aggressive buy as side 'B'."""
        trade = parse_trade(
            {"coin": "ETH", "side": "B", "px": "2000.5", "sz": "0.4", "time": 1_700_000_000_000}
        )
        assert trade.side is Side.BUY
        assert trade.symbol == "ETH"
        assert trade.size == Decimal("0.4")

    def test_sell_side_code_maps_to_sell(self) -> None:
        """Side 'A' means the ask was hit."""
        trade = parse_trade(
            {"coin": "ETH", "side": "A", "px": "2000.5", "sz": "0.4", "time": 1_700_000_000_000}
        )
        assert trade.side is Side.SELL

    def test_unknown_side_raises(self) -> None:
        """An unrecognised side code is a payload error."""
        with pytest.raises(PayloadError):
            parse_trade(
                {"coin": "ETH", "side": "X", "px": "1", "sz": "1", "time": 1_700_000_000_000}
            )


class TestCandleClosing:
    """The rollover rule that decides when a candle is final."""

    async def test_first_candle_is_not_published_as_closed(
        self, fake_producer: FakeProducer
    ) -> None:
        """The in-progress candle is held back, not published as closed."""
        collector = HyperliquidCollector(producer=fake_producer)
        await collector.handle_candle(CANDLE_PAYLOAD)
        assert fake_producer.events_on(Topics.MARKET_CANDLES) == []

    async def test_rollover_publishes_the_previous_candle_as_closed(
        self, fake_producer: FakeProducer
    ) -> None:
        """A later open time proves the previous candle is final."""
        collector = HyperliquidCollector(producer=fake_producer)
        await collector.handle_candle(CANDLE_PAYLOAD)
        await collector.handle_candle({**CANDLE_PAYLOAD, "c": "37030.0"})  # same candle, updated
        assert fake_producer.events_on(Topics.MARKET_CANDLES) == []

        next_candle = {
            **CANDLE_PAYLOAD,
            "t": CANDLE_PAYLOAD["t"] + 60_000,
            "T": CANDLE_PAYLOAD["T"] + 60_000,
        }
        await collector.handle_candle(next_candle)

        published = fake_producer.events_on(Topics.MARKET_CANDLES)
        assert len(published) == 1
        closed = published[0]
        assert isinstance(closed, Candle)
        assert closed.is_closed is True
        assert closed.close == Decimal("37030.0")  # the latest update, not the first

    async def test_stale_candle_update_is_ignored(self, fake_producer: FakeProducer) -> None:
        """An out-of-order update for an older candle never overwrites state."""
        collector = HyperliquidCollector(producer=fake_producer)
        await collector.handle_candle(CANDLE_PAYLOAD)
        await collector.handle_candle({**CANDLE_PAYLOAD, "t": CANDLE_PAYLOAD["t"] - 60_000})
        assert fake_producer.events_on(Topics.MARKET_CANDLES) == []

    async def test_trades_are_published_per_entry(self, fake_producer: FakeProducer) -> None:
        """A trades batch publishes one event per print."""
        collector = HyperliquidCollector(producer=fake_producer)
        await collector.handle_trades(
            [
                {"coin": "BTC", "side": "B", "px": "1", "sz": "1", "time": 1_700_000_000_000},
                {"coin": "BTC", "side": "A", "px": "2", "sz": "2", "time": 1_700_000_000_001},
            ]
        )
        assert len(fake_producer.events_on(Topics.MARKET_TRADES)) == 2

    async def test_malformed_trade_is_skipped_not_fatal(self, fake_producer: FakeProducer) -> None:
        """One bad print does not discard the good ones."""
        collector = HyperliquidCollector(producer=fake_producer)
        await collector.handle_trades(
            [
                {"coin": "BTC", "side": "?", "px": "1", "sz": "1", "time": 1_700_000_000_000},
                {"coin": "BTC", "side": "B", "px": "2", "sz": "2", "time": 1_700_000_000_001},
            ]
        )
        assert len(fake_producer.events_on(Topics.MARKET_TRADES)) == 1


class TestOrderBook:
    """Order book snapshot parsing and publishing."""

    def test_levels_are_parsed_best_first(self) -> None:
        """Bids arrive best (highest) first and keep that order."""
        book = parse_order_book(BOOK_PAYLOAD, depth=10)
        assert book.best_bid == Decimal("36999.0")
        assert book.best_ask == Decimal("37001.0")
        assert book.mid_price == Decimal("37000.0")

    def test_depth_truncation(self) -> None:
        """Only the requested number of levels is retained."""
        book = parse_order_book(BOOK_PAYLOAD, depth=1)
        assert len(book.bids) == 1
        assert len(book.asks) == 1

    def test_spread_in_basis_points(self) -> None:
        """The spread is reported relative to the mid price."""
        book = parse_order_book(BOOK_PAYLOAD, depth=10)
        spread = book.spread_bps
        assert spread is not None
        assert spread == pytest.approx(Decimal("0.5405"), abs=Decimal("0.001"))

    def test_imbalance_is_signed_toward_the_heavier_side(self) -> None:
        """More resting bid size than ask size gives a positive imbalance."""
        book = parse_order_book(BOOK_PAYLOAD, depth=10)
        assert book.imbalance(depth=1) == pytest.approx(0.2)

    def test_missing_levels_raises(self) -> None:
        """A payload without both sides cannot become a snapshot."""
        with pytest.raises(PayloadError):
            parse_order_book({"coin": "BTC", "levels": []}, depth=5)

    def test_non_list_side_raises(self) -> None:
        """A malformed side is a payload error."""
        with pytest.raises(PayloadError):
            parse_levels("not-a-list", depth=5)

    async def test_one_sided_book_is_dropped(self, fake_producer: FakeProducer) -> None:
        """A book with an empty side cannot price a fill, so it is not published."""
        collector = HyperliquidOrderBookCollector(producer=fake_producer)
        await collector.handle_book({**BOOK_PAYLOAD, "levels": [[], BOOK_PAYLOAD["levels"][1]]})
        assert fake_producer.events_on(Topics.MARKET_ORDERBOOK) == []

    async def test_throttle_suppresses_rapid_updates(self, fake_producer: FakeProducer) -> None:
        """The per-symbol throttle caps republish frequency."""
        collector = HyperliquidOrderBookCollector(
            producer=fake_producer, min_publish_interval_s=60.0
        )
        await collector.handle_book(BOOK_PAYLOAD)
        await collector.handle_book(BOOK_PAYLOAD)
        assert len(fake_producer.events_on(Topics.MARKET_ORDERBOOK)) == 1
        assert collector.snapshots_throttled == 1

    async def test_no_throttle_publishes_every_update(self, fake_producer: FakeProducer) -> None:
        """Setting the interval to zero disables throttling."""
        collector = HyperliquidOrderBookCollector(
            producer=fake_producer, min_publish_interval_s=0.0
        )
        await collector.handle_book(BOOK_PAYLOAD)
        await collector.handle_book(BOOK_PAYLOAD)
        assert len(fake_producer.events_on(Topics.MARKET_ORDERBOOK)) == 2
