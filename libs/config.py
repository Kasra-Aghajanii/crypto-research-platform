"""Central configuration for the trading platform.

Every tunable value in the platform is declared here and read from the
environment (or a local ``.env`` file).  Nothing outside this module may
hardcode an endpoint, a limit or a credential -- see the "no hardcoded values"
rule in ``CLAUDE.md``.

The module exposes a single process-wide ``settings`` instance::

    from libs.config import settings
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(StrEnum):
    """Execution mode of the platform.

    ``PAPER`` simulates every fill locally.  ``LIVE`` routes orders to the real
    Hyperliquid exchange and is gated behind the go-live checklist; the default
    is always ``PAPER``.
    """

    PAPER = "paper"
    LIVE = "live"


class KafkaSettings(BaseSettings):
    """Connection settings for the Redpanda/Kafka event bus."""

    model_config = SettingsConfigDict(env_prefix="KAFKA_", extra="ignore")

    bootstrap_servers: str = Field(
        default="localhost:19092",
        description="Comma-separated Kafka/Redpanda bootstrap servers.",
    )
    client_id: str = Field(default="trading-platform", description="Kafka client identifier.")
    consumer_group_prefix: str = Field(
        default="tp", description="Prefix applied to every consumer group id."
    )
    auto_offset_reset: str = Field(
        default="latest",
        description="Where a new consumer group starts reading: latest or earliest.",
    )
    max_poll_records: int = Field(default=500, ge=1, description="Max records per poll batch.")
    producer_linger_ms: int = Field(
        default=5, ge=0, description="Producer batching linger in milliseconds."
    )


class HyperliquidSettings(BaseSettings):
    """Endpoints and instrument selection for the Hyperliquid perp DEX.

    Market data endpoints are public -- no API key is required.  A wallet key is
    only needed for live order placement and must stay unset in paper mode.
    """

    model_config = SettingsConfigDict(env_prefix="HYPERLIQUID_", extra="ignore")

    ws_url: str = Field(
        default="wss://api.hyperliquid.xyz/ws", description="Hyperliquid WebSocket endpoint."
    )
    rest_url: str = Field(
        default="https://api.hyperliquid.xyz/info", description="Hyperliquid REST info endpoint."
    )
    symbols: list[str] = Field(
        default_factory=lambda: ["BTC", "ETH", "SOL"],
        description="Perp coins to subscribe to. Hyperliquid format: no slash, no USDT suffix.",
    )
    candle_intervals: list[str] = Field(
        default_factory=lambda: ["1m", "5m", "15m", "1h"],
        description="Candle intervals to collect, ordered from fastest to slowest.",
    )
    orderbook_depth: int = Field(
        default=20, ge=1, le=100, description="Order book levels retained per side."
    )
    publish_unclosed_candles: bool = Field(
        default=False,
        description="Also publish in-progress candle updates. Agents only act on closed "
        "candles, so this is off by default to keep bus volume down.",
    )
    ws_ping_interval_s: float = Field(
        default=30.0, gt=0, description="Interval between application-level WS pings."
    )
    ws_receive_timeout_s: float = Field(
        default=90.0,
        gt=0,
        description="Reconnect if no message arrives within this window (dead-socket detection).",
    )
    reconnect_initial_delay_s: float = Field(
        default=1.0, gt=0, description="First reconnect backoff delay."
    )
    reconnect_max_delay_s: float = Field(
        default=60.0, gt=0, description="Ceiling for exponential reconnect backoff."
    )
    wallet_private_key: SecretStr | None = Field(
        default=None,
        description="Wallet key for LIVE trading only. Must be None in paper mode.",
    )

    @field_validator("symbols", "candle_intervals", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow comma-separated environment values for list fields."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


class StorageSettings(BaseSettings):
    """TimescaleDB and Redis connection settings."""

    model_config = SettingsConfigDict(extra="ignore")

    timescale_dsn: str = Field(
        default="postgresql://trader:trader@localhost:5432/marketdata",
        description="TimescaleDB DSN for price history.",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0", description="Redis URL for cache and agent state."
    )


class PaperBrokerSettings(BaseSettings):
    """Fill-simulation parameters for the paper execution agent."""

    model_config = SettingsConfigDict(env_prefix="PAPER_", extra="ignore")

    starting_equity: float = Field(
        default=10_000.0, gt=0, description="Starting paper account equity in USD."
    )
    taker_fee_bps: float = Field(
        default=4.5, ge=0, description="Taker fee in basis points applied to every simulated fill."
    )
    slippage_bps: float = Field(
        default=1.0,
        ge=0,
        description="Extra adverse slippage in bps applied on top of the book price.",
    )
    max_book_age_s: float = Field(
        default=5.0,
        gt=0,
        description="Reject a simulated fill if the last order book is older than this.",
    )


class RiskSettings(BaseSettings):
    """Hard risk limits enforced by the Risk Manager agent."""

    model_config = SettingsConfigDict(env_prefix="RISK_", extra="ignore")

    max_position_notional: float = Field(
        default=2_500.0, gt=0, description="Maximum notional USD per single position."
    )
    max_portfolio_leverage: float = Field(
        default=3.0, gt=0, description="Maximum gross notional / equity across the portfolio."
    )
    max_open_positions: int = Field(
        default=3, ge=1, description="Maximum number of simultaneously open positions."
    )
    daily_loss_limit_pct: float = Field(
        default=3.0,
        gt=0,
        description="Kill switch: halt new entries after this percent of daily equity loss.",
    )
    risk_per_trade_pct: float = Field(
        default=1.0, gt=0, description="Percent of equity risked per trade for position sizing."
    )
    min_confidence: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Signals below this confidence are vetoed outright.",
    )
    default_stop_distance_pct: float = Field(
        default=1.5,
        gt=0,
        description="Fallback stop distance (percent) when a signal carries no stop level.",
    )
    min_order_notional: float = Field(
        default=10.0, gt=0, description="Do not emit orders smaller than this notional."
    )
    trailing_stop_pct: float | None = Field(
        default=None,
        description="Trailing stop distance in percent attached to new orders. None means "
        "a fixed stop only.",
    )
    min_stop_distance_pct: float = Field(
        default=0.15,
        gt=0,
        description="Stops closer than this percent of the entry price are rejected as "
        "degenerate: they would size the position enormously and be hit by noise.",
    )


class MarketAnalystSettings(BaseSettings):
    """Indicator windows and scoring weights for the Market Analyst agent."""

    model_config = SettingsConfigDict(env_prefix="ANALYST_", extra="ignore")

    warmup_candles: int = Field(
        default=200, ge=50, description="Candles buffered per timeframe before analysis starts."
    )
    rsi_period: int = Field(default=14, ge=2, description="RSI lookback period.")
    rsi_overbought: float = Field(default=70.0, description="RSI level considered overbought.")
    rsi_oversold: float = Field(default=30.0, description="RSI level considered oversold.")
    macd_fast: int = Field(default=12, ge=1, description="MACD fast EMA period.")
    macd_slow: int = Field(default=26, ge=2, description="MACD slow EMA period.")
    macd_signal: int = Field(default=9, ge=1, description="MACD signal EMA period.")
    ema_fast: int = Field(default=20, ge=1, description="Fast trend EMA period.")
    ema_slow: int = Field(default=50, ge=2, description="Slow trend EMA period.")
    bollinger_period: int = Field(default=20, ge=2, description="Bollinger band SMA period.")
    bollinger_std: float = Field(default=2.0, gt=0, description="Bollinger band standard devs.")
    atr_period: int = Field(default=14, ge=1, description="ATR lookback period.")
    volume_ma_period: int = Field(default=20, ge=1, description="Volume moving-average period.")
    swing_lookback: int = Field(
        default=3, ge=1, description="Bars on each side required to confirm a swing pivot."
    )
    support_resistance_levels: int = Field(
        default=3, ge=1, description="Number of support/resistance levels reported per side."
    )
    level_cluster_pct: float = Field(
        default=0.35, gt=0, description="Percent distance within which pivots merge into one level."
    )
    timeframe_weights: dict[str, float] = Field(
        default_factory=lambda: {"1m": 0.10, "5m": 0.20, "15m": 0.30, "1h": 0.40},
        description="Confluence weight per timeframe; higher timeframes dominate.",
    )
    signal_ttl_s: float = Field(
        default=90.0, gt=0, description="Seconds before an emitted AgentSignal is considered stale."
    )
    min_abs_score: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Absolute confluence score below which the analyst stays flat.",
    )
    full_conviction_score: float = Field(
        default=0.45,
        gt=0.0,
        le=1.0,
        description="Confluence score treated as full conviction. A composite of bounded "
        "components saturates well below 1.0, so confidence is scaled against this "
        "rather than read off the raw score.",
    )


class DecisionEngineSettings(BaseSettings):
    """Decision engine behaviour (Phase 2: single-agent passthrough)."""

    model_config = SettingsConfigDict(env_prefix="DECISION_", extra="ignore")

    passthrough_agent: str = Field(
        default="market_analyst",
        description="In passthrough mode, only this agent's signals produce decisions.",
    )
    min_confidence: float = Field(
        default=0.55, ge=0.0, le=1.0, description="Minimum confidence to emit a decision."
    )
    default_trust_weight: float = Field(
        default=1.0,
        gt=0,
        description="Trust weight used before Brier scoring has enough history (Phase 3).",
    )
    cooldown_s: float = Field(
        default=60.0,
        ge=0,
        description="Minimum seconds between two decisions for the same symbol.",
    )


class MonitorSettings(BaseSettings):
    """Position monitor behaviour: exit checks and trailing stops."""

    model_config = SettingsConfigDict(env_prefix="MONITOR_", extra="ignore")

    default_trailing_stop_pct: float | None = Field(
        default=None,
        description="Trailing stop distance in percent applied to positions whose order "
        "carried none. None disables trailing by default.",
    )
    reconcile_adopts_unguarded: bool = Field(
        default=True,
        description="Adopt positions seen in a portfolio snapshot that the monitor is not "
        "already guarding. Off means an unguarded position stays unguarded.",
    )


class LearningSettings(BaseSettings):
    """Outcome attribution and trust adaptation."""

    model_config = SettingsConfigDict(env_prefix="LEARNING_", extra="ignore")

    min_samples_for_weight: int = Field(
        default=30, ge=1, description="Scored outcomes required before a trust weight adapts."
    )
    score_window: int = Field(
        default=200, ge=10, description="Recent outcomes retained per agent for the mean Brier."
    )
    min_weight: float = Field(default=0.25, gt=0, description="Floor on an adapted trust weight.")
    max_weight: float = Field(default=2.0, gt=0, description="Ceiling on an adapted trust weight.")
    attribute_degraded_signals: bool = Field(
        default=False,
        description="Score neutral/degraded (confidence 0) signals. Off by default: they express "
        "no view, so scoring them would dilute the mean Brier with free 0.0s.",
    )


class RecordingSettings(BaseSettings):
    """Market-data recording.

    Order book depth and open interest have no historical endpoint on
    Hyperliquid, so these settings govern data that cannot be re-obtained.  The
    intervals trade storage against resolution; see
    ``docs/PHASE5_RECORDING.md`` for the sample-size arithmetic behind the
    defaults.
    """

    model_config = SettingsConfigDict(env_prefix="RECORDING_", extra="ignore")

    orderbook_interval_s: float = Field(
        default=5.0,
        ge=0.0,
        description="Minimum seconds between stored order book snapshots per symbol. "
        "0 stores every snapshot the collector publishes.",
    )
    perp_metrics_interval_s: float = Field(
        default=60.0,
        gt=0.0,
        description="Seconds between open-interest polls. Open interest moves slowly, "
        "so a minute is ample and keeps the table small.",
    )
    stored_book_levels: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Raw book levels retained per side, so a later change to the "
        "imbalance definition can be applied to already-recorded history.",
    )
    batch_size: int = Field(
        default=200, ge=1, description="Rows buffered before a flush is forced."
    )
    flush_interval_s: float = Field(
        default=10.0, gt=0, description="Seconds between periodic flushes."
    )
    max_retry_buffer: int = Field(
        default=50_000,
        ge=1,
        description="Rows retained in memory for retry when a write fails. Bounded so a "
        "long database outage cannot take the recorder down with it.",
    )


class Settings(BaseSettings):
    """Root settings object aggregating every configuration group."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    trading_mode: TradingMode = Field(
        default=TradingMode.PAPER,
        description="Platform-wide execution mode. Defaults to paper and must stay there "
        "until the go-live checklist is complete.",
    )
    log_level: str = Field(default="INFO", description="Root logging level.")
    persistence_enabled: bool = Field(
        default=True,
        description="Write state to TimescaleDB. Services still start when the database is "
        "unreachable; they log a warning and run in memory.",
    )
    service_name: str = Field(default="trading-platform", description="Logical service name.")

    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    hyperliquid: HyperliquidSettings = Field(default_factory=HyperliquidSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    paper: PaperBrokerSettings = Field(default_factory=PaperBrokerSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    analyst: MarketAnalystSettings = Field(default_factory=MarketAnalystSettings)
    decision: DecisionEngineSettings = Field(default_factory=DecisionEngineSettings)
    monitor: MonitorSettings = Field(default_factory=MonitorSettings)
    learning: LearningSettings = Field(default_factory=LearningSettings)
    recording: RecordingSettings = Field(default_factory=RecordingSettings)

    @property
    def is_paper(self) -> bool:
        """Return ``True`` when the platform is running in paper-trading mode."""
        return self.trading_mode is TradingMode.PAPER

    def require_paper_mode(self, component: str) -> None:
        """Guard used by paper-only components.

        Args:
            component: Human-readable name of the calling component.

        Raises:
            RuntimeError: If the platform is not in paper mode.
        """
        if not self.is_paper:
            raise RuntimeError(
                f"{component} is paper-only but TRADING_MODE={self.trading_mode.value}. "
                "Refusing to start."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build (once) and return the process-wide settings object."""
    return Settings()


settings: Settings = get_settings()

__all__ = [
    "DecisionEngineSettings",
    "HyperliquidSettings",
    "KafkaSettings",
    "LearningSettings",
    "MarketAnalystSettings",
    "MonitorSettings",
    "PaperBrokerSettings",
    "RecordingSettings",
    "RiskSettings",
    "Settings",
    "StorageSettings",
    "TradingMode",
    "get_settings",
    "settings",
]
