# CLAUDE.md — Read this first, every session.

## What this project is
AI-powered quantitative crypto trading platform. Fully autonomous — it
analyzes markets and places trades by itself. Paper trading by default,
live trading gated behind a go-live checklist.

## Current status
- Phase 0 (foundations) ✅ complete
- Phase 1 (data ingestion) ✅ complete
- Phase 2 (first agent vertical slice) 🔲 IN PROGRESS — start here

## Key decision: Hyperliquid, not Binance
We switched from Binance to Hyperliquid (decentralized perp DEX).
Reason: Binance blocks Iranian nationals (KYC/OFAC). Hyperliquid is
a smart contract — no KYC, connects via MetaMask wallet, full algo API.
All Binance references in Phase 1 files need replacing with Hyperliquid.

## Architecture rules — never violate these
- TRADING_MODE=paper until go-live checklist. Never live by default.
- No agent calls another agent directly. All comms via Kafka topics.
- Every agent publishes AgentSignal (confidence=0 neutral on error).
- AgentContext is pre-loaded by Orchestrator — agents never self-fetch.
- All events are immutable (frozen=True on BaseEvent).
- Brier score drives adaptive agent trust weights.

## What to build in Phase 2 (in this order)
1. services/data_ingestion/market_data/hyperliquid_collector.py
   — replaces collector.py (was Binance WS, now Hyperliquid WS)
2. services/data_ingestion/market_data/hyperliquid_orderbook.py
   — replaces orderbook.py
3. services/agents/market_analyst/agent.py
   — full Market Analyst Agent (RSI, MACD, EMA, multi-timeframe,
     support/resistance, bollinger bands, volume analysis)
4. services/decision_engine/engine.py
   — single-agent passthrough mode for now
5. services/agents/risk_manager/agent.py
   — veto logic, position sizing, daily loss limits
6. services/agents/execution/paper_broker.py
   — paper trading only, simulates fills at live Hyperliquid prices
7. services/agents/portfolio_manager/tracker.py
   — tracks paper portfolio equity, PnL, open positions

## Tech stack quick reference
- Python 3.12, Pydantic v2, asyncio
- Kafka: Redpanda (docker-compose.yml)
- DB: TimescaleDB (price data), Redis (cache/state)
- Agents: inherit BaseAgent from services/agents/common/base_agent.py
- Schemas: all in libs/schemas/ — import from there, never redefine
- Config: libs/config.py → settings singleton
- All Kafka topics listed in libs/kafka_client/__init__.py

## Hyperliquid API references
- WebSocket: wss://api.hyperliquid.xyz/ws
- REST: https://api.hyperliquid.xyz/info
- Docs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers
- No API key needed for market data (public endpoints)
- Trading requires wallet private key (paper mode uses simulation only)
- Symbols format: "BTC", "ETH", "SOL" (no slash, no USDT suffix)
- Perp markets only (perpetual futures, not spot)

## Folder structure reminder
trading-platform/
├── libs/schemas/          ← shared Pydantic models, never edit lightly
├── libs/kafka_client/     ← KafkaProducer, KafkaConsumer
├── libs/config.py         ← settings singleton
├── services/agents/common/base_agent.py  ← BaseAgent, AgentContext
├── services/agents/market_analyst/       ← build here next
├── services/agents/risk_manager/         ← build here next
├── services/agents/execution/            ← build here next
├── services/decision_engine/             ← build here next
├── services/data_ingestion/market_data/  ← replace with Hyperliquid
└── docker-compose.yml     ← run: docker compose up -d

## How to run (once Phase 2 is built)
1. docker compose up -d          # start infrastructure
2. python -m services.data_ingestion.market_data.hyperliquid_collector
3. python -m services.agents.market_analyst.agent
4. python -m services.decision_engine.engine
5. python -m services.agents.risk_manager.agent
6. python -m services.agents.execution.paper_broker

## Code standards
- Type hints everywhere, mypy strict
- Every class/function has a docstring
- No hardcoded values — everything in config or passed as args
- Production-quality, not prototypes
- Clean architecture, SOLID principles
