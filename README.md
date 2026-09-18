# Quantitative Crypto Research Platform

An event-driven research and paper-trading system for crypto perpetual futures, built to answer one question honestly: **does any of this actually predict price?**

Six phases of testing say no. This repository is the instrument that established that, and the record of how.

**Stack:** Python 3.12 · Kafka (Redpanda) · TimescaleDB · Redis · Pydantic v2 · asyncio · Docker
**Quality:** 445 tests passing · `mypy --strict` clean · `ruff` clean · integration tests against live PostgreSQL/TimescaleDB

---

## Why this repository is worth reading

Most trading repositories show an equity curve going up. This one shows a rigorous negative result, and the three separate occasions where a bug produced a false positive convincing enough to have been deployed.

That is the point. A research system that only ever confirms your hypothesis is not a research system.

**The false discovery that nearly got through:** a cross-sectional momentum strategy reported **+187 bps at p = 0.0031**, positive across four consecutive horizons, three of them surviving Bonferroni correction. Every statistical guard in the pipeline passed it.

It was an indexing bug. Symbols list on the exchange at different dates, so integer offset `100` was June 2023 for BTC and April 2024 for ETC. The strategy was ranking assets against each other **on different calendar dates**, and the entire signal was that misalignment. A sub-period robustness check caught it; correctly aligned, the same cell reads **−44.7 bps at p = 0.34**.

The fix aligns by timestamp and refuses to compare dates a symbol does not have. A regression test constructs six symbols on identical price paths with staggered listing dates — a correct implementation must return exactly zero spread, and the old code returned a large non-zero one.

---

## Architecture

```
   Exchange APIs (WebSocket + REST)
              │
     ┌────────▼────────┐
     │   Collectors     │  candles · order book · perp metrics · funding
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │  Kafka / Redpanda│  immutable, replayable event log
     └────────┬────────┘
              │
   ┌──────────┼──────────┐
   ▼          ▼          ▼
┌──────┐ ┌─────────┐ ┌──────────┐
│Agents│ │Recorder │ │ Research │
└──┬───┘ └────┬────┘ └────┬─────┘
   │          │           │
   ▼          ▼           ▼
┌─────────────────────────────┐
│  Decision Engine → Risk      │
│  Manager (veto) → Execution  │
│  (paper) → Position Monitor  │
└──────────────┬──────────────┘
               ▼
      ┌─────────────────┐
      │  TimescaleDB     │  time-series + attribution + audit
      └─────────────────┘
```

### Design decisions that mattered

**Event-sourced, replay-first.** The Kafka log is the source of truth; database tables are materialised views of it. Backtesting is not a separate code path — historical candles are replayed through the same topics the live collectors publish to, so agents cannot tell the difference. A bug in live trading is a bug in backtesting, by construction.

**Agents never call each other.** Every agent is a pure function of a pre-loaded context, publishing one typed signal to its own topic. No agent fetches its own data mid-analysis. This makes each one independently testable with a mock context, and prevents a slow external call inside one agent from stalling a decision cycle.

**Failure publishes, never silences.** An agent that crashes or times out emits a `confidence = 0.0` neutral signal rather than nothing. The decision cycle stays alive and the engine down-weights the failure, instead of hanging on a missing input.

**The risk manager holds a veto.** No order reaches execution without explicit approval. Paper mode is the infrastructure-level default; live trading requires a separate configuration flag and a manual checklist, not a code path that can be entered by accident.

**Collectors do not write to the database.** A separate recorder consumes Kafka and persists. If the recorder restarts, committed offsets resume it where it stopped. If the database goes down, failed rows are retried per-stream from a bounded buffer, so one stream's failure cannot discard another's. Order book history cannot be re-fetched, which makes this distinction load-bearing rather than academic.

---

## Research findings

Every result below is measured with non-overlapping samples, Benjamini–Hochberg FDR correction across the full search, and Bonferroni reported alongside. Gross and net returns are reported separately.

### Phase 3 — does the blended multi-agent analyst work?

**No.** Validated against real market data rather than synthetic series.

| | Synthetic | Real (1,124 trades) |
|---|---|---|
| Hit rate (net) | 100% | 34.0% |
| Brier score | 0.015 | **0.401** (0.25 = coin flip) |
| Overconfidence | −0.117 | +0.402 |

Stated confidence was worse than a coin flip, and inverted on one symbol. This matters because the risk manager sizes positions off confidence — the system was sizing up hardest precisely where it knew least.

**The synthetic test was a tautology:** a trend-following strategy evaluated on a series that trends by construction. No component is validated on generated data anywhere in this repository now.

### Phase 4 — do the individual signals work at 1h?

**No.** Eight signals × four horizons on real data, one signal at a time, no blending.

| | |
|---|---|
| Cells tested | 32 |
| Raw p < 0.05 | 10 |
| Expected false positives at that threshold | ~1.6 |
| Survive FDR correction | 6 |
| …and profitable after fees | **0** |

Every Brier score exceeded 0.2500. Not one signal's confidence beat a coin flip. All six FDR survivors were significant for hit rates *below* 50% — and inverting them fails, because at h=1 several combine a sub-50% hit rate with non-negative mean gross return. Hit rate and expectancy point in opposite directions, which is why the harness reports both.

**A methodology bug found here:** the first version tested significance on *net* return, and every cell at h=1 lit up. With no edge, mean net return converges on exactly minus the round-trip cost, and n≈15,000 makes that overwhelmingly significant. That test detects the fee, not the signal. Edge is now tested on hit rate and gross return only.

### Phase 5 — do maker fees rescue it? Is there structural carry?

**No, and no.**

Maker rebate rerun: 13 of 32 cells cross break-even — and **0** of them have a gross edge distinguishable from zero (all p > 0.2, most > 0.5). One cell becomes "profitable" while its measured gross edge is *negative*. A cell that turns profitable only because its cost fell is a rounding error with a smaller subtraction applied.

Funding carry (long spot, short perp, delta neutral): never profitable at taker cost across seven holding periods. At **zero cost** — no fees, no slippage, no adverse selection, no risk — the ceiling is **1.1–2.8% annualised**, below what the USDC collateral earns sitting still.

**The constraint that closed the short-horizon regime:** measured return volatility at a 1-minute horizon is **5.1 bps**; the taker round trip is **9.0 bps**. The entire standard deviation of the move is smaller than the cost of capturing it. Even a perfect 1-minute predictor is untradeable on the taker side — an arithmetic result, not an empirical one.

### Phase 6 — does anything work at 3–90 days?

**No.** Nine low-turnover signals × six horizons on 20 perps, plus cross-sectional momentum.

| | |
|---|---|
| Cells with firings | 54 |
| Testable (n_eff ≥ 30) | 45 |
| Edge surviving FDR | **0** |
| Smallest p-value in the matrix | 0.27 |

Two corrections did the heavy lifting:

**Pooling correlated symbols buys far less than it appears to.** Mean pairwise correlation across the 20 perps is 0.609, so `n_effective = (k·n)/(1+(k−1)ρ)` gives a **1.59× multiplier, not 20×**. Every p-value is computed at effective sample size. Nine cells fall below 30 effective observations and are reported UNTESTABLE — a statement about available history, not about those signals.

**The sample is a bull market.** The first run reported `momentum_200d @ 7d` at +96.5 bps, p = 0.022 — against an unconditional 7-day return of +88.1 bps over the same window. Almost pure drift. Every edge is now measured as excess over a direction-adjusted buy-and-hold benchmark, with hit rates tested against the market's own up-rate rather than against 50%. Testing against zero would have scored market beta as skill.

**Data honesty:** the venue backfills daily candles from an index source for dates before it traded. Those bars carry `volume = 0`, `trades = 0`, and no funding. They are real prices but not venue market data. Only traded bars are used — 1,211 median bars per symbol (~3.3 years) rather than the 2,213 available.

### What six phases established

The cost hypothesis is closed from both directions. Phase 5 showed that at 1h the round trip exceeds the entire move; Phase 6 tested the regime where costs are negligible against moves of hundreds of basis points, and found the same nothing.

**Transaction costs were never the binding constraint. The absence of predictive signal was.**

### What this does *not* establish

- Anything about 90-day-plus horizons — those cells are untestable on available history and need an external daily series.
- That these signals fail everywhere. One venue, ~3.3 years, mostly a bull market, one stated convention per signal.
- That the conventions tested are the right ones. `drawdown` was tested as buy-the-dip, `vol_regime` as risk-on; both have defensible opposite readings that would flip every sign.
- Anything about parameter tuning. These are conventional defaults; a search would need its own correction.

---

## Engineering: bugs the tests caught

Each of these would have been invisible in production, and two of them would have produced convincing false results.

**Serialization poisoned every consumer.** `model_dump_json()` includes computed fields, but every event model sets `extra="forbid"` — so candles, fills, signals, verdicts and snapshots serialised correctly and then failed to *decode* coming off Kafka. Every consumer would have logged and skipped them as poison messages. The existing tests never round-tripped through the transport. Fixed by converting 15 `@computed_field` decorators to plain properties, with round-trip tests now covering every registered topic model.

**The recorder discarded good data on a partial failure.** A flush that failed for one stream dropped every stream's buffer. Unacceptable for order book history, which cannot be re-fetched. Now retried per-stream against a bounded buffer.

**Confidence never cleared the decision floor.** A blend of bounded components saturates near 0.45, so every signal sat below the `min_confidence` threshold — the platform would have run indefinitely without ever placing a trade. Found by test, not by watching an empty positions table.

**Bollinger fought the trend.** Reading %B as mean-reversion inside a trend cancelled the trend score, so a strong uptrend scored identically to a flat market. Reversion is now scaled by `1 − |trend|`.

**Momentum read positive in downtrends.** MACD histogram *slope* rises toward zero during a steady decline. Switched to histogram level.

**Backfill silently returned nothing.** `fetch_range` gave up on the first empty window, so fine-interval backfills came back empty rather than erroring.

**O(n²) replay.** The first replay implementation would have taken hours; bounding the context window to the live buffer size brought a full run to 7 seconds.

---

## Repository layout

```
├── libs/
│   ├── schemas/          # Pydantic v2 event contracts, immutable, versioned
│   ├── kafka_client/     # async producer/consumer, idempotent, manual offsets
│   ├── persistence/      # TimescaleDB repositories, NUMERIC not float
│   ├── observability/    # structlog JSON logging, metrics
│   └── config.py         # pydantic-settings, single source
├── services/
│   ├── data_ingestion/   # collectors + recorder
│   ├── agents/           # BaseAgent SDK + individual agents
│   ├── decision_engine/  # ensemble + Brier-weighted trust
│   ├── execution/        # paper broker, position monitor
│   └── backtesting/      # replay engine
├── scripts/
│   ├── migration/        # schema migrations
│   └── research/         # signal matrix, maker analysis, funding carry,
│                         # long-horizon, sample planning
├── tests/                # 445 tests incl. live-database integration
└── docs/                 # phase reports (PHASE3–PHASE6)
```

---

## Running it

```bash
docker compose up -d                          # Redpanda, TimescaleDB, Redis
python -m scripts.migrate                     # apply schema
python -m pytest                              # 445 tests
python -m services.data_ingestion.recorder    # start recording
python -m scripts.research.long_horizon       # reproduce Phase 6
```

Every result in `docs/` is reproducible from a single command, listed at the bottom of each report.

---

## Status

Paper trading only. Nothing in six phases supports allocating capital, and the system has never been connected to live funds.

Order book imbalance and open interest are still recording — they cannot be backtested because the venue publishes only current snapshots, so the series has to be built forward from now. That is the one remaining hypothesis with a tractable timeline: **answerable in days to weeks at 1–15 minute horizons**, versus years at the horizons open interest plausibly operates on.

The honest expectation is that it fails too. It is recording because the data is free and irreplaceable, not because the prior is good.

---

## License

MIT

**Not financial advice.** This is a research system that concluded there was nothing to trade. Treat it as an engineering reference and a case study in negative results.

---

Built by **Kasra Aghajani** � [github.com/Kasra-Aghajanii](https://github.com/Kasra-Aghajanii)

Open to freelance work on real-time data pipelines and streaming infrastructure. Reach me through GitHub.
