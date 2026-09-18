# Phase 5 — recording, maker economics, structural scan

Three questions, in the order they were asked. Two of them come up empty, and
the third is a data-collection job that has now started.

## Summary

| Item | Result |
|---|---|
| 1. Record order book + open interest | **Running.** Both collectors and the recorder are built, verified against live TimescaleDB, and started from zero history. |
| 2. Maker-side rerun of Phase 4 | **Empty.** 13 cells cross break-even at a maker rebate; **0** have a gross edge distinguishable from noise. |
| 3. Funding carry (structural) | **Empty.** Ceiling is 1.1–2.8% annualised *before any cost*, and not significant at achievable sample sizes. |

Items 2 and 3 both come up empty. The recording in item 1 is the only thing here
that changes what is knowable later.

---

## 1. Recording the two untestable signals

### What runs

| Component | Role |
|---|---|
| `services/data_ingestion/market_data/hyperliquid_orderbook.py` | Existing collector; publishes L2 book snapshots to Kafka |
| `services/data_ingestion/market_data/hyperliquid_perp_metrics.py` | **New.** Polls `metaAndAssetCtxs` for open interest |
| `services/data_ingestion/recorder.py` | **New.** Consumes both topics and writes TimescaleDB |
| `scripts/migration/0002_market_recording.py` | **New.** `orderbook_snapshots`, `perp_metrics`, `funding_rates` |

Open interest arrives in the same payload as mark price, oracle price, premium
and funding, so all of it is recorded — the row costs the same, and premium
against oracle is the live basis that item 3 needs.

### Why a separate recorder rather than writing from the collectors

Two failure modes, handled differently:

- **Recorder restart** — Kafka offsets are committed as messages are consumed, so
  a restarted recorder resumes where it left off and Redpanda's retention covers
  the gap. A collector writing straight to the database would drop everything
  produced while it was down.
- **Database outage** — offsets have already advanced, so Kafka will not replay.
  Failed rows are therefore retained in memory and retried on the next flush,
  and each stream is written independently so a failure in one cannot discard
  the others. The buffer is bounded at 50,000 rows. The honest guarantee is
  *survives a transient outage*, not *survives any outage*.

### Verified, not assumed

Migration applied to the live TimescaleDB; collectors polled the live API; the
full pipeline ran end to end for 75 seconds:

```
orderbook_snapshots:  BTC/ETH/SOL  14 rows each over 69s  (5s throttle)
perp_metrics:         BTC/ETH/SOL   2 rows each over 60s  (60s poll)
```

18 integration tests run against the real database.

### Storage and defaults

| Setting | Default | Rationale |
|---|---|---|
| `RECORDING_ORDERBOOK_INTERVAL_S` | 5.0 | ~52k rows/day across 3 symbols |
| `RECORDING_PERP_METRICS_INTERVAL_S` | 60.0 | OI moves slowly; a minute is ample |
| `RECORDING_STORED_BOOK_LEVELS` | 10 | Raw levels kept so a later change to the imbalance definition can be applied to already-recorded history |

Expect roughly 15–20 MB per symbol per day for the order book, and a rounding
error for perp metrics.

### Time to a usable sample

Computed from the *measured* volatility of Hyperliquid forward returns
(`scripts/research/sample_plan.py`), at α = 0.05, power = 0.80, three symbols
recorded in parallel.

**On 1h bars** — matching the Phase 4 horizons:

| Effect to detect | Observations | h=1 | h=5 | h=20 | h=100 |
|---|---|---|---|---|---|
| 52% hit rate | 4,904 | 68d | 341d | 1,362d | 6,811d |
| 53% hit rate | 2,178 | 30d | 151d | 605d | 3,025d |
| 55% hit rate | 783 | 11d | 54d | 218d | 1,088d |

To detect a 10 bps mean edge at h=20 takes 5,910 observations — **1,642 days**.
Open interest, which changes slowly and is naturally a multi-hour signal, is
effectively unanswerable on a horizon anyone would wait for.

**On 1m bars** — the regime order book imbalance actually lives in:

| Effect | h=1 | h=5 | h=15 | h=60 |
|---|---|---|---|---|
| 52% hit rate | 1d | 6d | 17d | 68d |
| 53% hit rate | 1d | 3d | 8d | 30d |

So the two signals have very different prospects:

- **Order book imbalance is answerable in days to weeks** at 1–15 minute
  horizons. This is the one worth waiting for.
- **Open interest is answerable in years** at the horizons it plausibly operates
  on. Record it — it is free and irreplaceable — but do not plan around it.

There is a catch that connects directly to items 2 and 3. Measured return
volatility at a 1-minute horizon is **5.1 bps**, while the taker round trip is
**9.0 bps**. The entire standard deviation of the move is smaller than the cost
of capturing it. Even a perfect 1-minute signal is untradeable on the taker
side, which makes maker execution a precondition for that whole regime rather
than an optimisation.

---

## 2. Maker-side rerun — same data, changed arithmetic

**This is not new evidence.** Identical observations, identical signals,
identical period. Only the number subtracted from each has changed. Gross edge is
fee-independent and did not move.

Fee scenarios swept, per side: 4.5 bps (taker, the Phase 4 baseline), 1.5 bps
(base maker), 0.0, −0.3 bps (rebate). The exact maker tier depends on 14-day
volume and staking, so this is a sweep rather than an assertion.

| Cost per side | Profitable cells (of 32) |
|---|---|
| +4.50 (taker) | 5 |
| −0.30 (rebate) | 18 |
| **Newly crossing break-even** | **13** |

And the finding that matters:

> **Of the 13 cells that cross break-even, 0 have a gross edge distinguishable
> from zero.** Every one has p_gross > 0.2, and most are above 0.5.

Examples: `macd_histogram @ h=1` becomes profitable at +0.66 bps on a gross edge
of +0.06 bps (p = 0.90). `ema_cross @ h=1` becomes profitable at +0.50 bps on a
gross edge of **−0.10 bps** (p = 0.82) — profitable while its measured edge is
negative.

A cell that turns profitable only because its cost fell, while its gross edge
remains indistinguishable from noise, is not a discovery. It is a rounding error
with a smaller subtraction applied.

**Not modelled, and it decides everything:** a resting order fills when someone
crosses the spread into it, which is disproportionately when they know something
you do not. That adverse selection is a real cost and is absent from these
numbers. Any signal reacting to a move already underway cannot be executed
passively at all. Treat the table as an upper bound on maker economics.

---

## 3. Structural scan — funding carry

The one trade here that does not forecast anything: hold spot, short the perp
against it, collect funding. Mechanical, delta neutral, immune to the finding
that killed Phases 3 and 4.

### The spot leg exists

I initially read Hyperliquid's spot markets as having zero volume and was wrong —
the API's token-name indexing is inconsistent. Matching spot pairs to perps *by
price* instead of by name:

| Perp | Spot pair | 24h notional | Basis |
|---|---|---|---|
| BTC | `@144` | $33.5M | −2.4 bps |
| ETH | `@155` | $20.8M | −6.4 bps |
| SOL | `@160` | $7.8M | −5.4 bps |

So the hedge leg is executable on Hyperliquid alone, without a CEX.

### Funding is small and often pinned

240 days, hourly:

| Symbol | Mean/hr | Annualised | % positive | % at floor |
|---|---|---|---|---|
| BTC | 0.00000511 | **+4.48%** | 75.5% | 39.8% |
| ETH | 0.00000597 | **+5.23%** | 79.2% | 46.8% |
| SOL | −0.00000126 | **−1.11%** | 57.6% | 30.9% |

Roughly 40% of all hours sit *exactly* on Hyperliquid's 0.01%/8h base rate —
funding settles there whenever the premium term is small. The carry is largely
the protocol's floor rate, not a crowding premium.

### The trade does not pay

PnL = funding collected − basis change − costs. Pooled, non-overlapping windows:

| Hold | n | Funding | Gross | Net @ 18 bps | Net @ 6 bps | Net @ 0 |
|---|---|---|---|---|---|---|
| 8h | 2,157 | +0.26 | +0.26 | −17.74 | — | — |
| 168h (1w) | 102 | +5.45 | +5.34 | −12.66 | — | — |
| 336h (2w) | 51 | +10.89 | +10.69 | −7.31 | +4.69 | +10.69 |
| 720h (30d) | 21 | +16.14 | +15.80 | −2.20 | +9.80 | +15.80 |
| 1440h (60d) | 9 | +18.41 | +17.57 | −0.43 | +11.57 | +17.57 |

All figures bps per trade. 18 bps = taker round trip on both legs; 6 bps = both
legs rested.

- **At taker cost: never profitable.** Not at any holding period tested.
- **At maker cost: profitable from ~2 weeks**, at +1.2% annualised — and *not
  significant* (p = 0.18 at 336h, p = 0.30 at 720h, n = 21–51).
- **At zero cost — the theoretical ceiling — 1.1% to 2.8% annualised.**

That last row is the one that closes the question. Even with no trading costs at
all, no slippage, no adverse selection and no risk, this trade yields under 3%
per year. That is below what the USDC collateral would earn sitting still. The
opportunity cost alone is disqualifying, before considering that the short perp
needs margin and can be liquidated by a move that leaves the pair flat overall.

The sample-size problem is structural too: a hold long enough to amortise costs
leaves only 9–21 independent observations in 240 days. Establishing significance
would take years, by which point the funding regime will have changed.

---

## Honest bottom line

Items 2 and 3 are both empty, and neither is a near miss.

- Maker fees do not rescue Phase 4's signals, because those signals have no
  measurable gross edge to rescue.
- Funding carry is real but too small to matter, and it is bounded above by ~3%
  annualised before any cost or risk.

The single useful output of Phase 5 is that the order book and open-interest
recorders are now running. Order book imbalance becomes answerable in **days to
weeks** at 1–15 minute horizons; that is the only open question left with a
tractable timeline.

Before betting on it, note what item 1's arithmetic already showed: at a
1-minute horizon the round-trip taker cost (9.0 bps) is nearly twice the entire
standard deviation of the move (5.1 bps). If order book imbalance turns out to
predict anything, it will only be tradeable passively — which is exactly the
execution mode item 2 could not validate.

## Reproducing

```bash
docker compose up -d && .venv/Scripts/python -m scripts.migrate
```

```bash
.venv/Scripts/python -m services.data_ingestion.recorder
```

```bash
.venv/Scripts/python -m scripts.research.maker_analysis
```

```bash
.venv/Scripts/python -m scripts.research.funding_carry --hold 8 24 72 168 336 720 1440
```

```bash
.venv/Scripts/python -m scripts.research.sample_plan --interval 1m --horizons 1 5 15 60 --days 20
```

Raw outputs are in `data/research/`.
