# Phase 6 — long-horizon research

Nine low-turnover signals across six horizons (3–90 days) on daily Hyperliquid
candles for 20 perps, plus cross-sectional momentum. The regime where cost stops
being the binding constraint.

```bash
.venv/Scripts/python -m scripts.research.long_horizon --csv data/research/long_horizon.csv
```

## Headline

**Nothing survives.** Of 54 single-signal cells, 45 were testable and **0** show
an edge over buy-and-hold after correction. Of 18 cross-sectional cells, 12 were
testable and the best raw p-value is **0.17** — no correction is even needed to
conclude nothing.

The premise held: at these horizons the 9 bps round trip is negligible against
moves measured in hundreds of basis points. Costs were not what stood between
these signals and profitability. There was no edge to be blocked by costs.

## Data — reported before any testing

Hyperliquid daily candles go back to **2020-08-19**, far more than the two years
the brief assumed. But most of it is not Hyperliquid data:

| | |
|---|---|
| Total daily bars (BTC) | 2,213 (from 2020-08-19) |
| **Bars with actual trading** | **1,292 (from 2023-02-26)** |
| Backfilled, zero volume, zero trades | 921 |

Bars before 2023-02-26 carry `volume = 0` and `trades = 0`. They are real index
prices backfilled from elsewhere, not Hyperliquid market data, and no funding
exists for them. **Only traded bars are used.** `--include-synthetic` overrides
this if you want the longer, less honest sample.

Twenty perps have usable history:

| Metric | Value |
|---|---|
| Symbols with traded history | 20 of 20 requested |
| Median traded bars/symbol | 1,211 (~3.3 years) |
| Longest | 1,292 (BTC, ETH, ATOM) |
| Shortest included | 972 (ETC, from 2024-01-12) |
| Funding history | 28,602 hourly points/symbol from 2023-05 |

**This is enough history for 3–60 day horizons and not enough for 90.** No
alternative source is needed for the shorter horizons; for 90 days and beyond,
Hyperliquid alone cannot answer the question and an external daily source
(a spot index with a decade of history) would be required.

## Power — reported before any results

The obvious way to buy sample size is to pool 20 symbols. In crypto that mostly
does not work. Measured mean pairwise correlation of daily returns across the 20
symbols is **0.609**, so:

```
n_effective = (k · n) / (1 + (k − 1) · ρ)
```

| Horizon | n/symbol | Pooled raw | **n_effective** | Verdict |
|---|---|---|---|---|
| 3d | 403.7 | 8,073 | 642 | testable |
| 7d | 173.0 | 3,460 | 275 | testable |
| 14d | 86.5 | 1,730 | 138 | testable |
| 30d | 40.4 | 807 | 64 | testable |
| 60d | 20.2 | 404 | 32 | testable |
| **90d** | 13.5 | 269 | **21** | **UNTESTABLE** |

**Pooling 20 correlated symbols multiplies information by 1.59×, not 20×.**
Every p-value in this report is computed at `n_effective`, not at the raw pooled
count — the correlation is priced into the numbers, not just mentioned in prose.

Nine cells (every signal at 90 days) fall below 30 effective observations and are
reported as UNTESTABLE. That is a statement about the available history, not
about those signals.

## The correction that changes everything

The sample covers a period in which crypto rose substantially:

| Horizon | Unconditional mean return | Up-rate |
|---|---|---|
| 3d | +26 bps | 49.3% |
| 7d | +60 bps | 48.1% |
| 30d | +262 bps | 47.1% |
| 90d | +694 bps | 44.5% |

A signal that is simply long most of the time collects this drift whether or not
it knows anything. My first run reported `momentum_200d @ 7d` earning **+96.5 bps
with p = 0.022** — and BTC's unconditional 7-day return over the same window is
**+88.1 bps**. Essentially all of it was drift.

Every edge in this report is therefore measured as **excess over a
direction-adjusted buy-and-hold benchmark**, and hit rates are tested against the
market's own up-rate rather than against 50%. Testing gross return against zero
would have scored market beta as skill.

## Results

All 45 testable cells: **zero** survive FDR, **zero** survive Bonferroni. The
smallest p-value anywhere in the matrix is 0.27.

Representative rows (full matrix in `data/research/long_horizon.csv`):

| Signal | h | n_eff | long% | hit% | null% | bench | excess | p_hit | p_exc |
|---|---|---|---|---|---|---|---|---|---|
| momentum_60d | 30 | 64 | 42 | 52.6 | 52.2 | +3 | +143 | 0.90 | 0.72 |
| momentum_200d | 60 | 32 | 45 | 47.2 | 53.3 | +62 | −611 | 0.48 | 0.39 |
| tsmom_30d | 60 | 32 | 40 | 56.2 | 52.5 | −36 | +360 | 0.73 | 0.67 |
| tsmom_180d | 30 | 64 | 45 | 45.6 | 52.3 | +25 | −311 | 0.26 | 0.45 |
| vol_regime | 30 | 64 | 56 | 55.2 | 49.5 | +23 | +448 | 0.45 | 0.27 |
| drawdown | 30 | 64 | 100 | 45.5 | 46.9 | +262 | −137 | 0.90 | 0.69 |
| funding_extreme | 30 | 64 | 46 | 45.2 | 49.7 | −30 | −340 | 0.53 | 0.39 |

Two rows worth reading carefully:

- **`drawdown` is 100% long by construction** and shows a benchmark of +262 bps
  at 30 days with **negative** excess (−137). Buying the dip captured the drift
  and slightly underperformed simply holding.
- **`vol_regime` at 30 days** has the largest positive excess in the matrix
  (+448 bps) and is still not significant (p = 0.27, n_eff = 64). It is the only
  cell worth a second look if more history ever becomes available.

## The bug that would have produced a false discovery

The first cross-sectional run reported 30-day-lookback momentum earning
**+187 bps at h=7 with p = 0.0031**, positive at four consecutive horizons, three
of them surviving *Bonferroni*. It would have been the first real finding of the
entire project.

It was a bug. `cross_sectional_momentum` indexed every symbol's price series by
the same integer offset, but the symbols list at different times and so have
histories of different lengths. Index 100 was June 2023 for BTC and April 2024
for ETC. The strategy was ranking symbols against each other **on different
calendar dates**, and the "signal" was pure misalignment.

A sub-period robustness check caught it: the effect vanished when the series were
aligned by hand, at which point the same cell read −44.7 bps with p = 0.34.

The function now aligns by timestamp and refuses to compare dates a symbol does
not have. A regression test constructs six symbols on identical price paths with
staggered listing dates — correctly aligned, the long-short spread must be
exactly zero, and the old code produced a large non-zero spread.

Corrected results:

| Lookback | h | n | win% | gross bps | p |
|---|---|---|---|---|---|
| 30d | 3 | 418 | 52.6 | −9.2 | 0.64 |
| 30d | 7 | 179 | 55.3 | +35.1 | 0.44 |
| 90d | 14 | 85 | 57.6 | +108.2 | 0.17 |
| 90d | 30 | 39 | 61.5 | +203.8 | 0.36 |

Nothing significant. Six of 18 cells are untestable at 60–90 days.

## What this does and does not establish

**Establishes:** across 3–60 day horizons, on 3.3 years of real Hyperliquid
traded history over 20 perps, nine standard low-turnover signals and
cross-sectional momentum show no edge over buy-and-hold that survives correction
for the size of the search and for cross-symbol correlation.

**Does not establish:**

- Anything about 90-day horizons. Fifteen cells are untestable; that needs a
  longer daily series than Hyperliquid has.
- That these signals do not work anywhere. One venue, 3.3 years, mostly a bull
  market, one stated convention per signal.
- That the conventions tested are the right ones. `drawdown` was tested as
  buy-the-dip and `vol_regime` as risk-on; both have equally defensible opposite
  readings that would flip every sign.
- Anything about parameter tuning. These are conventional defaults, and a search
  would need its own correction.

## Where this leaves the project

Four phases have now returned nothing:

| Phase | Question | Answer |
|---|---|---|
| 3 | Does the blended analyst work? | No — 34% hit rate, Brier 0.40 |
| 4 | Do its components work individually at 1h? | No — 0 of 32 cells |
| 5 | Do maker fees rescue them? Is there structural carry? | No, and no |
| 6 | Does anything work at 3–90 days? | No — 0 of 45 testable cells |

The cost hypothesis is now closed. Phase 5 showed that at 1h the round trip
exceeded the entire move; Phase 6 tested the regime where that is emphatically
untrue and found the same nothing. **Transaction costs were never the binding
constraint. The absence of predictive signal was.**

What remains genuinely open:

1. **Order book imbalance**, still recording since Phase 5. The only untested
   hypothesis with a tractable timeline (days to weeks at 1–15 minute horizons).
2. **90-day-plus horizons**, which need an external daily price source.
3. **Conventions opposite to those tested here**, which is a second look at the
   same data and would need fresh data to confirm.

Nothing in six phases supports allocating capital.

## Reproducing

```bash
.venv/Scripts/python -m scripts.research.long_horizon --csv data/research/long_horizon.csv
```

```bash
.venv/Scripts/python -m scripts.research.long_horizon --horizons 3 7 14 30 --include-synthetic
```

Full output in `data/research/long_horizon.txt`, per-cell CSV in
`data/research/long_horizon.csv`.
