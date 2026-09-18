# Phase 4 single-signal research — results

Eight signals, four forward horizons, one signal at a time, no blending.
Real Hyperliquid 1h candles for BTC, ETH and SOL, 2026-02-12 → 2026-09-08
(~5,000 bars per symbol), costs at the Hyperliquid taker fee of 4.5 bps per side
(9.0 bps round trip).

Reproduce:

```bash
.venv/Scripts/python -m scripts.research.signal_matrix --days 240 --csv data/research/matrix_1h.csv
```

## Headline

**Nothing is tradeable.** Of 32 signal × horizon cells, zero have an edge that
survives correction for the size of the search *and* clears transaction costs.

| | |
|---|---|
| Cells tested | 32 |
| Expected false positives at raw p < 0.05 | 1.6 |
| Raw p < 0.05 on an edge test | 10 |
| Survive Benjamini–Hochberg FDR | 6 |
| …and profitable after fees | **0** |
| Survive Bonferroni on gross return | **0** |

That 10 → 6 → 0 funnel is the useful part. Ten cells looked significant; the
search was large enough to expect about two of those by chance; six survived FDR;
none of the six made money.

## The matrix

Brier score (0.2500 = coin flip, lower is better) over mean net return in bps
per independent observation. `*` = edge survives FDR correction.

| signal | h=1 | h=5 | h=20 | h=100 |
|---|---|---|---|---|
| rsi | 0.2586 / −14.34 | 0.2450 / −1.09 | 0.2543 / −41.41 | 0.2443 / −6.80 |
| rsi_momentum | 0.2708* / −8.68 | 0.2696 / −9.40 | 0.2665 / +3.43 | 0.2817* / −42.89 |
| macd_histogram | 0.2878* / −8.94 | 0.2859 / −7.74 | 0.2845 / +12.34 | 0.2807 / +16.13 |
| ema_cross | 0.3597* / −9.09 | 0.3645 / −10.42 | 0.3958* / −15.95 | 0.3757 / −67.03 |
| bollinger_percent_b | 0.2718 / −11.45 | 0.2700 / −14.69 | 0.2786 / −23.47 | 0.2571 / −45.89 |
| atr_momentum | 0.3855* / −8.68 | 0.3786 / −10.43 | 0.3867 / +3.50 | 0.4449 / −16.88 |
| obv_divergence | 0.2721 / −7.86 | 0.2664 / −4.18 | 0.2706 / +2.47 | 0.2498 / −24.64 |
| funding_rate | 0.2541 / −9.29 | 0.2549 / −11.24 | 0.2503 / −3.06 | 0.2520 / +0.35 |

**Every Brier score is above 0.2500.** Not one signal's stated confidence beat a
coin flip. The trend signals are the worst calibrated: `ema_cross` and
`atr_momentum` sit at 0.36–0.44, meaning they are confidently wrong — exactly the
failure mode Phase 3 found in the blend, now located in specific components.

## The six that survived correction — and why none of them help

All six survivors are significant on *hit rate*, and all six are significant for
being **below** 50%:

| cell | hit rate | gross bps | net bps | n | p_hit |
|---|---|---|---|---|---|
| rsi_momentum @ h=1 | 47.9% | +0.32 | −8.68 | 14,919 | 0.0000 |
| atr_momentum @ h=1 | 48.2% | +0.32 | −8.68 | 14,876 | 0.0000 |
| macd_histogram @ h=1 | 48.6% | +0.06 | −8.94 | 14,829 | 0.0005 |
| ema_cross @ h=1 | 48.8% | −0.09 | −9.09 | 14,829 | 0.0028 |
| ema_cross @ h=20 | 43.6% | −6.95 | −15.95 | 741 | 0.0005 |
| rsi_momentum @ h=100 | 37.4% | −33.89 | −42.89 | 147 | 0.0029 |

**Do not invert these.** The obvious reaction — "47.9% is reliably wrong, so trade
the opposite" — fails on three of the six. At h=1 those signals are right less
than half the time yet still have a *non-negative* mean gross return, because
their wins are bigger than their losses. Inverting `rsi_momentum @ h=1` would
raise the hit rate to 52.1% and turn +0.32 bps into −0.32 bps. Hit rate and
expectancy point in opposite directions here, which is precisely why this harness
reports both.

The h=1 results are also the least interesting economically: a 48% hit rate with
~0 bps gross is consistent with ordinary short-horizon mean reversion around the
bid-ask, not a tradeable inefficiency.

The two with genuinely negative gross returns (`ema_cross @ h=20`,
`rsi_momentum @ h=100`) are the only candidates where an inverse reading might
mean something. Both have small independent samples (741 and 147), and testing
their inverse on this same data would be a second look at the same evidence.
That needs fresh data, not a rerun.

## What "significant" was worth here

Four cells hit raw p < 0.05 and then failed correction:

- `rsi @ h=5` (p_hit = 0.0275)
- `rsi_momentum @ h=5` (p_hit = 0.0243)
- `macd_histogram @ h=20` (p_gross = 0.0197)
- `bollinger_percent_b @ h=1` (p_hit = 0.0276)

With 32 tests, ~1.6 false positives are expected at that threshold. Four is in
that neighbourhood. Any of these reported on its own would have looked like a
finding.

`macd_histogram @ h=20` is the one worth naming: +21.34 bps gross, +12.34 bps net,
p_gross = 0.0197 raw. It is the only cell in the matrix that is both profitable
after costs and significant before correction — and it does not survive FDR, on
741 observations. It is the single most likely candidate for a follow-up on
out-of-sample data, and it is not evidence of anything yet.

## Two signals could not be tested at all

| signal | needs | why not |
|---|---|---|
| `orderbook_imbalance` | order book depth | Hyperliquid's `l2Book` returns the current snapshot only. No historical depth endpoint exists. |
| `open_interest_change` | open interest | `metaAndAssetCtxs` reports current OI only. No historical series. |

I probed `openInterestHistory`, `oiHistory` and `l2BookHistory` directly; none
exist. Both signals are implemented and registered, and both are marked NO DATA
rather than dropped, so they appear in every run as an open question rather than
disappearing. Measuring them requires recording the series forward from now — the
existing `hyperliquid_orderbook` collector already produces book snapshots, but
nothing persists them yet.

**Funding rate was testable** — `fundingHistory` does return a series, hourly,
and it is included above.

## How the numbers were made trustworthy

Three decisions do most of the work here, and each one moves results toward
"nothing found":

**Non-overlapping samples.** A signal firing every bar with a 100-bar horizon
produces observations sharing 99 of their 100 bars. Counting those as independent
draws inflates significance enormously. Every p-value uses a subsample spaced at
least one horizon apart. The cost is visible in the tables: `funding_rate @ h=100`
has 14,706 firings but only 150 independent observations.

**Multiple-comparison correction.** 32 tests were run. Benjamini–Hochberg FDR is
applied across the whole matrix, and Bonferroni is reported alongside.

**Gross and net kept separate.** With no edge, mean net return converges on minus
the round-trip cost, and a large sample makes that reliably non-zero. A
"significant" net loss is usually detecting the fee, not the signal. 13 of 32
cells are significantly unprofitable; that number is close to meaningless on its
own, and the report says so. The edge tests are hit rate and *gross* return.

Also worth stating: the three symbols are correlated, so pooling them understates
variance and the pooled p-values are, if anything, optimistic. Per-symbol
breakdowns are printed by `single_signal.py`.

## What this does not say

- One 7-month window, three correlated majors, one interval (1h). This rejects
  "these readings have a usable edge here"; it does not prove no edge exists.
- Each signal was tested under **one** stated convention. RSI was tested both
  ways (`rsi` mean-reversion, `rsi_momentum` momentum) precisely because that
  choice flips the answer; the others were not.
- No parameter search. Periods are the conventional defaults. A search would also
  need its own correction, and with 32 tests already producing nothing, widening
  the search mostly widens the false-positive budget.
- Nothing here tests combinations. That is deliberate — Phase 3 showed a blend
  fails without telling you which part failed.

## Suggested next steps

1. **Do not build more agents on these signals.** That was already your call; the
   data supports it. None of the eight carries usable information at these
   horizons on this data.
2. **Start recording order book and open interest now.** They are the two signals
   with genuinely no evidence either way, and every day without a collector is a
   day of history that cannot be recovered. This is the highest-value cheap move.
3. If any single cell deserves a second look it is `macd_histogram @ h=20`, on
   **out-of-sample** data — a different period, or different symbols. Testing it
   again here proves nothing.
4. Consider whether 1h bars are the right resolution at all. Costs of 9 bps round
   trip dominate every result; a signal would need to predict a move substantially
   larger than that to be tradeable, which argues for longer horizons or lower
   turnover than anything tested here.
5. Test the maker side. Everything above assumes taker fees. Hyperliquid's maker
   rebate changes the arithmetic materially, and several cells are within 9 bps of
   break-even.
