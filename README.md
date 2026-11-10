# Kronos on BTC daily candles: a walk-forward evaluation

Does [Kronos](https://github.com/shiyu-coder/Kronos), a foundation model for
financial candlesticks, forecast Bitcoin's daily candles better than naive
baselines? This repo runs it walk-forward on BTCUSDT daily data and scores
every forecast against what actually happened.

**Short answer: no.** Across 15 configurations, Kronos never beats a random
walk on next-day close error, its direction calls are indistinguishable from a
coin flip, and its prediction intervals are far too narrow.

## Setup

- **Data:** BTCUSDT 1d candles from Binance's public API, 2022-01-01 to 2026-01-31 (`fetch_data.py`).
- **Test windows:** 2025-09-01 to 2026-01-31 (153 days) and 2024-07-01 to 2025-08-31 (427 days), 580 days in total.
- **Walk-forward:** for each target day the context is the `lookback` closed candles strictly before it, so nothing from the future leaks in.
- **Models:** Kronos-mini, -small and -base, with lookbacks of 40 (the paper's daily setting), 90 and 400 bars.
- **Sampling:** 30 sample paths per forecast (20 for the 7-day run; T=1.0, top-p 0.9 unless noted). `KronosPredictor` averages samples internally, so each series is repeated in the batch with `sample_count=1` to keep the full distribution.
- **Baselines:**
  - Random walk (tomorrow = today) for point error.
  - Gaussian with trailing 30-day volatility for CRPS and intervals.
  - Trailing 20-day mean high-low range for range forecasts.

## Results

Each row is one configuration at one horizon (`h1` = next day, `h7` = 7 days ahead). Ratios below 1 mean Kronos beats the baseline.

```
run                            n        MAE/RW         dir n       dir acc             p          corr       pred up        act up         cov80       CRPS/RW rangeMAE/base       L/S ret       B&H ret
small/h1                     153          1.14           153         54.2%          0.33        +0.065           59%           49%           62%          1.12          1.24           +0%          -27%
mini/h1                      153          1.14           153         51.6%          0.75        -0.075           56%           49%           65%          1.16          1.12          -25%          -27%
base/h1                      153          1.75           153         47.7%          0.63        -0.027           62%           49%           44%          1.69          1.66          -21%          -27%
small_ext/h1                 427          2.12           427         51.8%          0.50        +0.037           40%           51%           57%          1.94          2.15          -11%          +72%
small_all/h1                 580          1.86           580         52.4%          0.26        +0.022           45%           51%           59%          1.72          1.90          -10%          +25%
mini_lb40/h1                 153          1.14           153         49.7%          1.00        -0.058           56%           49%           57%          1.16          1.05          -32%          -27%
small_lb40/h1                153          1.03           153         52.9%          0.52        +0.017           36%           49%           56%          1.05          1.02           -7%          -27%
base_lb40/h1                 153          1.07           153         50.3%          1.00        -0.044           37%           49%           53%          1.10          1.05          -31%          -27%
mini_all_lb40/h1             580          1.11           580         49.3%          0.77        -0.016           57%           51%           55%          1.11          1.03          -54%          +25%
small_all_lb40/h1            580          1.07           580         51.4%          0.53        +0.081           46%           51%           56%          1.07          1.02           -2%          +25%
base_all_lb40/h1             580          1.08           580         49.7%          0.90        +0.081           40%           51%           52%          1.09          1.01          -42%          +25%
small_lb40_topp1/h1           153          1.04           153         56.2%          0.15        -0.010           35%           49%           61%          1.04          1.09           -3%          -27%
small_lb90/h1                153          1.09           153         47.1%          0.52        +0.037           55%           49%           57%          1.11          1.12           -7%          -27%
small_all_lb90/h1            580          1.12           580         49.7%          0.90        +0.051           52%           51%           58%          1.12          1.04          -18%          +25%
small_7d_lb40/h1             153          1.07           153         49.7%          1.00        -0.022           35%           49%           54%          1.08          1.01          -10%          -27%
small_7d_lb40/h7             147          0.93            21         47.6%          1.00        +0.150           55%           45%           44%          1.01          0.93             -             -
```

Columns:

- `n`: number of forecasts.
- `MAE/RW`: close mean absolute error ÷ random-walk error.
- `dir n` / `dir acc` / `p`: direction accuracy on non-overlapping windows and its two-sided binomial p-value against a coin flip.
- `corr`: correlation of predicted and actual returns.
- `pred up` / `act up`: share of days predicted up and actually up.
- `cov80`: how often the actual close fell inside Kronos's 10–90% band (should be 80%).
- `CRPS/RW`: continuous ranked probability score ÷ Gaussian baseline.
- `rangeMAE/base`: high-low range error ÷ trailing-range baseline.
- `L/S ret` and `B&H ret`: total return of a long/short strategy on the predicted sign, and of buy-and-hold (0.1% fee per side).

Per-run plots are in `eval_*.png`, per-day rows in `daily_*.csv`, and every metric in `results.json`.

## Findings

1. **No point-forecast skill.** Next-day close error is 1.03× the random walk's at best (small, 40-bar lookback) and up to 2.1× with a 400-bar lookback.
2. **No directional skill.** Accuracy ranges from 47% to 56%, and nothing is significant (lowest p = 0.15, out of 15 configurations tried).
3. **Badly overconfident intervals.** This is the most consistent result: the nominal 80% band covers the actual close 44–65% of the time. A plain volatility-scaled Gaussian covers 82–84%.
4. **Range forecasts are no better than a trailing average.** They come in at 0.93–2.15× its error.
5. **Long lookbacks hurt.** With 400 bars the per-window normalisation spans a long trend, the token resolution gets coarse and forecasts mostly revert to the window mean.
6. **The 7-day row looks better than it is.** Its 0.93 error ratio is averaged over overlapping windows. On the 21 independent windows, direction accuracy is 47.6%.

## Caveats

- **Only one asset at daily frequency.** These results say nothing about Kronos on intraday data or other markets.
- **Possible training overlap.** The 2024-07 to 2025-08 window may overlap Kronos's pretraining data. Kronos does worse there, so this doesn't change the conclusion.
- **Single seed.** All runs use seed 0 with 30 samples per forecast (20 for the 7-day run).

## Related reports

Other evaluations in the Kronos issue tracker reach the same conclusion on different data:

- [#387](https://github.com/shiyu-coder/Kronos/issues/387): walk-forward backtests on 5m, 15m and 1h crypto futures. Worse than a random walk, with forecast cones too narrow.
- [#282](https://github.com/shiyu-coder/Kronos/issues/282): Kronos-small on 30 years of daily data for 20 major assets. Negative.
- [#323](https://github.com/shiyu-coder/Kronos/issues/323): a reproducible direction benchmark of Kronos-base.
- [#355](https://github.com/shiyu-coder/Kronos/issues/355): Kronos-mini on 30-minute US equities.
- [#375](https://github.com/shiyu-coder/Kronos/issues/375): cross-sectional ranking on 182 US equities. IC +0.022 (t=1.39), which doesn't survive trading costs.

This repo adds daily BTC, all three public model sizes and a lookback sweep, and scores the full sample distribution with CRPS and interval coverage, not just the mean forecast.

The sample-averaging workaround in `run_forecasts.py` won't be needed once [#395](https://github.com/shiyu-coder/Kronos/pull/395) or [#321](https://github.com/shiyu-coder/Kronos/pull/321) is merged; both expose the individual sample paths.

## Reproducing

```bash
git clone https://github.com/shiyu-coder/Kronos   # or set KRONOS_PATH to an existing checkout
pip install -r Kronos/requirements.txt
python fetch_data.py                               # optional, btcusdt_1d.csv is included
PYTHON=python ./run_all.sh && ./run_lb40.sh        # several CPU hours; forecasts_*.npz are included
python evaluate.py && python summary.py
```
