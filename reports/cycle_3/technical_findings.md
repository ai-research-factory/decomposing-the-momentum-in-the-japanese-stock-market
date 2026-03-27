# Cycle 3: Evaluation Framework — Technical Findings

## Summary

Implemented a walk-forward evaluation framework for the total momentum long-short strategy. The framework integrates the existing decomposition algorithm (Cycle 1) with the data pipeline (Cycle 2) and the provided backtest utilities (`WalkForwardValidator`, `calculate_costs`, `compute_metrics`, `generate_metrics_json`).

## Implementation

### New module: `src/evaluation.py`

Key components:

1. **`compute_momentum_scores()`** — Wraps `decompose_momentum()` to produce clean momentum scores with NaN rows dropped.

2. **`form_long_short_portfolio()`** — Ranks stocks by momentum score and assigns long positions to the top quantile and short positions to the bottom quantile. Supports both quantile-based and fixed-count portfolio construction.

3. **`compute_portfolio_return()`** — Calculates equal-weighted daily returns for a long-short portfolio by matching prices across consecutive dates.

4. **`run_walk_forward_evaluation()`** — Core evaluation loop:
   - Uses `WalkForwardValidator` to split the time series into train/test windows.
   - Computes momentum scores on the training window only (no future leakage).
   - Forms a static long-short portfolio from the latest training-window scores.
   - Computes daily returns across the test window.
   - Applies transaction costs via `calculate_costs()`.
   - Produces `BacktestResult` objects with gross/net Sharpe, annual return, max drawdown, hit rate, and PnL series.

5. **`evaluate_momentum_strategy()`** — End-to-end pipeline that runs walk-forward and outputs ARF-standard `metrics.json`.

### Walk-forward configuration

- 5 walk-forward splits (`n_splits=5`)
- Minimum training size: 60 trading days (`min_train_size=60`)
- 1-day gap between train and test to prevent leakage
- Transaction costs: 10 bps fee + 5 bps slippage

## Results on Real Data

| Metric | Value |
|--------|-------|
| Gross Sharpe Ratio | -0.5047 |
| Net Sharpe Ratio | -0.5240 |
| Annual Return | -5.18% |
| Max Drawdown | -24.75% |
| Hit Rate | 47.08% |
| Total Trades | 72 |
| Walk-Forward Windows | 4 |
| Positive Windows | 1 of 4 |

### Interpretation

The total momentum strategy shows **negative risk-adjusted returns** over the 2021-2026 period with 33 Japanese stocks. This is consistent with several known phenomena:

1. **Short lookback period**: Using a 12-day lookback captures short-term reversal effects rather than the 12-month momentum documented in the academic literature. The paper's methodology uses monthly data over 12-month windows.

2. **Momentum crash risk**: The 2021-2026 period includes significant regime changes (post-COVID recovery, monetary policy shifts) that can cause momentum crashes.

3. **Small universe**: With only 33 stocks across 7 industries, the strategy has limited diversification and is sensitive to individual stock movements.

4. **Daily frequency**: Academic momentum strategies typically operate at monthly rebalancing frequency. Daily implementation amplifies noise and transaction costs.

## Tests

24 new tests in `tests/test_evaluation.py` covering:

- **Momentum score computation** (4 tests): column presence, NaN handling, directional correctness
- **Portfolio formation** (6 tests): position assignment, long/short balance, winner/loser allocation, edge cases
- **Portfolio return calculation** (4 tests): long/short return signs, hedging, empty portfolio
- **Walk-forward evaluation** (5 tests): result types, window count, no future leakage, valid metrics, PnL series
- **End-to-end metrics** (5 tests): schema compliance, transaction cost structure, walk-forward structure, custom metrics

Total test count: 50 (10 decomposition + 16 pipeline + 24 evaluation), all passing.

## Next Steps (Phase 4)

Phase 4 will decompose the strategy into three variants:
- Total momentum (baseline — implemented here)
- Industry momentum
- Stock-specific (residual) momentum

This will allow direct comparison of the paper's core hypothesis that industry momentum is the dominant component.
