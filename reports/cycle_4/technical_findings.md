# Cycle 4: Decomposed Strategy Backtest

## Objective

Integrate the momentum decomposition algorithm into the evaluation framework and backtest all three strategies (total, industry, stock-specific) using walk-forward validation.

## Implementation

### New Functions in `src/evaluation.py`

- `compute_factor_correlations()` — Computes time-series correlations between decomposed momentum factors (industry vs stock-specific, total vs industry, total vs stock-specific).
- `run_decomposed_backtest()` — Runs walk-forward backtest for all 3 strategies using the same configuration and windows.
- `compare_strategies()` — Generates per-strategy ARF-schema metrics for side-by-side comparison.
- `evaluate_decomposed_strategies()` — End-to-end pipeline: runs all 3 strategies, computes correlations, selects the best strategy, and outputs combined metrics.

### Test Coverage

- 17 new tests in `tests/test_decomposed_backtest.py` covering correlations, decomposed backtest, strategy comparison, and end-to-end evaluation.
- Total: 67 tests (10 decomposition + 16 pipeline + 24 evaluation + 17 decomposed backtest), all passing.

## Results

### Strategy Comparison (Walk-Forward, 4 Windows)

| Strategy | Gross Sharpe | Net Sharpe | Ann. Return | Max DD | Hit Rate | Positive Windows |
|---|---|---|---|---|---|---|
| Total Momentum | -0.50 | -0.52 | -5.18% | -24.75% | 47.1% | 1/4 |
| Industry Momentum | -0.26 | -0.28 | -2.54% | -13.47% | 47.3% | 1/4 |
| **Stock-Specific Momentum** | **0.54** | **0.52** | **4.94%** | **-10.90%** | **51.6%** | **3/4** |

### Best Strategy

Stock-specific (residual) momentum is the clear winner:
- Only strategy with positive net Sharpe ratio (0.52)
- 3 out of 4 walk-forward windows are profitable
- Lowest maximum drawdown (-10.9%)
- Hit rate above 50%

### Factor Correlations

| Pair | Correlation |
|---|---|
| Industry vs Stock-Specific | 0.026 |
| Total vs Industry | 1.000 |
| Total vs Stock-Specific | 0.026 |

### Key Findings

1. **Stock-specific momentum outperforms total and industry momentum.** This is the opposite of the paper's finding that industry momentum is the dominant driver. The divergence likely comes from using daily frequency with a 12-day lookback (short-term reversal regime) instead of the paper's monthly frequency with 12-month lookback.

2. **Industry momentum dominates total momentum composition.** The near-perfect correlation (1.0) between total and industry momentum confirms that at the cross-sectional average level, total momentum is almost entirely explained by industry effects. This aligns with the paper.

3. **Near-zero correlation between industry and stock-specific momentum.** The correlation of 0.026 is close to zero, which is expected since stock-specific momentum is derived from OLS residuals (orthogonal by construction within each window). The paper reports a negative correlation, which may emerge with longer lookback periods.

4. **Total and industry momentum show negative Sharpe.** With daily data and short lookback, these strategies capture short-term reversal rather than momentum, resulting in negative performance. The stock-specific component, being purged of industry effects, captures genuine stock-level mean-reversion patterns.

## Configuration

- Walk-forward: 5 splits, minimum 60 training days
- Lookback: 12 periods (trading days)
- Portfolio: 30% quantile long/short, equal-weighted
- Transaction costs: 10 bps fee + 5 bps slippage
- Universe: 33 Japanese stocks, 7 industries, ~5 years daily data

## Next Steps

- Phase 5: Implement transaction cost model and compare gross vs net returns in detail
- Consider testing with monthly resampling and 252-day lookback to better align with the paper's methodology
- Explore dynamic rebalancing within walk-forward windows
