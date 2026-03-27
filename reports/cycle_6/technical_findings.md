# Cycle 6: Hyperparameter Optimization — Technical Findings

## Objective

Optimize the key hyperparameters (momentum lookback window and portfolio quantile threshold) for all three decomposed momentum strategies using walk-forward validation, and compare optimized performance against the baseline (lookback=12, quantile=0.3).

## Implementation

### New Functions (added to `src/evaluation.py`)

1. **`run_parameter_grid_search()`** — Grid search over lookback x quantile combinations for a given strategy. Evaluates each combination using full walk-forward validation and returns results sorted by net Sharpe.

2. **`analyze_parameter_sensitivity()`** — Marginalizes grid results to compute average net Sharpe by lookback (across quantiles) and by quantile (across lookbacks), identifying the best/worst parameter combinations and total Sharpe range.

3. **`evaluate_hyperparameter_optimization()`** — Full Phase 6 pipeline: runs grid search for all 3 strategies, compares optimized vs baseline, and produces ARF-standard metrics.

### Search Space

- **Lookback values**: [5, 10, 12, 15, 20, 30] trading days
- **Quantile values**: [0.1, 0.2, 0.3, 0.4]
- **Strategies**: total_momentum, industry_momentum, stock_specific_momentum
- **Total combinations evaluated**: 72 (6 lookbacks x 4 quantiles x 3 strategies)
- **Walk-forward windows**: 4 per combination (n_splits=5, min_train_size=60)

## Key Results

### Best Overall: Stock-Specific Momentum (lookback=10, quantile=0.4)

| Metric | Baseline (12, 0.3) | Optimized (10, 0.4) | Change |
|--------|-------------------|---------------------|--------|
| Net Sharpe | 0.5203 | **0.8265** | +0.3062 |
| Gross Sharpe | 0.5428 | 0.8596 | +0.3168 |
| Annual Return | 4.94% | 4.47% | -0.47% |
| Max Drawdown | -10.9% | -4.13% | +6.77% |
| Positive Windows | 3/4 (75%) | **4/4 (100%)** | +25% |

### Best Per Strategy

| Strategy | Optimal Lookback | Optimal Quantile | Net Sharpe | vs Baseline |
|----------|-----------------|------------------|------------|-------------|
| Stock-Specific | 10 | 0.4 | **0.8265** | +0.3062 |
| Industry | 30 | 0.3 | 0.6741 | +0.9579 |
| Total | 5 | 0.1 | -0.0351 | +0.4889 |

### Top 5 Parameter Combinations (across all strategies)

| Rank | Strategy | Lookback | Quantile | Net Sharpe | Positive Windows |
|------|----------|----------|----------|------------|-----------------|
| 1 | stock_specific | 10 | 0.4 | 0.8265 | 4/4 |
| 2 | industry | 30 | 0.3 | 0.6741 | 3/4 |
| 3 | stock_specific | 20 | 0.2 | 0.5289 | 2/4 |
| 4 | stock_specific | 20 | 0.3 | 0.5267 | 3/4 |
| 5 | stock_specific | 12 | 0.3 | 0.5203 | 3/4 |

## Parameter Sensitivity Analysis

### Lookback Sensitivity

**Stock-Specific Momentum** — Shorter lookbacks (5-12 days) outperform longer ones. Performance degrades significantly beyond 20 days, with lookback=30 producing negative average Sharpe (-0.16). Optimal at 5-10 days, suggesting short-term residual momentum effect.

| Lookback | Avg Net Sharpe |
|----------|---------------|
| 5 | 0.3883 |
| 10 | 0.2862 |
| 12 | 0.2784 |
| 15 | 0.0412 |
| 20 | 0.1457 |
| 30 | -0.1611 |

**Industry Momentum** — Opposite pattern: longer lookbacks perform better. Lookback=30 is the best (avg 0.33), while lookback=5 is worst (avg -0.72). This aligns with the paper's finding that industry momentum operates on a longer timescale.

| Lookback | Avg Net Sharpe |
|----------|---------------|
| 5 | -0.7242 |
| 10 | 0.0328 |
| 12 | -0.1980 |
| 15 | -0.2942 |
| 20 | -0.3346 |
| 30 | 0.3258 |

**Total Momentum** — Shorter lookbacks are less negative. Best at lookback=5 (avg -0.10), worst at lookback=15 (avg -0.64). No configuration produces positive average Sharpe.

### Quantile Sensitivity

**Stock-Specific Momentum** — Higher quantiles (more concentrated portfolios) perform better: quantile=0.4 (avg 0.41) vs quantile=0.1 (avg -0.16). The wider portfolio captures more of the residual momentum signal.

**Industry Momentum** — Quantile=0.4 is best (avg -0.002, near zero), while quantile=0.2 is worst (avg -0.50). Concentrated portfolios are less effective for industry momentum.

## Key Observations

1. **Optimization significantly improves stock-specific momentum**: Net Sharpe increased from 0.52 to 0.83 (+59%), and all 4 walk-forward windows became profitable (vs 3/4 at baseline). The max drawdown also improved dramatically from -10.9% to -4.1%.

2. **Industry momentum becomes profitable with longer lookback**: At lookback=30 (approx 6 weeks), industry momentum achieves positive Sharpe (0.67). This supports the paper's thesis that industry momentum operates on a longer timescale than stock-specific momentum.

3. **Total momentum remains unprofitable across all parameters**: Even the best configuration (lookback=5, quantile=0.1) only achieves -0.04 net Sharpe. This may indicate that at daily frequency with ~33 stocks, total momentum is dominated by short-term reversal effects.

4. **Lookback and quantile interact differently per strategy**: Stock-specific momentum favors shorter lookback + broader portfolio, while industry momentum favors longer lookback. This supports the decomposition approach — the two components capture genuinely different market dynamics.

5. **Sharpe range across parameters is large**: The gap between best and worst parameter combinations is 0.93 (total), 1.82 (industry), and 1.43 (stock-specific) Sharpe units. This highlights the importance of parameter selection and the risk of overfitting to in-sample results.

## Tests

- 29 new tests added in `tests/test_hyperparameter_optimization.py`
- All 129 tests pass (100 existing + 29 new)
- Tests cover: grid search mechanics, result sorting, sensitivity analysis computation, ARF schema compliance, baseline comparison structure, and parameter validation

## Limitations & Caveats

- **Overfitting risk**: With 72 parameter combinations and only 4 walk-forward windows, the optimized parameters may not generalize. The baseline (12, 0.3) is a more conservative choice.
- **Grid resolution**: The search grid is coarse. Finer grids around the optima (e.g., lookback 8-12) might yield further improvements but increase overfitting risk.
- **No cross-validation of parameters**: Parameters are selected based on average OOS Sharpe across windows, but not validated on a held-out period. Phase 7 (robustness validation) will address this.
