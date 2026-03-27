# Cycle 1: Core Decomposition Algorithm — Technical Findings

## Implementation Summary

Implemented the core momentum decomposition algorithm that separates total stock momentum into industry momentum and stock-specific (residual) momentum components.

### Algorithm

1. **Total returns**: Period-over-period percentage change in price for each stock.
2. **Industry returns**: Equal-weighted mean of stock returns within each industry at each date.
3. **Residual returns**: Rolling OLS regression of stock returns on industry returns over a lookback window. The residual (epsilon) captures the stock-specific component: `stock_return = alpha + beta * industry_return + epsilon`.
4. **Momentum scores**: Rolling sum of each return component over the lookback window.

### Data

- Source: ARF Data API (10 Japanese stocks, 3 TOPIX-17 industries)
- Period: 2021-03-26 to 2026-03-27 (~5 years of daily data)
- Stocks: 7203.T, 7267.T, 7974.T (auto), 6758.T, 6501.T, 8035.T, 6857.T (electronics), 8306.T, 8316.T, 8411.T (finance)
- Total observations: 12,220 (1,222 per stock)
- Valid (non-NaN) momentum observations: 11,980

### Results

| Metric | Total Momentum | Industry Momentum | Stock-Specific Momentum |
|--------|---------------|-------------------|------------------------|
| Mean   | 0.0149        | 0.0149            | -0.0000                |
| Std    | 0.0732        | 0.0596            | 0.0300                 |
| Min    | -0.4762       | -0.3222           | -0.1355                |
| Max    | 0.4290        | 0.2361            | 0.1541                 |

### Key Observations

1. **Industry momentum dominates**: The mean and standard deviation of industry momentum are close to total momentum, confirming that industry effects are the primary driver of total stock momentum in this sample.

2. **Stock-specific momentum is mean-zero**: As expected from the regression residuals, the stock-specific momentum has a mean near zero (-0.000007).

3. **Near-zero correlation**: The correlation between industry momentum and stock-specific momentum is 0.0002, indicating near-orthogonality. This is consistent with the regression-based decomposition but does not yet show the negative correlation reported in the paper (which may emerge with longer historical data and monthly rebalancing).

4. **Variance decomposition**: Industry momentum variance (0.0596^2 = 0.00355) accounts for most of total momentum variance (0.0732^2 = 0.00536), with stock-specific contributing the remainder (0.0300^2 = 0.00090).

### Limitations

- Uses daily data with daily lookback=12 (trading days), not the monthly lookback=12 (months) specified in the paper. Future phases should implement monthly resampling.
- Only 5 years of data available from the API (2021-2026), vs. the paper's 1980-2022 analysis period.
- No backtesting or portfolio construction yet (deferred to Phase 3-4).

### Unit Tests

10 tests implemented and passing:
- Output column and shape validation
- Non-NaN value verification after lookback period
- Directional tests (uptrend → positive momentum, downtrend → negative)
- Within-industry momentum consistency
- Lookback parameter sensitivity
- Edge cases (empty DataFrame, single stock)
