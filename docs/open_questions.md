# Open Questions

## Data Limitations

- **Limited historical period**: ARF Data API provides ~5 years of daily data (2021-2026). The paper analyzes 1980-2022. This limits our ability to reproduce long-term findings and period-by-period analysis (Phase 9).
- **Stock universe size**: 33 Japanese stocks available across 7 industries. The paper uses the full TOPIX constituent universe (~2000 stocks). Broader coverage would improve factor robustness and statistical power.
- **Uneven industry sizes**: Electronics has 8 stocks while auto/pharma/retail have only 3. This imbalance may bias industry momentum estimates for smaller sectors.

## Methodological Questions

- **Daily vs monthly lookback**: The current implementation uses a lookback of 12 trading days. The paper uses 12-month lookback. Future phases should resample to monthly frequency or use a lookback of ~252 trading days for closer alignment.
- **Industry classification granularity**: Using 7 industry groups derived from TOPIX-17 sectors. The paper likely uses the full TOPIX-17 classification. More granular sectors could change the decomposition results.
- **Negative correlation**: The paper reports negative correlation between industry and stock-specific momentum. Our decomposition shows near-zero correlation (~0.0). This may be due to: (a) shorter time period, (b) daily vs monthly frequency, (c) smaller stock universe, or (d) OLS residuals being orthogonal to regressors by construction at each point.
- **Outlier handling**: 129 outliers flagged (z>5) but not removed. Need to decide whether to winsorize, clip, or exclude these in downstream analysis (Phase 3+).

## Implementation Notes

- The regression uses a simple rolling OLS without intercept constraints. The paper may use a different specification.
- Equal-weighted industry returns are used. Value-weighted returns might better reflect actual industry momentum.
- Forward-fill strategy assumes that a stock's last observed price is the best estimate for missing days. This is standard but may introduce slight bias during volatile periods.

## Evaluation Framework Questions (Cycle 3)

- **Negative total momentum Sharpe**: The total momentum strategy shows negative Sharpe (-0.50) on real data. This likely results from using a 12-day lookback at daily frequency, which captures short-term reversal rather than the 12-month momentum effect in the paper. Phase 4+ should explore monthly rebalancing and longer lookback.
- **Static vs dynamic portfolios**: The current implementation forms a static portfolio at the end of each training window and holds it through the test window. The paper may rebalance monthly. Dynamic rebalancing will be explored in later phases.
- **Portfolio weighting**: Currently equal-weighted. The paper may use signal-weighted or risk-parity weighting.
- **Quantile threshold sensitivity**: Using 30% quantile for long/short legs. Different thresholds may significantly affect results with only 33 stocks.

## Decomposed Backtest Questions (Cycle 4)

- **Stock-specific momentum outperforms**: The backtest shows stock-specific momentum (net Sharpe 0.52) outperforming total (-0.52) and industry (-0.28) momentum. This contradicts the paper's finding that industry momentum is the dominant driver. The likely explanation is the use of 12-day lookback at daily frequency, which captures short-term reversal/mean-reversion rather than the 12-month momentum effect studied in the paper.
- **Total-industry correlation near 1.0**: The cross-sectional average correlation between total and industry momentum is ~1.0, meaning industry effects dominate total momentum composition. This aligns with the paper but needs verification at the individual stock level.
- **Industry vs stock-specific correlation near zero**: The 0.026 correlation is expected from OLS residuals but may differ from the paper's reported negative correlation when using monthly frequency and longer lookback.
- **Frequency mismatch**: The largest open question remains the daily vs monthly frequency gap. Phases 6+ should explore monthly resampling with 12-month lookback to align with the paper's methodology.

## Transaction Cost Questions (Cycle 5)

- **Low turnover masks cost sensitivity**: Because portfolios are static within each walk-forward window (no intra-window rebalancing), turnover and cost drag are minimal. A dynamic rebalancing strategy would reveal greater cost sensitivity.
- **Linear cost model**: The current model assumes constant cost per trade (fee + slippage in bps). In practice, market impact is nonlinear — it increases with trade size and decreases with liquidity. The `calculate_costs_detailed()` function provides the foundation for more sophisticated models.
- **Breakeven cost ceiling**: Stock-specific momentum's breakeven exceeds 100 bps, the grid-search upper bound. The true breakeven may be significantly higher, but this is academic since realistic costs for Japanese large-caps are well below 50 bps.
- **Gross vs net return equality**: Annual returns appear identical across cost scenarios because costs affect daily returns but the annualized return calculation in `compute_metrics` may round away small differences. The Sharpe ratio, which uses daily return means and standard deviations, better captures cost effects.

## Hyperparameter Optimization Questions (Cycle 6)

- **Overfitting risk**: With 72 parameter combinations tested across only 4 walk-forward windows, the selected optimal parameters (lookback=10, quantile=0.4) may overfit. The improvement from Sharpe 0.52 to 0.83 is substantial but should be validated on unseen data in Phase 7.
- **Optimal lookback differs by strategy**: Stock-specific momentum favors short lookback (5-10 days) while industry momentum favors longer lookback (30 days). This suggests the two components genuinely capture different market dynamics, but also means a single "optimal lookback" doesn't exist — each strategy should be independently parameterized.
- **Quantile sensitivity**: Stock-specific momentum improves monotonically with higher quantile (0.1 to 0.4), but this may be an artifact of the small universe (33 stocks). With quantile=0.4, the long/short legs contain ~13 stocks each, leaving only ~7 stocks "neutral." In a larger universe, this pattern might not hold.
- **Total momentum still negative**: No parameter combination produces positive Sharpe for total momentum. This persistent underperformance at daily frequency reinforces the need for monthly resampling to capture the classic 12-month momentum effect.
- **Grid resolution**: The search grid is coarse (6 lookback x 4 quantile values). Finer search around the optima could yield marginal improvements but increases overfitting risk with limited walk-forward windows.

## Robustness Verification Questions (Cycle 7)

- **Walk-forward configuration sensitivity**: Performance varies substantially across different walk-forward configurations (n_splits, min_train_size). Only 42% of 12 tested configurations produce positive Sharpe. The best result (0.83) occurs with n_splits=5, min_train_size=60, but nearby configurations like n_splits=5, min_train_size=80 produce Sharpe -0.82. This suggests the strategy's apparent profitability is partially dependent on evaluation methodology.
- **Holdout parameter instability**: The holdout validation selects different optimal parameters (lookback=20, quantile=0.1) than the full-period optimization (lookback=10, quantile=0.4). Despite this, the stock-specific momentum strategy itself remains profitable on holdout data (Sharpe 0.45), indicating strategy-level robustness even with parameter-level instability.
- **Sub-period limitations**: Each of 3 sub-periods contains only ~400 dates and generates just 1 walk-forward window, producing very noisy per-period Sharpe estimates. The -1.84 Sharpe in period 1 (2021-03 to 2022-11) may reflect the post-COVID market regime rather than strategy failure. More data would be needed for reliable sub-period analysis.
- **Bootstrap CI width**: The 95% CI width of 0.46 Sharpe units reflects only 4 walk-forward windows in the bootstrap. While the entire CI is above zero, higher confidence would require more independent evaluation windows.
- **Baseline outperforms on holdout**: Stock-specific momentum with baseline parameters (lookback=12, quantile=0.3) achieved Sharpe 0.74 on holdout, vs 0.45 for holdout-optimized parameters. This raises the question of whether grid search optimization adds value or introduces overfitting when the number of walk-forward windows is small.

## ARF Data API Notes

- All 33 requested tickers returned data successfully (no API errors).
- Only 1 missing data point across the entire panel (0.08%), indicating high data quality from the API.
- API provides ~5 years of daily data per ticker. No option for longer historical periods was observed.
