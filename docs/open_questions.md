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

## ARF Data API Notes

- All 33 requested tickers returned data successfully (no API errors).
- Only 1 missing data point across the entire panel (0.08%), indicating high data quality from the API.
- API provides ~5 years of daily data per ticker. No option for longer historical periods was observed.
