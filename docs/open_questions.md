# Open Questions

## Data Limitations

- **Limited historical period**: ARF Data API provides ~5 years of daily data (2021-2026). The paper analyzes 1980-2022. This limits our ability to reproduce long-term findings and period-by-period analysis (Phase 9).
- **Small stock universe**: Only 10 Japanese stocks available across 3 industries. The paper uses the full TOPIX constituent universe. Broader coverage would improve factor robustness.

## Methodological Questions

- **Daily vs monthly lookback**: The current implementation uses a lookback of 12 trading days. The paper uses 12-month lookback. Future phases should resample to monthly frequency or use a lookback of ~252 trading days for closer alignment.
- **Industry classification granularity**: Using 3 broad industry groups (auto, electronics, finance). The paper likely uses finer TOPIX-17 sector classification. More granular sectors could change the decomposition results.
- **Negative correlation**: The paper reports negative correlation between industry and stock-specific momentum. Our decomposition shows near-zero correlation (0.0002). This may be due to: (a) shorter time period, (b) daily vs monthly frequency, (c) smaller stock universe, or (d) different market regime.

## Implementation Notes

- The regression uses a simple rolling OLS without intercept constraints. The paper may use a different specification.
- Equal-weighted industry returns are used. Value-weighted returns might better reflect actual industry momentum.
