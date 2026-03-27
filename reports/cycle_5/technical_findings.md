# Cycle 5: Transaction Cost Analysis

## Objective

Implement a transaction cost model with multiple cost scenarios and compare gross vs net returns across all three decomposed momentum strategies.

## Implementation

### New in `src/backtest.py`

- `COST_SCENARIOS` — Five pre-defined cost regimes: zero (0 bps), low (8 bps), medium (15 bps), high (30 bps), very_high (50 bps).
- `CostBreakdown` dataclass — Detailed per-window cost breakdown: fee vs slippage split, turnover, cost per trade, annualized cost drag.
- `calculate_costs_detailed()` — Extended cost calculator returning both net returns and a full `CostBreakdown`.
- `compute_turnover()` — Portfolio-level turnover calculation from position histories.

### New in `src/evaluation.py`

- `run_cost_sensitivity_analysis()` — Runs all 3 strategies across 5 cost scenarios (15 evaluations), reusing a single set of walk-forward PnL series.
- `compute_cost_impact()` — Quantifies Sharpe degradation and return drag from zero-cost baseline for each strategy/scenario pair.
- `compute_breakeven_cost()` — Grid search over cost levels to find the bps at which each strategy's net Sharpe drops to zero.
- `evaluate_transaction_costs()` — End-to-end Phase 5 pipeline combining sensitivity, impact, and breakeven analysis.

### Test Coverage

- 33 new tests in `tests/test_transaction_costs.py` covering detailed cost calculation, turnover, sensitivity analysis, cost impact, breakeven costs, and end-to-end evaluation.
- Total: 100 tests (10 decomposition + 16 pipeline + 24 evaluation + 17 decomposed backtest + 33 transaction costs), all passing.

## Results

### Gross vs Net Comparison (Medium Cost: 15 bps)

| Strategy | Gross Sharpe | Net Sharpe | Sharpe Degradation | Breakeven (bps) |
|---|---|---|---|---|
| Total Momentum | -0.50 | -0.52 | 0.019 | 0.0 |
| Industry Momentum | -0.26 | -0.28 | 0.019 | 0.0 |
| **Stock-Specific** | **0.54** | **0.52** | **0.023** | **100+** |

### Cost Sensitivity (Best Strategy: Stock-Specific Momentum)

| Scenario | Total Cost (bps) | Net Sharpe | Sharpe Delta |
|---|---|---|---|
| Zero | 0 | 0.543 | -- |
| Low | 8 | 0.531 | -0.012 |
| Medium | 15 | 0.520 | -0.023 |
| High | 30 | 0.498 | -0.045 |
| Very High | 50 | 0.468 | -0.075 |

### Key Findings

1. **Stock-specific momentum is highly robust to transaction costs.** Even under the very_high cost scenario (50 bps round-trip), the strategy retains a net Sharpe of 0.47. The breakeven cost exceeds 100 bps, indicating strong profitability relative to realistic cost levels.

2. **Cost impact is modest due to low turnover.** The static portfolio (formed once per walk-forward window) incurs costs only at entry, not through rebalancing. This buy-and-hold approach within each window keeps cost drag minimal (~2.3 bps Sharpe degradation at 15 bps costs).

3. **Total and industry momentum remain unprofitable regardless of costs.** Both strategies have negative gross Sharpe, so transaction costs only worsen their performance. Their breakeven cost is 0 bps (never profitable).

4. **Sharpe degradation scales linearly with cost level.** For stock-specific momentum, each additional 10 bps of cost reduces net Sharpe by approximately 0.015. This linear relationship reflects the static portfolio structure.

5. **Cost structure matters less than strategy selection.** The gap between gross and net Sharpe (0.02-0.08) is an order of magnitude smaller than the gap between strategies (stock-specific at 0.54 vs total at -0.50). Strategy alpha dwarfs cost drag in this setup.

## Configuration

- Walk-forward: 5 splits, minimum 60 training days
- Lookback: 12 periods (trading days)
- Portfolio: 30% quantile long/short, equal-weighted
- Cost scenarios: 0, 8, 15, 30, 50 bps total (fee:slippage ratio varies)
- Universe: 33 Japanese stocks, 7 industries, ~5 years daily data

## Next Steps

- Phase 6: Hyperparameter optimization of lookback period and holding period
- Consider monthly resampling to better align with the paper's methodology
- Implement dynamic rebalancing within windows to test the effect on turnover and costs
