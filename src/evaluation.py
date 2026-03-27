"""
Walk-forward evaluation framework for momentum strategies.

Evaluates long-short portfolios formed on total momentum scores
using the WalkForwardValidator and backtest utilities.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import (
    BacktestConfig,
    BacktestResult,
    WalkForwardValidator,
    calculate_costs,
    compute_metrics,
    generate_metrics_json,
)
from src.decomposition import decompose_momentum

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def compute_momentum_scores(panel: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    """
    Compute momentum scores for each stock at each date.

    Uses the decomposition module to get total, industry, and stock-specific
    momentum. Returns a wide-format DataFrame indexed by date with stock_id
    columns containing total_momentum scores.

    Args:
        panel: Panel with columns [date, stock_id, industry_id, price].
        lookback: Rolling lookback window for momentum calculation.

    Returns:
        DataFrame with columns: date, stock_id, total_momentum,
        industry_momentum, stock_specific_momentum.
    """
    result = decompose_momentum(panel, lookback=lookback)
    return result.dropna(subset=["total_momentum"])


def form_long_short_portfolio(
    scores: pd.DataFrame,
    date: pd.Timestamp,
    momentum_col: str = "total_momentum",
    n_long: int | None = None,
    n_short: int | None = None,
    quantile: float = 0.3,
) -> pd.DataFrame:
    """
    Form a long-short portfolio for a given date based on momentum scores.

    Goes long the top quantile and short the bottom quantile of stocks
    ranked by momentum score.

    Args:
        scores: DataFrame with date, stock_id, and momentum columns.
        date: The date for which to form the portfolio.
        momentum_col: Column name to rank stocks on.
        n_long: Number of stocks to go long (overrides quantile).
        n_short: Number of stocks to go short (overrides quantile).
        quantile: Fraction of stocks for each leg if n_long/n_short not set.

    Returns:
        DataFrame with columns: stock_id, position (-1 or +1).
    """
    day_scores = scores[scores["date"] == date].copy()
    if day_scores.empty:
        return pd.DataFrame(columns=["stock_id", "position"])

    day_scores = day_scores.dropna(subset=[momentum_col])
    n_stocks = len(day_scores)
    if n_stocks < 2:
        return pd.DataFrame(columns=["stock_id", "position"])

    n_l = n_long or max(1, int(n_stocks * quantile))
    n_s = n_short or max(1, int(n_stocks * quantile))

    ranked = day_scores.sort_values(momentum_col, ascending=False)
    longs = ranked.head(n_l)["stock_id"].values
    shorts = ranked.tail(n_s)["stock_id"].values

    positions = []
    for sid in longs:
        positions.append({"stock_id": sid, "position": 1.0})
    for sid in shorts:
        positions.append({"stock_id": sid, "position": -1.0})

    return pd.DataFrame(positions)


def compute_portfolio_return(
    panel: pd.DataFrame,
    portfolio: pd.DataFrame,
    date: pd.Timestamp,
    next_date: pd.Timestamp,
) -> float:
    """
    Compute the equal-weighted return of a long-short portfolio over one period.

    Args:
        panel: Full panel with date, stock_id, price columns.
        portfolio: DataFrame with stock_id, position columns.
        date: Portfolio formation date.
        next_date: Date at which to measure return.

    Returns:
        Portfolio return (float). Long leg contributes positively,
        short leg negatively.
    """
    if portfolio.empty:
        return 0.0

    prices_now = panel[panel["date"] == date][["stock_id", "price"]]
    prices_next = panel[panel["date"] == next_date][["stock_id", "price"]]

    merged = (
        portfolio.merge(prices_now, on="stock_id", how="inner")
        .rename(columns={"price": "price_now"})
        .merge(prices_next, on="stock_id", how="inner")
        .rename(columns={"price": "price_next"})
    )

    if merged.empty:
        return 0.0

    merged["stock_return"] = (merged["price_next"] - merged["price_now"]) / merged["price_now"]
    # Equal-weighted: each leg gets 1/n_total weight, direction from position sign
    n_positions = len(merged)
    merged["weighted_return"] = merged["position"] * merged["stock_return"] / n_positions
    return float(merged["weighted_return"].sum())


def run_walk_forward_evaluation(
    panel: pd.DataFrame,
    config: BacktestConfig | None = None,
    lookback: int = 12,
    momentum_col: str = "total_momentum",
    quantile: float = 0.3,
) -> list[BacktestResult]:
    """
    Run walk-forward evaluation of a momentum strategy.

    For each walk-forward window:
    1. Use the training period to compute momentum scores.
    2. At each date in the test period, form a long-short portfolio
       based on the most recent momentum scores available.
    3. Compute daily portfolio returns and metrics.

    Args:
        panel: Preprocessed panel with [date, stock_id, industry_id, price].
        config: Backtest configuration. Uses defaults if None.
        lookback: Momentum lookback window in periods.
        momentum_col: Which momentum column to use for portfolio formation.
        quantile: Fraction of stocks for long/short legs.

    Returns:
        List of BacktestResult, one per walk-forward window.
    """
    config = config or BacktestConfig()
    validator = WalkForwardValidator(config)

    # Get unique sorted dates
    dates = sorted(panel["date"].unique())
    date_index = pd.DataFrame({"date": dates}).reset_index()

    results = []
    window_num = 0

    for train_idx, test_idx in validator.split(pd.DataFrame(index=range(len(dates)))):
        train_dates = [dates[i] for i in train_idx]
        test_dates = [dates[i] for i in test_idx]

        if len(train_dates) < lookback + 2 or len(test_dates) < 2:
            continue

        # Compute momentum scores on training data
        train_panel = panel[panel["date"].isin(train_dates)].copy()
        scores = compute_momentum_scores(train_panel, lookback=lookback)

        if scores.empty:
            continue

        # Get the latest available momentum scores (end of training period)
        latest_score_date = scores["date"].max()

        # Form portfolio based on latest training scores
        portfolio = form_long_short_portfolio(
            scores, latest_score_date,
            momentum_col=momentum_col, quantile=quantile,
        )

        if portfolio.empty:
            continue

        # Compute daily returns for the test period
        test_panel = panel[panel["date"].isin(test_dates)].copy()
        test_dates_sorted = sorted(test_dates)

        daily_returns = []
        positions_series = []

        for i in range(len(test_dates_sorted) - 1):
            d_now = test_dates_sorted[i]
            d_next = test_dates_sorted[i + 1]
            ret = compute_portfolio_return(panel, portfolio, d_now, d_next)
            daily_returns.append(ret)

            # Track position changes (static portfolio = same positions each day)
            n_pos = len(portfolio)
            positions_series.append(n_pos)

        if not daily_returns:
            continue

        returns_series = pd.Series(daily_returns, dtype=float)

        # Position series for cost calculation: constant positions (no rebalancing within window)
        pos_series = pd.Series(
            [1.0] * len(daily_returns), dtype=float
        )
        # First day has a position change (entering), rest are held
        pos_series.iloc[0] = 0.0  # signals initial entry

        # Compute gross metrics
        gross_metrics = compute_metrics(returns_series)

        # Compute net returns
        net_returns = calculate_costs(returns_series, pos_series, config)
        net_metrics = compute_metrics(net_returns)

        total_trades = len(portfolio)  # initial entry trades

        result = BacktestResult(
            window=window_num,
            train_start=str(pd.Timestamp(train_dates[0]).date()),
            train_end=str(pd.Timestamp(train_dates[-1]).date()),
            test_start=str(pd.Timestamp(test_dates_sorted[0]).date()),
            test_end=str(pd.Timestamp(test_dates_sorted[-1]).date()),
            gross_sharpe=gross_metrics["sharpeRatio"],
            net_sharpe=net_metrics["sharpeRatio"],
            annual_return=gross_metrics["annualReturn"],
            max_drawdown=gross_metrics["maxDrawdown"],
            total_trades=total_trades,
            hit_rate=gross_metrics["hitRate"],
            pnl_series=returns_series,
        )
        results.append(result)
        window_num += 1

        logger.info(
            "Window %d: train=%s..%s test=%s..%s gross_sharpe=%.3f net_sharpe=%.3f",
            result.window, result.train_start, result.train_end,
            result.test_start, result.test_end,
            result.gross_sharpe, result.net_sharpe,
        )

    return results


def evaluate_momentum_strategy(
    panel: pd.DataFrame,
    config: BacktestConfig | None = None,
    lookback: int = 12,
    momentum_col: str = "total_momentum",
    quantile: float = 0.3,
) -> dict:
    """
    Full evaluation pipeline: run walk-forward and generate metrics JSON.

    Args:
        panel: Preprocessed panel data.
        config: Backtest configuration.
        lookback: Momentum lookback window.
        momentum_col: Momentum column for ranking.
        quantile: Long/short quantile.

    Returns:
        Dict matching the ARF metrics.json schema.
    """
    config = config or BacktestConfig()
    results = run_walk_forward_evaluation(
        panel, config=config, lookback=lookback,
        momentum_col=momentum_col, quantile=quantile,
    )

    custom_metrics = {
        "strategy": f"long_short_{momentum_col}",
        "lookback_periods": lookback,
        "quantile": quantile,
        "momentum_column": momentum_col,
    }

    metrics = generate_metrics_json(results, config, custom_metrics=custom_metrics)
    return metrics


def save_metrics(metrics: dict, cycle: int = 3) -> Path:
    """Save metrics.json to the reports directory."""
    out_dir = REPORTS_DIR / f"cycle_{cycle}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", out_path)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load panel data
    from src.data_fetcher import build_panel, DATA_DIR

    panel_path = DATA_DIR / "jp_stocks_panel.csv"
    if panel_path.exists():
        print("Loading cached panel data...")
        panel = pd.read_csv(panel_path, parse_dates=["date"])
    else:
        print("Building panel from API...")
        panel, report = build_panel()
        panel.to_csv(panel_path, index=False)

    # Run evaluation
    print("\nRunning walk-forward evaluation of total momentum strategy...")
    config = BacktestConfig(n_splits=5, min_train_size=60)
    metrics = evaluate_momentum_strategy(panel, config=config, lookback=12)

    # Save results
    save_metrics(metrics, cycle=3)
    print("\n=== Metrics ===")
    print(json.dumps(metrics, indent=2))
