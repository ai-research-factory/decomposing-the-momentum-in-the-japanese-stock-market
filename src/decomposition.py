"""
Core momentum decomposition algorithm.

Decomposes total stock momentum into industry momentum and
stock-specific (residual) momentum components via cross-sectional
regression of stock returns on industry returns.
"""
import numpy as np
import pandas as pd


def _compute_returns(prices: pd.Series, period: int = 1) -> pd.Series:
    """Compute period-over-period returns."""
    return prices.pct_change(period)


def decompose_momentum(
    df: pd.DataFrame,
    lookback: int = 12,
    return_period: int = 1,
) -> pd.DataFrame:
    """
    Decompose stock momentum into industry and stock-specific components.

    For each stock at each date:
      1. Compute total return over the past `return_period` months.
      2. Compute industry return (equal-weighted average of stock returns
         within the same industry).
      3. Regress each stock's return on its industry return over the past
         `lookback` periods to obtain residuals.
      4. Compute momentum as the cumulative sum of each component over
         the lookback window.

    Args:
        df: DataFrame with columns [date, stock_id, industry_id, price].
            `date` should be parseable as datetime; data should be sorted
            by date within each stock.
        lookback: Number of periods for momentum calculation.
        return_period: Number of periods for return calculation.

    Returns:
        DataFrame with columns:
            date, stock_id, industry_id,
            total_momentum, industry_momentum, stock_specific_momentum
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # Step 1: Compute per-stock returns
    df["return"] = df.groupby("stock_id")["price"].transform(
        lambda x: _compute_returns(x, return_period)
    )

    # Step 2: Compute industry returns (equal-weighted mean of stock returns)
    industry_returns = (
        df.groupby(["date", "industry_id"])["return"]
        .mean()
        .rename("industry_return")
    )
    df = df.merge(industry_returns, on=["date", "industry_id"], how="left")

    # Step 3: Regress stock returns on industry returns and compute residuals
    # For each stock, run a rolling OLS: return_i = alpha + beta * industry_return + epsilon
    # The residual (epsilon) is the stock-specific return.
    df["residual_return"] = np.nan

    for stock_id, group in df.groupby("stock_id"):
        mask = group["return"].notna() & group["industry_return"].notna()
        valid = group[mask]
        if len(valid) < lookback + 1:
            continue

        stock_ret = valid["return"].values
        ind_ret = valid["industry_return"].values
        residuals = np.full(len(valid), np.nan)

        for i in range(lookback, len(valid)):
            window_stock = stock_ret[i - lookback : i + 1]
            window_ind = ind_ret[i - lookback : i + 1]

            # Simple OLS: y = alpha + beta * x
            x = window_ind
            y = window_stock
            x_mean = x.mean()
            y_mean = y.mean()
            denom = ((x - x_mean) ** 2).sum()
            if denom < 1e-15:
                # Industry return is constant; residual = stock return - mean
                residuals[i] = y[-1] - y_mean
            else:
                beta = ((x - x_mean) * (y - y_mean)).sum() / denom
                alpha = y_mean - beta * x_mean
                residuals[i] = y[-1] - (alpha + beta * x[-1])

        df.loc[valid.index, "residual_return"] = residuals

    # Step 4: Compute momentum scores as rolling sum of returns
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    df["total_momentum"] = df.groupby("stock_id")["return"].transform(
        lambda x: x.rolling(window=lookback, min_periods=lookback).sum()
    )

    df["industry_momentum"] = df.groupby("stock_id")["industry_return"].transform(
        lambda x: x.rolling(window=lookback, min_periods=lookback).sum()
    )

    df["stock_specific_momentum"] = df.groupby("stock_id")["residual_return"].transform(
        lambda x: x.rolling(window=lookback, min_periods=lookback).sum()
    )

    # Select output columns
    result = df[
        [
            "date",
            "stock_id",
            "industry_id",
            "total_momentum",
            "industry_momentum",
            "stock_specific_momentum",
        ]
    ].copy()

    return result
