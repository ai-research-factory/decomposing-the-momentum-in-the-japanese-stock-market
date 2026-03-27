"""
Japanese stock data pipeline using the ARF Data API.

Fetches OHLCV data for Japanese stocks across TOPIX-17 sector classifications,
preprocesses it (date alignment, missing data handling, outlier detection),
and produces a clean panel DataFrame for momentum decomposition.
"""
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

API_BASE = "https://ai-research-factory-618707091171.asia-northeast1.run.app/api/data/ohlcv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# TOPIX-17 sector mapping for all available Japanese stocks
STOCK_INDUSTRY_MAP: dict[str, str] = {
    # 自動車・輸送機器 (Automobiles & Transportation Equipment)
    "7203.T": "auto",
    "7267.T": "auto",
    "7974.T": "auto",
    # 電機・精密 (Electronics & Precision)
    "6758.T": "electronics",
    "6861.T": "electronics",
    "6902.T": "electronics",
    "6501.T": "electronics",
    "6594.T": "electronics",
    "8035.T": "electronics",
    "6857.T": "electronics",
    "6723.T": "electronics",
    # 金融 (Finance)
    "8306.T": "finance",
    "8316.T": "finance",
    "8411.T": "finance",
    "8766.T": "finance",
    "8591.T": "finance",
    # 商社・素材 (Trading & Materials)
    "8058.T": "trading",
    "8031.T": "trading",
    "8001.T": "trading",
    "8053.T": "trading",
    "8002.T": "trading",
    # 通信・IT・サービス (Telecom/IT/Services)
    "9984.T": "telecom_it",
    "9432.T": "telecom_it",
    "9433.T": "telecom_it",
    "9434.T": "telecom_it",
    "4661.T": "telecom_it",
    "6098.T": "telecom_it",
    # 医薬・ヘルスケア (Pharma/Healthcare)
    "4502.T": "pharma",
    "4519.T": "pharma",
    "4568.T": "pharma",
    # 消費・小売 (Consumer/Retail)
    "9983.T": "retail",
    "2914.T": "retail",
    "3382.T": "retail",
}

# Minimum stocks per industry to include in the panel
MIN_STOCKS_PER_INDUSTRY = 2

# Outlier detection: daily returns beyond this z-score are flagged
OUTLIER_ZSCORE_THRESHOLD = 5.0


@dataclass
class DataQualityReport:
    """Summary of data quality after pipeline processing."""
    total_stocks_requested: int = 0
    total_stocks_fetched: int = 0
    failed_tickers: list[str] = field(default_factory=list)
    n_industries: int = 0
    date_range_start: str = ""
    date_range_end: str = ""
    total_observations: int = 0
    missing_price_count: int = 0
    missing_price_pct: float = 0.0
    forward_filled_count: int = 0
    outlier_count: int = 0
    stocks_per_industry: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_stocks_requested": self.total_stocks_requested,
            "total_stocks_fetched": self.total_stocks_fetched,
            "failed_tickers": self.failed_tickers,
            "n_industries": self.n_industries,
            "date_range": f"{self.date_range_start} to {self.date_range_end}",
            "total_observations": self.total_observations,
            "missing_price_pct": round(self.missing_price_pct, 4),
            "forward_filled_count": self.forward_filled_count,
            "outlier_count": self.outlier_count,
            "stocks_per_industry": self.stocks_per_industry,
        }


def fetch_stock_data(
    ticker: str,
    interval: str = "1d",
    period: str = "5y",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single ticker from the ARF Data API.

    Uses local file cache to avoid redundant API calls.

    Args:
        ticker: Stock ticker symbol (e.g. "7203.T").
        interval: Data interval ("1d", "1h", etc.).
        period: Historical period ("5y", "10y", etc.).
        cache_dir: Directory to cache CSV files. Defaults to data/.

    Returns:
        DataFrame with columns from the API (timestamp, open, high, low, close, volume).

    Raises:
        ValueError: If the API returns no data for the ticker.
        requests.HTTPError: If the API request fails.
    """
    cache_dir = cache_dir or DATA_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_ticker = ticker.replace("/", "_")
    cache_path = cache_dir / f"{safe_ticker}_{interval}_{period}.csv"

    if cache_path.exists():
        logger.debug("Cache hit for %s at %s", ticker, cache_path)
        df = pd.read_csv(cache_path, parse_dates=["timestamp"])
        return df

    url = f"{API_BASE}?ticker={ticker}&interval={interval}&period={period}&format=json"
    logger.info("Fetching %s from API...", ticker)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", [])
    if not data:
        raise ValueError(f"No data returned for {ticker}")

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.to_csv(cache_path, index=False)
    return df


def _align_dates(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Align all stocks to a common set of trading dates.

    Creates a full (date x stock) grid so that missing dates for any stock
    become explicit NaN rows, which can then be forward-filled.
    """
    all_dates = panel["date"].unique()
    all_stocks = panel["stock_id"].unique()

    full_index = pd.MultiIndex.from_product(
        [all_dates, all_stocks], names=["date", "stock_id"]
    )
    panel = panel.set_index(["date", "stock_id"]).reindex(full_index).reset_index()

    # Propagate industry_id for filled rows
    stock_industry = (
        panel.dropna(subset=["industry_id"])
        .drop_duplicates("stock_id")[["stock_id", "industry_id"]]
    )
    panel = panel.drop(columns=["industry_id"]).merge(stock_industry, on="stock_id", how="left")
    return panel.sort_values(["stock_id", "date"]).reset_index(drop=True)


def _forward_fill_prices(panel: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Forward-fill missing prices within each stock's time series.

    Returns the filled panel and the count of forward-filled values.
    """
    missing_before = panel["price"].isna().sum()
    panel["price"] = panel.groupby("stock_id")["price"].transform(
        lambda s: s.ffill()
    )
    filled_count = missing_before - panel["price"].isna().sum()
    return panel, int(filled_count)


def _detect_outliers(panel: pd.DataFrame, z_threshold: float = OUTLIER_ZSCORE_THRESHOLD) -> pd.DataFrame:
    """
    Flag daily returns that exceed a z-score threshold as outliers.

    Adds an 'is_outlier' boolean column. Does NOT remove outliers —
    downstream consumers decide how to handle them.
    """
    panel = panel.copy()
    panel["daily_return"] = panel.groupby("stock_id")["price"].pct_change()

    mean_ret = panel["daily_return"].mean()
    std_ret = panel["daily_return"].std()

    if std_ret > 0:
        panel["return_zscore"] = (panel["daily_return"] - mean_ret) / std_ret
        panel["is_outlier"] = panel["return_zscore"].abs() > z_threshold
    else:
        panel["return_zscore"] = 0.0
        panel["is_outlier"] = False

    panel = panel.drop(columns=["daily_return", "return_zscore"])
    return panel


def _validate_panel(panel: pd.DataFrame) -> None:
    """Validate that the panel has the expected structure."""
    required_cols = {"date", "stock_id", "industry_id", "price"}
    missing = required_cols - set(panel.columns)
    if missing:
        raise ValueError(f"Panel missing required columns: {missing}")

    if panel.empty:
        raise ValueError("Panel is empty — no data was fetched")

    # Check for duplicate (date, stock_id) pairs
    dupes = panel.duplicated(subset=["date", "stock_id"], keep=False)
    if dupes.any():
        n_dupes = dupes.sum()
        logger.warning("Found %d duplicate (date, stock_id) rows — dropping duplicates", n_dupes)
        panel.drop_duplicates(subset=["date", "stock_id"], keep="first", inplace=True)


def build_panel(
    stock_map: dict[str, str] | None = None,
    interval: str = "1d",
    period: str = "5y",
    cache_dir: Path | None = None,
    rate_limit_sec: float = 0.5,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Fetch all stocks, preprocess, and build a clean panel DataFrame.

    Pipeline steps:
      1. Fetch raw OHLCV data for each ticker (with caching).
      2. Align all stocks to a common date grid.
      3. Forward-fill small price gaps.
      4. Drop stocks that still have excessive missing data (>10%).
      5. Detect outliers (flag but do not remove).
      6. Validate and return.

    Args:
        stock_map: Dict mapping ticker -> industry_id. Defaults to STOCK_INDUSTRY_MAP.
        interval: Data interval for the API.
        period: Historical period for the API.
        cache_dir: Cache directory for raw CSV files.
        rate_limit_sec: Seconds to sleep between API calls.

    Returns:
        Tuple of (panel DataFrame, DataQualityReport).
    """
    stock_map = stock_map or STOCK_INDUSTRY_MAP
    report = DataQualityReport(total_stocks_requested=len(stock_map))
    cache_dir = cache_dir or DATA_DIR

    # Step 1: Fetch raw data
    frames = []
    for ticker, industry in stock_map.items():
        try:
            df = fetch_stock_data(ticker, interval, period, cache_dir=cache_dir)
            row = pd.DataFrame({
                "date": pd.to_datetime(df["timestamp"]).dt.normalize(),
                "stock_id": ticker,
                "industry_id": industry,
                "price": df["close"].astype(float),
            })
            frames.append(row)
            logger.info("Fetched %s (%s): %d rows", ticker, industry, len(df))
            time.sleep(rate_limit_sec)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", ticker, e)
            report.failed_tickers.append(ticker)

    if not frames:
        raise RuntimeError("Failed to fetch any stock data from the API")

    panel = pd.concat(frames, ignore_index=True)
    report.total_stocks_fetched = panel["stock_id"].nunique()

    # Step 2: Align dates
    panel = _align_dates(panel)

    # Count raw missing
    total_cells = len(panel)
    raw_missing = int(panel["price"].isna().sum())
    report.missing_price_count = raw_missing
    report.missing_price_pct = raw_missing / total_cells if total_cells > 0 else 0.0

    # Step 3: Forward-fill prices
    panel, filled_count = _forward_fill_prices(panel)
    report.forward_filled_count = filled_count

    # Step 4: Drop stocks with >10% missing after forward-fill
    stock_missing = panel.groupby("stock_id")["price"].apply(lambda s: s.isna().mean())
    drop_stocks = stock_missing[stock_missing > 0.10].index.tolist()
    if drop_stocks:
        logger.warning("Dropping stocks with >10%% missing data: %s", drop_stocks)
        panel = panel[~panel["stock_id"].isin(drop_stocks)].reset_index(drop=True)
        report.failed_tickers.extend(drop_stocks)

    # Drop any remaining rows without prices (start-of-series before first observation)
    panel = panel.dropna(subset=["price"]).reset_index(drop=True)

    # Step 5: Ensure minimum stocks per industry
    industry_counts = panel.groupby("industry_id")["stock_id"].nunique()
    thin_industries = industry_counts[industry_counts < MIN_STOCKS_PER_INDUSTRY].index.tolist()
    if thin_industries:
        logger.warning("Dropping industries with fewer than %d stocks: %s", MIN_STOCKS_PER_INDUSTRY, thin_industries)
        panel = panel[~panel["industry_id"].isin(thin_industries)].reset_index(drop=True)

    # Step 6: Detect outliers
    panel = _detect_outliers(panel)
    report.outlier_count = int(panel["is_outlier"].sum())

    # Step 7: Validate
    _validate_panel(panel)

    # Finalize report
    report.n_industries = panel["industry_id"].nunique()
    report.date_range_start = str(panel["date"].min().date())
    report.date_range_end = str(panel["date"].max().date())
    report.total_observations = len(panel)
    report.stocks_per_industry = (
        panel.groupby("industry_id")["stock_id"].nunique().to_dict()
    )

    # Sort for consistent output
    panel = panel.sort_values(["date", "stock_id"]).reset_index(drop=True)
    return panel, report


def save_panel(panel: pd.DataFrame, filename: str = "jp_stocks_panel.csv") -> Path:
    """Save the panel to the data directory. Returns the file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / filename
    panel.to_csv(out_path, index=False)
    logger.info("Saved panel (%d rows) to %s", len(panel), out_path)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Building Japanese stock data panel from ARF Data API...")
    panel, report = build_panel()
    save_panel(panel)

    print(f"\n=== Data Quality Report ===")
    for k, v in report.to_dict().items():
        print(f"  {k}: {v}")
    print(f"\nPanel shape: {panel.shape}")
    print(f"Columns: {list(panel.columns)}")
