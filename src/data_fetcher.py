"""
Fetch Japanese stock data from the ARF Data API for momentum decomposition.

Downloads OHLCV data for stocks across 3 industries (TOPIX sector classification)
and prepares a panel DataFrame with columns: date, stock_id, industry_id, price.
"""
import pandas as pd
import requests
import os
import time

API_BASE = "https://ai-research-factory-618707091171.asia-northeast1.run.app/api/data/ohlcv"

# 3 industries, ~10 stocks total
STOCK_INDUSTRY_MAP = {
    # 自動車・輸送機器 (Automobiles & Transportation Equipment)
    "7203.T": "auto",
    "7267.T": "auto",
    "7974.T": "auto",
    # 電機・精密 (Electronics & Precision)
    "6758.T": "electronics",
    "6501.T": "electronics",
    "8035.T": "electronics",
    "6857.T": "electronics",
    # 金融 (Finance)
    "8306.T": "finance",
    "8316.T": "finance",
    "8411.T": "finance",
}

CACHE_DIR = os.path.join(os.path.dirname(__file__))


def fetch_stock_data(ticker: str, interval: str = "1d", period: str = "5y") -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker from ARF Data API."""
    cache_path = os.path.join(CACHE_DIR, f"{ticker.replace('/', '_')}_{interval}_{period}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["timestamp"])
        return df

    url = f"{API_BASE}?ticker={ticker}&interval={interval}&period={period}&format=json"
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


def build_panel(interval: str = "1d", period: str = "5y") -> pd.DataFrame:
    """
    Fetch all stocks and build a panel DataFrame.

    Returns:
        DataFrame with columns: date, stock_id, industry_id, price
    """
    frames = []
    for ticker, industry in STOCK_INDUSTRY_MAP.items():
        try:
            df = fetch_stock_data(ticker, interval, period)
            panel = pd.DataFrame({
                "date": df["timestamp"],
                "stock_id": ticker,
                "industry_id": industry,
                "price": df["close"],
            })
            frames.append(panel)
            print(f"  Fetched {ticker} ({industry}): {len(df)} rows")
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"  WARNING: Failed to fetch {ticker}: {e}")

    if not frames:
        raise RuntimeError("Failed to fetch any stock data")

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["date", "stock_id"]).reset_index(drop=True)
    return result


if __name__ == "__main__":
    print("Fetching Japanese stock data from ARF Data API...")
    panel = build_panel()
    out_path = os.path.join(CACHE_DIR, "jp_stocks_panel.csv")
    panel.to_csv(out_path, index=False)
    print(f"\nSaved panel data to {out_path}")
    print(f"Shape: {panel.shape}")
    print(f"Stocks: {panel['stock_id'].nunique()}")
    print(f"Industries: {panel['industry_id'].nunique()}")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
