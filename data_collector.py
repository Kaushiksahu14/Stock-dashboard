"""
Stock Data Collection & Preparation Module
==========================================
Fetches stock market data using yfinance (or generates realistic mock data as fallback).
Cleans, transforms, and stores data in SQLite database.

Calculated Metrics:
  - Daily Return = (CLOSE - OPEN) / OPEN
  - 7-day Moving Average
  - 52-week High / Low
  - Volatility Score (custom: 30-day rolling std of daily returns)
  - Momentum Index (custom: RSI-like indicator)
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "stocks.db"

# Indian NSE stocks to track
COMPANIES = {
    "RELIANCE": {"name": "Reliance Industries Ltd", "sector": "Energy"},
    "TCS": {"name": "Tata Consultancy Services", "sector": "IT"},
    "INFY": {"name": "Infosys Ltd", "sector": "IT"},
    "HDFCBANK": {"name": "HDFC Bank Ltd", "sector": "Banking"},
    "ICICIBANK": {"name": "ICICI Bank Ltd", "sector": "Banking"},
    "HINDUNILVR": {"name": "Hindustan Unilever Ltd", "sector": "FMCG"},
    "SBIN": {"name": "State Bank of India", "sector": "Banking"},
    "BHARTIARTL": {"name": "Bharti Airtel Ltd", "sector": "Telecom"},
    "ITC": {"name": "ITC Ltd", "sector": "FMCG"},
    "WIPRO": {"name": "Wipro Ltd", "sector": "IT"},
}

# Realistic base prices (approx INR values)
BASE_PRICES = {
    "RELIANCE": 2450, "TCS": 3800, "INFY": 1550, "HDFCBANK": 1650,
    "ICICIBANK": 1050, "HINDUNILVR": 2500, "SBIN": 620,
    "BHARTIARTL": 1150, "ITC": 430, "WIPRO": 480,
}


def fetch_data_yfinance() -> dict[str, pd.DataFrame]:
    """Try fetching real data from yfinance. Returns dict of symbol -> DataFrame."""
    try:
        import yfinance as yf
        data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=400)  # ~13 months for 52-week calc

        for symbol in COMPANIES:
            ticker = f"{symbol}.NS"
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not df.empty:
                df = df.reset_index()
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df = df.rename(columns={
                    "Date": "date", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume"
                })
                df["symbol"] = symbol
                data[symbol] = df[["date", "symbol", "open", "high", "low", "close", "volume"]]
                print(f"  ✓ Fetched {len(df)} rows for {symbol}")
        return data
    except Exception as e:
        print(f"  ⚠ yfinance unavailable ({e}), using mock data generator")
        return {}


def generate_mock_data() -> dict[str, pd.DataFrame]:
    """Generate realistic mock stock data with random walks and market patterns."""
    print("  Generating realistic mock stock data...")
    np.random.seed(42)
    data = {}
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=400)

    # Generate trading days (exclude weekends)
    all_days = pd.bdate_range(start=start_date, end=end_date)

    for symbol, base_price in BASE_PRICES.items():
        n = len(all_days)
        # Geometric Brownian Motion for realistic price simulation
        drift = 0.0003  # slight upward bias
        volatility = np.random.uniform(0.015, 0.028)

        # Generate daily returns with mean-reverting component
        daily_returns = np.random.normal(drift, volatility, n)

        # Add some market-wide correlation (simulate market events)
        market_shock = np.zeros(n)
        for _ in range(5):  # 5 random market events
            shock_day = np.random.randint(0, n)
            shock_magnitude = np.random.normal(0, 0.03)
            market_shock[shock_day] = shock_magnitude
        daily_returns += market_shock * 0.5

        # Build price series
        close_prices = [base_price]
        for r in daily_returns[1:]:
            close_prices.append(close_prices[-1] * (1 + r))
        close_prices = np.array(close_prices)

        # Generate OHLV from close
        open_prices = close_prices * (1 + np.random.normal(0, 0.005, n))
        high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.008, n)))
        low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.008, n)))
        volumes = np.random.lognormal(mean=np.log(5_000_000), sigma=0.5, size=n).astype(int)

        df = pd.DataFrame({
            "date": all_days[:n],
            "symbol": symbol,
            "open": np.round(open_prices, 2),
            "high": np.round(high_prices, 2),
            "low": np.round(low_prices, 2),
            "close": np.round(close_prices, 2),
            "volume": volumes,
        })
        data[symbol] = df
        print(f"  ✓ Generated {len(df)} rows for {symbol}")

    return data


def clean_and_transform(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Clean data and add calculated metrics:
      - daily_return: (close - open) / open
      - ma_7: 7-day moving average of close
      - high_52w: 52-week rolling high
      - low_52w: 52-week rolling low
      - volatility: 30-day rolling std of daily returns (custom metric)
      - momentum: RSI-like momentum index (custom metric)
    """
    print("\n📊 Cleaning and transforming data...")
    all_frames = []

    for symbol, df in data.items():
        df = df.copy()

        # --- Cleaning ---
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Handle missing values
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

        # Ensure high >= max(open, close) and low <= min(open, close)
        df["high"] = df[["high", "open", "close"]].max(axis=1)
        df["low"] = df[["low", "open", "close"]].min(axis=1)

        # --- Calculated Metrics ---
        # Daily Return
        df["daily_return"] = ((df["close"] - df["open"]) / df["open"]).round(6)

        # 7-day Moving Average
        df["ma_7"] = df["close"].rolling(window=7, min_periods=1).mean().round(2)

        # 52-week High/Low (252 trading days ≈ 1 year)
        df["high_52w"] = df["high"].rolling(window=252, min_periods=1).max().round(2)
        df["low_52w"] = df["low"].rolling(window=252, min_periods=1).min().round(2)

        # --- Custom Metrics ---
        # Volatility Score: 30-day rolling std of daily returns (annualized)
        df["volatility"] = (
            df["daily_return"].rolling(window=30, min_periods=5).std() * np.sqrt(252)
        ).round(4)

        # Momentum Index (simplified RSI)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df["momentum"] = (100 - (100 / (1 + rs))).round(2)
        df["momentum"] = df["momentum"].fillna(50)

        all_frames.append(df)
        print(f"  ✓ Transformed {symbol}: {len(df)} records, {len(df.columns)} columns")

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\n  Total records: {len(combined)}")
    return combined


def store_to_database(df: pd.DataFrame):
    """Store cleaned data and company info in SQLite."""
    print("\n💾 Storing data in SQLite database...")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    # Companies table
    companies_df = pd.DataFrame([
        {"symbol": sym, "name": info["name"], "sector": info["sector"]}
        for sym, info in COMPANIES.items()
    ])
    companies_df.to_sql("companies", conn, if_exists="replace", index=False)

    # Stock data table
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_sql("stock_data", conn, if_exists="replace", index=False)

    # Create indexes for fast queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_symbol ON stock_data(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_date ON stock_data(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_symbol_date ON stock_data(symbol, date)")

    conn.commit()
    conn.close()
    print(f"  ✓ Database saved to {DB_PATH}")


def run_collection():
    """Main data pipeline: fetch → clean → store."""
    print("=" * 60)
    print("🚀 Stock Data Collection Pipeline")
    print("=" * 60)

    # Try real data first, fall back to mock
    print("\n📡 Attempting to fetch real data via yfinance...")
    data = fetch_data_yfinance()
    if not data:
        data = generate_mock_data()

    # Clean and transform
    df = clean_and_transform(data)

    # Store in database
    store_to_database(df)

    # Print summary
    print("\n" + "=" * 60)
    print("✅ Data collection complete!")
    print(f"   Companies: {len(COMPANIES)}")
    print(f"   Total records: {len(df)}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Columns: {', '.join(df.columns)}")
    print("=" * 60)


if __name__ == "__main__":
    run_collection()
