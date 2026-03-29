"""
Stock Data Intelligence Dashboard — FastAPI Backend
====================================================
REST API endpoints for stock market data access and analysis.

Endpoints:
  GET  /companies                          → List all companies
  GET  /data/{symbol}                      → Last 30 days of stock data
  GET  /data/{symbol}?days=90              → Custom date range
  GET  /summary/{symbol}                   → 52-week high, low, avg close
  GET  /compare?symbol1=INFY&symbol2=TCS   → Compare two stocks
  GET  /top-movers                         → Top gainers and losers today
  GET  /sectors                            → Sector-wise performance
  GET  /predict/{symbol}                   → ML-based 7-day price prediction

Swagger docs available at /docs
"""

import sqlite3
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

DB_PATH = Path(__file__).parent / "data" / "stocks.db"


# ─── Database Helper ─────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


# ─── App Setup ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure database exists on startup."""
    if not DB_PATH.exists():
        print("Database not found. Running data collector...")
        from data_collector import run_collection
        run_collection()
    yield


app = FastAPI(
    title="Stock Data Intelligence Dashboard API",
    description="REST API for Indian stock market data with analytics and insights.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ─── Routes ──────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the dashboard frontend."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Stock Data Intelligence Dashboard API", "docs": "/docs"}


@app.get("/companies", tags=["Companies"])
async def get_companies():
    """
    Returns a list of all available companies with their symbol, name, and sector.
    Also includes the latest closing price and daily return for each.
    """
    df = query_df("""
        SELECT c.symbol, c.name, c.sector,
               s.close AS latest_price, s.daily_return, s.date AS latest_date,
               s.momentum, s.volatility
        FROM companies c
        LEFT JOIN stock_data s ON c.symbol = s.symbol
            AND s.date = (SELECT MAX(date) FROM stock_data WHERE symbol = c.symbol)
        ORDER BY c.symbol
    """)
    return {
        "count": len(df),
        "companies": df.to_dict(orient="records")
    }


@app.get("/data/{symbol}", tags=["Stock Data"])
async def get_stock_data(
    symbol: str,
    days: int = Query(default=30, ge=1, le=365, description="Number of trading days to return")
):
    """
    Returns stock data for a given symbol.
    Default: last 30 trading days. Use `days` parameter to customize.
    Includes OHLCV data plus all calculated metrics.
    """
    symbol = symbol.upper()

    # Validate symbol exists
    check = query_df("SELECT 1 FROM companies WHERE symbol = ?", (symbol,))
    if check.empty:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    df = query_df("""
        SELECT date, symbol, open, high, low, close, volume,
               daily_return, ma_7, high_52w, low_52w, volatility, momentum
        FROM stock_data
        WHERE symbol = ?
        ORDER BY date DESC
        LIMIT ?
    """, (symbol, days))

    return {
        "symbol": symbol,
        "count": len(df),
        "data": df.to_dict(orient="records")
    }


@app.get("/summary/{symbol}", tags=["Analytics"])
async def get_summary(symbol: str):
    """
    Returns 52-week summary for a given stock:
    high, low, average close, current price, total return, volatility, and momentum.
    """
    symbol = symbol.upper()

    check = query_df("SELECT name, sector FROM companies WHERE symbol = ?", (symbol,))
    if check.empty:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    df = query_df("""
        SELECT date, open, high, low, close, volume,
               daily_return, ma_7, volatility, momentum
        FROM stock_data
        WHERE symbol = ?
        ORDER BY date DESC
    """, (symbol,))

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for '{symbol}'")

    latest = df.iloc[0]
    oldest = df.iloc[-1]

    return {
        "symbol": symbol,
        "name": check.iloc[0]["name"],
        "sector": check.iloc[0]["sector"],
        "summary": {
            "high_52w": round(float(df["high"].max()), 2),
            "low_52w": round(float(df["low"].min()), 2),
            "avg_close": round(float(df["close"].mean()), 2),
            "current_price": round(float(latest["close"]), 2),
            "total_return_pct": round(
                ((float(latest["close"]) - float(oldest["close"])) / float(oldest["close"])) * 100, 2
            ),
            "avg_volume": int(df["volume"].mean()),
            "current_volatility": round(float(latest["volatility"]) if pd.notna(latest["volatility"]) else 0, 4),
            "current_momentum": round(float(latest["momentum"]) if pd.notna(latest["momentum"]) else 50, 2),
            "data_points": len(df),
            "date_range": {
                "from": df["date"].iloc[-1],
                "to": df["date"].iloc[0],
            }
        }
    }


@app.get("/compare", tags=["Analytics"])
async def compare_stocks(
    symbol1: str = Query(..., description="First stock symbol"),
    symbol2: str = Query(..., description="Second stock symbol"),
):
    """
    Compare two stocks' performance over the available data range.
    Returns side-by-side metrics and correlation analysis.
    """
    symbol1, symbol2 = symbol1.upper(), symbol2.upper()

    for sym in [symbol1, symbol2]:
        check = query_df("SELECT 1 FROM companies WHERE symbol = ?", (sym,))
        if check.empty:
            raise HTTPException(status_code=404, detail=f"Symbol '{sym}' not found")

    df1 = query_df(
        "SELECT date, close, daily_return, volatility, momentum FROM stock_data WHERE symbol = ? ORDER BY date",
        (symbol1,)
    )
    df2 = query_df(
        "SELECT date, close, daily_return, volatility, momentum FROM stock_data WHERE symbol = ? ORDER BY date",
        (symbol2,)
    )

    # Merge on common dates
    merged = pd.merge(df1, df2, on="date", suffixes=(f"_{symbol1}", f"_{symbol2}"))

    # Calculate correlation
    correlation = round(
        merged[f"daily_return_{symbol1}"].corr(merged[f"daily_return_{symbol2}"]), 4
    )

    def stock_stats(df, sym):
        return {
            "symbol": sym,
            "current_price": round(float(df["close"].iloc[-1]), 2),
            "total_return_pct": round(
                ((float(df["close"].iloc[-1]) - float(df["close"].iloc[0])) / float(df["close"].iloc[0])) * 100, 2
            ),
            "avg_daily_return": round(float(df["daily_return"].mean()) * 100, 4),
            "max_daily_gain": round(float(df["daily_return"].max()) * 100, 2),
            "max_daily_loss": round(float(df["daily_return"].min()) * 100, 2),
            "volatility": round(float(df["volatility"].dropna().iloc[-1]) if not df["volatility"].dropna().empty else 0, 4),
        }

    # Normalized price series for charting (base 100)
    base1 = float(merged[f"close_{symbol1}"].iloc[0])
    base2 = float(merged[f"close_{symbol2}"].iloc[0])
    chart_data = []
    for _, row in merged.iterrows():
        chart_data.append({
            "date": row["date"],
            f"{symbol1}": round((float(row[f"close_{symbol1}"]) / base1) * 100, 2),
            f"{symbol2}": round((float(row[f"close_{symbol2}"]) / base2) * 100, 2),
        })

    return {
        "comparison": {
            "stock1": stock_stats(df1, symbol1),
            "stock2": stock_stats(df2, symbol2),
            "correlation": correlation,
            "correlation_interpretation": (
                "Strong positive" if correlation > 0.7 else
                "Moderate positive" if correlation > 0.3 else
                "Weak/No correlation" if correlation > -0.3 else
                "Moderate negative" if correlation > -0.7 else
                "Strong negative"
            ),
            "common_trading_days": len(merged),
        },
        "normalized_prices": chart_data,
    }


@app.get("/top-movers", tags=["Analytics"])
async def get_top_movers():
    """Returns today's top gainers and losers based on daily return."""
    df = query_df("""
        SELECT s.symbol, c.name, s.close, s.daily_return, s.volume, s.date
        FROM stock_data s
        JOIN companies c ON s.symbol = c.symbol
        WHERE s.date = (SELECT MAX(date) FROM stock_data)
        ORDER BY s.daily_return DESC
    """)

    if df.empty:
        return {"gainers": [], "losers": []}

    records = df.to_dict(orient="records")
    return {
        "date": records[0]["date"] if records else None,
        "gainers": records[:3],
        "losers": list(reversed(records[-3:])),
    }


@app.get("/sectors", tags=["Analytics"])
async def get_sector_performance():
    """Returns average performance grouped by sector."""
    df = query_df("""
        SELECT c.sector,
               AVG(s.daily_return) AS avg_daily_return,
               AVG(s.volatility) AS avg_volatility,
               COUNT(DISTINCT c.symbol) AS stock_count
        FROM stock_data s
        JOIN companies c ON s.symbol = c.symbol
        WHERE s.date = (SELECT MAX(date) FROM stock_data)
        GROUP BY c.sector
        ORDER BY avg_daily_return DESC
    """)
    return {"sectors": df.to_dict(orient="records")}


@app.get("/predict/{symbol}", tags=["AI / ML"])
async def predict_stock(
    symbol: str,
    days: int = Query(default=7, ge=1, le=30, description="Number of days to predict"),
):
    """
    Predicts future stock prices using a Linear Regression model.

    **Model**: scikit-learn LinearRegression trained on recent 90 days of data.
    **Features**: Day index, 7-day MA, momentum (RSI), volatility, and lagged close prices.
    **Output**: Predicted closing prices for the next N trading days.

    ⚠️ Disclaimer: This is a simple ML model for demonstration purposes only.
    It should NOT be used for actual trading decisions.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    symbol = symbol.upper()

    check = query_df("SELECT name FROM companies WHERE symbol = ?", (symbol,))
    if check.empty:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    # Fetch last 120 days for training (gives buffer for feature engineering)
    df = query_df("""
        SELECT date, close, ma_7, momentum, volatility, daily_return
        FROM stock_data
        WHERE symbol = ?
        ORDER BY date ASC
    """, (symbol,))

    if len(df) < 30:
        raise HTTPException(status_code=400, detail="Not enough data to train prediction model")

    # ── Feature Engineering ──
    df["day_index"] = np.arange(len(df))
    df["close_lag1"] = df["close"].shift(1)
    df["close_lag3"] = df["close"].shift(3)
    df["close_lag7"] = df["close"].shift(7)
    df["return_ma5"] = df["daily_return"].rolling(5).mean()
    df = df.dropna()

    feature_cols = ["day_index", "ma_7", "momentum", "volatility", "close_lag1", "close_lag3", "close_lag7", "return_ma5"]
    X = df[feature_cols].values
    y = df["close"].values

    # Use last 90 days for training (more recent = more relevant)
    train_size = min(90, len(X))
    X_train = X[-train_size:]
    y_train = y[-train_size:]

    # ── Train Model ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Training R² score
    train_score = round(model.score(X_train_scaled, y_train), 4)

    # ── Predict Future Days ──
    predictions = []
    last_row = df.iloc[-1].copy()
    last_date = pd.to_datetime(df["date"].iloc[-1])
    current_close = float(last_row["close"])
    current_ma7 = float(last_row["ma_7"])
    current_momentum = float(last_row["momentum"])
    current_volatility = float(last_row["volatility"])
    lag1 = current_close
    lag3 = float(df["close"].iloc[-3])
    lag7 = float(df["close"].iloc[-7])
    return_ma5 = float(last_row["return_ma5"])
    day_idx = float(last_row["day_index"])

    for i in range(1, days + 1):
        day_idx += 1
        # Advance to next business day
        if i == 1:
            pred_date = last_date + timedelta(days=1)
        else:
            pred_date = pred_date + timedelta(days=1)
        while pred_date.weekday() >= 5:
            pred_date += timedelta(days=1)

        features = np.array([[day_idx, current_ma7, current_momentum, current_volatility, lag1, lag3, lag7, return_ma5]])
        features_scaled = scaler.transform(features)
        predicted_price = round(float(model.predict(features_scaled)[0]), 2)

        predictions.append({
            "date": pred_date.strftime("%Y-%m-%d"),
            "predicted_close": predicted_price,
            "day": i,
        })

        # Roll forward features for next prediction
        lag7 = lag3
        lag3 = lag1
        lag1 = predicted_price
        # Update moving average estimate
        current_ma7 = round((current_ma7 * 6 + predicted_price) / 7, 2)

    # Historical data for chart context (last 30 days)
    recent = df.tail(30)[["date", "close"]].to_dict(orient="records")

    return {
        "symbol": symbol,
        "name": check.iloc[0]["name"],
        "model": "Linear Regression (scikit-learn)",
        "features_used": feature_cols,
        "training_days": train_size,
        "r2_score": train_score,
        "current_price": current_close,
        "predictions": predictions,
        "historical": recent,
        "disclaimer": "This is a basic ML model for demonstration. Not financial advice.",
    }


# ─── Run ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
