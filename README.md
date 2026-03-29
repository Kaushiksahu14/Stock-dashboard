# 📊 StockPulse — Stock Data Intelligence Dashboard

A mini financial data platform that collects, analyzes, and visualizes Indian stock market (NSE) data through a clean REST API and an interactive web dashboard.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite)
![Chart.js](https://img.shields.io/badge/Charts-Chart.js-FF6384?logo=chart.js)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ installed
- pip package manager

### Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/stock-dashboard.git
cd stock-dashboard

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Collect and prepare stock data
python data_collector.py

# 5. Start the server
python app.py
```

Open **http://localhost:8000** in your browser to see the dashboard.  
API docs available at **http://localhost:8000/docs** (Swagger UI).

### Docker (Alternative)

```bash
docker build -t stock-dashboard .
docker run -p 8000:8000 stock-dashboard
```

---

## 🏗️ Project Structure

```
stock-dashboard/
├── app.py                  # FastAPI backend — all REST endpoints
├── data_collector.py       # Data pipeline — fetch, clean, transform, store
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container deployment
├── data/
│   └── stocks.db           # SQLite database (generated)
└── static/
    └── index.html          # Interactive dashboard frontend
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/companies` | GET | List all 10 NSE companies with latest price & daily return |
| `/data/{symbol}` | GET | Stock OHLCV data + metrics. `?days=30` (default), up to 365 |
| `/summary/{symbol}` | GET | 52-week high, low, avg close, volatility, momentum |
| `/compare?symbol1=X&symbol2=Y` | GET | Side-by-side comparison with correlation & normalized chart data |
| `/predict/{symbol}` | GET | ML-based 7-day price prediction using Linear Regression |
| `/top-movers` | GET | Today's top 3 gainers and losers |
| `/sectors` | GET | Sector-wise average performance |
| `/docs` | GET | Interactive Swagger API documentation |

### Example Requests

```bash
# Get all companies
curl http://localhost:8000/companies

# Get TCS stock data for last 90 days
curl http://localhost:8000/data/TCS?days=90

# Get Reliance summary
curl http://localhost:8000/summary/RELIANCE

# Compare Infosys vs Wipro
curl "http://localhost:8000/compare?symbol1=INFY&symbol2=WIPRO"

# Today's top movers
curl http://localhost:8000/top-movers
```

---

## 📊 Data Pipeline — Logic & Approach

### Data Collection (`data_collector.py`)

1. **Source**: Attempts real-time data via `yfinance` (Yahoo Finance API for NSE stocks). Falls back to a **realistic mock data generator** using Geometric Brownian Motion if yfinance is unavailable.

2. **Companies Tracked**: 10 major NSE stocks across 4 sectors:
   - **IT**: TCS, Infosys, Wipro
   - **Banking**: HDFC Bank, ICICI Bank, SBI
   - **Energy**: Reliance Industries
   - **FMCG**: Hindustan Unilever, ITC
   - **Telecom**: Bharti Airtel

3. **Data Range**: ~400 calendar days (~280 trading days) for full 52-week calculations.

### Data Cleaning
- Forward-fill and back-fill missing values
- Coerce invalid numeric entries
- Enforce OHLC consistency (`high ≥ max(open, close)`, `low ≤ min(open, close)`)
- Proper datetime parsing and sorting

### Calculated Metrics

| Metric | Formula | Purpose |
|---|---|---|
| **Daily Return** | `(close - open) / open` | Intra-day performance |
| **7-Day Moving Average** | Rolling mean of close (7 periods) | Short-term trend |
| **52-Week High/Low** | Rolling max/min over 252 trading days | Annual price range |
| **Volatility Score** ⭐ | 30-day rolling std of daily returns × √252 | Annualized risk measure |
| **Momentum Index** ⭐ | RSI-based (14-period relative strength) | Overbought/oversold signal |

⭐ = Custom metrics added beyond the assignment requirements.

### Storage
- **SQLite** database with indexed tables for fast queries
- Indexes on `symbol`, `date`, and composite `(symbol, date)`

---

## 🎨 Frontend Dashboard

The dashboard is a **single-page application** built with vanilla HTML/CSS/JS and Chart.js:

- **Company Sidebar**: Live list with prices and daily change percentages
- **Interactive Charts**: Closing price + 7-day MA with 30D/90D/180D/1Y timeframe buttons
- **Stat Cards**: 52-week high/low, total return, volatility, momentum (RSI), avg volume
- **Top Movers**: Today's top gainers and losers at a glance
- **Stock Comparison**: Select any two stocks to see normalized performance, correlation, and side-by-side statistics
- **AI Prediction Panel**: Generate 7-day forecasts with a single click, visualized as an extension of the price chart
- **Responsive Design**: Works on desktop and mobile

---

## 🧠 AI / ML — Price Prediction

The `/predict/{symbol}` endpoint uses **scikit-learn's Linear Regression** to forecast stock prices for the next 7 trading days.

### How It Works

1. **Feature Engineering**: The model uses 8 features — day index, 7-day moving average, momentum (RSI), volatility, 3 lagged close prices (1-day, 3-day, 7-day), and a 5-day return moving average.
2. **Training**: The model trains on the most recent 90 trading days (more recent data = more relevant patterns). Features are standardized using `StandardScaler`.
3. **Prediction**: For each future day, the model predicts a closing price, then rolls the predicted value forward as input for the next day's prediction. Moving averages are recalculated iteratively.
4. **Output**: Returns predicted prices, the R² training score, feature list, and a combined historical + forecast chart.

### Why Linear Regression?

For a demonstration project, Linear Regression provides interpretability (you can inspect feature weights), fast training, and reasonable short-term forecasts. The R² score gives immediate feedback on model quality. More complex models (LSTM, XGBoost) could improve accuracy but would add significant complexity without changing the core approach.

### Disclaimer

This is a basic ML model for educational/demonstration purposes only. Real stock price prediction requires far more sophisticated approaches and should never be used for actual trading decisions.

---

## 🧠 Design Decisions & Insights

1. **Why SQLite over PostgreSQL?**  
   For a project of this scale (10 stocks × ~280 days = ~2,800 rows), SQLite provides zero-config simplicity while still supporting indexed queries. Easy to switch to PostgreSQL for production via SQLAlchemy.

2. **Why Geometric Brownian Motion for mock data?**  
   GBM is the standard model in quantitative finance for simulating stock prices. It produces realistic-looking charts with proper volatility clustering and drift, unlike simple random walks.

3. **Volatility Score as a custom metric**  
   Annualized volatility (σ × √252) is the industry-standard risk measure used by portfolio managers. A stock with 30% volatility means its price is expected to fluctuate ±30% annually.

4. **Momentum / RSI as a custom metric**  
   The Relative Strength Index helps identify potential reversals. RSI > 70 suggests overbought conditions (possible pullback), while RSI < 30 suggests oversold (possible bounce).

5. **Normalized comparison charts (Base 100)**  
   When comparing two stocks at different price levels (e.g., ₹3800 TCS vs ₹480 Wipro), absolute prices are misleading. Normalizing to base 100 shows *relative performance* clearly.

---

## ⚡ Bonus Features Implemented

- ✅ **Swagger API Documentation** — Auto-generated at `/docs`
- ✅ **Stock Comparison Endpoint** — With correlation analysis
- ✅ **AI Price Prediction** — Linear Regression model with 7-day forecast and interactive chart
- ✅ **Top Gainers / Losers** — Additional analytics endpoint
- ✅ **Sector Performance** — Aggregate sector analysis
- ✅ **Custom Metrics** — Volatility Score + Momentum Index (beyond required metrics)
- ✅ **Dockerization** — Ready-to-deploy Dockerfile
- ✅ **Interactive Dashboard** — Full UI with charts, filters, comparison, and ML predictions
- ✅ **Database Indexing** — Optimized queries with composite indexes
- ✅ **Async-ready API** — FastAPI's native async support

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Backend | FastAPI + Uvicorn |
| Database | SQLite |
| Data Processing | Pandas, NumPy |
| ML / AI | scikit-learn (Linear Regression) |
| Data Source | yfinance / Mock Generator |
| Frontend | HTML5, CSS3, JavaScript |
| Charts | Chart.js 4.x |
| Containerization | Docker |

---

## 📝 License

This project was built as part of the JarNox Software Internship Assignment.
