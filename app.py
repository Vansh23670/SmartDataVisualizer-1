import streamlit as st
import requests
import aiohttp
import asyncio
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import time
import threading
from queue import Queue
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from scipy.optimize import minimize
from datetime import datetime
from websocket import create_connection, WebSocketConnectionClosedException
import os
import logging
from cryptography.fernet import Fernet
import ast
import nest_asyncio
from functools import wraps
from plotly.subplots import make_subplots

# External library for sortable widgets
try:
    from streamlit_sortables import sortables
except ImportError:
    st.warning("⚠️ Optional dependency 'streamlit-sortables' missing. Install with 'pip install streamlit-sortables' to enable dashboard customization.")
    sortables = None

nest_asyncio.apply()

try:
    import pandas_ta as ta
except ImportError as e:
    ta = None
    if "cannot import name 'NaN'" in str(e):
        st.error("⚠️ Failed to import pandas_ta due to NumPy compatibility. Using limited indicators.")
    else:
        st.error("⚠️ Failed to import pandas_ta: limited indicators enabled.")

df_live = None
current_price = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    "api_base_url": "https://api.binance.com",
    "ws_base_url": "wss://stream.binance.com:9443",
    "default_limit": 100,
    "default_interval": "1m",
    "supported_cryptos": ["BTC", "ETH", "XRP", "ADA", "SOL"],
    "max_retries": 10,
    "cache_ttl": 15,
    "valid_timeframes": ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"],
}

PORTFOLIO_KEY_FILE = "portfolio_key.bin"
if 'encryption_key' not in st.session_state:
    if os.path.exists(PORTFOLIO_KEY_FILE):
        with open(PORTFOLIO_KEY_FILE, 'rb') as f:
            st.session_state.encryption_key = f.read()
    else:
        st.session_state.encryption_key = Fernet.generate_key()
        with open(PORTFOLIO_KEY_FILE, 'wb') as f:
            f.write(st.session_state.encryption_key)
cipher = Fernet(st.session_state.encryption_key)

price_cache = {}

if "pandas_ta_missing" not in st.session_state:
    st.session_state.pandas_ta_missing = ta is None
    if ta is None:
        st.warning("⚠️ Technical indicators disabled due to missing pandas_ta. Basic SMA/RSI shown.")

if "portfolio" not in st.session_state:
    st.session_state.portfolio = {crypto: 0 for crypto in CONFIG["supported_cryptos"]}

if 'ws_thread_running' not in st.session_state:
    st.session_state.ws_thread_running = False
if 'ws_data' not in st.session_state:
    st.session_state.ws_data = None
if 'ws_queue' not in st.session_state:
    st.session_state.ws_queue = Queue()
if 'ws_symbol' not in st.session_state:
    st.session_state.ws_symbol = None
if 'ws_errors' not in st.session_state:
    st.session_state.ws_errors = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = 0
if 'timeframe' not in st.session_state:
    st.session_state.timeframe = CONFIG["default_interval"]
if 'crypto' not in st.session_state:
    st.session_state.crypto = CONFIG["supported_cryptos"][0]

PORTFOLIO_FILE = "portfolio.json"

def save_portfolio() -> None:
    encrypted_data = cipher.encrypt(json.dumps(st.session_state.portfolio).encode())
    with open(PORTFOLIO_FILE, 'wb') as f:
        f.write(encrypted_data)

def load_portfolio() -> None:
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'rb') as f:
                encrypted_data = f.read()
            decrypted_data = cipher.decrypt(encrypted_data).decode()
            st.session_state.portfolio = json.loads(decrypted_data)
        except Exception as e:
            st.error(f"Failed to load portfolio: {e}")
            st.session_state.portfolio = {crypto: 0 for crypto in CONFIG["supported_cryptos"]}
    else:
        st.session_state.portfolio = {crypto: 0 for crypto in CONFIG["supported_cryptos"]}

load_portfolio()

# --------------------------
# API Fetching Functions with async
# --------------------------

async def fetch_historical_klines_async(symbol: str, interval: str = "1m", limit: int = 100) -> pd.DataFrame:
    """Fetch historical kline data from Binance API with rate limiting."""
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        url = f"{CONFIG['api_base_url']}/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.json()
            df = pd.DataFrame(
                data,
                columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
                ],
            )
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            return df

async def fetch_all_historical_data(symbols: list, interval: str = "1d", limit: int = 30) -> dict:
    """Fetch historical data for multiple symbols concurrently."""
    tasks = [fetch_historical_klines_async(symbol + "USDT", interval, limit) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return {symbol: result for symbol, result in zip(symbols, results) if not isinstance(result, Exception)}

async def fetch_current_price_async(symbol: str) -> float:
    """Fetch current price from Binance API with caching."""
    cache_key = f"{symbol}_price"
    if cache_key in price_cache and (time.time() - price_cache[cache_key]["timestamp"]) < CONFIG["cache_ttl"]:
        return price_cache[cache_key]["price"]
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        url = f"{CONFIG['api_base_url']}/api/v3/ticker/price?symbol={symbol.upper()}"
        async with session.get(url) as resp:
            resp.raise_for_status()
            price_data = await resp.json()
            price = float(price_data["price"])
            price_cache[cache_key] = {"price": price, "timestamp": time.time()}
            return price

@st.cache_data(ttl=10)
def fetch_order_book(symbol: str, limit: int = 100):
    """Fetch order book bids and asks for depth chart."""
    url = f"{CONFIG['api_base_url']}/api/v3/depth?symbol={symbol}&limit={limit}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

@st.cache_data(ttl=60)
def fetch_dex_data(crypto: str) -> dict:
    """Fetch DEX data from real sources or show empty state."""
    try:
        # In a real implementation, this would connect to DEX APIs
        # For now, return empty state since no real DEX API is configured
        return {
            "status": "No DEX data available",
            "message": "Configure DEX API endpoints to view liquidity data"
        }
    except Exception as e:
        return {
            "status": "Error fetching DEX data",
            "message": f"Failed to connect to DEX APIs: {str(e)}"
        }

@st.cache_data(ttl=60)
def fetch_wallet_balance(crypto: str) -> dict:
    """Fetch wallet balance from real sources or show empty state."""
    try:
        # In a real implementation, this would connect to wallet APIs
        # For now, return empty state since no real wallet API is configured
        return {
            "status": "No wallet connected",
            "message": "Connect your wallet to view balance information"
        }
    except Exception as e:
        return {
            "status": "Error fetching wallet data",
            "message": f"Failed to connect to wallet: {str(e)}"
        }

@st.cache_data(ttl=300)
def fetch_ico_data() -> list:
    """Fetch ICO data from real sources or show empty state."""
    try:
        # In a real implementation, this would connect to ICO tracking APIs
        # For now, return empty state since no real ICO API is configured
        return []
    except Exception as e:
        return []

@st.cache_data(ttl=30)
def fetch_exchange_prices(crypto: str) -> dict:
    """Fetch prices from multiple exchanges."""
    try:
        # Fetch from Binance (already implemented)
        binance_url = f"{CONFIG['api_base_url']}/api/v3/ticker/price?symbol={crypto}USDT"
        binance_response = requests.get(binance_url)
        binance_price = float(binance_response.json()["price"]) if binance_response.status_code == 200 else None
        
        # For other exchanges, show empty state since APIs not configured
        return {
            f"{crypto}USDT": {
                "Binance": binance_price,
                "Other exchanges": "Configure additional exchange APIs"
            }
        }
    except Exception as e:
        return {
            f"{crypto}USDT": {
                "Error": f"Failed to fetch prices: {str(e)}"
            }
        }

@st.cache_data(ttl=30)
def fetch_transaction_data(crypto: str) -> dict:
    """Fetch transaction data from blockchain APIs or show empty state."""
    try:
        # In a real implementation, this would connect to blockchain APIs
        # For now, return empty state since no real blockchain API is configured
        return {
            "status": "No blockchain data available",
            "message": "Configure blockchain API endpoints to view transaction data"
        }
    except Exception as e:
        return {
            "status": "Error fetching transaction data",
            "message": f"Failed to connect to blockchain APIs: {str(e)}"
        }

# --------------------------
# Sentiment Analysis
# --------------------------

@st.cache_data(ttl=60)
def fetch_social_media_posts(query: str, limit: int = 10) -> list:
    """Fetch social media posts for sentiment analysis."""
    try:
        # In a real implementation, this would connect to social media APIs
        # For now, return empty state since no real social media API is configured
        return []
    except Exception as e:
        return []

@st.cache_resource
def get_sentiment_analyzer():
    """Cached SentimentIntensityAnalyzer instance."""
    return SentimentIntensityAnalyzer()

@st.cache_data(ttl=60)
def analyze_sentiment(crypto: str) -> dict:
    """Analyze sentiment of social media posts for a cryptocurrency."""
    analyzer = get_sentiment_analyzer()
    posts = fetch_social_media_posts(f"{crypto} price")
    
    if not posts:
        return {
            "status": "No social media data available",
            "message": "Configure social media API endpoints to view sentiment analysis",
            "compound": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0
        }
    
    sentiments = [analyzer.polarity_scores(post["text"]) for post in posts]
    if sentiments:
        avg_sentiment = {
            "compound": np.mean([s["compound"] for s in sentiments]),
            "positive": np.mean([s["pos"] for s in sentiments]),
            "negative": np.mean([s["neg"] for s in sentiments]),
            "neutral": np.mean([s["neu"] for s in sentiments])
        }
        return avg_sentiment
    return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 0.0}

# --------------------------
# Indicator Calculation & Chart
# --------------------------

def calculate_sma_fallback(df: pd.DataFrame, length: int) -> pd.Series:
    """Calculate Simple Moving Average as a fallback."""
    return df["close"].rolling(window=length).mean()

def calculate_rsi_fallback(df: pd.DataFrame, length: int) -> pd.Series:
    """Calculate RSI as a fallback when pandas_ta is unavailable."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=length).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=length).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_indicators(df: pd.DataFrame, sma_length: int, rsi_length: int, macd_fast: int, macd_slow: int, macd_signal: int, theme: str = "plotly") -> go.Figure:
    """Calculate technical indicators and generate a Plotly chart with Bollinger Bands."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Price with SMA & Bollinger Bands", "RSI & MACD"))
    fig.add_trace(go.Scatter(x=df["close_time"], y=df["close"], mode="lines", name="Close Price"), row=1, col=1)
    
    if ta is not None:
        try:
            df["SMA"] = ta.sma(df["close"], length=sma_length)
            fig.add_trace(go.Scatter(x=df["close_time"], y=df["SMA"], mode="lines", name=f"SMA({sma_length})"), row=1, col=1)
            
            bbands = ta.bbands(df["close"], length=sma_length, std=2)
            upper_band_col = next((col for col in bbands.columns if col.startswith("BBU")), None)
            lower_band_col = next((col for col in bbands.columns if col.startswith("BBL")), None)

            if upper_band_col and lower_band_col:
                fig.add_trace(go.Scatter(x=df["close_time"], y=bbands[upper_band_col], mode="lines", name="Upper BB"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df["close_time"], y=bbands[lower_band_col], mode="lines", name="Lower BB", fill="tonexty"), row=1, col=1)
            else:
                st.warning("Bollinger Bands columns not found, skipping BB plot.")

            rsi = ta.rsi(df["close"], length=rsi_length)
            fig.add_trace(go.Scatter(x=df["close_time"], y=rsi, mode="lines", name=f"RSI({rsi_length})"), row=2, col=1)
            
            macd_df = ta.macd(df["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
            macd_col = next((col for col in macd_df.columns if col.startswith("MACD_")), None)
            signal_col = next((col for col in macd_df.columns if col.startswith("MACDs_")), None)
            if macd_col and signal_col:
                fig.add_trace(go.Scatter(x=df["close_time"], y=macd_df[macd_col], mode="lines", name="MACD"), row=2, col=1)
                fig.add_trace(go.Scatter(x=df["close_time"], y=macd_df[signal_col], mode="lines", name="Signal"), row=2, col=1)
        except Exception as e:
            st.error(f"Error calculating indicators with pandas_ta: {e}")
            # Fallback to basic indicators
            df["SMA"] = calculate_sma_fallback(df, sma_length)
            fig.add_trace(go.Scatter(x=df["close_time"], y=df["SMA"], mode="lines", name=f"SMA({sma_length})"), row=1, col=1)
            
            rsi = calculate_rsi_fallback(df, rsi_length)
            fig.add_trace(go.Scatter(x=df["close_time"], y=rsi, mode="lines", name=f"RSI({rsi_length})"), row=2, col=1)
    else:
        # Use fallback indicators
        df["SMA"] = calculate_sma_fallback(df, sma_length)
        fig.add_trace(go.Scatter(x=df["close_time"], y=df["SMA"], mode="lines", name=f"SMA({sma_length})"), row=1, col=1)
        
        rsi = calculate_rsi_fallback(df, rsi_length)
        fig.add_trace(go.Scatter(x=df["close_time"], y=rsi, mode="lines", name=f"RSI({rsi_length})"), row=2, col=1)

    fig.update_layout(height=600, template=theme)
    return fig

def create_candlestick_chart(df: pd.DataFrame, theme: str = "plotly") -> go.Figure:
    """Create a candlestick chart with volume."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Candlestick Chart", "Volume"))
    
    fig.add_trace(go.Candlestick(x=df["close_time"], open=df["open"], high=df["high"],
                                low=df["low"], close=df["close"], name="Price"), row=1, col=1)
    
    fig.add_trace(go.Bar(x=df["close_time"], y=df["volume"], name="Volume"), row=2, col=1)
    
    fig.update_layout(height=600, template=theme, xaxis_rangeslider_visible=False)
    return fig

def create_order_book_chart(order_book_data: dict, theme: str = "plotly") -> go.Figure:
    """Create order book depth chart."""
    bids = pd.DataFrame(order_book_data["bids"], columns=["price", "quantity"])
    asks = pd.DataFrame(order_book_data["asks"], columns=["price", "quantity"])
    
    bids["price"] = pd.to_numeric(bids["price"])
    bids["quantity"] = pd.to_numeric(bids["quantity"])
    asks["price"] = pd.to_numeric(asks["price"])
    asks["quantity"] = pd.to_numeric(asks["quantity"])
    
    bids["cumulative"] = bids["quantity"].cumsum()
    asks["cumulative"] = asks["quantity"].cumsum()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bids["price"], y=bids["cumulative"], mode="lines", name="Bids", fill="tozeroy"))
    fig.add_trace(go.Scatter(x=asks["price"], y=asks["cumulative"], mode="lines", name="Asks", fill="tozeroy"))
    
    fig.update_layout(title="Order Book Depth", xaxis_title="Price", yaxis_title="Cumulative Volume", template=theme)
    return fig

# --------------------------
# WebSocket for Real-time Data
# --------------------------

def websocket_thread(symbol: str):
    """WebSocket thread function for real-time price updates."""
    if st.session_state.ws_thread_running:
        return
    
    st.session_state.ws_thread_running = True
    st.session_state.ws_symbol = symbol
    
    def run_websocket():
        ws_url = f"{CONFIG['ws_base_url']}/ws/{symbol.lower()}@ticker"
        retry_count = 0
        
        while st.session_state.ws_thread_running and retry_count < CONFIG["max_retries"]:
            try:
                ws = create_connection(ws_url)
                retry_count = 0
                
                while st.session_state.ws_thread_running:
                    try:
                        result = ws.recv()
                        data = json.loads(result)
                        st.session_state.ws_queue.put(data)
                        st.session_state.last_update = time.time()
                    except WebSocketConnectionClosedException:
                        break
                    except json.JSONDecodeError:
                        continue
                        
                ws.close()
                
            except Exception as e:
                retry_count += 1
                error_msg = f"WebSocket error (attempt {retry_count}): {str(e)}"
                st.session_state.ws_errors.append(error_msg)
                logger.error(error_msg)
                time.sleep(min(retry_count * 2, 30))
        
        st.session_state.ws_thread_running = False
    
    thread = threading.Thread(target=run_websocket, daemon=True)
    thread.start()

def stop_websocket():
    """Stop the WebSocket connection."""
    st.session_state.ws_thread_running = False
    st.session_state.ws_symbol = None

def get_websocket_data():
    """Get the latest WebSocket data."""
    if not st.session_state.ws_queue.empty():
        try:
            data = st.session_state.ws_queue.get_nowait()
            st.session_state.ws_data = data
            return data
        except:
            pass
    return st.session_state.ws_data

# --------------------------
# Machine Learning Prediction
# --------------------------

@st.cache_resource
def train_ml_model(df: pd.DataFrame):
    """Train ML model for price prediction."""
    if len(df) < 20:
        return None, "Insufficient data for training"
    
    try:
        # Prepare features
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=5).std()
        df["sma_ratio"] = df["close"] / df["close"].rolling(window=10).mean()
        
        # Create features and target
        features = ["returns", "volatility", "sma_ratio", "volume"]
        df = df.dropna()
        
        if len(df) < 10:
            return None, "Insufficient data after preprocessing"
        
        X = df[features].values
        y = df["close"].shift(-1).fillna(method="ffill").values
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate feature importance
        importance = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
        
        return model, {
            "features": features,
            "importance": importance.importances_mean,
            "score": model.score(X_test, y_test)
        }
    except Exception as e:
        return None, f"Model training failed: {str(e)}"

def predict_price(model, df: pd.DataFrame, info: dict) -> dict:
    """Make price prediction using trained model."""
    if model is None:
        return {"error": "No trained model available"}
    
    try:
        # Prepare latest features
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=5).std()
        df["sma_ratio"] = df["close"] / df["close"].rolling(window=10).mean()
        
        latest_features = df[info["features"]].iloc[-1:].values
        prediction = model.predict(latest_features)[0]
        current_price = df["close"].iloc[-1]
        
        return {
            "prediction": prediction,
            "current_price": current_price,
            "change_pct": ((prediction - current_price) / current_price) * 100,
            "confidence": info["score"]
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# --------------------------
# Portfolio Optimization
# --------------------------

def optimize_portfolio(historical_data: dict, risk_tolerance: float = 0.5) -> dict:
    """Optimize portfolio allocation using modern portfolio theory."""
    try:
        # Calculate returns for each asset
        returns_data = {}
        for symbol, df in historical_data.items():
            if isinstance(df, pd.DataFrame) and len(df) > 1:
                returns_data[symbol] = df["close"].pct_change().dropna()
        
        if len(returns_data) < 2:
            return {"error": "Need at least 2 assets for optimization"}
        
        # Create returns matrix
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 10:
            return {"error": "Insufficient data for optimization"}
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns_df.mean() * 252  # Annualized
        cov_matrix = returns_df.cov() * 252
        
        # Optimization objective
        def objective(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            # Risk-adjusted return (higher risk_tolerance = more aggressive)
            return -(portfolio_return - risk_tolerance * portfolio_variance)
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(expected_returns)))
        
        # Initial guess
        initial_guess = [1/len(expected_returns)] * len(expected_returns)
        
        # Optimize
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = dict(zip(expected_returns.index, result.x))
            portfolio_return = np.sum(expected_returns * result.x)
            portfolio_risk = np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x)))
            
            return {
                "weights": optimal_weights,
                "expected_return": portfolio_return,
                "risk": portfolio_risk,
                "sharpe_ratio": portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            }
        else:
            return {"error": "Optimization failed"}
            
    except Exception as e:
        return {"error": f"Portfolio optimization failed: {str(e)}"}

# --------------------------
# Main Dashboard
# --------------------------

def main():
    st.set_page_config(
        page_title="Crypto Trading Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Comprehensive Cryptocurrency Trading Dashboard")
    st.markdown("Real-time data, technical analysis, portfolio management, and sentiment analysis")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Crypto selection
    crypto = st.sidebar.selectbox(
        "Select Cryptocurrency",
        CONFIG["supported_cryptos"],
        index=CONFIG["supported_cryptos"].index(st.session_state.crypto)
    )
    st.session_state.crypto = crypto
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        CONFIG["valid_timeframes"],
        index=CONFIG["valid_timeframes"].index(st.session_state.timeframe)
    )
    st.session_state.timeframe = timeframe
    
    # Data limit
    data_limit = st.sidebar.slider("Data Points", 50, 1000, 100)
    
    # Technical indicator settings
    st.sidebar.subheader("Technical Indicators")
    sma_length = st.sidebar.slider("SMA Length", 5, 50, 20)
    rsi_length = st.sidebar.slider("RSI Length", 5, 30, 14)
    macd_fast = st.sidebar.slider("MACD Fast", 5, 20, 12)
    macd_slow = st.sidebar.slider("MACD Slow", 20, 50, 26)
    macd_signal = st.sidebar.slider("MACD Signal", 5, 15, 9)
    
    # Chart theme
    chart_theme = st.sidebar.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark"])
    
    # Real-time data toggle
    enable_realtime = st.sidebar.checkbox("Enable Real-time Data", value=False)
    
    # Main content
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(f"💰 Current Price: {crypto}")
        
        # Fetch current price
        try:
            current_price = asyncio.run(fetch_current_price_async(f"{crypto}USDT"))
            st.metric(label=f"{crypto}/USDT", value=f"${current_price:,.2f}")
        except Exception as e:
            st.error(f"Failed to fetch current price: {e}")
            current_price = 0
    
    with col2:
        st.subheader("🔄 Real-time Updates")
        if enable_realtime:
            if not st.session_state.ws_thread_running:
                if st.button("Start Live Data"):
                    websocket_thread(f"{crypto}USDT")
                    st.success("Live data started!")
                    st.rerun()
            else:
                if st.button("Stop Live Data"):
                    stop_websocket()
                    st.success("Live data stopped!")
                    st.rerun()
                
                # Show live data
                ws_data = get_websocket_data()
                if ws_data:
                    st.json({"price": ws_data.get("c", "N/A"), "volume": ws_data.get("v", "N/A")})
        else:
            st.info("Enable real-time data to see live updates")
    
    with col3:
        st.subheader("📊 Quick Stats")
        if st.session_state.ws_data:
            st.metric("24h Change", f"{float(st.session_state.ws_data.get('P', 0)):.2f}%")
            st.metric("24h Volume", f"{float(st.session_state.ws_data.get('v', 0)):,.0f}")
    
    # Fetch historical data
    st.subheader("📈 Historical Data & Technical Analysis")
    
    try:
        with st.spinner("Fetching historical data..."):
            df = asyncio.run(fetch_historical_klines_async(f"{crypto}USDT", timeframe, data_limit))
        
        if df is not None and len(df) > 0:
            # Chart tabs
            chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📊 Candlestick", "📈 Technical Indicators", "📉 Order Book"])
            
            with chart_tab1:
                candlestick_fig = create_candlestick_chart(df, chart_theme)
                st.plotly_chart(candlestick_fig, use_container_width=True)
            
            with chart_tab2:
                indicators_fig = calculate_indicators(df, sma_length, rsi_length, macd_fast, macd_slow, macd_signal, chart_theme)
                st.plotly_chart(indicators_fig, use_container_width=True)
            
            with chart_tab3:
                try:
                    order_book = fetch_order_book(f"{crypto}USDT")
                    order_book_fig = create_order_book_chart(order_book, chart_theme)
                    st.plotly_chart(order_book_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to fetch order book: {e}")
        else:
            st.error("No historical data available")
    
    except Exception as e:
        st.error(f"Failed to fetch historical data: {e}")
        df = None
    
    # Portfolio Management
    st.subheader("💼 Portfolio Management")
    
    portfolio_col1, portfolio_col2 = st.columns(2)
    
    with portfolio_col1:
        st.write("**Current Holdings**")
        
        total_value = 0
        for crypto_symbol in CONFIG["supported_cryptos"]:
            amount = st.session_state.portfolio.get(crypto_symbol, 0)
            if amount > 0:
                try:
                    price = asyncio.run(fetch_current_price_async(f"{crypto_symbol}USDT"))
                    value = amount * price
                    total_value += value
                    st.write(f"{crypto_symbol}: {amount:.4f} units (${value:,.2f})")
                except:
                    st.write(f"{crypto_symbol}: {amount:.4f} units (Price unavailable)")
        
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")
    
    with portfolio_col2:
        st.write("**Add/Remove Holdings**")
        
        selected_crypto = st.selectbox("Select Crypto", CONFIG["supported_cryptos"])
        action = st.radio("Action", ["Add", "Remove"])
        amount = st.number_input("Amount", min_value=0.0, step=0.0001, format="%.4f")
        
        if st.button("Update Portfolio"):
            if action == "Add":
                st.session_state.portfolio[selected_crypto] += amount
            else:
                st.session_state.portfolio[selected_crypto] = max(0, st.session_state.portfolio[selected_crypto] - amount)
            
            save_portfolio()
            st.success(f"{action}ed {amount} {selected_crypto}")
            st.rerun()
    
    # Portfolio Optimization
    st.subheader("🎯 Portfolio Optimization")
    
    risk_tolerance = st.slider("Risk Tolerance", 0.0, 1.0, 0.5, 0.1)
    
    if st.button("Optimize Portfolio"):
        with st.spinner("Optimizing portfolio..."):
            try:
                historical_data = asyncio.run(fetch_all_historical_data(CONFIG["supported_cryptos"]))
                optimization_result = optimize_portfolio(historical_data, risk_tolerance)
                
                if "error" in optimization_result:
                    st.error(optimization_result["error"])
                else:
                    st.success("Portfolio optimization completed!")
                    
                    opt_col1, opt_col2 = st.columns(2)
                    
                    with opt_col1:
                        st.write("**Optimal Weights**")
                        for symbol, weight in optimization_result["weights"].items():
                            st.write(f"{symbol}: {weight:.2%}")
                    
                    with opt_col2:
                        st.write("**Expected Performance**")
                        st.metric("Expected Return", f"{optimization_result['expected_return']:.2%}")
                        st.metric("Expected Risk", f"{optimization_result['risk']:.2%}")
                        st.metric("Sharpe Ratio", f"{optimization_result['sharpe_ratio']:.2f}")
                        
            except Exception as e:
                st.error(f"Portfolio optimization failed: {e}")
    
    # Machine Learning Prediction
    st.subheader("🤖 AI Price Prediction")
    
    if df is not None and len(df) > 20:
        if st.button("Train ML Model & Predict"):
            with st.spinner("Training model..."):
                model, info = train_ml_model(df)
                
                if model is None:
                    st.error(f"Model training failed: {info}")
                else:
                    st.success(f"Model trained successfully! Score: {info['score']:.3f}")
                    
                    # Make prediction
                    prediction_result = predict_price(model, df, info)
                    
                    if "error" in prediction_result:
                        st.error(prediction_result["error"])
                    else:
                        pred_col1, pred_col2, pred_col3 = st.columns(3)
                        
                        with pred_col1:
                            st.metric("Current Price", f"${prediction_result['current_price']:.2f}")
                        
                        with pred_col2:
                            st.metric("Predicted Price", f"${prediction_result['prediction']:.2f}")
                        
                        with pred_col3:
                            change_pct = prediction_result['change_pct']
                            st.metric("Expected Change", f"{change_pct:+.2f}%")
                        
                        st.write(f"**Model Confidence:** {prediction_result['confidence']:.3f}")
    else:
        st.info("Need more historical data for ML prediction")
    
    # Sentiment Analysis
    st.subheader("😊 Sentiment Analysis")
    
    sentiment_data = analyze_sentiment(crypto)
    
    if "status" in sentiment_data:
        st.warning(f"{sentiment_data['status']}: {sentiment_data['message']}")
    else:
        sent_col1, sent_col2, sent_col3, sent_col4 = st.columns(4)
        
        with sent_col1:
            st.metric("Overall Sentiment", f"{sentiment_data['compound']:.2f}")
        
        with sent_col2:
            st.metric("Positive", f"{sentiment_data['positive']:.2f}")
        
        with sent_col3:
            st.metric("Negative", f"{sentiment_data['negative']:.2f}")
        
        with sent_col4:
            st.metric("Neutral", f"{sentiment_data['neutral']:.2f}")
    
    # Additional Data Sections
    st.subheader("📊 Additional Market Data")
    
    data_tab1, data_tab2, data_tab3, data_tab4 = st.tabs(["🏦 Exchange Prices", "💧 DEX Data", "👛 Wallet", "🚀 ICO Data"])
    
    with data_tab1:
        exchange_prices = fetch_exchange_prices(crypto)
        st.json(exchange_prices)
    
    with data_tab2:
        dex_data = fetch_dex_data(crypto)
        if "status" in dex_data:
            st.warning(f"{dex_data['status']}: {dex_data['message']}")
        else:
            st.json(dex_data)
    
    with data_tab3:
        wallet_data = fetch_wallet_balance(crypto)
        if "status" in wallet_data:
            st.warning(f"{wallet_data['status']}: {wallet_data['message']}")
        else:
            st.json(wallet_data)
    
    with data_tab4:
        ico_data = fetch_ico_data()
        if ico_data:
            for ico in ico_data:
                st.write(f"**{ico['name']}** - {ico['date']} - {ico['raised']}")
        else:
            st.info("No ICO data available")
    
    # Transaction Data
    st.subheader("🔗 Transaction Data")
    
    tx_data = fetch_transaction_data(crypto)
    if "status" in tx_data:
        st.warning(f"{tx_data['status']}: {tx_data['message']}")
    else:
        st.json(tx_data)
    
    # Auto-refresh
    if enable_realtime:
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()
