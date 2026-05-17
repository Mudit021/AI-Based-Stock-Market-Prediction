import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Stock Market Predictor",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title(" AI-Powered Stock Market Predictor")
st.markdown("Analyze historical stock data and predict future prices using Machine Learning.")

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("Data/cleaned_sp500.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    model = joblib.load("models/xgboost_stock_model.pkl")
    return model

model = load_model()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(" Stock Selection")

stocks = sorted(df['Ticker'].unique())

selected_stock = st.sidebar.selectbox(
    "Choose Stock",
    stocks
)

# =========================================================
# FILTER STOCK DATA
# =========================================================

stock_df = df[df['Ticker'] == selected_stock].copy()

stock_df = stock_df.sort_values("Date")

# =========================================================
# FEATURE ENGINEERING
# =========================================================

# Moving Averages
stock_df['MA10'] = stock_df['Close'].rolling(10).mean()
stock_df['MA50'] = stock_df['Close'].rolling(50).mean()
stock_df['MA200'] = stock_df['Close'].rolling(200).mean()

# Daily Return
stock_df['Daily_Return'] = stock_df['Close'].pct_change()

# Volatility
stock_df['Volatility'] = stock_df['Daily_Return'].rolling(10).std()

# RSI
delta = stock_df['Close'].diff()

gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

stock_df['RSI'] = 100 - (100 / (1 + rs))

# MACD
ema12 = stock_df['Close'].ewm(span=12, adjust=False).mean()
ema26 = stock_df['Close'].ewm(span=26, adjust=False).mean()

stock_df['MACD'] = ema12 - ema26

# Bollinger Bands
stock_df['BB_Middle'] = stock_df['Close'].rolling(20).mean()

std = stock_df['Close'].rolling(20).std()

stock_df['BB_Upper'] = stock_df['BB_Middle'] + (2 * std)
stock_df['BB_Lower'] = stock_df['BB_Middle'] - (2 * std)

# Drop NaN values
stock_df = stock_df.dropna()

# =========================================================
# STOCK METRICS
# =========================================================

latest_close = stock_df['Close'].iloc[-1]
highest_price = stock_df['High'].max()
lowest_price = stock_df['Low'].min()
volatility = stock_df['Daily_Return'].std()

st.subheader(f" {selected_stock} Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Current Price",
    f"${latest_close:.2f}"
)

col2.metric(
    "Highest Price",
    f"${highest_price:.2f}"
)

col3.metric(
    "Lowest Price",
    f"${lowest_price:.2f}"
)

col4.metric(
    "Volatility",
    f"{volatility:.4f}"
)

# =========================================================
# PRICE CHART
# =========================================================

st.subheader(" Closing Price Trend")

fig = px.line(
    stock_df,
    x='Date',
    y='Close',
    title=f"{selected_stock} Closing Price"
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# MOVING AVERAGE CHART
# =========================================================

st.subheader(" Moving Averages")

fig_ma = go.Figure()

fig_ma.add_trace(go.Scatter(
    x=stock_df['Date'],
    y=stock_df['Close'],
    mode='lines',
    name='Close Price'
))

fig_ma.add_trace(go.Scatter(
    x=stock_df['Date'],
    y=stock_df['MA10'],
    mode='lines',
    name='MA10'
))

fig_ma.add_trace(go.Scatter(
    x=stock_df['Date'],
    y=stock_df['MA50'],
    mode='lines',
    name='MA50'
))

fig_ma.add_trace(go.Scatter(
    x=stock_df['Date'],
    y=stock_df['MA200'],
    mode='lines',
    name='MA200'
))

fig_ma.update_layout(
    title=f"{selected_stock} Moving Averages",
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(fig_ma, use_container_width=True)

# =========================================================
# CANDLESTICK CHART
# =========================================================

st.subheader(" Candlestick Chart")

fig_candle = go.Figure(data=[go.Candlestick(
    x=stock_df['Date'],
    open=stock_df['Open'],
    high=stock_df['High'],
    low=stock_df['Low'],
    close=stock_df['Close']
)])

fig_candle.update_layout(
    title=f"{selected_stock} Candlestick Chart",
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(fig_candle, use_container_width=True)

# =========================================================
# AI PREDICTION
# =========================================================

st.subheader(" AI Prediction")

features = [
    'Open',
    'High',
    'Low',
    'Volume',
    'MA10',
    'MA50',
    'MA200',
    'Daily_Return',
    'Volatility',
    'RSI',
    'MACD',
    'BB_Upper',
    'BB_Lower'
]

latest_data = stock_df[features].iloc[-1:]

prediction = model.predict(latest_data)[0]

st.metric(
    "Predicted Next Closing Price",
    f"${prediction:.2f}"
)

# =========================================================
# BUY / SELL SIGNAL
# =========================================================

st.subheader(" Trading Signal")

current_price = stock_df['Close'].iloc[-1]

if prediction > current_price:
    st.success(" BUY SIGNAL — Predicted price is higher than current price.")
elif prediction < current_price:
    st.error(" SELL SIGNAL — Predicted price is lower than current price.")
else:
    st.warning(" HOLD SIGNAL — Market appears stable.")

# =========================================================
# RSI ANALYSIS
# =========================================================

st.subheader(" RSI Indicator")

latest_rsi = stock_df['RSI'].iloc[-1]

st.metric("Current RSI", f"{latest_rsi:.2f}")

if latest_rsi > 70:
    st.warning(" Stock may be OVERBOUGHT.")
elif latest_rsi < 30:
    st.warning(" Stock may be OVERSOLD.")
else:
    st.success(" RSI is in a normal range.")

# =========================================================
# RECENT DATA
# =========================================================

st.subheader(" Recent Historical Data")

st.dataframe(
    stock_df.tail(20),
    use_container_width=True
)

st.markdown("---")
st.markdown("Built by Mudit using Python, Streamlit, XGBoost, and Plotly")