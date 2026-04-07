import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta

# Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

# =============================
# CONFIG
# =============================

st.set_page_config(page_title="Stock Prediction Platform", layout="wide")
st.title("📈 Stock Prediction Platform (Next 30 Days)")

# =============================
# PATH SETUP
# =============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models/stock_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models/scaler.pkl")
default_file = os.path.join(BASE_DIR, "AAPL (4).xls")

# =============================
# LOAD DL MODEL (OPTIONAL)
# =============================

dl_model = None
scaler = None

if os.path.exists(model_path) and os.path.exists(scaler_path):
    dl_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

# =============================
# FILE UPLOAD / DEFAULT DATA
# =============================
if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, engine="openpyxl")

    elif uploaded_file.name.endswith(".xls"):
        df = pd.read_excel(uploaded_file, engine="xlrd")

else:
    # Default dataset
    try:
        df = pd.read_excel(default_file, engine="xlrd")
        st.info("Using default dataset: AAPL (4).xls")
    except:
        st.error("Default dataset not found or engine issue")
        st.stop()

# =============================
# VALIDATION
# =============================

required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

if not all(col in df.columns for col in required_cols):
    st.error("Dataset must contain Date, Open, High, Low, Close, Volume")
    st.stop()

# =============================
# PREPROCESS
# =============================

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df.sort_values('Date', inplace=True)

st.subheader("📋 Last 5 Rows")
st.write(df.tail())

# =============================
# MODEL SELECT
# =============================

st.subheader("🤖 Select Model")

model_choice = st.selectbox(
    "Choose Model",
    ["Deep Learning", "ARIMA", "SARIMA", "XGBoost"]
)

# =============================
# PREDICTION
# =============================

inv_preds = None

try:

    # -------------------------
    # DEEP LEARNING
    # -------------------------
    if model_choice == "Deep Learning":

        if dl_model is None or scaler is None:
            st.error("DL model not found. Add models folder.")
            st.stop()

        last_30 = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(30)

        scaled = scaler.transform(last_30)
        X_input = scaled.reshape(1, -1)

        predictions_scaled = []

        for _ in range(30):
            pred = dl_model.predict(X_input)[0]
            predictions_scaled.append(pred)

            seq = X_input.reshape(30, 5)
            last = seq[-1]

            new_row = np.vstack([seq[1:], [
                last[0], last[1], last[2], pred, last[4]
            ]])

            X_input = new_row.reshape(1, -1)

        preds = np.array(predictions_scaled).reshape(-1, 1)

        dummy = np.zeros((30, 5))
        dummy[:, 3] = preds[:, 0]

        inv_preds = scaler.inverse_transform(dummy)[:, 3]

    # -------------------------
    # ARIMA
    # -------------------------
    elif model_choice == "ARIMA":

        series = df['Close']
        model_arima = ARIMA(series, order=(5,1,0))
        model_fit = model_arima.fit()

        forecast = model_fit.forecast(steps=30)
        inv_preds = forecast.values

    # -------------------------
    # SARIMA
    # -------------------------
    elif model_choice == "SARIMA":

        series = df['Close']
        model_sarima = SARIMAX(
            series,
            order=(1,1,1),
            seasonal_order=(1,1,1,12)
        )

        model_fit = model_sarima.fit(disp=False)

        forecast = model_fit.forecast(steps=30)
        inv_preds = forecast.values

    # -------------------------
    # XGBOOST
    # -------------------------
    elif model_choice == "XGBoost":

        df_ml = df.copy()

        for i in range(1,6):
            df_ml[f"lag_{i}"] = df_ml["Close"].shift(i)

        df_ml.dropna(inplace=True)

        X = df_ml.drop(columns=["Date", "Close"])
        y = df_ml["Close"]

        model_xgb = XGBRegressor(n_estimators=200)
        model_xgb.fit(X, y)

        last_row = X.iloc[-1].values.reshape(1, -1)

        preds = []

        for _ in range(30):
            pred = model_xgb.predict(last_row)[0]
            preds.append(pred)

            last_row = np.roll(last_row, -1)
            last_row[0, -1] = pred

        inv_preds = np.array(preds)

except Exception as e:
    st.error(f"Model failed: {e}")
    st.stop()

# =============================
# OUTPUT
# =============================

last_date = df['Date'].iloc[-1]

future_dates = [
    last_date + timedelta(days=i+1) for i in range(30)
]

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Close': inv_preds
})

st.subheader("📈 Forecasted Prices")
st.write(forecast_df)

# =============================
# CHART
# =============================

hist = df[['Date', 'Close']].tail(60)

combined = pd.concat([
    hist.rename(columns={'Close': 'Price'}).set_index('Date'),
    forecast_df.rename(columns={'Predicted_Close': 'Price'}).set_index('Date')
])

st.subheader("📊 Historical + Prediction")
st.line_chart(combined)
