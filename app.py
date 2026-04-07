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

# Metrics
from sklearn.metrics import mean_squared_error

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
# FILE UPLOAD
# =============================
uploaded_file = st.file_uploader(
    "Upload CSV / Excel",
    type=["csv", "xlsx", "xls"]
)

def load_data(file):
    try:
        # Try CSV first
        return pd.read_csv(file)
    except:
        try:
            return pd.read_excel(file, engine="openpyxl")
        except:
            return pd.read_excel(file, engine="xlrd")

# =============================
# LOAD DATA
# =============================

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success(f"Using uploaded file: {uploaded_file.name}")

else:
    try:
        df = load_data(default_file)
        st.info("Using default dataset: AAPL (4).xls")
    except:
        st.error("Default dataset not readable. Please upload file.")
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
# =============================
# MODEL COMPARISON
# =============================

from sklearn.metrics import mean_squared_error

st.subheader("🏆 Model Comparison")

results = []

try:
    # Use last 30 actual values for evaluation
    actual = df['Close'].tail(30).values

    # -------- DL --------
    if dl_model is not None:
        dl_preds = inv_preds  # current selected model
        rmse = np.sqrt(mean_squared_error(actual, dl_preds[:len(actual)]))
        results.append(["Deep Learning", rmse])

    # -------- ARIMA --------
    try:
        arima_model = ARIMA(df['Close'], order=(5,1,0)).fit()
        arima_pred = arima_model.forecast(steps=30).values
        rmse = np.sqrt(mean_squared_error(actual, arima_pred[:len(actual)]))
        results.append(["ARIMA", rmse])
    except:
        pass

    # -------- SARIMA --------
    try:
        sarima_model = SARIMAX(df['Close'],
                              order=(1,1,1),
                              seasonal_order=(1,1,1,12)).fit(disp=False)
        sarima_pred = sarima_model.forecast(steps=30).values
        rmse = np.sqrt(mean_squared_error(actual, sarima_pred[:len(actual)]))
        results.append(["SARIMA", rmse])
    except:
        pass

    # -------- XGBOOST --------
    try:
        df_ml = df.copy()
        for i in range(1,6):
            df_ml[f"lag_{i}"] = df_ml["Close"].shift(i)
        df_ml.dropna(inplace=True)

        X = df_ml.drop(columns=["Date","Close"])
        y = df_ml["Close"]

        xgb_model = XGBRegressor(n_estimators=200)
        xgb_model.fit(X, y)

        last_row = X.iloc[-1].values.reshape(1, -1)
        preds = []

        for _ in range(30):
            pred = xgb_model.predict(last_row)[0]
            preds.append(pred)
            last_row = np.roll(last_row, -1)
            last_row[0, -1] = pred

        rmse = np.sqrt(mean_squared_error(actual, preds[:len(actual)]))
        results.append(["XGBoost", rmse])
    except:
        pass

except:
    st.warning("Comparison failed")

# =============================
# TABLE
# =============================

if len(results) > 0:

    comp_df = pd.DataFrame(results, columns=["Model", "RMSE"])

    comp_df = comp_df.sort_values("RMSE")

    st.dataframe(comp_df)

    # Best model
    best_model = comp_df.iloc[0]

    st.success(f"🏆 Best Model: {best_model['Model']} (RMSE: {best_model['RMSE']:.2f})")

    # Chart
    import plotly.express as px
    fig = px.bar(comp_df, x="Model", y="RMSE", title="Model Comparison")
    st.plotly_chart(fig)
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
