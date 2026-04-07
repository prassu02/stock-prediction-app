
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
st.title("📈 Stock Prediction Platform")

# =============================
# PATH SETUP
# =============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models/stock_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models/scaler.pkl")
default_file = os.path.join(BASE_DIR, "AAPL (4).xls")

# =============================
# LOAD DL MODEL
# =============================

dl_model, scaler = None, None
if os.path.exists(model_path) and os.path.exists(scaler_path):
    dl_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

# =============================
# FILE UPLOAD
# =============================

uploaded_file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])

def load_data(file):
    try:
        return pd.read_csv(file)
    except:
        try:
            return pd.read_excel(file, engine="openpyxl")
        except:
            return pd.read_excel(file, engine="xlrd")

# =============================
# LOAD DATA
# =============================

if uploaded_file:
    df = load_data(uploaded_file)
    st.success(f"Using uploaded file: {uploaded_file.name}")
else:
    try:
        df = load_data(default_file)
        st.info("Using default dataset: AAPL (4).xls")
    except:
        st.error("Default dataset not found")
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
# MODEL FUNCTIONS
# =============================

def predict_dl(df):
    if dl_model is None or scaler is None:
        return None

    last_30 = df[['Open','High','Low','Close','Volume']].tail(30)
    scaled = scaler.transform(last_30)
    X_input = scaled.reshape(1, -1)

    preds = []
    for _ in range(30):
        p = dl_model.predict(X_input)[0]
        preds.append(p)

        seq = X_input.reshape(30,5)
        last = seq[-1]

        new_row = np.vstack([seq[1:], [last[0], last[1], last[2], p, last[4]]])
        X_input = new_row.reshape(1,-1)

    dummy = np.zeros((30,5))
    dummy[:,3] = preds
    return scaler.inverse_transform(dummy)[:,3]


def predict_arima(df):
    model = ARIMA(df['Close'], order=(5,1,0)).fit()
    return model.forecast(30).values


def predict_sarima(df):
    model = SARIMAX(df['Close'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    return model.forecast(30).values


def predict_xgb(df):
    df_ml = df.copy()

    for i in range(1,6):
        df_ml[f"lag_{i}"] = df_ml["Close"].shift(i)

    df_ml.dropna(inplace=True)

    X = df_ml.drop(columns=["Date","Close"])
    y = df_ml["Close"]

    model = XGBRegressor(n_estimators=200)
    model.fit(X,y)

    last = X.iloc[-1].values.reshape(1,-1)
    preds = []

    for _ in range(30):
        p = model.predict(last)[0]
        preds.append(p)
        last = np.roll(last,-1)
        last[0,-1] = p

    return np.array(preds)

# =============================
# RUN ALL MODELS
# =============================

st.subheader("🤖 Running Models...")

predictions = {}

try:
    dl_preds = predict_dl(df)
    if dl_preds is not None:
        predictions["Deep Learning"] = dl_preds
except: pass

try:
    predictions["ARIMA"] = predict_arima(df)
except: pass

try:
    predictions["SARIMA"] = predict_sarima(df)
except: pass

try:
    predictions["XGBoost"] = predict_xgb(df)
except: pass

# =============================
# MODEL COMPARISON
# =============================

st.subheader("🏆 Model Comparison")

results = []
actual = df['Close'].tail(30).values

for name, preds in predictions.items():
    try:
        rmse = np.sqrt(mean_squared_error(actual, preds[:len(actual)]))
        results.append([name, rmse])
    except:
        pass

if len(results) == 0:
    st.error("No models worked")
    st.stop()

comp_df = pd.DataFrame(results, columns=["Model","RMSE"]).sort_values("RMSE")

st.dataframe(comp_df)

best_model = comp_df.iloc[0]["Model"]
st.success(f"🏆 Best Model: {best_model}")

# =============================
# FINAL OUTPUT
# =============================

final_preds = predictions[best_model]

last_date = df['Date'].iloc[-1]
future_dates = [last_date + timedelta(days=i+1) for i in range(30)]

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Close": final_preds
})

st.subheader("📈 Forecast (Best Model)")
st.write(forecast_df)

# =============================
# CHART
# =============================

hist = df[['Date','Close']].tail(60)

combined = pd.concat([
    hist.rename(columns={'Close':'Price'}).set_index('Date'),
    forecast_df.rename(columns={'Predicted_Close':'Price'}).set_index('Date')
])

st.subheader("📊 Chart")
st.line_chart(combined)
