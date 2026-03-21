import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta

st.set_page_config(page_title="Stock Prediction App", layout="wide")

# Safe path loading (Cloud compatible)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models/stock_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models/scaler.pkl")

# Load model
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error("❌ Model files not found. Run train_model.py first.")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.title("📈 Apple Stock Price Prediction (Next 30 Days)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        st.error("CSV must contain Date, Open, High, Low, Close, Volume")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    st.subheader("📋 Last 5 Rows")
    st.write(df.tail())

    # Prepare last 30 days
    last_30 = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(30)

    scaled = scaler.transform(last_30)
    X_input = scaled.reshape(1, -1)

    predictions_scaled = []

    # Predict next 30 days
    for _ in range(30):
        pred = model.predict(X_input)[0]
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

    # Create future dates
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(30)]

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': inv_preds
    })

    st.subheader("📈 Forecasted Prices")
    st.write(forecast_df)

    # Chart
    hist = df[['Date', 'Close']].tail(60)

    combined = pd.concat([
        hist.rename(columns={'Close': 'Price'}).set_index('Date'),
        forecast_df.rename(columns={'Predicted_Close': 'Price'}).set_index('Date')
    ])

    st.subheader("📊 Historical + Prediction")
    st.line_chart(combined)
