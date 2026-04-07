
# 🚀 AI Stock Prediction Platform (Deep Learning + Time Series + ML)

An end-to-end **AI-powered stock analytics and forecasting platform** built using Streamlit.
This application enables users to upload stock datasets or use a default dataset to generate **30-day future price predictions** using multiple advanced models.

---

## 📌 Key Features

✔ Default dataset support (`AAPL (4).xls`)
✔ Upload custom datasets (CSV / Excel)
✔ Multi-model prediction system
✔ 30-day future forecasting
✔ Interactive visualization dashboard
✔ Streamlit Cloud deployment ready
✔ Error-handled and production-safe pipeline

---

## 🤖 Models Implemented

* Deep Learning Model (Sequence-based prediction using trained model)
* ARIMA (Trend-based forecasting)
* SARIMA (Seasonal pattern forecasting)
* XGBoost (Machine Learning regression)

---

## 📂 Project Structure

```id="r6k8ka"
AI-DL-Platform/
│── app.py
│── requirements.txt
│── AAPL (4).xls
│── models/
│    ├── stock_model.pkl
│    ├── scaler.pkl
```

---

## 📊 Dataset Requirements

The dataset must contain the following columns:

* Date
* Open
* High
* Low
* Close
* Volume

---

## ⚙️ Installation (Local)

```id="rbp0e0"
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deployment

Deploy easily using:

👉 Streamlit Community Cloud

### Steps:

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Select `app.py`
4. Deploy 🚀

---

## 📈 Output

* Forecasted stock prices for next 30 days
* Combined historical + predicted visualization
* Model-driven predictions based on selected algorithm

---

## ⚠️ Notes

* Deep Learning model is optional:

  * Requires `models/stock_model.pkl`
  * Requires `models/scaler.pkl`

* If not available, use:

  * ARIMA / SARIMA / XGBoost models

---

## 🔥 Advanced Capabilities

* Handles both uploaded and default datasets
* Supports multiple forecasting techniques
* Designed for scalability and deployment
* Modular architecture for future upgrades

---

## 🚀 Future Enhancements

* Live stock API integration (real-time data)
* Auto model selection (best model suggestion)
* Hyperparameter tuning (Optuna)
* Trading signal generation (Buy/Sell)
* Model performance comparison dashboard
* REST API backend integration

---

## 🧠 Tech Stack

* Python
* Streamlit
* Pandas / NumPy
* Scikit-learn
* Statsmodels
* XGBoost

---

## 📌 Author

**Prasanna Kumar**
AI & Data Science

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---
