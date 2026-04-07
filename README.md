
# 🚀 AI Stock Prediction Platform (Deep Learning + Time Series + ML)

An end-to-end **AI-powered stock price forecasting platform** built with Streamlit.
This app supports multiple models including Deep Learning, ARIMA, SARIMA, and XGBoost for predicting stock prices for the next 30 days.

---

## 📌 Features

✔ Upload CSV / Excel stock data
✔ Automated data validation
✔ Multi-model prediction system
✔ Interactive dashboard visualization
✔ 30-day future forecasting
✔ Clean and deployable Streamlit UI

---

## 🤖 Models Used

* Deep Learning Model (Custom trained using historical sequences)
* ARIMA (AutoRegressive Integrated Moving Average)
* SARIMA (Seasonal ARIMA)
* XGBoost Regressor

---

## 📂 Project Structure

```
AI-DL-Platform/
│── app.py
│── requirements.txt
│── models/
│    ├── stock_model.pkl
│    ├── scaler.pkl
```

---

## 📊 Input Data Format

Your dataset must include the following columns:

* Date
* Open
* High
* Low
* Close
* Volume

Example:

| Date | Open | High | Low | Close | Volume |
| ---- | ---- | ---- | --- | ----- | ------ |

---

## ⚙️ Installation

1. Clone the repository:

```
git clone <your-repo-link>
cd AI-DL-Platform
```

2. Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Run the App

```
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 🌐 Deployment

You can deploy this app easily using:

👉 Streamlit Community Cloud

Steps:

1. Push code to GitHub
2. Connect repository in Streamlit Cloud
3. Select `app.py`
4. Deploy 🚀

---

## ⚠️ Important Notes

* Deep Learning model requires:

  * `models/stock_model.pkl`
  * `models/scaler.pkl`

* If not available, use:

  * ARIMA / SARIMA / XGBoost models

---

## 📈 Output

* Forecasted stock prices (next 30 days)
* Combined historical + predicted chart
* Model-based predictions

---

## 🔥 Future Improvements

* Live stock data integration (API)
* Auto model selection
* Hyperparameter tuning (Optuna)
* Trading signals (Buy/Sell)
* Model performance comparison dashboard

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
AI & Data Science Enthusiast


## ⭐ If you like this project

Give it a ⭐ on GitHub and share it!


