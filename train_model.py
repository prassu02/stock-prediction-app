import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# Create folders
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load dataset
df = pd.read_csv('data/apple_stock.csv')

df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

features = ['Open', 'High', 'Low', 'Close', 'Volume']
data = df[features]

# Scale
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
X, y = [], []

for i in range(30, len(scaled_data)):
    X.append(scaled_data[i-30:i].flatten())
    y.append(scaled_data[i][3])  # Close

X = np.array(X)
y = np.array(y)

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Save
joblib.dump(model, 'models/stock_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ Model + Scaler saved successfully!")
