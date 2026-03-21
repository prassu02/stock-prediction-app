{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "503d3b19-86d7-4220-b13e-c9db70dac52a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Model + Scaler saved successfully!\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "import os\n",
    "\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "\n",
    "# Create folders\n",
    "os.makedirs('models', exist_ok=True)\n",
    "os.makedirs('data', exist_ok=True)\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv('apple_stock.csv')\n",
    "\n",
    "df['Date'] = pd.to_datetime(df['Date'])\n",
    "df.sort_values('Date', inplace=True)\n",
    "\n",
    "features = ['Open', 'High', 'Low', 'Close', 'Volume']\n",
    "data = df[features]\n",
    "\n",
    "# Scale\n",
    "scaler = MinMaxScaler()\n",
    "scaled_data = scaler.fit_transform(data)\n",
    "\n",
    "# Create sequences\n",
    "X, y = [], []\n",
    "\n",
    "for i in range(30, len(scaled_data)):\n",
    "    X.append(scaled_data[i-30:i].flatten())\n",
    "    y.append(scaled_data[i][3])  # Close\n",
    "\n",
    "X = np.array(X)\n",
    "y = np.array(y)\n",
    "\n",
    "# Train model\n",
    "model = RandomForestRegressor(n_estimators=100)\n",
    "model.fit(X, y)\n",
    "\n",
    "# Save\n",
    "joblib.dump(model, 'models/stock_model.pkl')\n",
    "joblib.dump(scaler, 'models/scaler.pkl')\n",
    "\n",
    "print(\"✅ Model + Scaler saved successfully!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3ad53392-6268-4497-96eb-25732c7860f7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fcededf7-3e7d-4f2f-b95d-e0119f965c5e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (tf_env)",
   "language": "python",
   "name": "tf_env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
