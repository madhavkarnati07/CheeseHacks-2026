# train_model.py - Multi-feature LSTM

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# loading data
data = pd.read_csv("processed_data.csv")

# use multiple features so the model has real signals to learn from
FEATURES = ["Electricity_Consumed", "Hour", "Month", "Temperature", "Avg_Past_Consumption"]
TARGET    = "Electricity_Consumed"

df = data[FEATURES].copy()

# scale all features 
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# save scaler
joblib.dump(scaler, "scaler.save")
print("Scaler saved.")

# creates sequences of 48 steps (24 hours) to predict the next step's consumption
STEPS = 48  # 24 hours of context at 30-min intervals

def create_sequences(data, steps=48):
    X, y = [], []
    for i in range(len(data) - steps):
        X.append(data[i:i+steps])          # all features
        y.append(data[i+steps, 0])         # predict only Electricity_Consumed (index 0)
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, STEPS)
print(f"Total sequences: {len(X)}  |  Input shape: {X.shape}")

# train test 
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# build model 
n_features = X_train.shape[2]

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(STEPS, n_features)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation="linear")
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

# train 
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# evaluate
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss (MSE): {test_loss:.6f}")
print(f"Test RMSE      : {test_loss**0.5:.4f}")

# save 
model.save("energy_lstm_model.h5")
print("\nModel saved to energy_lstm_model.h5")
print("Scaler saved to scaler.save")