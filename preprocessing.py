"""
preprocess.py
Converts the UCI Household Electric Power Consumption dataset
into the processed_data.csv format expected by train_model.py
"""

import kagglehub
import pandas as pd
import numpy as np
from pathlib import Path

# downloading dataset 
print("Downloading dataset via kagglehub...")
path = kagglehub.dataset_download("uciml/electric-power-consumption-data-set")
data_file = Path(path) / "household_power_consumption.txt"
print(f"Dataset path: {data_file}")

# loading raw data 
print("Loading data...")
df = pd.read_csv(
    data_file,
    sep=";",
    na_values=["?"],          #  uses '?' for missing values
    low_memory=False
)

# parsing timestamps and setting index 
df["Timestamp"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S"
)
df = df.drop(columns=["Date", "Time"])
df = df.set_index("Timestamp")

# convert all columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")

print(f"Raw rows: {len(df):,}  |  Missing: {df['Global_active_power'].isna().sum():,}")

# 30 minute intervals 
# global_active_power is in kilowatts — mean over 30 min ≈ kWh consumed
df_30 = df["Global_active_power"].resample("30min").mean()

# drop any remaining NaNs from gaps in the original data
df_30 = df_30.dropna()

print(f"Rows after 30-min resample & dropna: {len(df_30):,}")

# build feature columns to match train_model.py expectations
result = pd.DataFrame(index=df_30.index)
result["Timestamp"] = df_30.index
result["Electricity_Consumed"] = df_30.values          # raw kWh (NOT pre-scaled)

ts = df_30.index
result["Hour"]    = ts.hour
result["Month"]   = ts.month
result["Day"]     = ts.day
result["Year"]    = ts.year

# temperature / Humidity / Wind not in this dataset — use Hour-based proxies
# so the model at least has some weather-like signal.
# (Replace with real weather data later if desired.)
np.random.seed(42)
n = len(result)

# simple temperature proxy: daily cycle + seasonal variation + small noise
day_of_year = ts.day_of_year
result["Temperature"] = (
    15                                            # base °C
    + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)   # seasonal
    + 4  * np.sin(2 * np.pi * ts.hour / 24)               # daily cycle (warmer afternoon)
    + np.random.normal(0, 1, n)                            # noise
)
result["Humidity"] = np.clip(
    60
    - 10 * np.sin(2 * np.pi * ts.hour / 24)               # lower midday
    + np.random.normal(0, 5, n),
    20, 100
)
result["Wind_Speed"] = np.abs(np.random.normal(3, 2, n))  # m/s, always positive

# rolling 48-step (24h) average of past consumption
result["Avg_Past_Consumption"] = (
    result["Electricity_Consumed"]
    .rolling(window=48, min_periods=1)
    .mean()
)

# anomaly label: flag anything > 3 std deviations from rolling mean
rolling_std = result["Electricity_Consumed"].rolling(48, min_periods=1).std().fillna(0)
result["Anomaly_Label"] = (
    (result["Electricity_Consumed"] - result["Avg_Past_Consumption"]).abs()
    > 3 * rolling_std
).astype(int)

# scale Electricity_Consumed to 0-1 (matching old format) ───────────────
e_min = result["Electricity_Consumed"].min()
e_max = result["Electricity_Consumed"].max()
result["Electricity_Consumed"] = (result["Electricity_Consumed"] - e_min) / (e_max - e_min)
result["Avg_Past_Consumption"]  = (result["Avg_Past_Consumption"]  - e_min) / (e_max - e_min)

# column order (matches original processed_data.csv) ───────────────
result = result[[
    "Timestamp", "Electricity_Consumed", "Temperature", "Humidity",
    "Wind_Speed", "Avg_Past_Consumption", "Anomaly_Label",
    "Year", "Month", "Day", "Hour"
]]

# save 
out_path = "processed_data.csv"
result.to_csv(out_path, index=False)

print(f"\n✅ Saved {len(result):,} rows → {out_path}")
print(f"   Date range : {result['Timestamp'].iloc[0]}  →  {result['Timestamp'].iloc[-1]}")
print(f"   kWh range  : {result['Electricity_Consumed'].min():.3f} – {result['Electricity_Consumed'].max():.3f}")
print(f"   Anomalies  : {result['Anomaly_Label'].sum():,} flagged")
print("\nCorrelation with Electricity_Consumed:")
print(result[["Electricity_Consumed","Hour","Month","Temperature","Avg_Past_Consumption"]].corr()["Electricity_Consumed"])
print("\nHourly averages (should show real daily pattern):")
print(result.groupby("Hour")["Electricity_Consumed"].mean().round(3).to_string())