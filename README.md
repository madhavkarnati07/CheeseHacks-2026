# WattSense

WattSense is an energy forecasting and optimization project built for the CheeseHacks 2026 hackathon. It combines a time-series LSTM model with utility rate-aware scheduling to forecast 24-hour electricity usage and reduce cost by shifting demand away from expensive peak periods.

## Overview

This repository contains:

- a Python-based LSTM forecasting pipeline for smart meter consumption data
- city-specific time-of-use (TOU) pricing logic for energy optimization
- generated CSV/JSON outputs for the next 24 hours of usage and costs
- an interactive HTML dashboard for viewing forecast and optimized schedule results

## Project goals

- predict household or smart-meter electricity usage over the next day
- model peak-vs-off-peak pricing behavior
- reduce energy costs by strategically shifting flexible load away from expensive periods
- provide an easy-to-use dashboard and script-based workflow

## Repository structure

- `train_model.py` — trains the LSTM model from processed smart-meter data
- `prediction.py` — loads the trained model, forecasts 24-hour demand, and optimizes the schedule
- `preprocessing.py` — prepares the raw smart-meter dataset for time-series modeling
- `energy_optimizer.html` — front-end dashboard for visualizing forecasted energy and savings
- `processed_data.csv` — cleaned and processed modeling dataset
- `smart_meter_data.csv` — original input data used for preprocessing
- `predictions_24h.csv` — baseline 24-hour predicted consumption schedule
- `optimized_schedule.csv` — cost-optimized 24-hour schedule
- `predictions_summary.json` — forecast summary used by the UI
- `energy_lstm_model.h5` — trained LSTM model
- `scaler.save` — saved MinMaxScaler used during training and inference
- `energy_forecast_chart.png` — generated forecast visualization
- `wattsense_logo_transparent.png` — project branding asset

## Tech stack

- Python
- Pandas
- NumPy
- scikit-learn
- TensorFlow / Keras
- Joblib
- Matplotlib
- HTML/CSS/JavaScript for the dashboard

## Setup

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:

```bash
pip install pandas numpy scikit-learn tensorflow joblib matplotlib
```

## Usage

### 1) Preprocess data

```bash
python preprocessing.py
```

### 2) Train the model

```bash
python train_model.py
```

### 3) Generate forecast and optimized schedule

```bash
python prediction.py --city madison
```

Available city options:

- `madison`
- `milwaukee`
- `greenbay`

You can skip plotting with:

```bash
python prediction.py --city madison --no-plot
```

### 4) Open the dashboard

Open `energy_optimizer.html` in a browser to explore the forecast, rate periods, and optimized schedule.

## Notes

- The model predicts the next 24 hours of energy demand in 30-minute intervals.
- The optimization logic shifts a portion of peak-hour demand into lower-cost periods based on city-specific TOU pricing.
- Forecast and optimization outputs are saved to CSV and JSON files for downstream use in the UI.

## Example output

The project generates:

- `predictions_24h.csv` — original predicted load profile
- `optimized_schedule.csv` — optimized load profile with lower cost
- `predictions_summary.json` — summary metrics including cost savings and peak exposure

## License

This repository does not currently include a license file. If you plan to publish or share the project publicly, consider adding an open-source license.
