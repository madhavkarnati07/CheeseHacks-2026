# prediction.py
# Usage:
#   python prediction.py                          # defaults: city=madison, plots shown
#   python prediction.py --city milwaukee         # city-specific TOU rates
#   python prediction.py --city greenbay --no-plot
#   python prediction.py --city madison --no-plot

import argparse
import json
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# CLI args
parser = argparse.ArgumentParser(description="WattSense LSTM Energy Optimizer")
parser.add_argument(
    "--city",
    choices=["madison", "milwaukee", "greenbay"],
    default="madison",
    help="City whose TOU rate structure to apply (default: madison)",
)
parser.add_argument(
    "--no-plot",
    action="store_true",
    help="Skip matplotlib chart generation (useful when running headless / serving via API)",
)
args = parser.parse_args()

CITY = args.city
NO_PLOT = args.no_plot

# City TOU rate tables 
#   Each entry has:
#     prices     
#     flat_rate  – blended flat-rate equivalent shown in the UI
#     label      
#     peak_hours 
#     shift_frac 
#
# Rates sourced from:
#   Madison   
#   Milwaukee 
#   Green Bay 
CITY_CONFIG = {
    "madison": {
        "label": "Madison (MG&E Rg-2A)",
        "flat_rate": 0.1885,
        "shift_frac": 0.45,          # moderate peak premium shift 45%
        "prices": [
            0.139, 0.139, 0.139, 0.139, 0.139, 0.139, 0.139, 0.139, 0.139, 0.139,  # 0-9
            0.234, 0.234, 0.234,                                                      # 10-12
            0.248, 0.248, 0.248, 0.248, 0.248, 0.248,                                # 13-18
            0.244, 0.244, 0.244,                                                      # 19-21
            0.139, 0.139,                                                             # 22-23
        ],
        "peak_hours":    list(range(13, 22)),   # 1pm–9pm  (prices ≥ 0.244)
        "offpeak_hours": list(range(0, 10)) + [22, 23],  # midnight–9am + 10pm–midnight
    },
    "milwaukee": {
        "label": "Milwaukee (We Energies)",
        "flat_rate": 0.1934,
        "shift_frac": 0.55,          # large peak/off-peak spread shift 55%
        "prices": [
            0.104, 0.104, 0.104, 0.104, 0.104, 0.104, 0.104,  # 0-6
            0.304, 0.304, 0.304, 0.304, 0.304, 0.304, 0.304,  # 7-13
            0.304, 0.304, 0.304, 0.304, 0.304,                 # 14-18
            0.104, 0.104, 0.104, 0.104, 0.104,                 # 19-23
        ],
        "peak_hours":    list(range(7, 19)),
        "offpeak_hours": list(range(0, 7)) + list(range(19, 24)),
    },
    "greenbay": {
        "label": "Green Bay (WPS 3-Tier)",
        "flat_rate": 0.148,
        "shift_frac": 0.65,          # extreme peak rate shift 65%
        "prices": [
            0.081, 0.081, 0.081, 0.081, 0.081, 0.081, 0.081,  # 0-6
            0.144, 0.144, 0.144, 0.144, 0.144, 0.144, 0.144,  # 7-13
            0.144, 0.144,                                       # 14-15
            0.325, 0.325, 0.325, 0.325,                        # 16-19
            0.144, 0.144,                                       # 20-21
            0.081, 0.081,                                       # 22-23
        ],
        "peak_hours":    list(range(16, 20)),
        "offpeak_hours": list(range(0, 7)) + [22, 23],
    },
}

cfg = CITY_CONFIG[CITY]
PRICES_BY_HOUR = cfg["prices"]   # indexed by hour-of-day (0-23)
SHIFT_FRACTION = cfg["shift_frac"]
PEAK_HOURS     = cfg["peak_hours"]
OFFPEAK_HOURS  = cfg["offpeak_hours"]

def get_rate(hour: int) -> tuple[float, str]:
    """Return ($/kWh, period_label) for a given hour."""
    price = PRICES_BY_HOUR[hour]
    if hour in PEAK_HOURS:
        return price, "PEAK"
    elif hour in OFFPEAK_HOURS:
        return price, "OFF-PEAK"
    else:
        return price, "MID-PEAK"

print(f"\n{'='*62}")
print(f"  WattSense Energy Optimizer — {cfg['label']}")
print(f"  Shift fraction : {SHIFT_FRACTION*100:.0f}%  |  Flat rate : ${cfg['flat_rate']:.4f}/kWh")
print(f"{'='*62}\n")

#  load model & scaler 
model  = load_model("energy_lstm_model.h5")
scaler = joblib.load("scaler.save")
print("Model and scaler loaded.")

# load processed data 
data = pd.read_csv("processed_data.csv")

FEATURES = ["Electricity_Consumed", "Hour", "Month", "Temperature", "Avg_Past_Consumption"]
STEPS    = 48

df_features = data[FEATURES].copy()
scaled_all  = scaler.transform(df_features)
seed        = scaled_all[-STEPS:]

# forecast next 48 half hour steps (24 hours) 
last_ts   = pd.to_datetime(data["Timestamp"].iloc[-1])
future_ts = [last_ts + pd.Timedelta(minutes=30 * (i + 1)) for i in range(48)]

future_features_raw          = data[FEATURES].iloc[-STEPS:].copy()
future_features_raw["Hour"]  = [ts.hour  for ts in future_ts]
future_features_raw["Month"] = [ts.month for ts in future_ts]
future_features_scaled       = scaler.transform(future_features_raw)

predictions_scaled = []
current_seq = seed.copy()

for i in range(48):
    pred = model.predict(current_seq.reshape(1, STEPS, len(FEATURES)), verbose=0)[0, 0]
    predictions_scaled.append(pred)
    next_row    = future_features_scaled[i].copy()
    next_row[0] = pred
    current_seq = np.append(current_seq[1:], [next_row], axis=0)

dummy = np.zeros((48, len(FEATURES)))
dummy[:, 0] = predictions_scaled
predictions_kwh = scaler.inverse_transform(dummy)[:, 0]
predictions_kwh = np.clip(predictions_kwh, 0, None)

print(f"Prediction range : {predictions_kwh.min():.3f} – {predictions_kwh.max():.3f} kWh")

hours = [ts.hour for ts in future_ts]

# build original (unoptimized) schedule 
original_rows = []
for ts, kwh in zip(future_ts, predictions_kwh):
    rate, period = get_rate(ts.hour)
    original_rows.append({
        "Timestamp": ts.strftime("%Y-%m-%d %H:%M"),
        "Hour":      ts.hour,
        "kWh":       round(float(kwh), 4),
        "Period":    period,
        "Rate":      rate,
        "Cost":      round(float(kwh) * rate, 4),
    })
df_orig = pd.DataFrame(original_rows)

# build optimized schedule 
#  strategy: shift SHIFT_FRACTION of peak load to the cheapest off peak slots.
#  total energy consumed is conserved only the timing changes.

optimized_kwh = predictions_kwh.copy()

# indices into the 48-slot array
peak_idx    = [i for i, h in enumerate(hours) if h in PEAK_HOURS]
offpeak_idx = [i for i, h in enumerate(hours) if h in OFFPEAK_HOURS]

total_to_shift = sum(optimized_kwh[i] for i in peak_idx) * SHIFT_FRACTION

# reduce peak slots
for i in peak_idx:
    optimized_kwh[i] *= (1 - SHIFT_FRACTION)

# distribute shifted load proportionally across off-peak slots
# (weight by existing off-peak usage so the curve stays realistic)
offpeak_base = np.array([optimized_kwh[i] for i in offpeak_idx], dtype=float)
offpeak_weights = offpeak_base / offpeak_base.sum() if offpeak_base.sum() > 0 \
                  else np.ones(len(offpeak_idx)) / len(offpeak_idx)

for idx, i in enumerate(offpeak_idx):
    optimized_kwh[i] += total_to_shift * offpeak_weights[idx]

opt_rows = []
for ts, kwh in zip(future_ts, optimized_kwh):
    rate, period = get_rate(ts.hour)
    opt_rows.append({
        "Timestamp": ts.strftime("%Y-%m-%d %H:%M"),
        "Hour":      ts.hour,
        "kWh":       round(float(kwh), 4),
        "Period":    period,
        "Rate":      rate,
        "Cost":      round(float(kwh) * rate, 4),
    })
df_opt = pd.DataFrame(opt_rows)

# save CSVs 
df_orig.to_csv("predictions_24h.csv", index=False)
df_opt.to_csv("optimized_schedule.csv", index=False)
print("Saved → predictions_24h.csv  |  optimized_schedule.csv")

# compute summary stats 
orig_cost  = df_orig["Cost"].sum()
opt_cost   = df_opt["Cost"].sum()
savings    = orig_cost - opt_cost
savings_pct = savings / orig_cost * 100 if orig_cost > 0 else 0
total_kwh  = df_orig["kWh"].sum()

peak_kwh      = df_orig.loc[df_orig["Period"] == "PEAK", "kWh"].sum()
peak_exposure = peak_kwh / total_kwh * 100 if total_kwh > 0 else 0

# hurly averages for the consumption chart (48 slots - 24 hourly means)
hourly_actual    = df_orig.groupby("Hour")["kWh"].mean().reindex(range(24), fill_value=0).tolist()
hourly_optimized = df_opt.groupby("Hour")["kWh"].mean().reindex(range(24), fill_value=0).tolist()

# save JSON summary (consumed by the frontend) 
summary = {
    "city":             CITY,
    "city_label":       cfg["label"],
    "flat_rate":        cfg["flat_rate"],
    "prices_by_hour":   PRICES_BY_HOUR,
    "shift_fraction":   SHIFT_FRACTION,
    "total_kwh":        round(total_kwh, 3),
    "original_cost":    round(orig_cost, 4),
    "optimized_cost":   round(opt_cost, 4),
    "savings":          round(savings, 4),
    "savings_pct":      round(savings_pct, 1),
    "peak_exposure_pct": round(peak_exposure, 1),
    "hourly_actual":    [round(v, 4) for v in hourly_actual],
    "hourly_optimized": [round(v, 4) for v in hourly_optimized],
    # full 48-slot arrays for detailed charts
    "slots_original":  df_orig[["Timestamp","Hour","kWh","Period","Rate","Cost"]].to_dict(orient="records"),
    "slots_optimized": df_opt[["Timestamp","Hour","kWh","Period","Rate","Cost"]].to_dict(orient="records"),
}

with open("predictions_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved → predictions_summary.json")

# print report 
print(f"\n{'='*62}")
print(f"  24-HOUR ENERGY OPTIMIZATION REPORT  [{cfg['label']}]")
print(f"{'='*62}")
print(f"  Total kWh (conserved)  : {total_kwh:.3f} kWh")
print(f"  Original Cost          : ${orig_cost:.4f}")
print(f"  Optimized Cost         : ${opt_cost:.4f}")
print(f"  SAVINGS                : ${savings:.4f}  ({savings_pct:.1f}% reduction)")
print(f"  Peak Exposure          : {peak_exposure:.1f}% of total usage in peak hours")
print(f"{'='*62}")
print(f"  Strategy: Shift {SHIFT_FRACTION*100:.0f}% of peak-hour load to off-peak slots")
print(f"  Peak hours             : {PEAK_HOURS}")
print(f"  Off-peak hours         : {OFFPEAK_HOURS}")
print(f"{'='*62}\n")

# optional matplotlib charts 
if NO_PLOT:
    print("--no-plot flag set. Skipping chart generation.")
else:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch

    def slot_colors(hours_list):
        return ["#e74c3c" if h in PEAK_HOURS
                else "#f39c12" if h not in OFFPEAK_HOURS
                else "#2ecc71"
                for h in hours_list]

    colors  = slot_colors(hours)
    x       = np.arange(48)
    labels  = [ts.strftime("%H:%M") for ts in future_ts]
    ticks   = [i for i in range(48) if i % 4 == 0]

    legend_patches = [
        Patch(facecolor="#e74c3c", label="Peak"),
        Patch(facecolor="#f39c12", label="Mid-Peak"),
        Patch(facecolor="#2ecc71", label="Off-Peak"),
    ]

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    # Row 1: kWh bars
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x, df_orig["kWh"], color=colors, alpha=0.85, edgecolor="white")
    ax1.set_title("Original Predicted Consumption", fontweight="bold")
    ax1.set_ylabel("Energy (kWh)")
    ax1.set_xticks(ticks); ax1.set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right", fontsize=7)
    ax1.legend(handles=legend_patches, fontsize=7)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x, df_opt["kWh"], color=colors, alpha=0.85, edgecolor="white")
    ax2.set_title(f"Optimized Consumption (Shifted {SHIFT_FRACTION*100:.0f}% of Peak)", fontweight="bold")
    ax2.set_ylabel("Energy (kWh)")
    ax2.set_xticks(ticks); ax2.set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right", fontsize=7)
    ax2.set_ylim(ax1.get_ylim())

    # row 2: overlay line chart
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(x, df_orig["kWh"], color="#3498db", lw=2, marker="o", ms=3, label="Original")
    ax3.plot(x, df_opt["kWh"],  color="#e74c3c", lw=2, marker="o", ms=3, label="Optimized", ls="--")
    ax3.fill_between(x, df_orig["kWh"], df_opt["kWh"],
                     where=df_orig["kWh"] > df_opt["kWh"], alpha=0.15, color="#e74c3c", label="Load reduced")
    ax3.fill_between(x, df_orig["kWh"], df_opt["kWh"],
                     where=df_orig["kWh"] < df_opt["kWh"], alpha=0.15, color="#2ecc71", label="Load added")
    for i in range(48):
        if hours[i] in PEAK_HOURS:
            ax3.axvspan(i - 0.5, i + 0.5, alpha=0.06, color="#e74c3c")
    ax3.set_title(f"Original vs Optimized — {cfg['label']}", fontweight="bold")
    ax3.set_ylabel("Energy (kWh)")
    ax3.set_xticks(ticks); ax3.set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right", fontsize=7)
    ax3.legend(fontsize=8)

    # row 3: cost bars
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.bar(x, df_orig["Cost"], color=colors, alpha=0.85, edgecolor="white")
    ax4.set_title(f"Original Cost  (Total: ${orig_cost:.4f})", fontweight="bold")
    ax4.set_ylabel("Cost ($)"); ax4.set_xlabel("Time")
    ax4.set_xticks(ticks); ax4.set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right", fontsize=7)

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.bar(x, df_opt["Cost"], color=colors, alpha=0.85, edgecolor="white")
    ax5.set_title(f"Optimized Cost  (Total: ${opt_cost:.4f}  |  Save ${savings:.4f})", fontweight="bold")
    ax5.set_ylabel("Cost ($)"); ax5.set_xlabel("Time")
    ax5.set_xticks(ticks); ax5.set_xticklabels([labels[i] for i in ticks], rotation=45, ha="right", fontsize=7)
    ax5.set_ylim(ax4.get_ylim())

    fig.suptitle(
        f"24-Hour Energy Forecast & Cost Optimization\n{cfg['label']} · 30-min intervals",
        fontsize=15, fontweight="bold"
    )
    plt.savefig("energy_forecast_chart.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Chart saved → energy_forecast_chart.png")