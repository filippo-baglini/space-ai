import os
import numpy as np
import pandas as pd

# Path to your CSV
csv_path = "datasets/OPSSAT/data/segments.csv"
df = pd.read_csv(csv_path, parse_dates=["timestamp"])

# Compute each channel’s median sampling interval
deltas = (
    df
    .sort_values(["channel", "timestamp"])
    .groupby("channel")["timestamp"]
    .diff()
    .dropna()
)
median_dt = deltas.groupby(df["channel"]).median()
common_dt = median_dt.min() 

# Prepare output dirs
root = "datasets/OPSSAT/data"
train_dir = os.path.join(root, "train")
test_dir  = os.path.join(root, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir,  exist_ok=True)

anomaly_dict = {}

for channel, ch_df in df.groupby("channel"):
    # Sort by time, set index
    ch = ch_df.sort_values("timestamp").set_index("timestamp")

    # Remove or aggregate duplicate timestamps
    ch = ch[~ch.index.duplicated(keep="first")]

    # Resample to uniform grid, forward-fill
    ch_resampled = ch.resample(common_dt).ffill()

    # Split by 'train' flag
    train_vals = ch_resampled.loc[ch_resampled["train"] == 1, "value"].astype(np.float32).values
    test_vals  = ch_resampled.loc[ch_resampled["train"] == 0, "value"].astype(np.float32).values

    np.save(os.path.join(train_dir, f"{channel}.npy"), train_vals)
    np.save(os.path.join(test_dir,  f"{channel}.npy"), test_vals)

    # Build anomaly intervals on test split
    flags = ch_resampled.loc[ch_resampled["train"] == 0, "anomaly"].astype(int).values
    intervals, in_anom = [], False
    for i, f in enumerate(flags):
        if f == 1 and not in_anom:
            in_anom, start = True, i
        elif f == 0 and in_anom:
            intervals.append((start, i - 1))
            in_anom = False
    if in_anom:
        intervals.append((start, len(flags) - 1))

    anomaly_dict[channel] = intervals

# Write anomalies.csv
an_rows = [
    {"channel": ch, "anomaly_sequences": str(seq)}
    for ch, seq in anomaly_dict.items()
]
pd.DataFrame(an_rows).to_csv(
    os.path.join(test_dir, "anomalies.csv"),
    index=False
)

print("Done: resampled at", common_dt)
print("Train: ", train_dir)
print("Test: ", test_dir, "(+ anomalies.csv)")
