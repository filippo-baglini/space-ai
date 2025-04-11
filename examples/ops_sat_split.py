import os
import numpy as np
import pandas as pd

# Path to the segments.csv file
csv_path = "datasets/OPS-SAT/data/segments.csv" 

# Read the CSV, parsing the timestamp column as datetime
df = pd.read_csv(csv_path, parse_dates=["timestamp"])

# Create output directories similar to the NASA structure:
# OPS-SAT/data/train and OPS-SAT/data/test
output_root = "datasets/OPS-SAT"
data_dir = os.path.join(output_root, "data")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Dictionary to store anomaly intervals for each channel (for the test split)
anomaly_dict = {}

# Process each channel separately
channels = df["channel"].unique()
for channel in channels:
    # Filter data for the channel and sort by timestamp
    channel_df = df[df["channel"] == channel].copy()
    channel_df.sort_values("timestamp", inplace=True)
    
    total_rows = len(channel_df)
    # Determine split index: first 80% is training, last 20% is testing.
    split_index = int(total_rows * 0.8)
    
    train_df = channel_df.iloc[:split_index]
    test_df = channel_df.iloc[split_index:]
    
    # Save the 'value' column as a NumPy array.
    # Depending on your downstream needs, you might want to include more columns.
    train_array = train_df["value"].values.astype(np.float32)
    test_array = test_df["value"].values.astype(np.float32)
    
    np.save(os.path.join(train_dir, f"{channel}.npy"), train_array)
    np.save(os.path.join(test_dir, f"{channel}.npy"), test_array)
    
    # Now extract anomaly intervals from the test set.
    test_reset = test_df.reset_index(drop=True)
    anomaly_flags = test_reset["anomaly"].astype(int).values
    
    intervals = []
    in_interval = False
    start_idx = None
    for i, flag in enumerate(anomaly_flags):
        if flag == 1 and not in_interval:
            # Start a new anomaly interval
            in_interval = True
            start_idx = i
        elif flag != 1 and in_interval:
            # End the current anomaly interval
            end_idx = i - 1
            intervals.append((start_idx, end_idx))
            in_interval = False
    # If the last data point is anomalous, close the interval
    if in_interval:
        intervals.append((start_idx, len(anomaly_flags) - 1))
    
    anomaly_dict[channel] = intervals

# Create anomalies.csv file in the test directory.
anomaly_list = []
for channel, intervals in anomaly_dict.items():
    anomaly_list.append({
        "channel": channel,
        "anomaly_sequences": str(intervals)
    })

anomaly_df = pd.DataFrame(anomaly_list)
anomaly_csv_path = os.path.join(test_dir, "anomalies.csv")
anomaly_df.to_csv(anomaly_csv_path, index=False)

print("Data processing complete.")
print(f"Train files saved in: {train_dir}")
print(f"Test files and anomalies.csv saved in: {test_dir}")
