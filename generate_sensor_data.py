import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters for dataset creation
num_samples = 10000
start_date = datetime(2024, 1, 1)

# Generate timestamps
timestamps = [start_date + timedelta(hours=i) for i in range(num_samples)]

# Generate synthetic sensor data
np.random.seed(42)
temperature = np.random.normal(loc=70, scale=5, size=num_samples)
vibration = np.random.normal(loc=0.02, scale=0.01, size=num_samples)
pressure = np.random.normal(loc=30, scale=2, size=num_samples)
failure = np.random.choice([0, 1], size=num_samples, p=[0.98, 0.02])  # 2% failure rate

# Create DataFrame
data = pd.DataFrame({
    'timestamp': timestamps,
    'temperature': temperature,
    'vibration': vibration,
    'pressure': pressure,
    'failure': failure
})

# Save to CSV
data.to_csv('sensor_data.csv', index=False)