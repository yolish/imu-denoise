import numpy as np
import pandas as pd
"""
Script for preparing labels for location recognition from IMU
"""
# Completely senseless dataset - just for initial debugging
window_size = 100
n_samples = 10
vals = np.random.random((n_samples*window_size,6))
d = {}
for i in range(6):
    d["x{}".format(i)] = vals[:, i]
d["cls"] = np.zeros(n_samples*window_size)
df = pd.DataFrame(d)
df.to_csv("nonsense_imu_lr_dataset.csv", index=False)
