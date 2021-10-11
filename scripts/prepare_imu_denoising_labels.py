"""
Script for preparing labels for IMU denoising task
"""
import numpy as np
import pandas as pd

# Completely senseless dataset - just for initial debugging
window_size = 100
n_samples = 10
vals = np.random.random((n_samples*window_size,12))
d = {}
for i in range(12):
    d["x{}".format(i)] = vals[:, i]
df = pd.DataFrame(d)
df.to_csv("nonsense_imu_denoising_dataset.csv", index=False)
