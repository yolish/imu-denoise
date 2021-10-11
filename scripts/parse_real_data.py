import pandas as pd
import matplotlib.pyplot as plt

# CSV file = p40
p40_data = pd.read_csv('..\datasets\Real_Data\set1\p40.csv')
accel_p40 = p40_data[['accX', 'accY', 'accZ']]


#txt file = MRU
m200_data = pd.read_csv('..\datasets\Real_Data\set1\m200.csv')
accell_m200 = m200_data[['Acc_X', 'Acc_Y', 'Acc_Z']]

plt.subplot(2, 1, 1)
plt.plot(accel_p40['accX'], label='X')
plt.plot(accel_p40['accY'], label='Y')
plt.plot(accel_p40['accZ'], label='Z')
plt.title('P40 Smartphone')

plt.subplot(2, 1, 2)
plt.plot(accell_m200['Acc_X'], label='X')
plt.plot(accell_m200['Acc_Y'], label='Y')
plt.plot(accell_m200['Acc_Z'], label='Z')
plt.title('MRU Device')
plt.show()


a = 5