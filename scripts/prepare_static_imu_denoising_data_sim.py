import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
number_of_samples = 2000
plot_on = False


gyro_vec_gt = np.zeros((number_of_samples,3))

# adding noise to w_vec
# generating offset
b_w = np.pi/180
# generating random noise
mu_ww = 0
sigma_ww = 2*(0.03*10*np.pi)/180
w_w = np.random.normal(mu_ww, sigma_ww, (number_of_samples, 3))
w_noise = b_w + w_w
gyro_vec_noised = gyro_vec_gt + w_noise   #noised

#plotting
# plt.subplot(2, 1, 1)
# plt.plot(gyro_vec_gt[:, 0], label='phi')
# plt.plot(gyro_vec_gt[:, 1], label='theta')
# plt.plot(gyro_vec_gt[:, 2], label='psy')
# plt.title('Omega')
# plt.xlabel('Num of samples')
# plt.ylabel('[rad/sec]')
# plt.legend()
# plt.grid()
#
# plt.subplot(2, 1, 2)
# plt.plot(gyro_vec_noised[:, 0], label='phi')
# plt.plot(gyro_vec_noised[:, 1], label='theta')
# plt.plot(gyro_vec_noised[:, 2], label='psy')
# plt.title('Omega Noised')
# plt.xlabel('Num of samples')
# plt.ylabel('[rad/sec]')
# plt.legend()
# plt.grid()
# if plot_on:
#     plt.show()


# Saving the data
# data_dir_path = 'C:\\masters\\git\\imu-transformers\\datasets\\'
# data_name = '21_12_20_static_gyro_2K.csv'
#
# concat_data = np.hstack((gyro_vec_noised, gyro_vec_gt))
# dff = pd.DataFrame(data=concat_data)
# dff.to_csv(data_dir_path + data_name, header=['gyro_x_noised', 'gyro_y_noised', 'gyro_z_noised','gyro_x', 'gyro_y', 'gyro_z'], index=False)

### Prepare acceleration vector - different angle sequence simlation
# num_samples = 100000
# np.random.seed(20)
# phi = np.random.rand()*np.pi/2
# print(phi)
# np.random.seed(70)
# theta = np.random.rand()*np.pi/2
# print(theta)
# g = 9.806
# angles_vec = np.array([-np.sin(theta), np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta)])
# print(angles_vec)
# accel_vec = angles_vec*g
# print(np.linalg.norm(accel_vec))
# accel_seq = np.vstack([accel_vec]*num_samples)
# print(accel_seq)


# plt.plot(accel_seq[:, 0], label='accel_x')
# plt.plot(accel_seq[:, 1], label='accel_y')
# plt.plot(accel_seq[:, 2], label='accel_z')
# plt.title('Omega')
# plt.xlabel('Num of samples')
# plt.ylabel('[m/sec^2]')
# plt.legend()
# plt.grid()
# plt.show()

def generate_accel_rand(num_samples):
    np.random.seed(40)
    phi = np.random.rand() * np.pi/3
    np.random.seed(41)
    theta = np.random.rand() * np.pi/3
    g = 9.806
    angles_vec = np.array([-np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)])
    accel_vec = angles_vec * g
    accel_seq = np.vstack([accel_vec] * num_samples)
    return accel_seq

def moving_average(seq, window_size):
    i = 0
    averaged = seq.copy()
    while i < len(seq)-window_size:
        averaged[i][0] = np.mean(seq[i:i + window_size,0])
        averaged[i][1] = np.mean(seq[i:i + window_size,1])
        averaged[i][2] = np.mean(seq[i:i + window_size,2])
        i = i + 1
    return averaged

def add_noise(seq):
    # b_f = (20*(1e-3))/9.81
    b_f = 0 # no offset
    # generating white noise
    mu_wf = 0
    sigma_wf = (20 * (1e-4)) / 9.81
    # sigma_wf = 500*0.2*(1e-2)/9.81
    w_f = np.random.normal(mu_wf, sigma_wf, (len(seq), 3))
    f_noise = b_f + w_f
    plt.plot(f_noise[:, 0], label='accel_x')
    # plt.plot(f_noise[:, 1], label='accel_y')
    # plt.plot(f_noise[:, 2], label='accel_z')
    plt.title('Noise')
    # plt.show()

    seq_noised = seq + f_noise
    return seq_noised


num_of_seq = 19
num_samples = 1000
avg_window_size = 20
accel_seq_all = generate_accel_rand(num_samples)
accel_seq_all_noised = add_noise(accel_seq_all)       # Adding noise
accel_seq_all_averaged = moving_average(accel_seq_all_noised, avg_window_size)# Averaging for baseline


#Generating noise
for i in range(num_of_seq):
    current_seq = generate_accel_rand(num_samples)
    noised_seq = add_noise(current_seq)
    averaged_seq = moving_average(noised_seq, avg_window_size)

    accel_seq_all = np.vstack([accel_seq_all, current_seq])
    accel_seq_all_noised = np.vstack([accel_seq_all_noised, noised_seq])
    accel_seq_all_averaged = np.vstack([accel_seq_all_averaged, averaged_seq])


# # b_f = (20*(1e-3))/9.81
# b_f = 0 # no offset
# # generating white noise
# mu_wf = 0
# sigma_wf = (20*(1e-4))/9.81
# # sigma_wf = 500*0.2*(1e-2)/9.81
# w_f = np.random.normal(mu_wf, sigma_wf, (len(accel_seq_all), 3))
# f_noise = b_f + w_f
# accel_seq_all_noised = accel_seq_all + f_noise  #noised


plt.subplot(3, 1, 1)
plt.plot(accel_seq_all[:, 0], label='accel_x')
plt.plot(accel_seq_all[:, 1], label='accel_y')
plt.plot(accel_seq_all[:, 2], label='accel_z')
plt.title('Acceleration')
plt.xlabel('Num of samples')
plt.ylabel('[m/sec^2]')
plt.legend()
plt.grid()


plt.subplot(3, 1, 2)
plt.plot(accel_seq_all_noised[:, 0], label='accel_x')
plt.plot(accel_seq_all_noised[:, 1], label='accel_y')
plt.plot(accel_seq_all_noised[:, 2], label='accel_z')
plt.title('Acceleration Noised')
plt.xlabel('Num of samples')
plt.ylabel('[m/sec^2]')
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(accel_seq_all_averaged[:, 0], label='accel_x')
plt.plot(accel_seq_all_averaged[:, 1], label='accel_y')
plt.plot(accel_seq_all_averaged[:, 2], label='accel_z')
plt.title('Acceleration Averaged')
plt.xlabel('Num of samples')
plt.ylabel('[m/sec^2]')
plt.legend()
plt.grid()

plt.show()

dist = np.linalg.norm(accel_seq_all_averaged-accel_seq_all)
print("Dist Numpy: ", dist)

t1 = torch.from_numpy(accel_seq_all_averaged)
t2 = torch.from_numpy(accel_seq_all)
dist_torch = torch.norm(t1-t2)
print("Dist Torch: ", dist_torch)

# Saving the data
save_data = False
if save_data:
    data_dir_path = 'C:\\masters\\git\\imu-transformers\\datasets\\'
    data_name = '07_03_21_static_random_accel_20_seq_of_1k_constant_test_4.csv'

    concat_data = np.hstack((accel_seq_all_noised, accel_seq_all))
    dff = pd.DataFrame(data=concat_data)
    dff.to_csv(data_dir_path + data_name, header=['acell_x_noised', 'acell_y_noised', 'acell_z_noised','acell_x', 'acell_y', 'acell_z'], index=False)

#generate noise
b_f = 0 # no offset
# generating white noise
mu_wf = 0
sigma_wf = (20 * (1e-4)) / 9.81
# sigma_wf = 500*0.2*(1e-2)/9.81
w_f_1 = np.random.normal(mu_wf, sigma_wf, (10000, 3))
f_noise_1 = b_f + w_f_1

w_f_2 = np.random.normal(mu_wf, sigma_wf, (20000, 3))
f_noise_2 = b_f + w_f_2
plt.subplot(2,1,1)
plt.plot(f_noise_1[:, 0], label='accel_x')
# plt.plot(f_noise[:, 1], label='accel_y')
# plt.plot(f_noise[:, 2], label='accel_z')
plt.title('Noise 1')
plt.subplot(2,1,2)
plt.plot(f_noise_2[:, 0], label='accel_x')
plt.title('Noise 1')

plt.show()

a = 5