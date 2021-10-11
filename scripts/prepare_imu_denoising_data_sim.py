# Alex Test Git
import numpy as np
import transforms3d
import matplotlib.pyplot as plt
import pandas as pd
import random

#Generate IMU outputs
def calculate_T (angle_vec):
    T = np.array([[1, np.sin(angle_vec[0]) * np.tan(angle_vec[1]), np.cos(angle_vec[0]) * np.tan(angle_vec[1])],
                    [0, np.cos(angle_vec[0]), -1 * np.sin(angle_vec[0])],
                    [0, np.sin(angle_vec[0]) / np.cos(angle_vec[1]), np.cos(angle_vec[0]) / np.cos(angle_vec[1])]])
    return T

def t_n_b (angle_vec):
    T = np.array([[1, np.sin(angle_vec[0]) * np.tan(angle_vec[1]), np.cos(angle_vec[0]) * np.tan(angle_vec[1])],
                    [0, np.cos(angle_vec[0]), -1 * np.sin(angle_vec[0])],
                    [0, np.sin(angle_vec[0]) / np.cos(angle_vec[1]), np.cos(angle_vec[0]) / np.cos(angle_vec[1])]])
    return T

def update_angle (angle_vec, angular_accel_vec, delta_t):
    updated_angle = angle_vec + delta_t*angular_accel_vec
    return updated_angle

def calc_w(T, angular_accel_vec):
    print(T.dtype)
    print(angular_accel_vec.dtype)
    return np.matmul(T, angular_accel_vec)

def calc_f(linear_accel_vec, T_n_b, g):
    f = linear_accel_vec + np.matmul(T_n_b, g)
    return f

plot_on = True
time_vec = np.arange(0, 20, 0.01)
num_of_samples = len(time_vec)

phi_0 = 0
theta_0 = 0
psy_0 = 0
angle_vec_0 = np.array([phi_0 , theta_0, psy_0])

#Here gonna be the sine input vec
# phi_tag = 0.0
# theta_tag = 0.0
# psy_tag = 0.0
# angular_accel_vec_one = np.array([phi_tag, theta_tag, psy_tag])

# Data descriptions:
# set1: a=8, b = 10,  a = 3, b = 7
a = 8
b = 10

# phi_tag = (np.sin(a*time_vec+b) + (np.sin(2*a*time_vec + b/3) +np.abs((np.sin(3*a*time_vec+b/2)))))
# theta_tag = np.sin(a*time_vec+b + np.pi) + (np.sin(2*a*time_vec + b/3 + np.pi) +np.abs((np.sin(3*a*time_vec+b/2 + np.pi))))
# psy_tag = 0.5*(np.sin(a*time_vec+b) + (np.sin(2*a*time_vec + b/3) +np.abs((np.sin(3*a*time_vec+b/2)))))

phi_tag = np.sin(2*np.pi*time_vec)
theta_tag = np.sin(2*np.pi*time_vec + np.pi)
psy_tag = np.sin(2*np.pi*time_vec + np.pi/4)

angular_velocity_vec = np.array([phi_tag, theta_tag, psy_tag])
angular_velocity_vec = angular_velocity_vec.T

plt.plot(phi_tag, label='phi')
plt.plot(theta_tag, label='theta')
plt.plot(psy_tag, label='psy')
plt.legend()
plt.title("Angular Velocity")
plt.legend()
plt.xlabel('Num of samples')
plt.ylabel('[rad / sec]')
plt.grid()
if plot_on:
    plt.show()


#Here gonna be the sine input vec
# accel_x = 0.1
# accel_y = 0.2
# accel_z = 0.3
# linear_accel_vec = np.array([accel_x, accel_y, accel_z])

accel_x = np.sin(2*np.pi*time_vec)
accel_y = np.sin(2*np.pi*time_vec + np.pi)
accel_z = np.sin(2*np.pi*time_vec + np.pi/4)
a = 3
b = 7
# accel_x = np.sin(a*time_vec+b) + (np.sin(2*a*time_vec + b/3) +np.abs((np.sin(3*a*time_vec+b/2))))
# accel_y = np.sin(a*time_vec+b + np.pi) + (np.sin(2*a*time_vec + b/3 + np.pi) +np.abs((np.sin(3*a*time_vec+b/2 + np.pi))))
# accel_z = 0.5*(np.sin(a*time_vec+b) + (np.sin(2*a*time_vec + b/3) +np.abs((np.sin(3*a*time_vec+b/2)))))
linear_accel_vec = np.array([accel_x, accel_y, accel_z])
linear_accel_vec = linear_accel_vec.T

plt.plot(accel_x, label='X')
plt.plot(accel_y, label='Y')
plt.plot(accel_z, label='Z')
plt.title("Linear Acceleration")
plt.legend()
plt.xlabel('Num of samples')
plt.ylabel('[m/sec^2]')
plt.grid()
if plot_on:
    plt.show()

# Generate
x0 = 0
y0 = 0
z0 = 0
pose_vec_0 = np.array([x0 , y0, z0])

vx_0 = 0
vy_0 = 0
vz_0 = 0
velocity_vec_0 = np.array([vx_0 , vy_0, vz_0])

delta_t = 0.01
g = np.array([0, 0, 9.81])

w_vec = np.zeros((num_of_samples,3))  #true
f_vec = np.zeros((num_of_samples,3))  # true
pose_vec = np.zeros((num_of_samples,3))
quat_vec = np.zeros((num_of_samples,4))
quat_vec_noised = np.zeros((num_of_samples,4))
velocity_vec = np.zeros((num_of_samples,3))

for i,w in enumerate(w_vec):
    T = calculate_T(angle_vec_0)
    #w_vec[i][:] = calc_w(T, angular_accel_vec_one)
    w_vec[i][:] = calc_w(T, angular_velocity_vec[i][:])
    quat = transforms3d.euler.euler2quat(angle_vec_0[0], angle_vec_0[1], angle_vec_0[2])
    T_n_b = transforms3d.quaternions.quat2mat(quat)
    f_vec[i][:] = calc_f(linear_accel_vec[i][:], T_n_b, g)

    if i < num_of_samples -1:
        #velocity_vec[i+1][:] = velocity_vec[i][:] + delta_t * linear_accel_vec[i][:]
        velocity_vec[i + 1][:] = velocity_vec[i][:] + delta_t * (np.matmul(T_n_b,f_vec[i][:]) + g)
        pose_vec[i+1][:] = pose_vec[i][:] + velocity_vec[i][:]
        quat_vec[i+1][:] = transforms3d.euler.euler2quat(angle_vec_0[0], angle_vec_0[1], angle_vec_0[2])
        angle_noise_elem = np.random.normal(0, 0.285, (3, )) #0.5 deg noise
        angle_vec_0_noised = angle_vec_0 + angle_noise_elem
        quat_vec_noised[i+1][:] = transforms3d.euler.euler2quat(angle_vec_0_noised[0], angle_vec_0_noised[1], angle_vec_0_noised[2])
        #Updating for next step
    angle_vec_0 = update_angle(angle_vec_0, angular_velocity_vec[i][:], delta_t)
    #angle_vec_0 = update_angle(angle_vec_0, angular_accel_vec_one, delta_t)

# #display pose
# plt.plot(pose_vec [:, 0], label='X')
# plt.plot(pose_vec [:, 1], label='Y')
# plt.plot(pose_vec [:, 2], label='Z')
# plt.title('Pose ')
# plt.xlabel('Num of samples')
# plt.ylabel('[m]')
# plt.legend()
# plt.grid()
# if plot_on:
#     plt.show()

# Adding noise to pose_vec
mu_pose = 0
sigma_pose = 0.115 # Equivalnet to 0.2m error
pose_noise = np.random.normal(mu_pose, sigma_pose, (num_of_samples, 3))
# #display pose noise
# plt.plot(pose_noise [:, 0], label='X')
# plt.plot(pose_noise [:, 1], label='Y')
# plt.plot(pose_noise [:, 2], label='Z')
# plt.title('Pose Noise')
# plt.xlabel('Num of samples')
# plt.ylabel('[m]')
# plt.legend()
# plt.grid()
# if plot_on:
#     plt.show()

pose_vec_noised = pose_vec + pose_noise

# #display pose noise
# plt.plot(pose_vec_noised[:, 0], label='X')
# plt.plot(pose_vec_noised[:, 1], label='Y')
# plt.plot(pose_vec_noised[:, 2], label='Z')
# plt.title('Pose Noised samples')
# plt.xlabel('Num of samples')
# plt.ylabel('[m]')
# plt.legend()
# plt.grid()
# if plot_on:
#     plt.show()

# Adding noise to quat vec
mu_agle = 0
sigma_angle = 0.005 # equivalent to 1 deg error
angle_noised = np.random.normal(mu_pose, sigma_pose, (num_of_samples, 4))

# #display angle noise
# plt.plot(quat_vec[:, 0], 'b--', label='q1')
# plt.plot(quat_vec[:, 1], 'g-', label='q2')
# plt.plot(quat_vec[:, 2], 'm-', label='q3')
# plt.plot(quat_vec[:, 3], 'y-', label='q4')
#
# plt.plot(quat_vec_noised[:, 0], 'b-', label='q1_noise')
# plt.plot(quat_vec_noised[:, 1], 'g--', label='q2_noise')
# plt.plot(quat_vec_noised[:, 2], 'm--', label='q3_noise')
# plt.plot(quat_vec_noised[:, 3], 'y-',label='q4_noise')

# plt.title('Quaternions Noised and not noised')
# plt.xlabel('Num of samples')
# plt.ylabel('[quat]')
# plt.legend()
# plt.grid()
# if plot_on:
#     plt.show()

w_vec[w_vec > 10] = 10
w_vec[w_vec < -10] = -10
# Adding noise - optional
# adding noise to f_vec
# generating offset offset
# b_f = (20*(1e-3))/9.81
b_f = 0
# generating white noise
mu_wf = 0
# sigma_wf = 500*0.2*(1e-2)/9.81
sigma_wf = 200*(1e-4)/9.81
w_f = np.random.normal(mu_wf, sigma_wf, (num_of_samples, 3))
f_noise = b_f + w_f
f_vec_noised = f_vec + f_noise  #noised

# adding noise to w_vec
# generating offset
b_w = np.pi/180
# generating random noise
mu_ww = 0
sigma_ww = 10*(0.03*10*np.pi)/180
w_w = np.random.normal(mu_ww, sigma_ww, (num_of_samples, 3))
w_noise = b_w + w_w
w_vec_noised = w_vec + w_noise   #noised


plt.plot(f_vec[:, 0], label='X')
plt.plot(f_vec[:, 1], label='Y')
plt.plot(f_vec[:, 2], label='Z')
plt.title('Specific Force')
plt.xlabel('Num of samples')
plt.ylabel('[m/sec^2]')
plt.legend()
plt.grid()
if plot_on:
    plt.show()

plt.plot(f_vec_noised[:, 0], label='X')
plt.plot(f_vec_noised[:, 1], label='Y')
plt.plot(f_vec_noised[:, 2], label='Z')
plt.title('Specific Force Noised')
plt.xlabel('Num of samples')
plt.ylabel('[m/sec^2]')
plt.legend()
plt.grid()
if plot_on:
    plt.show()

# plt.plot(f_noise[:, 0], label='X')
# plt.plot(f_noise[:, 1], label='Y')
# plt.plot(f_noise[:, 2], label='Z')
# plt.title('Specific Force Noise')
# plt.xlabel('Num of samples')
# plt.ylabel('[m/sec^2]')
# plt.legend()
# plt.grid()
# if plot_on:
#     plt.show()

plt.subplot(2, 1, 1)
plt.plot(w_vec[:, 0], label='phi')
plt.plot(w_vec[:, 1], label='theta')
plt.plot(w_vec[:, 2], label='psy')
plt.title('Omega')
plt.xlabel('Num of samples')
plt.ylabel('[rad/sec]')
plt.legend()
plt.grid()
# if plot_on:
#     plt.show()

plt.subplot(2, 1, 2)
plt.plot(w_vec_noised[:, 0], label='phi')
plt.plot(w_vec_noised[:, 1], label='theta')
plt.plot(w_vec_noised[:, 2], label='psy')
plt.title('Omega Noised')
plt.xlabel('Num of samples')
plt.ylabel('[rad/sec]')
plt.legend()
plt.grid()
if plot_on:
    plt.show()

# plt.plot(w_noise[:, 0], label='phi')
# plt.plot(w_noise[:, 1], label='theta')
# plt.plot(w_noise[:, 2], label='psy')
# plt.title('Omega Noise')
# plt.xlabel('Num of samples')
# plt.ylabel('[rad/sec]')
# plt.legend()
# plt.grid()
# if plot_on:
#     plt.show()

# draw sinus
# a = 8
# b = 10
# t = np.linspace(0,10,1000)
# print(t)
# combined_sin = np.sin(a*t+b) + (np.sin(2*a*t + b/3) +np.abs((np.sin(3*a*t+b/2))))
# plt.plot(t, combined_sin)
# plt.show()

# #Plotting trajectory
# plt.scatter(pose_vec[:, 0], pose_vec[:, 1])
# plt.title('XY Trajectory')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.grid()
# if plot_on:
#     plt.show()

def moving_average(seq, window_size):
    i = 0
    averaged = seq.copy()
    while i < len(seq)-window_size:
        averaged[i][0] = np.mean(seq[i:i+window_size,0])
        averaged[i][1] = np.mean(seq[i:i + window_size,1])
        averaged[i][2] = np.mean(seq[i:i + window_size,2])
        i = i + 1
    return averaged

grav_movig_averaged = moving_average(f_vec_noised, 20)

# plt.subplot(2, 1, 1)
plt.plot(f_vec_noised[:, 0], label='X_noised')
plt.plot(f_vec_noised[:, 1], label='Y_noised')
plt.plot(f_vec_noised[:, 2], label='Z_noised')
plt.title('Specific Force Noised')
plt.xlabel('Num of samples')
plt.ylabel('[m/sec^2]')
plt.legend()
plt.grid()

# plt.subplot(2, 1, 2)
plt.plot(grav_movig_averaged[:, 0], label='X_averaged')
plt.plot(grav_movig_averaged[:, 1], label='Y_averaged')
plt.plot(grav_movig_averaged[:, 2], label='Z_averaged')
plt.title('Acceleration Averaged')
plt.xlabel('Num of samples')
plt.ylabel('[m/sec^2]')
plt.legend()
plt.grid()

plt.show()

dist = np.linalg.norm(grav_movig_averaged-f_vec)
print("Dist Numpy: ", dist)

data_dir_path = 'C:\\masters\\git\\imu-transformers\\datasets\\'
save_data = False
if save_data:
    data_name = '08_02_21_simple_sin_dinamic_sine_50k_train.csv'
    concat_data = np.hstack((f_vec_noised, f_vec))
    dff = pd.DataFrame(data=concat_data)
    # dff.to_csv(data_dir_path + data_name, index_label=['sample_num'], header=['gyro_x_noised', 'gyro_y_noised', 'gyro_z_noised', 'grav_x_noised', 'grav_y_noised', 'grav_z_noised', 'gyro_x', 'gyro_y', 'gyro_z', 'grav_x', 'grav_y', 'grav_z'])
    dff.to_csv(data_dir_path + data_name, header=['grav_x_noised', 'grav_y_noised', 'grav_z_noised', 'grav_x', 'grav_y', 'grav_z'], index=False)



# data_name = '15_20_20_simple_sin_2000_samples_05deg_02m.csv'
# concat_data = np.hstack((f_vec_noised,w_vec_noised, f_vec, w_vec))
# dff = pd.DataFrame(data=concat_data)
# # dff.to_csv(data_dir_path + data_name, index_label=['sample_num'], header=['gyro_x_noised', 'gyro_y_noised', 'gyro_z_noised', 'grav_x_noised', 'grav_y_noised', 'grav_z_noised', 'gyro_x', 'gyro_y', 'gyro_z', 'grav_x', 'grav_y', 'grav_z'])
# dff.to_csv(data_dir_path + data_name, header=['gyro_x_noised', 'gyro_y_noised', 'gyro_z_noised', 'grav_x_noised', 'grav_y_noised', 'grav_z_noised', 'gyro_x', 'gyro_y', 'gyro_z', 'grav_x', 'grav_y', 'grav_z'], index=False)


# data_name = '23_02_20_simple_sin_2000_05deg_02m.csv'
# concat_data = np.hstack((w_vec, f_vec, pose_vec, quat_vec))
# dff = pd.DataFrame(data=concat_data)
# dff.to_csv('/home/maintenance/inputs/' + data_name, index_label=['sample_num'], header=['gyro_x', 'gyro_y', 'gyro_z', 'grav_x', 'grav_y', 'grav_z', 'pos_x', 'pos_y', 'pos_z','ori_w', 'ori_x', 'ori_y', 'ori_z'])
#
# data_name = '23_02_20_simple_sin_2000_05deg_02m_noised.csv'
# concat_data_noised = np.hstack((w_vec, f_vec, pose_vec_noised, quat_vec_noised))
# dff = pd.DataFrame(data=concat_data_noised)
# dff.to_csv('/home/maintenance/inputs/' + data_name, index_label=['sample_num'], header=['gyro_x', 'gyro_y', 'gyro_z', 'grav_x', 'grav_y', 'grav_z', 'pos_x', 'pos_y', 'pos_z','ori_w', 'ori_x', 'ori_y', 'ori_z'])

print('Finished Generating Data')