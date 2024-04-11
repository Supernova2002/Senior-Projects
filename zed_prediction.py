
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import math
import matplotlib.animation as animation
from sympy import Symbol, solve
from math import*
from pykalman import KalmanFilter
from metran.kalmanfilter import SPKalmanFilter

projectile_df = pd.read_csv("position_data1712690560.7402215.csv")        
projectile_df = projectile_df.dropna()
basketball_x = projectile_df['x'].values
basketball_x = basketball_x[:130]
basketball_x = basketball_x *25.4 / 1000
basketball_y = projectile_df['y'].values
basketball_y = basketball_y[:130]
basketball_y = basketball_y *25.4 / 1000
basketball_z = projectile_df['z'].values
basketball_z = basketball_z[:130]
basketball_z = basketball_z *25.4/1000


framerate = 60
initial_distance = basketball_y[1]-basketball_y[0]
initial_velocity = (basketball_y[1]-basketball_y[0]) * framerate 
dT = 1 / framerate
g = 9.81
#initial_state = np.asarray([basketball_x[0],basketball_y[0],basketball_z[0],0,0,0,0,0,-1*g])
initial_state = [basketball_x[0],basketball_y[0],basketball_z[0],0,0,0,0,0,-1*g]
transition_matrix = np.asarray(
    [
        [1., 0., 0., dT, 0., 0., 0., 0., 0.], # x pos
        [0., 1., 0., 0., dT, 0., 0., 0., 0.], # y pos
        [0., 0., 1., 0., 0., dT, 0., 0., 0.], # z pos
        [0., 0., 0., 1., 0., 0., 0., 0., 0.], # x velocity
        [0., 0., 0., 0., 1., 0., 0., dT, 0.], # y velocity
        [0., 0., 0., 0., 0., 1., 0., 0., 0.], # z velocity
        [0., 0., 0., 0., 0., 0., 1., 0., 0.], # x accel
        [0., 0., 0., 0., 0., 0., 0., 1., 0.], # y accel
        [0., 0., 0., 0., 0., 0., 0., 0., 1.] # z accel
    ]
)
observation_matrix = np.asarray(
    [
        [1, 0, 0, 0, 0, 0,0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0]
    ]
)






#kf1 = KalmanFilter(transition_matrices = transition_matrix,
#                observation_matrices = observation_matrix,
#                initial_state_mean = initial_state)
measurements = []

for values in zip(basketball_x,basketball_y,basketball_z):
    measurements.append(values)

shrink_factor =4
kf1 = SPKalmanFilter()
kf1.set_matrices(transition_matrix=transition_matrix,transition_covariance= np.cov(transition_matrix),observation_matrix=observation_matrix,observation_variance=np.cov(observation_matrix))
kf1.set_observations(measurements)
kf1.run_filter(initial_state)
sys.exit(1)
measurements = measurements[0::int(len(measurements)/60)]
time1 = time.time()
#kf1 = kf1.em(measurements[:int(len(measurements)//shrink_factor)], n_iter=20)
print(len(measurements[:int(len(measurements)//shrink_factor)]))
(smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements[:int(len(measurements)//shrink_factor)])
next_mean = smoothed_state_means[-1]
next_covar = smoothed_state_covariances[-1]
second_half = measurements[int(len(measurements)//shrink_factor):]
predicted_error = []
predicted_state_means = []
predicated_state_covariances = []
pitch = []
yaw = []
for measure in second_half:
    next_mean, next_covar = kf1.filter_update(next_mean,next_covar)
    predicted_state_means.append(next_mean)
    predicated_state_covariances.append(next_covar)
    error = next_mean[0:3] - measure
    predicted_error.append(error)
    yaw.append(math.degrees(np.arctan(next_mean[1]/next_mean[0])))
    pitch.append(90-math.degrees(np.arctan(next_mean[3]/(np.sqrt(np.power(next_mean[0],2) + np.power(next_mean[1],2))))))
time2 = time.time()
print("Time for kalman filter is " + str(time2-time1) + " seconds")
#print(predicted_error)
print(predicted_error[-1])
print(yaw)
print(pitch)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(basketball_x, basketball_z, basketball_y, c = 'red', label = "basketball")
ax.legend()
ax.set_xlabel("Depth (meters)")
ax.set_zlabel("Height (meters)")
ax.set_ylabel("Width (meters)")