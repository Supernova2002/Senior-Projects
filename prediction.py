
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import matplotlib.animation as animation
from sympy import Symbol, solve
from math import*
from pykalman import KalmanFilter


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--csv":
        projectile_df = pd.read_csv(args[1],index_col='Frame')        
        time1 = time.time()
        projectile_df = projectile_df.dropna()
        basketball_x = projectile_df['TX'].values
        basketball_x = basketball_x / 1000
        basketball_x = basketball_x[140:230]
        basketball_y = projectile_df['TZ'].values
        basketball_y = basketball_y / 1000
        basketball_y = basketball_y[140:230]
        basketball_z = projectile_df['TY'].values
        basketball_z = basketball_z / 1000
        basketball_z = basketball_z[140:230]


        framerate = 100 
        initial_distance = basketball_y[1]-basketball_y[0]
        initial_velocity = (basketball_y[1]-basketball_y[0]) * framerate 
        dT = 1 / framerate
        g = 9.81
        initial_state = np.asarray([basketball_x[0],basketball_y[0],basketball_z[0],0,0,0,0,-1*g,0])
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





        kf1 = KalmanFilter(transition_matrices = transition_matrix,
                        observation_matrices = observation_matrix,
                        initial_state_mean = initial_state)
        measurements = []

        for values in zip(basketball_x,basketball_y,basketball_z):
            measurements.append(values)


        shrink_factor =4

        measurements = measurements[0::int(len(measurements)/60)]
        print(len(measurements[:int(len(measurements)//shrink_factor)]))

        kf1 = kf1.em(measurements[:int(len(measurements)//shrink_factor)], n_iter=20)

        (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements[:int(len(measurements)//shrink_factor)])
        next_mean = smoothed_state_means[-1]
        next_covar = smoothed_state_covariances[-1]
        second_half = measurements[int(len(measurements)//shrink_factor):]
        predicted_error = []
        predicted_state_means = []
        predicated_state_covariances = []
        for measure in second_half:
            next_mean, next_covar = kf1.filter_update(next_mean,next_covar)
        predicted_state_means.append(next_mean)
        predicated_state_covariances.append(next_covar)
        error = next_mean[0:3] - measure
        predicted_error.append(error)
        time2 = time.time()
        print("Time for kalman filter is " + str(time2-time1) + " seconds")
        print(predicted_error[-1])




