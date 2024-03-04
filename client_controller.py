from __future__ import print_function

import airsim
import os
import argparse
import logging
import random
import time
import pandas as pd
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.interpolate import interp1d
from model_predictive_control import MPCController
from PIL import Image
import io
import cv2

client = airsim.CarClient()
client.confirmConnection()

initial_pose = airsim.Pose(
    airsim.Vector3r(-0.03267443925142288, -0.0007991418242454529, 2.680995464324951),
    airsim.Quaternionr(0.003961396403610706, 0.0008345289970748127, -0.04337545484304428, -0.9990506172180176)
)

car_name = "PhysXCar"
client.simSetVehiclePose(initial_pose, True, "PhysXCar")

client.enableApiControl(True)
car_controls = airsim.CarControls()
print('AirSimClient connected')


car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0


vehicle_name = "PhysXCar"
STEER_BOUND = 1.0
STEER_BOUNDS = (-STEER_BOUND, STEER_BOUND)
IMAGE_SIZE = (144, 256)
IMAGE_DECIMATION = 4
MIN_SPEED = 10
DTYPE = 'float32'
STEER_NOISE = lambda: random.uniform(-0.1, 0.1)
THROTTLE_NOISE = lambda: random.uniform(-0.05, 0.05)
STEER_NOISE_NN = lambda: 0 #random.uniform(-0.05, 0.05)
THROTTLE_NOISE_NN = lambda: 0 #random.uniform(-0.05, 0.05)
IMAGE_CLIP_LOWER = IMAGE_SIZE[0]
IMAGE_CLIP_UPPER = 0


def done():
    car_controls.steering = 0
    car_controls.throttle = 0
    car_controls.brake = 1

    client.setCarControls(car_controls)
    print("Done.")



def run_airsim_client(args):
    frames_per_episode = 10000
    spline_points = 10000

    report = {
        'num_episodes': args.num_episodes,
        'controller_name': args.controller_name,
        'distances': [],
        'target_speed': args.target_speed,
    }

    track_DF = pd.read_csv('path_data.txt', header=None)
    # The track data are rescaled by 100x with relation to Carla measurements
    # track_DF = track_DF / 100

    # pts_2D = track_DF.loc[:, [0, 1]].values
    # tck, u = splprep(pts_2D.T, u=None, s=4, per=1, k=3)
    # u_new = np.linspace(u.min(), u.max(), spline_points)
    # x_new, y_new = splev(u_new, tck, der=0)
    # pts_2D = np.c_[x_new, y_new]

    pts_2D = track_DF.iloc[:, [0, 1]].values

    # Generating parameter values for interpolation
    # spline_points = 100  # You can adjust this value based on your needs
    u = np.arange(pts_2D.shape[0])

    # Interpolating x and y coordinates separately
    f_x = interp1d(u, pts_2D[:, 0], kind='linear', fill_value='extrapolate')
    f_y = interp1d(u, pts_2D[:, 1], kind='linear', fill_value='extrapolate')

    # Generating new parameter values for interpolation
    u_new = np.linspace(u.min(), u.max(), spline_points)

    # Evaluating the interpolating functions at new parameter values
    x_new = f_x(u_new)
    y_new = f_y(u_new)

    # Updating the 2D points with interpolated values
    pts_2D = np.column_stack((x_new, y_new))

    car_controls.steering = 0.0
    car_controls.throttle = 0.5

    depth_array = None

    if args.controller_name == 'mpc':
        weather_id = 2
        controller = MPCController(args.target_speed)
    elif args.controller_name == 'pid':
        weather_id = 1
        controller = PDController(args.target_speed)
    # elif args.controller_name == 'pad':
    #     weather_id = 5
    #     controller = PadController()
    elif args.controller_name == 'nn':
        # Import it here because importing TensorFlow is time consuming
        from neural_network_controller import NNController  # noqa
        weather_id = 11
        controller = NNController(
            args.target_speed,
            args.model_dir_name,
            args.which_model,
            args.throttle_coeff_A,
            args.throttle_coeff_B,
            args.ensemble_prediction,
        )
        report['model_dir_name'] = args.model_dir_name
        report['which_model'] = args.which_model
        report['throttle_coeff_A'] = args.throttle_coeff_A
        report['throttle_coeff_B'] = args.throttle_coeff_B
        report['ensemble_prediction'] = args.ensemble_prediction

    episode = 0
    num_fails = 0

    while episode < args.num_episodes:
        # Start a new episode

        if args.store_data:
            depth_storage = np.zeros((
                (IMAGE_CLIP_LOWER-IMAGE_CLIP_UPPER) // IMAGE_DECIMATION,
                IMAGE_SIZE[1] // IMAGE_DECIMATION,
                frames_per_episode
            )).astype(DTYPE)
            log_dicts = frames_per_episode * [None]
        else:
            depth_storage = None
            log_dicts = None

        status, depth_storage, one_log_dict, log_dicts, distance_travelled = run_episode(
            client,
            car_controls,
            controller,
            pts_2D,
            depth_storage,
            log_dicts,
            frames_per_episode,
            args.controller_name,
            args.store_data
        )

        # status = 'TRUE'

        if 'FAIL' in status:
            num_fails += 1
            print(status)
            continue
        else:
            print('SUCCESS: ' + str(episode))
            # report['distances'].append(distance_travelled)
            if args.store_data:
                np.save('depth_data/{}_depth_data{}.npy'.format(args.controller_name, episode), depth_storage)
                # pd.DataFrame(log_dicts).to_csv(
                #     'logs/{}_racetrack{}_log{}.txt'.format(args.controller_name, args.racetrack, episode), index=False)
            episode += 1

    report['num_fails'] = num_fails

    # report_output = os.path.join('reports', args.report_filename)
    # pd.to_pickle(report, report_output)


def run_episode(client, car_controls, controller, pts_2D, depth_storage, log_dicts, frames_per_episode, controller_name, store_data):
    num_laps = 0
    curr_closest_waypoint = None
    prev_closest_waypoint = None
    num_waypoints = pts_2D.shape[0]
    num_steps_below_min_speed = 0

    MIN_DEPTH_METERS = 0
    MAX_DEPTH_METERS = 50

    for frame in range(frames_per_episode):
        response = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])[0]

        # Reshape to a 2d array with correct width and height
        depth_img_in_meters = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
        depth_img_in_meters = depth_img_in_meters.reshape(response.height, response.width, 1)

        # Lerp 0..100m to 0..255 gray values
        depth_8bit_lerped = np.interp(depth_img_in_meters, (MIN_DEPTH_METERS, MAX_DEPTH_METERS), (0, 255))
        cv2.imwrite("2.png", depth_8bit_lerped.astype('uint8'))

        # Get depth data from the response
        depth_data = np.array(response.image_data_float, dtype=np.float32)

        # Reshape the depth data to match the image dimensions
        depth_height = response.height
        depth_width = response.width
        depth_array = depth_data.reshape((depth_height, depth_width))
        depth_array = depth_array[IMAGE_CLIP_UPPER:IMAGE_CLIP_LOWER, :][::IMAGE_DECIMATION, ::IMAGE_DECIMATION]

        # Get vehicle pose
        pose = client.simGetVehiclePose()

        # Get car state
        car_state = client.getCarState()

        one_log_dict = controller.control(pts_2D, car_state, pose, depth_array)

        prev_closest_waypoint = curr_closest_waypoint
        curr_closest_waypoint = one_log_dict['which_closest']

        # Check if we made a whole lap
        if prev_closest_waypoint is not None:
            # It's possible for `prev_closest_waypoint` to be larger than `curr_closest_waypoint`
            # but if `0.9*prev_closest_waypoint` is larger than `curr_closest_waypoint`
            # it definitely means that we completed a lap (or the car had been teleported)
            if 0.9 * prev_closest_waypoint > curr_closest_waypoint:
                num_laps += 1

        steer, throttle = one_log_dict['steer'], one_log_dict['throttle']
        car_controls.steering = steer
        car_controls.throttle = throttle

        # Send control commands to the vehicle
        client.setCarControls(car_controls)

        if store_data:
            depth_storage[..., frame] = depth_array
            one_log_dict['frame'] = frame
            log_dicts[frame] = one_log_dict

        distance_travelled = num_laps + curr_closest_waypoint / float(num_waypoints)
        return 'SUCCESS', depth_storage, one_log_dict, log_dicts, distance_travelled


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-e', '--num_episodes',
        default=400,
        type=int,
        dest='num_episodes',
        help='Number of episodes')
    argparser.add_argument(
        '-s', '--speed',
        default=7,
        type=float,
        dest='target_speed',
        help='Target speed')
    argparser.add_argument(
        '-cont', '--controller_name',
        default='mpc',
        dest='controller_name',
        help='Controller name')
    argparser.add_argument(
        '-sd', '--store_data',
        action='store_false',
        dest='store_data',
        help='Should the data be stored?')

    # For the NN controller
    argparser.add_argument(
        '-mf', '--model_dir_name',
        default=None,
        dest='model_dir_name',
        help='NN model directory name')
    argparser.add_argument(
        '-w', '--which_model',
        default='best',
        dest='which_model',
        help='Which model to load (5, 10, 15, ..., or: "best")')
    argparser.add_argument(
        '-tca', '--throttle_coeff_A',
        default=1.0,
        type=float,
        dest='throttle_coeff_A',
        help='Coefficient by which NN throttle predictions will be multiplied by')
    argparser.add_argument(
        '-tcb', '--throttle_coeff_B',
        default=0.0,
        type=float,
        dest='throttle_coeff_B',
        help='Coefficient by which NN throttle predictions will be shifted by')
    argparser.add_argument(
        '-ens', '--ensemble-prediction',
        action='store_true',
        dest='ensemble_prediction',
        help='Whether predictions for steering should be aggregated')

    args = argparser.parse_args()
    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    while True:
        # try:
        run_airsim_client(args)

        done()
        return


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
