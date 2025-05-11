# FILE MUST RUN IN PYTHON 3.6.8, Tensorflow 1.14.0 (GPU version requires CUDA 10), Keras 2.2.5, h5py 2.10.0, CARLA 0.9.13, Numpy 1.16.4, OpenCV 4.5.3

from platform import python_version

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

try:
    import carla
except ImportError:
    print("\nCarla module not found")
    print("Make sure to run the command 'pip install carla'")

    if python_version() != "3.6.8":
        print("\nPython 3.6.8 is required to run this program")
        print("Current Python Version: " + python_version())

        print("\nIf using VS code, use the command 'Ctrl + Shift + P'")
        print("Type 'Python: Select Interpreter'") 
        print("Select Python 3.6.8")

        print("\nIf you are using the command line, type 'python3.6' or 'py -3.6' instead of 'python'")

    input("\nPress Enter to exit")
    exit()

import tensorflow as tf
# Use TensorFlow's compatibility mode for deprecated APIs
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import math
import random
import time
import numpy as np
import cv2
from collections import deque

from keras.applications.xception import Xception 
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Flatten, AveragePooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
import keras.backend.tensorflow_backend as backend   # Must use older version of Tensorflow

import threading
from threading import Thread

from tqdm import tqdm

print("\nPreparing Client...\n")

# model_setup
MODEL_NAME = "CNN1"
# MODEL_NAME = "64x3"
# MODEL_NAME = "Xception"

IMG_WIDTH, IMG_HEIGHT = 640, 480
# RL parameters
EPISODES = 5000
EPISODE_LENGTH = 10 #seconds
LEARNING_RATE = 0.001
REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_SIZE = 1000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
# Epsilon
DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.995 ## tend towards 1.0 depeninding on # of steps
MIN_EPSILON = 0.1
# Q_WEIGHTS = [1.0, 0.8, 0.9, 0.6, 0.6] # [left, right, speed up, stay the same, slow down] 
ACTION_NAMES = ["left", "right", "speed up", "stay the same", "slow down"]
Q_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0] 

# Debugging
THREADED = True
SHOW_PREVIEW = False
PRINT_ACTIONS = False
PRINT_NUM_ACTIONS = False
PRINT_QS = False
PRINT_TIMES = False
PRINT_TRAINING = False
PRINT_THREADS = False

# Simulation parameters
POPULATE_CARS = True # True: Cars on highway, False: spawn obstacles
AUTOPILOT = True # spawn cars with autopilot
CARS_ON_HIGHWAY = 20 # cars on highway
WAYPOINTS_PER_LANE = 16 # waypoints on highway
WAYPOINT_SEPARATION = 7.0 # distance between waypoints

# GPU / Memory 
MEMORY_FRACTION = 0.8 # GPU allocation (if using GPU)
NUM_TRAINING_THREADS = 1 # number of threads to run in parallel

MIN_REWARD = 20 # export model if stayed alive for 20% of the episode
AGGREGATE_STATS_EVERY = 10 # tensorboard stats
# run "tensorboard --logdir=logs/" in terminal to view tensorboard
# http://localhost:6006/

def export_constants_to_txt():
    filename = f"models/Constants_{MODEL_NAME}_{int(time.time)}.txt"

    constants = {
        "MODEL_NAME": MODEL_NAME,
        "IMG_WIDTH": IMG_WIDTH,
        "IMG_HEIGHT": IMG_HEIGHT,
        "THREADED": THREADED,
        "\n" : "",
        "LEARNING_RATE": LEARNING_RATE,
        "REPLAY_MEMORY_SIZE": REPLAY_MEMORY_SIZE,
        "MIN_REPLAY_SIZE": MIN_REPLAY_SIZE,
        "MINIBATCH_SIZE": MINIBATCH_SIZE,
        "PREDICTION_BATCH_SIZE": PREDICTION_BATCH_SIZE,
        "TRAINING_BATCH_SIZE": TRAINING_BATCH_SIZE,
        "UPDATE_TARGET_EVERY": UPDATE_TARGET_EVERY,
        "MEMORY_FRACTION": MEMORY_FRACTION,
        "MIN_REWARD": MIN_REWARD,
        "EPISODES": EPISODES,
        "EPISODE_LENGTH": EPISODE_LENGTH,
        "DISCOUNT": DISCOUNT,
        "epsilon": epsilon,
        "EPSILON_DECAY": EPSILON_DECAY,
        "MIN_EPSILON": MIN_EPSILON,
        "AGGREGATE_STATS_EVERY": AGGREGATE_STATS_EVERY,
        "Q_WEIGHTS": Q_WEIGHTS,
        "\n" : "",
        "POPULATE_CARS": POPULATE_CARS,
        "AUTOPILOT": AUTOPILOT,
        "CARS_ON_HIGHWAY": CARS_ON_HIGHWAY,
        "WAYPOINTS_PER_LANE": WAYPOINTS_PER_LANE,
        "WAYPOINT_SEPARATION": WAYPOINT_SEPARATION,
    }

    # Write constants to the file
    with open(filename, "w") as file:
        for key, value in constants.items():
            file.write(f"{key}: {value}\n")

class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class CarEnvironment:
    SHOW_CAMERA = SHOW_PREVIEW
    STEER_AMOUNT = 1.0

    image_width = IMG_WIDTH
    image_height = IMG_HEIGHT
    actor_list = []
    auto_list = []

    front_camera = None
    collision_list = []

    # functions
    def collision_data(self, collision):
        self.collision_list.append(collision)

    def process_img(self, image):
        i = np.array(image.raw_data) # flatten image to array
        i2 = i.reshape((self.image_height, self.image_width, 4)) # 4 channels: RGBA
        i3 = i2[:, :, :3] # remove alpha channel (height, width, channels(3))
        i4 = i3/255.0 # normalise 

        if self.SHOW_CAMERA:
            cv2.imshow("", i4) # show image
            cv2.waitKey(10)

        # init camera
        self.front_camera = i3
        # return i4

    def generate_lane_waypoints_50_50(self, num_waypoints=WAYPOINTS_PER_LANE, separation=WAYPOINT_SEPARATION):
        # Get the current waypoint of the vehicle
        spawn_points = self.world.get_map().get_spawn_points()
        current_transform = spawn_points[312]
        current_waypoint = self.world.get_map().get_waypoint(current_transform.location)

        # Initialize dictionaries to store waypoints for each lane
        waypoints = {
            "lane2": [],  # Current lane
            "lane1": [],  # Left lane
            "lane3": [],  # Right lane
            "lane4": []   # Second right lane
        }

        # Generate waypoints for lane2 (current lane)
        waypoint = current_waypoint
        for _ in range(num_waypoints // 2):  # Generate waypoints behind
            previous_waypoints = waypoint.previous(separation)
            if previous_waypoints:
                waypoint = previous_waypoints[0]
                waypoints["lane2"].insert(0, waypoint)  # Insert at the beginning
            else:
                break

        waypoint = current_waypoint
        for _ in range(num_waypoints // 2):  # Generate waypoints in front
            next_waypoints = waypoint.next(separation)
            if next_waypoints:
                waypoint = next_waypoints[0]
                waypoints["lane2"].append(waypoint)
            else:
                break

        # Generate waypoints for lane1 (left lane)
        left_lane = current_waypoint.get_left_lane()
        if left_lane:
            waypoint = left_lane
            for _ in range(num_waypoints // 2):  # Generate waypoints behind
                previous_waypoints = waypoint.previous(separation)
                if previous_waypoints:
                    waypoint = previous_waypoints[0]
                    waypoints["lane1"].insert(0, waypoint)
                else:
                    break

            waypoint = left_lane
            for _ in range(num_waypoints // 2):  # Generate waypoints in front
                next_waypoints = waypoint.next(separation)
                if next_waypoints:
                    waypoint = next_waypoints[0]
                    waypoints["lane1"].append(waypoint)
                else:
                    break

        # Generate waypoints for lane3 (right lane)
        right_lane = current_waypoint.get_right_lane()
        if right_lane:
            waypoint = right_lane
            for _ in range(num_waypoints // 2):  # Generate waypoints behind
                previous_waypoints = waypoint.previous(separation)
                if previous_waypoints:
                    waypoint = previous_waypoints[0]
                    waypoints["lane3"].insert(0, waypoint)
                else:
                    break

            waypoint = right_lane
            for _ in range(num_waypoints // 2):  # Generate waypoints in front
                next_waypoints = waypoint.next(separation)
                if next_waypoints:
                    waypoint = next_waypoints[0]
                    waypoints["lane3"].append(waypoint)
                else:
                    break

        # Generate waypoints for lane4 (second right lane)
        if right_lane:
            second_right_lane = right_lane.get_right_lane()
            if second_right_lane:
                waypoint = second_right_lane
                for _ in range(num_waypoints // 2):  # Generate waypoints behind
                    previous_waypoints = waypoint.previous(separation)
                    if previous_waypoints:
                        waypoint = previous_waypoints[0]
                        waypoints["lane4"].insert(0, waypoint)
                    else:
                        break

                waypoint = second_right_lane
                for _ in range(num_waypoints // 2):  # Generate waypoints in front
                    next_waypoints = waypoint.next(separation)
                    if next_waypoints:
                        waypoint = next_waypoints[0]
                        waypoints["lane4"].append(waypoint)
                    else:
                        break

        return waypoints
    
    def generate_lane_waypoints_25_75(self, num_waypoints=WAYPOINTS_PER_LANE, separation=WAYPOINT_SEPARATION):
        # Get the current waypoint of the vehicle
        spawn_points = self.world.get_map().get_spawn_points()
        current_transform = spawn_points[312]
        current_waypoint = self.world.get_map().get_waypoint(current_transform.location)

        # Initialize dictionaries to store waypoints for each lane
        waypoints = {
            "lane2": [],  # Current lane
            "lane1": [],  # Left lane
            "lane3": [],  # Right lane
            "lane4": []   # Second right lane
        }

        # Generate waypoints for lane2 (current lane)
        waypoint = current_waypoint
        for _ in range(num_waypoints // 4):  # Generate 1/4 waypoints behind
            previous_waypoints = waypoint.previous(separation)
            if previous_waypoints:
                waypoint = previous_waypoints[0]
                waypoints["lane2"].insert(0, waypoint)  # Insert at the beginning
            else:
                break

        waypoint = current_waypoint
        for _ in range(3 * num_waypoints // 4):  # Generate 3/4 waypoints ahead
            next_waypoints = waypoint.next(separation)
            if next_waypoints:
                waypoint = next_waypoints[0]
                waypoints["lane2"].append(waypoint)
            else:
                break

        # Generate waypoints for lane1 (left lane)
        left_lane = current_waypoint.get_left_lane()
        if left_lane:
            waypoint = left_lane
            for _ in range(num_waypoints // 4):  # Generate 1/4 waypoints behind
                previous_waypoints = waypoint.previous(separation)
                if previous_waypoints:
                    waypoint = previous_waypoints[0]
                    waypoints["lane1"].insert(0, waypoint)
                else:
                    break

            waypoint = left_lane
            for _ in range(3 * num_waypoints // 4):  # Generate 3/4 waypoints ahead
                next_waypoints = waypoint.next(separation)
                if next_waypoints:
                    waypoint = next_waypoints[0]
                    waypoints["lane1"].append(waypoint)
                else:
                    break

        # Generate waypoints for lane3 (right lane)
        right_lane = current_waypoint.get_right_lane()
        if right_lane:
            waypoint = right_lane
            for _ in range(num_waypoints // 4):  # Generate 1/4 waypoints behind
                previous_waypoints = waypoint.previous(separation)
                if previous_waypoints:
                    waypoint = previous_waypoints[0]
                    waypoints["lane3"].insert(0, waypoint)
                else:
                    break

            waypoint = right_lane
            for _ in range(3 * num_waypoints // 4):  # Generate 3/4 waypoints ahead
                next_waypoints = waypoint.next(separation)
                if next_waypoints:
                    waypoint = next_waypoints[0]
                    waypoints["lane3"].append(waypoint)
                else:
                    break

        # Generate waypoints for lane4 (second right lane)
        if right_lane:
            second_right_lane = right_lane.get_right_lane()
            if second_right_lane:
                waypoint = second_right_lane
                for _ in range(num_waypoints // 4):  # Generate 1/4 waypoints behind
                    previous_waypoints = waypoint.previous(separation)
                    if previous_waypoints:
                        waypoint = previous_waypoints[0]
                        waypoints["lane4"].insert(0, waypoint)
                    else:
                        break

                waypoint = second_right_lane
                for _ in range(3 * num_waypoints // 4):  # Generate 3/4 waypoints ahead
                    next_waypoints = waypoint.next(separation)
                    if next_waypoints:
                        waypoint = next_waypoints[0]
                        waypoints["lane4"].append(waypoint)
                    else:
                        break

        return waypoints

    def generate_lane_waypoints_0_100(self, num_waypoints=WAYPOINTS_PER_LANE, separation=WAYPOINT_SEPARATION):
        spawn_points = self.world.get_map().get_spawn_points()
        current_transform = spawn_points[312]
        current_waypoint = self.world.get_map().get_waypoint(current_transform.location)

        waypoints = {
            "lane2": [],
            "lane1": [],
            "lane3": [],
            "lane4": []
        }

        waypoint = current_waypoint
        for _ in range(num_waypoints):
            next_waypoints = waypoint.next(separation)
            if next_waypoints:
                waypoint = next_waypoints[0]
                waypoints["lane2"].append(waypoint)
            else:
                break

        left_lane = current_waypoint.get_left_lane()
        if left_lane:
            waypoint = left_lane
            for _ in range(num_waypoints):
                next_waypoints = waypoint.next(separation)
                if next_waypoints:
                    waypoint = next_waypoints[0]
                    waypoints["lane1"].append(waypoint)
                else:
                    break

        right_lane = current_waypoint.get_right_lane()
        if right_lane:
            waypoint = right_lane
            for _ in range(num_waypoints):
                next_waypoints = waypoint.next(separation)
                if next_waypoints:
                    waypoint = next_waypoints[0]
                    waypoints["lane3"].append(waypoint)
                else:
                    break

            second_right_lane = right_lane.get_right_lane()
            if second_right_lane:
                waypoint = second_right_lane
                for _ in range(num_waypoints):
                    next_waypoints = waypoint.next(separation)
                    if next_waypoints:
                        waypoint = next_waypoints[0]
                        waypoints["lane4"].append(waypoint)
                    else:
                        break

        return waypoints

    def populate_autopilot_cars(self, all_waypoints, num_cars=10):
        # Shuffle the waypoints to randomize spawn locations
        selected_waypoints = [random.choice(row) for row in all_waypoints if row]

        # Spawn cars at random waypoints
        for i in range(min(num_cars, len(selected_waypoints))):
            # Choose a random waypoint
            spawn_waypoint = selected_waypoints[i]

            # Define the spawn transform for the vehicle
            spawn_transform = spawn_waypoint.transform
            spawn_transform.location.z += 1.0

            # Choose a random vehicle blueprint
            vehicle_bp = random.choice(self.bp_lib.filter('vehicle.toyota.prius'))

            # Try to spawn the vehicle
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
            if vehicle:
                # Enable autopilot
                if AUTOPILOT:
                    vehicle.set_autopilot(True)

                # Add the vehicle to the actor list for cleanup later
                self.actor_list.append(vehicle)
                self.auto_list.append(vehicle)
                # print(f"Autopilot car spawned at {spawn_transform.location}.")
            # else:
            #     print(f"Failed to spawn autopilot car at {spawn_transform.location}.")

    def spawn_obstacles(self, all_waypoints, num_cars=10):
        selected_waypoints = [random.choice(row) for row in all_waypoints if row]

        for i in range(min(num_cars, len(selected_waypoints))):
            spawn_waypoint = selected_waypoints[i]

            spawn_transform = spawn_waypoint.transform
            spawn_transform.location.z += 1.0

            vehicle_bp = random.choice(self.bp_lib.filter('static.prop.vendingmachine'))


            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
            if vehicle:
                self.actor_list.append(vehicle)

    def set_spectator(self):
        # Define the fixed location
        fixed_location = self.fixed_location

        # Calculate the spectator's position behind and above the fixed location
        offset_distance = 15.0  # Distance behind the fixed location
        offset_height = 25.0    # Height above the fixed location
        forward_vector = carla.Vector3D(0, -1, 0)  # Assume a forward vector pointing along the x-axis

        spectator_location = carla.Location(
            x=fixed_location.x - offset_distance * forward_vector.x,
            y=fixed_location.y - offset_distance * forward_vector.y,
            z=fixed_location.z + offset_height
        )
        spectator_rotation = carla.Rotation(
            pitch=-30,  # Slightly tilted down
            yaw=-90,  
            roll=0
        )

        # Get the spectator and set its transform
        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))
        #print("Spectator position updated based on fixed location.")
    # RL functions
    def reset(self):
        # reset lists
        self.collision_list = []
        self.actor_list = []
        self.auto_list = []

        # spawn agent
        spawn_points = self.world.get_map().get_spawn_points()

        self.transform = spawn_points[312] # start of highway

        self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.transform)
        self.actor_list.append(self.vehicle)

        if POPULATE_CARS:
            self.populate_autopilot_cars(self.all_waypoints, num_cars=CARS_ON_HIGHWAY) # populate highway (high numbers may peak CPU usage)
        else:
            self.spawn_obstacles(self.all_waypoints, num_cars=10) # spawn obstacles 

        time.sleep(0.5) # wait for cars to spawn

        if AUTOPILOT:
            for car in self.auto_list:
                try:
                    car.set_autopilot(True)
                except:
                    pass

        # initalise rgb camera
        self.camera_bp = self.bp_lib.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', f'{self.image_width}')
        self.camera_bp.set_attribute('image_size_y', f'{self.image_height}')
        self.camera_bp.set_attribute('fov', '110')

        # set camera spawn position
        camera_pos = carla.Transform(carla.Location(z=2))

        # spawn camera
        self.camera = self.world.spawn_actor(self.camera_bp, camera_pos, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        # collect camera data
        self.camera.listen(lambda image: self.process_img(image))

        # send vehicle control to improve agent response time
        if AUTOPILOT:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0)) # match speed of other cars
        else:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, brake=0.0))

        # avoid taking input during init
        time.sleep(4)

        # spawn collision sensor
        collision_sensor= self.bp_lib.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_sensor, camera_pos, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        # collect collision data
        self.collision_sensor.listen(lambda collision: self.collision_data(collision))

        # wait for camera initlisation
        while (self.front_camera is None):
            time.sleep(0.01)

        # used to control step length
        self.episode_start = time.time()

        # send vehicle control to improve agent response time
        if AUTOPILOT:
            self.vehicle.apply_control(carla.VehicleControl(throttle=2.0, brake=0.0)) # match speed of other cars
        else:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, brake=0.0))

        return self.front_camera
    
    def step(self, action): 
        current_transform = self.vehicle.get_transform()
        current_location = current_transform.location
        current_waypoint = self.world.get_map().get_waypoint(current_location)
        next_waypoint = None

        # action_define action_num
        if PRINT_ACTIONS:
            # Store the previous action as an instance variable
            if not hasattr(self, 'previous_action'):
                self.previous_action = None

            # Only print the action if it is different from the previous action
            if action != self.previous_action: 
                action_descriptions = {
                    0: "Change to the left lane",
                    1: "Change to the right lane",
                    2: "Speed up",
                    3: "Stay the same",
                    4: "Slow down"
                }
                print(f"\nCurrent Action: {action} - {action_descriptions.get(action, 'Unknown action')}")
            # Update the previous action
            self.previous_action = action

        if PRINT_QS:
            # Get current Q-values
            current_qs = agent.get_qs(self.front_camera)
            
            # Initialize previous_qs if not already done
            if not hasattr(self, 'previous_qs'):
                self.previous_qs = current_qs
                print(f"\nInitial Q-values: {current_qs}")
            # Check if Q-values have changed
            elif not np.array_equal(current_qs, self.previous_qs):
                print(f"\nQ-values changed:")
                print(f"Previous: {self.previous_qs}")
                print(f"Current:  {current_qs}")
                print(f"Difference: {current_qs - self.previous_qs}")
                self.previous_qs = current_qs

        # change action numbers to match first input
        if action == 0:  # Change to the left lane
            next_waypoint = current_waypoint.get_left_lane()
            if next_waypoint:
                self.vehicle.set_transform(next_waypoint.transform)

        elif action == 1:  # Change to the right lane
            next_waypoint = current_waypoint.get_right_lane()
            if next_waypoint:
                self.vehicle.set_transform(next_waypoint.transform)

        elif action == 2:  # Speed up
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 3:  # Stay the same
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=0))
        elif action == 4:  # Slow down
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.5))



        velocity = self.vehicle.get_velocity()
        speed_kmh = int(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))

        # Reward logic
        if len(self.collision_list) != 0:
            done = True
            reward = -1  # Penalty for collision
        elif next_waypoint is not None and (next_waypoint.lane_id == -5 or next_waypoint.lane_id == 1): # Check if the vehicle is off the road
            done = True
            reward = -1
        elif speed_kmh == 0: # TODO: check for non autopilot
            done = True
            reward = -1
        elif speed_kmh < 10:
            done = False
            reward = -0.5
        elif speed_kmh < 50:
            done = False
            reward = -0.3  # Penalty for slow speed
        else:
            done = False
            reward = 1  # Reward for maintaining good speed

        reward += 0.1 # Reward for survival

        # End simulation after EPISODE_LENGTH
        if self.episode_start + EPISODE_LENGTH < time.time():
            done = True

        return self.front_camera, reward, done, None 

    def __init__(self):

        print("Initialising Environment...\n")

        self.client = carla.Client("localhost", 2000)

        # Load the world
        attempts = 0
        while attempts < 3:
            try:
                self.world = self.client.load_world('Town04_OPT', carla.MapLayer.Buildings|carla.MapLayer.ParkedVehicles)
                break  # Exit the loop if no error occurs
            except Exception as e:
                if attempts != 0:
                    print(f"Carla Environment not responding: \n{e}") 
                attempts += 1
                if attempts < 3:
                    print(f"Retrying Response in 10 seconds... (Attempt {attempts}/3)")
                    time.sleep(10)
                else:
                    user_input = input(f"\nFailed response after {attempts} attempts reached. \nPlease close any instances of CARLA and reopen\n Press 'y' to retry or Enter to exit: ")
                    if user_input.lower() == 'y':
                        attempts = 0  # Reset attempts and retry
                    else:
                        print("Exiting...")
                        exit()

        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.world.unload_map_layer(carla.MapLayer.Buildings)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)

        self.bp_lib = self.world.get_blueprint_library()
        self.vehicle_bp = self.bp_lib.find('vehicle.nissan.micra')

        self.fixed_location = carla.Location(x=9.497402, y=214.450912, z=0.281942) #spawn 312

        if AUTOPILOT:
            lane_waypoints = self.generate_lane_waypoints_25_75()
        else:
            lane_waypoints = self.generate_lane_waypoints_0_100()

        self.all_waypoints = []
        for i in range(WAYPOINTS_PER_LANE):
            batch = [
                lane_waypoints["lane1"][i],
                lane_waypoints["lane2"][i],
                lane_waypoints["lane3"][i],
                lane_waypoints["lane4"][i]
            ]
            self.all_waypoints.append(batch)

        self.set_spectator()

        if AUTOPILOT:
            # Warm up the Traffic Manager
            vehicle_bp = self.bp_lib.find('vehicle.toyota.prius')  # Use any vehicle blueprint
            spawn_points = self.world.get_map().get_spawn_points()

            # Spawn a temporary vehicle
            temp_vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[0])
            if temp_vehicle:
                temp_vehicle.set_autopilot(True)  # Enable autopilot
                time.sleep(2)  # Allow time for the Traffic Manager to initialize
                temp_vehicle.destroy()  # Destroy the temporary vehicle 

class DQNAgent:
    def create_model_x(self): # Xception model
        base_model = Xception(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH,3) )

        # MODEL_NAME = "Xception"

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        predictions = Dense(5, activation="linear")(x) # Output layer == action_num
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=["accuracy"])
        return model

    def create_model_64(self):
        model = Sequential([
            Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'),

            Conv2D(64, (3, 3), padding='same', activation='relu'),
            AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'),

            Conv2D(64, (3, 3), padding='same', activation='relu'),
            AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'),

            Flatten(),
            Dense(512, activation='relu'),
            Dense(5, activation='linear')  # Output layer for 5 actions
        ])

        # Compile the model
        model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=["accuracy"])

        return model

    def cnn_1(self):
        model = Sequential([
            Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            BatchNormalization(),
            Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
            BatchNormalization(),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.2),
            Dense(5, activation='linear')
        ])
        model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=["accuracy"])
        return model

    def create_model_dt(self):# DeepTraffic
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH,3)))  # Input layer

        # Fully connected layers with 'tanh' activation
        model.add(tf.keras.layers.Dense(36, activation='tanh'))
        model.add(tf.keras.layers.Dense(24, activation='tanh'))
        model.add(tf.keras.layers.Dense(24, activation='tanh'))
        model.add(tf.keras.layers.Dense(24, activation='tanh'))

        # Output layer with linear activation for regression
        model.add(tf.keras.layers.Dense(3, activation='linear'))

        predictions = tf.keras.layers.Dense(3, activation='linear')(model.output)
        model = Model(inputs=model.input, outputs=predictions)

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            metrics=['accuracy']
        )

        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def get_qs(self, state): # can cause silent exit on first call
        try:
            if THREADED and NUM_TRAINING_THREADS > 1:
                with self.tf_lock:
                    with self.graph.as_default():
                        with self.session.as_default():
                            qs= self.model.predict(np.array(state).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 3) / 255)[0]
            else:
                qs = self.model.predict(np.array(state).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 3) / 255)[0]
        except Exception as e:
            print(f"Error in agent.get_qs(): {e}")
        # add weights to values action_

        qs *= Q_WEIGHTS
        return qs
        
    def train_in_loop(self):
        self.training_initialised = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_SIZE:
            # if len(self.replay_memory) % 100 == 0:
            #     print(f"\nReplay memory size: {len(self.replay_memory)}")
            return
        
        if PRINT_THREADS:
            print(f"\nTraining running on thread: {threading.current_thread().name}")

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255

        start_qs = time.time()
        if THREADED:
            # with self.tf_lock:
                with self.graph.as_default():
                    with self.session.as_default():
                        # predict current Q values for all states in minibatch
                        current_qs_list = self.model.predict(current_states, batch_size=PREDICTION_BATCH_SIZE)
        else:
            current_qs_list = self.model.predict(current_states, batch_size=PREDICTION_BATCH_SIZE)
        end_qs = time.time()

        new_current_states = np.array([transition[3] for transition in minibatch])/255

        start_future_qs = time.time()
        if THREADED:
            # with self.tf_lock:
                with self.graph.as_default():
                    with self.session.as_default():
                        # predict future Q values for all states in minibatch
                        future_qs_list = self.target_model.predict(new_current_states, batch_size=PREDICTION_BATCH_SIZE)
        else: 
            future_qs_list = self.target_model.predict(new_current_states, batch_size=PREDICTION_BATCH_SIZE)
        end_future_qs = time.time()

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)
            
        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        start_fit = time.time()
        if THREADED and NUM_TRAINING_THREADS > 1:
            with self.tf_lock:
                with self.graph.as_default():
                    with self.session.as_default():
                        self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)
        elif THREADED:
            with self.graph.as_default():
                with self.session.as_default():
                    self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)
        else:
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)
        end_fit = time.time()

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            if THREADED and NUM_TRAINING_THREADS > 1:
                with self.tf_lock:
                    with self.graph.as_default():
                        with self.session.as_default():
                            self.target_model.set_weights(self.model.get_weights())
            elif THREADED:
                with self.graph.as_default():
                    with self.session.as_default():
                        self.target_model.set_weights(self.model.get_weights())
            else:
                self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        if PRINT_TIMES or PRINT_TRAINING:
            print(f"\nTime taken for Q prediction: {end_qs - start_qs:.2f}s")
            print(f"Time taken for future Q prediction: {end_future_qs - start_future_qs:.2f}s")
            print(f"Time taken for model fit: {end_fit - start_fit:.2f}s")
            print(f"Time taken for training: {end_fit - start_qs:.2f}s")

    def __init__(self):
        # model_setup
        if MODEL_NAME == "64x3":
            self.model = self.create_model_64() # create model
            self.target_model = self.create_model_64() # create target model

        elif MODEL_NAME == "Xception":
            self.model = self.create_model_x() # create model
            self.target_model = self.create_model_x() # create target model

        elif MODEL_NAME == "CNN1":
            self.model = self.cnn_1()
            self.target_model = self.cnn_1()
        else:
            print(f"Model \"{MODEL_NAME}\" not found")
            exit()

        print(f"Model: {MODEL_NAME} \n")

        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}") # by @sentdex, minimises unnecessary updates and exports, imporving performance
        self.target_update_counter = 0

        if THREADED:
            self.graph = tf.get_default_graph()
            self.session = backend.get_session()
            with self.graph.as_default():
                with self.session.as_default():
                    self.session.run(tf.global_variables_initializer())

        if THREADED:
            self.training_threads = []
            self.num_training_threads = NUM_TRAINING_THREADS 
            self.tf_lock = threading.Lock()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialised = False

if __name__ == "__main__":
    try:
        ep_rewards = [-200]

        # set equal for repeatable results
        random.seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)

        # required for multiple agents)
        if THREADED:
            # add check for GPU
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION) 
            # backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
            config = tf.ConfigProto(
                device_count={'GPU': 0},
                intra_op_parallelism_threads=4,
                inter_op_parallelism_threads=4)
            backend.set_session(tf.Session(config=config))


        if not os.path.exists("models"):
            os.makedirs("models")

        agent = DQNAgent()
        
        env = CarEnvironment()

        qs = agent.get_qs(np.ones((env.image_height, env.image_width, 3))) # can silent exit

        if THREADED:
            for i in range(agent.num_training_threads):
                t = Thread(target=agent.train_in_loop, daemon=True, name=f"training_thread_{i+1}")
                t.start()
                agent.training_threads.append(t)

            while not agent.training_initialised:
                time.sleep(0.01)

            if PRINT_THREADS:
                print("\nActive threads after starting trainer_thread:")
                for t in threading.enumerate():
                    print(f"Thread name: {t.name}, Alive: {t.is_alive()}")

        print("\nStarting Training...\n")
        q_values_log = []
        for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):
            env.collision_list = []
            agent.tensorboard.step = episode
            episode_reward = 0
            step = 1
            current_state = env.reset()
            done = False

            episode_start_time = time.time()
            action_count = 0  

            total_actions = 0
            total_time = 0                

            # choose action
            while True:
                if np.random.random() < epsilon:
                    if PRINT_TIMES:
                        start_pred = time.time()
                        action = np.argmax(agent.get_qs(current_state))
                        end_pred = time.time()
                        print(f"\nPrediction time: {end_pred - start_pred:.2f}s")
                    else:
                        action = np.argmax(agent.get_qs(current_state))
                else:
                    action = np.random.randint(0, 5) # random action action_num
                    time.sleep(0.01) # wait for random action to be chosen

                action_count += 1

                new_state, reward, done, _ = env.step(action)
                episode_reward += reward

                if PRINT_TIMES:
                    update_time = time.time()
                    agent.update_replay_memory((current_state, action, reward, new_state, done))
                    end_update = time.time()
                    print(f"\nUpdate time: {end_update - update_time:.2f}s")
                else:
                    agent.update_replay_memory((current_state, action, reward, new_state, done))
                current_state = new_state  # Update state for next decision

                if len(agent.replay_memory) % MIN_REPLAY_SIZE == 0:
                    if PRINT_THREADS:
                        print("\nExploration phase finished")

                        print("\nActive threads:")
                        for t in threading.enumerate():
                            print(f"Thread name: {t.name}, Alive: {t.is_alive()}")

                if not THREADED:
                    agent.train()

                step += 1  

                if done:
                    break
            
            episode_time = time.time() - episode_start_time
            total_actions += action_count
            total_time += episode_time

            if AUTOPILOT:    
                for car in env.auto_list:
                    try:
                        car.set_autopilot(False) # must be false to destroy
                    except:
                        pass

            for actor in env.actor_list:
                actor.destroy()

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon, episode_time=episode_time, actions_per_second=(action_count/episode_time))

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    agent.model.save(f'models/W_{MODEL_NAME}_E{episode}_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

            if episode % 1000 == 0 or episode == 500 or episode == 250 or episode == 100:
                agent.model.save(f'models/{MODEL_NAME}_E{episode}_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
                # view model performance by running "tensorboard --logdir=logs" in the command line

            if episode == 500:
                export_constants_to_txt()
            
            if episode % 100 == 0 or episode == 1:
                current_qs = agent.get_qs(current_state)
                q_values_log.append((episode, list(zip(ACTION_NAMES, current_qs.tolist()))))

            if PRINT_NUM_ACTIONS:
                print(f"\nNumber of actions per second: {action_count/episode_time:.2f} actions/s")
        # Set termination flag for training thread and wait for it to finish
        if THREADED:
            agent.terminate = True
            for trainer_thread in agent.training_threads:
                trainer_thread.join()
    except KeyboardInterrupt:
        print("\nInterrupted by user.\n")
    finally:
        if THREADED:
            agent.terminate = True
            for trainer_thread in agent.training_threads:
                trainer_thread.join()
        if q_values_log:
            with open(f"models/q_values_log_{MODEL_NAME}_{int(time.time)}.txt", "w") as f:
                for episode, qs_action_pairs in q_values_log:
                    qs_str = ", ".join([f"{name}: {value:.4f}" for name, value in qs_action_pairs])
                    f.write(f"Episode {episode}: [{qs_str}]\n")