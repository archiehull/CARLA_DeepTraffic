# FILE MUST RUN IN PYTHON 3.8
from platform import python_version

try:
    import carla
except ImportError:
    print("\nCarla module not found")
    print("Make sure to run the command 'pip install carla'")

    if python_version() != "3.8.0":
        print("\nPython 3.8.0 is required to run this program")
        print("Current Python Version: " + python_version())

        print("\nIf using VS code, use the command 'Ctrl + Shift + P'")
        print("Type 'Python: Select Interpreter'") 
        print("Select Python 3.8.0")

        print("\nIf you are using the command line, type 'python3.8' or 'py -3.8' instead of 'python'")

    input("\nPress Enter to exit")
    exit()

import math
import random
import time
import numpy as np
import cv2
from collections import deque
import tensorflow as tf
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model

# CONSTANTS

# Image parameters
SHOW_PREVIEW = False
IMG_WIDTH, IMG_HEIGHT = 640, 480

# RL parameters
EPISODE_LENGTH = 10 #seconds

REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_SIZE = 1000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception" ## TODO: INVESTIGATE MODELS

MEMORY_FRACTION = 0.8 # allocates 80% of processing power to avoid overconsuption (test)
MIN_REWARD = -200

EPISODES = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95 ## tend towards 1.0 depeninding on # of steps
MIN_EPSILON = 0.001

AGGRIGATE_STATS_EVERY = 10

class CarEnvironment:
    SHOW_CAMERA = SHOW_PREVIEW
    STEER_AMOUNT = 1.0

    image_width = IMG_WIDTH
    image_height = IMG_HEIGHT
    actor_list = []

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


    # RL functions
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()
        self.vehicle_bp = self.bp_lib.find('vehicle.nissan.micra')

    def reset(self):
        # reset lists
        self.collision_list = []
        self.actor_list = []

        # spawn agent
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.transform)
        self.actor_list.append(self.vehicle)

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
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
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
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def step(self, action):
        if action == 0:         # turn left (-1)
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= -1 * self.STEER_AMOUNT))
        elif action == 1:       # straight (0)
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:       # turn right (1)
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 1 * self.STEER_AMOUNT))

        velocity = self.vehicle.get_velocity()
        speed_kmh = int(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))


        ## EXPERMIENT WITH DIFFERENT REWARD VALS
        if len(self.collision_list) != 0:
            # end sim and punish for crashing
            done = True
            reward = -200

        elif speed_kmh < 50:
            # slight punishment for slow speeds
            done = False
            reward = -1
        
        else:
            # slight reward for good speed
            done = False
            reward = 1

        # end sim after EPISODE_LENGTH elapsed
        if self.episode_start + EPISODE_LENGTH < time.time():
            done = True
        

        # self.collision_list = []  #remove?

        return self.front_camera, reward, done, None
    

    class DQNAgent:
        def __init__(self):
            self.model = self.create_model()
            self.target_model = self.create_model()
            self.target_model.set_weights(self.model.get_weights())

            self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

            self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}") # by @sentdex, minimises unnecessary updates and exports, imporving performance
            self.target_update_counter = 0
            self.graph = tf.get_default.graph()

            self.terminate = False
            self.last_logged_episode = 0
            self.training_initialised = False

        def create_model(self):
            # model can be built here manually
            '''
            model = tf.Sequential()
            model.add()
            '''
            base_model = Xception(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH,3) )

            x = base_model.output
            x = GlobalAveragePooling2D()(x)

            predictions = Dense(3, activation="linear")(x)
            model = Model(inputs=base_model.input, outputs=predictions)
            model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
            return model