# FILE MUST RUN IN PYTHON 3.6.8, Tensorflow 1.14.0, Keras 2.2.4, CARLA 0.9.13

from platform import python_version

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
from keras.callbacks import TensorBoard
import os
from threading import Thread

import keras.backend.tensorflow_backend as backend   # Must use older version of Tensorflow

from tqdm import tqdm

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

AGGREGATE_STATS_EVERY = 10

# Own Tensorboard class
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
        # self.world = self.client.get_world
        self.world = self.client.load_world('Town04_OPT', carla.MapLayer.Buildings|carla.MapLayer.ParkedVehicles)

        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        self.world.unload_map_layer(carla.MapLayer.Buildings)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)

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
            reward = -2

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
        self.graph = tf.get_default_graph()

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

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        with self.graph.as_default():
            # predict current Q values for all states in minibatch
            current_qs_list = self.model.predict(current_states, batch_size=PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        with self.graph.as_default():
            # predict future Q values for all states in minibatch
            future_qs_list = self.target_model.predict(new_current_states, batch_size=PREDICTION_BATCH_SIZE)

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

        with self.graph.as_default():
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 3) / 255)[0]
        
    def train_in_loop(self):
        # first train is always slow, so simulate dummy train
        X = np.random.uniform(size=(1, IMG_HEIGHT, IMG_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X, y, batch_size=1, verbose=0)
        
        self.training_initialised = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

if __name__ == "__main__":
    FPS = 60 # MODIFY THIS TO CHANGE FPS
    ep_rewards = [-200]

    # set equal for repeatable results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    # required for multiple agents
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION) 
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    if not os.path.exists("models"):
        os.makedirs("models")

    agent = DQNAgent()
    env = CarEnvironment()

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()

    while not agent.training_initialised:
        time.sleep(0.01)

    agent.get_qs(np.ones((env.image_height, env.image_width, 3)))

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):
        env.collision_list = []
        agent.tensorboard.step = episode
        episode_reward = 0
        step = 1
        current_state = env.reset()
        done = False
        episode_start = time.time()

        while True:
            if np.random.random() < epsilon:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0, 3)
                time.sleep(1/FPS)

            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            agent.update_replay_memory((current_state, action, reward, new_state, done))

            step += 1

            if done:
                break

        for action in env.actor_list:
            action.destroy()

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

            # Set termination flag for training thread and wait for it to finish
        agent.terminate = True
        trainer_thread.join()

        if episode % 25 == 0:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # view models by running "tensorboard --logdir=logs" in the command line
        
