import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
import h5py
import argparse


import keras.backend.tensorflow_backend as backend
from keras.models import load_model
from CarlaClient_3 import CarEnvironment, MEMORY_FRACTION


MODEL_PATH = 'C:\Temp\old_models\models\Xception___-29.00max_-114.50avg_-200.00min__1743198607.model'

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the CARLA DeepTraffic agent.")
    parser.add_argument('--model_path', type=str, required=False, default=MODEL_PATH, help="Path to the trained model file.")
    args = parser.parse_args()

    MODEL_PATH = args.model_path

    # Memory fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Load the model
    model = load_model(MODEL_PATH)

    # Create environment
    env = CarEnvironment()

    # For agent speed measurements - keeps last 60 frametimes
    fps_counter = deque(maxlen=60)

    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    model.predict(np.ones((1, env.image_height, env.image_width, 3)))

    # Loop over episodes
    while True:

        print('Restarting episode')

        # Reset environment and get initial state
        current_state = env.reset()
        env.collision_hist = []

        done = False

        # Loop over steps
        while True:

            # For FPS counter
            step_start = time.time()

            # Show current frame
            cv2.imshow(f'Agent - preview', current_state)
            cv2.waitKey(1)

            # Predict an action based on current observation space
            qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
            action = np.argmax(qs)

            # Step environment (additional flag informs environment to not break an episode by time limit)
            new_state, reward, done, _ = env.step(action)

            # Set current step for next loop iteration
            current_state = new_state

            # If done - agent crashed, break an episode
            if done:
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}] {action}')

        # Destroy an actor at end of episode
        for actor in env.actor_list:
            actor.destroy()

# References:
# 
# Kinsley, Harrison (2019) Reinforcement Learning in Action - Self-driving cars with Carla and Python
# https://pythonprogramming.net/reinforcement-learning-self-driving-autonomous-cars-carla-python/ [Accessed 14 May 2025]