import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
import argparse

import keras.backend.tensorflow_backend as backend
from keras.models import load_model
from CarlaClient_5 import CarEnvironment, MEMORY_FRACTION

# cant leverage gpu on tensorflow without downgrading cuda version (not possible on my rtx3070)

MODEL_PATH = 'C:\\Temp\\models\\Xception___-27.00max_-113.50avg_-200.00min__1745582883.model'

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
    print('Loading model...')
    
    try:
        model.predict(np.ones((1, env.image_height, env.image_width, 3)))
    except Exception as e:
        print(f"Error during model initialization: {e}")
        exit()

    
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
            # cv2.imshow(f'Agent - preview', current_state)
            # cv2.waitKey(1)

            # Predict an action based on current observation space
            qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
            print(f"Prediction time: {time.time() - step_start:.2f}s")

            action = np.argmax(qs)

            # Step environment (additional flag informs environment to not break an episode by time limit)
            # step_start = time.time()
            new_state, reward, done, _ = env.step(action)
            # print(f"Environment step time: {time.time() - step_start:.2f}s")

            # Set current step for next loop iteration
            current_state = new_state

            # If done - agent crashed, break an episode
            if done:
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | 'f'Action: [U:{qs[0]:>5.2f}, S:{qs[1]:>5.2f}, D:{qs[2]:>5.2f}, L:{qs[3]:>5.2f}, R:{qs[4]:>5.2f}] {action}')

        # Destroy an actor at end of episode
        for actor in env.actor_list:
            try:    
                actor.set_autopilot(False)
            except:
                pass
        
            actor.destroy()