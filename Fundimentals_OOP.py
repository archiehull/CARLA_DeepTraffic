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

IMG_WIDTH, IMG_HEIGHT = 640, 480
SHOW_PREVIEW = False
EPISODE_LENGTH = 10 #seconds

class CarEnvironment:
    SHOW_CAMERA = SHOW_PREVIEW
    STEER_AMOUNT = 1.0

    image_width = IMG_WIDTH
    image_height = IMG_HEIGHT

    front_camera = None

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

print("yo")