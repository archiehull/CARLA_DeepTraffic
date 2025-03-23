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
import subprocess
import numpy as np
import cv2
import matplotlib.pyplot as plt

def run_carla():
    exe_filepath = "C:\Temp\CARLA_0.9.15\WindowsNoEditor\CarlaUE4.exe"

    # Run the executable in a fully detached process
    subprocess.Popen(
        exe_filepath,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True
    )

def manual_control():
    
    # Manual control UI
    manual_fp = "C:\Temp\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\manual_control.py"

    # Run the executable in a fully detached process
    subprocess.Popen(
        manual_fp,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True
    )

# run_carla()
# manual_control()

actor_list = []

## CONNECT TO CARLA SIMULATOR

# connect to client
client = carla.Client('localhost', 2000) #ip and port - should be localhost and 2000 by default
# provide access to sim assests 

world = client.get_world()


## SPAWN OBJECTS
# provides access to blueprint for creating objects
bp_lib = world.get_blueprint_library()
# 3d location to spawn objects
spawn_points = world.get_map().get_spawn_points()

# spawn a vehicle
vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

actor_list.append(vehicle)

# Retrieve the spectator object
spectator = world.get_spectator()

transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)), vehicle.get_transform().rotation)
# Get the location and rotation of the spectator through its transform
spectator.set_transform(transform)

## SPAWN NPC
for i in range(10):
    vehicle_bp = random.choice(bp_lib.filter('vehicle'))
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    actor_list.append(npc)


## AUTOPILOT
for v in world.get_actors().filter('*vehicle*'):
    v.set_autopilot(True)

actor_list[0].set_autopilot(True)

## DATA COLLECTION
IMG_WIDTH, IMG_HEIGHT = 640, 480

camera_bp = bp_lib.find('sensor.camera.rgb')

camera_bp.set_attribute('image_size_x', f'{IMG_WIDTH}')
camera_bp.set_attribute('image_size_y', f'{IMG_HEIGHT}')
camera_bp.set_attribute('fov', '110')

camera_init_trans = carla.Transform(carla.Location(z=2))

camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

actor_list.append(camera)


def process_img(image):
    i = np.array(image.raw_data) # flatten image to array
    i2 = i.reshape((IMG_HEIGHT, IMG_WIDTH, 4)) # 4 channels: RGBA
    i3 = i2[:, :, :3] # remove alpha channel (height, width, channels(3))
    i4 = i3/255.0 # normalise 
    cv2.imshow("", i4) # show image

    cv2.waitKey(10)
    return i4

camera.listen(lambda image: process_img(image))
time.sleep(20)
camera.stop()

for actor in actor_list:
    actor.destroy() 