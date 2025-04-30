import sys
import glob
import os

# Add CARLA PythonAPI and agents module to the Python path
sys.path.append(r"C:\Temp\CARLA_0.9.13\WindowsNoEditor\PythonAPI\carla")
sys.path.append(r"C:\Temp\CARLA_0.9.13\WindowsNoEditor\PythonAPI")

import carla
import time
from agents.navigation.controller import VehiclePIDController
import math
import random
import numpy as np


VEHICLE_VEL = 50  # Target velocity for the vehicle


class Player:
    def __init__(self, world, bp, vel_ref=VEHICLE_VEL, max_throt=0.75, max_brake=0.3, max_steer=0.8):
        self.world = world
        self.max_throt = max_throt
        self.max_brake = max_brake
        self.max_steer = max_steer
        self.vehicle = None
        self.bp = bp
        self.vel_ref = vel_ref
        self.waypointsList = []

        # Spawn the vehicle
        while self.vehicle is None:
            spawn_points = world.get_map().get_spawn_points()
            spawn_point = spawn_points[312]  # Use spawn point 312
            self.vehicle = world.try_spawn_actor(bp, spawn_point)

        self.spectator = world.get_spectator()

          # Apply initial control to start moving the vehicle
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))  # Adjust throttle as needed



        # PID controller setup
        dt = 1.0 / 20.0
        args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': dt}
        args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': dt}
        offset = 0

        self.controller = VehiclePIDController(self.vehicle,
                                                args_lateral=args_lateral_dict,
                                                args_longitudinal=args_longitudinal_dict,
                                                offset=offset,
                                                max_throttle=max_throt,
                                                max_brake=max_brake,
                                                max_steering=max_steer)

        self.current_pos = self.vehicle.get_transform().location
        self.past_pos = self.vehicle.get_transform().location

    def dist2Waypoint(self, waypoint):
        vehicle_transform = self.vehicle.get_transform()
        vehicle_x = vehicle_transform.location.x
        vehicle_y = vehicle_transform.location.y
        waypoint_x = waypoint.transform.location.x
        waypoint_y = waypoint.transform.location.y
        return math.sqrt((vehicle_x - waypoint_x) ** 2 + (vehicle_y - waypoint_y) ** 2)

    def go2Waypoint(self, waypoint, draw_waypoint=True, threshold=0.3):
        if draw_waypoint:
            self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                         color=carla.Color(r=255, g=0, b=0), life_time=10.0,
                                         persistent_lines=True)

        current_pos_np = np.array([self.current_pos.x, self.current_pos.y])
        past_pos_np = np.array([self.past_pos.x, self.past_pos.y])
        waypoint_np = np.array([waypoint.transform.location.x, waypoint.transform.location.y])
        vec2wp = waypoint_np - current_pos_np
        motion_vec = current_pos_np - past_pos_np
        dot = np.dot(vec2wp, motion_vec)
        if dot >= 0:
            while self.dist2Waypoint(waypoint) > threshold:
                control_signal = self.controller.run_step(self.vel_ref, waypoint)
                self.vehicle.apply_control(control_signal)
                self.update_spectator()

    def getLeftLaneWaypoints(self, offset=2 * VEHICLE_VEL, separation=0.3):
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        left_lane = current_waypoint.get_left_lane()
        self.waypointsList = left_lane.next(offset)[0].next_until_lane_end(separation)

    def getRightLaneWaypoints(self, offset=2 * VEHICLE_VEL, separation=0.3):
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        right_lane = current_waypoint.get_right_lane()
        self.waypointsList = right_lane.next(offset)[0].next_until_lane_end(separation)

    def do_left_lane_change(self):
        self.getLeftLaneWaypoints()
        for i in range(len(self.waypointsList) - 1):
            self.current_pos = self.vehicle.get_location()
            self.go2Waypoint(self.waypointsList[i])
            self.past_pos = self.current_pos
            self.update_spectator()

    def do_right_lane_change(self):
        self.getRightLaneWaypoints()
        for i in range(len(self.waypointsList) - 1):
            self.current_pos = self.vehicle.get_location()
            self.go2Waypoint(self.waypointsList[i])
            self.past_pos = self.current_pos
            self.update_spectator()

    def update_spectator(self):
        new_yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
        spectator_transform = self.vehicle.get_transform()
        spectator_transform.location += carla.Location(x=-10 * math.cos(new_yaw), y=-10 * math.sin(new_yaw), z=5.0)

        self.spectator.set_transform(spectator_transform)
        self.world.tick()

    def draw_waypoints(self):
        for waypoint in self.waypointsList:
            self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                         color=carla.Color(r=255, g=0, b=0), life_time=10.0,
                                         persistent_lines=True)



def main():
    vehicle = None  # Initialize the vehicle reference
    try:
        # Connect to the CARLA server
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        # Load the world
        world = client.load_world('Town04_OPT') 

        # Define the fixed location
        fixed_location = carla.Location(x=9.497402, y=214.450912, z=0.281942)  # Spawn 312

        # Set the spectator's position
        set_spectator(world, fixed_location)

        # Get the spawn points
        spawn_points = world.get_map().get_spawn_points()

        # Use spawn point 312 for the main vehicle
        transform = spawn_points[312]  # Start of highway
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.toyota.prius')

         # Spawn another vehicle ahead at spawn point 296
        spawn_vehicle_ahead(world, spawn_points[296])


        # Create the player
        player = Player(world, vehicle_bp)

        # Perform lane change maneuvers
        while True:
            player.update_spectator()
            maneuver = input("Enter maneuver (l: left lane change, r: right lane change, d: draw waypoints): ")
            if maneuver == "l":
                player.do_left_lane_change()
            elif maneuver == "r":
                player.do_right_lane_change()
            elif maneuver == "d":
                player.getLeftLaneWaypoints()
                player.draw_waypoints()

       


        # Keep the script running to observe the spectator and vehicle
        print("Spectator and vehicle setup complete. Press Ctrl+C to exit.")

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nScript terminated by user.")
    finally:
        # Destroy the vehicle if it exists
        if vehicle is not None:
            print("Destroying vehicle...")
            vehicle.destroy()
            print("Vehicle destroyed.")


def set_spectator(world, fixed_location):

    # Calculate the spectator's position behind and above the fixed location
    offset_distance = 50.0  # Distance behind the fixed location
    offset_height = 50.0    # Height above the fixed location
    forward_vector = carla.Vector3D(0, -1, 0)  # Assume a forward vector pointing along the x-axis

    spectator_location = carla.Location(
        x=fixed_location.x - offset_distance * forward_vector.x,
        y=fixed_location.y - offset_distance * forward_vector.y,
        z=fixed_location.z + offset_height
    )
    spectator_rotation = carla.Rotation(
        pitch=-30,
        yaw=-90,   
        roll=0
    )

    # Get the spectator and set its transform
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))
    print("Spectator position updated based on fixed location.")

def spawn_vehicle_ahead(world, spawn_point):
       # Define the spawn transform for the new vehicle
    spawn_transform = carla.Transform(
        location=spawn_point.location,
        rotation=spawn_point.rotation
    )

    # Get the blueprint library
    blueprint_library = world.get_blueprint_library()

    # Find a vehicle blueprint
    vehicle_bp = blueprint_library.find('vehicle.toyota.prius')  # Replace with desired object for demo

    # Try to spawn the vehicle
    new_vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
    if new_vehicle:
        print(f"Vehicle spawned successfully at location: {spawn_point.location}")
        return new_vehicle
    else:
        print(f"Failed to spawn vehicle at location: {spawn_point.location}")
        return None


if __name__ == "__main__":
    main()

    #https://stackoverflow.com/questions/71128092/how-to-implement-a-lane-change-manoeuver-on-carla