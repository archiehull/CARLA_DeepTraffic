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

CARS_ON_HIGHWAY = 10 # number of cars on highway
WAYPOINTS_PER_LANE = 8 # number of waypoints on highway
WAYPOINT_SEPARATION = 10.0 # distance between waypoints


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

def generate_lane_waypoints(world, num_waypoints=WAYPOINTS_PER_LANE, separation=WAYPOINT_SEPARATION):
    spawn_points = world.get_map().get_spawn_points()
    current_transform = spawn_points[312]
    current_waypoint = world.get_map().get_waypoint(current_transform.location)

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

    # Skip adding the current waypoint to the list
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

def generate_lane_waypoints_infront(world, num_waypoints=WAYPOINTS_PER_LANE, separation=WAYPOINT_SEPARATION):
    spawn_points = world.get_map().get_spawn_points()
    current_transform = spawn_points[312]
    current_waypoint = world.get_map().get_waypoint(current_transform.location)

    # Initialize dictionaries to store waypoints for each lane
    waypoints = {
        "lane2": [],  # Current lane
        "lane1": [],  # Left lane
        "lane3": [],  # Right lane
        "lane4": []   # Second right lane
    }

    # Generate waypoints for lane2 (current lane)
    waypoint = current_waypoint
    for _ in range(num_waypoints):  # Generate waypoints in front
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
        for _ in range(num_waypoints):  # Generate waypoints in front
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
        for _ in range(num_waypoints):  # Generate waypoints in front
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
            for _ in range(num_waypoints):  # Generate waypoints in front
                next_waypoints = waypoint.next(separation)
                if next_waypoints:
                    waypoint = next_waypoints[0]
                    waypoints["lane4"].append(waypoint)
                else:
                    break

    return waypoints

def populate_autopilot_cars(world, all_waypoints, actor_list, auto_list, num_cars=CARS_ON_HIGHWAY):
 
        selected_waypoints = [random.choice(row) for row in all_waypoints if row]

        # print(len(all_waypoints), "waypoints available for spawning cars.")

        # Spawn cars at random waypoints
        for i in range(min(num_cars, len(selected_waypoints))):
            # Choose a random waypoint
            spawn_waypoint = selected_waypoints[i]

            # Define the spawn transform for the vehicle
            spawn_transform = spawn_waypoint.transform
            spawn_transform.location.z += 1.0

            blueprint_library = world.get_blueprint_library()
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.toyota.prius'))
            # vehicle_bp = blueprint_library.filter('static.prop.gnome')

            # Try to spawn the vehicle
            start_time = time.time()
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
            
            if vehicle:
                # Enable autopilot
                # vehicle.set_autopilot(True)

                # Add the vehicle to the actor list for cleanup later
                actor_list.append(vehicle)
                auto_list.append(vehicle)
                end_time = time.time()
                # print(f"Autopilot car {i} spawned at {spawn_transform.location}. Time taken: {end_time - start_time:.2f} seconds.")

            else:
                print(f"Failed to spawn autopilot car at {spawn_transform.location}.")
            
def spawn_waypoint_indicators(world, waypoints, actor_list):
    blueprint_library = world.get_blueprint_library()
    sphere_bp = blueprint_library.find('static.prop.gnome')  # Use a visible prop as an indicator

    for lane, lane_waypoints in waypoints.items():
        for waypoint in lane_waypoints:
            spawn_transform = waypoint.transform
            spawn_transform.location.z += 1.0  # Raise the indicator slightly above the ground
            indicator = world.try_spawn_actor(sphere_bp, spawn_transform)
            if indicator:
                actor_list.append(indicator)
          

def main():
    vehicle = None  # Initialize the vehicle reference
    try:
        actor_list = []  # List to keep track of spawned actors
        auto_list = []  # List to keep track of autopilot cars

        # Connect to the CARLA server
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        # Load the world
        world = client.load_world('Town04_OPT') 
        world.set_weather(carla.WeatherParameters.ClearNoon)


        # Define the fixed location
        fixed_location = carla.Location(x=9.497402, y=214.450912, z=0.281942)  # Spawn 312

        # Set the spectator's position
        set_spectator(world, fixed_location)

        # Get the spawn points
        spawn_points = world.get_map().get_spawn_points()

        # Use spawn point 312 for the main vehicle
        transform = spawn_points[312]  # Start of highway
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.nissan.micra')
        world.try_spawn_actor(vehicle_bp, transform)

        waypoints = generate_lane_waypoints_infront(world)
        print(f"Spawned {len(waypoints)} waypoint indicators ahead of the vehicle.")
        spawn_waypoint_indicators(world, waypoints, actor_list)

        while True:
            num = input("Enter number of waypoints to spawn: ")
            sep = input("Enter separation between waypoints: ")
            for actor in actor_list:
                actor.destroy()

            actor_list = []  # List to keep track of spawned actors

            if num == "y":
                break

            sep = float(sep)
            num = int(num) 

            waypoints = generate_lane_waypoints_infront(world, num, sep)
            spawn_waypoint_indicators(world, waypoints, actor_list)


        combined_waypoints = waypoints["lane1"] + waypoints["lane2"] + waypoints["lane3"] + waypoints["lane4"]
        all_waypoints = []
        num_waypoints_per_lane = len(waypoints["lane1"])  # Assuming all lanes have the same number of waypoints

        for i in range(num_waypoints_per_lane):
            batch = [
                waypoints["lane1"][i],
                waypoints["lane2"][i],
                waypoints["lane3"][i],
                waypoints["lane4"][i]
            ]
            all_waypoints.append(batch)
        

        while True:
            populate_autopilot_cars(world, all_waypoints, actor_list, auto_list)

            input("")
            for actor in actor_list:
                actor.destroy()
            
            actor_list = []  # List to keep track of spawned actors
        
        # Spawn another vehicle ahead at spawn point 296
        # spawn_vehicle_ahead(world, spawn_points[296])


        # Create the player
        player = Player(world, vehicle_bp)

        populate_autopilot_cars(world, waypoints, actor_list, auto_list, num_cars=10)

        # Perform lane change maneuvers
        while False:
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
            for actor in actor_list:
                actor.destroy()
            print("Vehicle destroyed.")


def set_spectator(world, fixed_location):

    # Calculate the spectator's position behind and above the fixed location
    offset_distance = 20.0  # Distance behind the fixed location
    offset_height = 20.0    # Height above the fixed location
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

