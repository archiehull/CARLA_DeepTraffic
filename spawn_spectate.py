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

WAYPOINT_INDICATORS = False
POPULATE_HIGHWAY = True
CHANGE_LANE = False
RANDOM_OLD = False

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

def generate_all_waypoints_old(world, num_waypoints=WAYPOINTS_PER_LANE, separation=WAYPOINT_SEPARATION):
    spawn_points = world.get_map().get_spawn_points()
    current_transform = spawn_points[312]
    current_waypoint = world.get_map().get_waypoint(current_transform.location)
    all_waypoints = []

    # List of starting waypoints for each lane (if they exist)
    lane_starts = [current_waypoint]
    left_lane = current_waypoint.get_left_lane()
    right_lane = current_waypoint.get_right_lane()
    if left_lane:
        lane_starts.append(left_lane)
    if right_lane:
        lane_starts.append(right_lane)
        second_right_lane = right_lane.get_right_lane()
        if second_right_lane:
            lane_starts.append(second_right_lane)

    # For each lane, collect waypoints in front
    for lane_start in lane_starts:
        waypoint = lane_start
        for _ in range(num_waypoints):
            next_waypoints = waypoint.next(separation)
            if next_waypoints:
                waypoint = next_waypoints[0]
                all_waypoints.append(waypoint)
            else:
                break

    return all_waypoints

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
        if RANDOM_OLD:
            selected_waypoints = random.sample(all_waypoints, min(num_cars, len(all_waypoints)))
        else:
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
          
def change_lane(vehicle, world, direction):
    current_transform = vehicle.get_transform()
    current_location = current_transform.location
    current_waypoint = world.get_map().get_waypoint(current_location)

    if direction == 'l':  # Change to the left lane
        next_waypoint = current_waypoint.get_left_lane()
        if next_waypoint:
            vehicle.set_transform(next_waypoint.transform)
            print(f"Vehicle moved to the left lane: {next_waypoint.lane_id}")
        else:
            print("No left lane available.")

    elif direction == 'r':  # Change to the right lane
        next_waypoint = current_waypoint.get_right_lane()
        if next_waypoint:
            vehicle.set_transform(next_waypoint.transform)
            print(f"Vehicle moved to the right lane: {next_waypoint.lane_id}")
        else:
            print("No right lane available.")

    else:
        print("Invalid direction. Use 'l' for left or 'r' for right.")



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
        actor = world.try_spawn_actor(vehicle_bp, transform)

        waypoints = generate_lane_waypoints_infront(world)
        print(f"Spawned {len(waypoints)} waypoint indicators ahead of the vehicle.")
        
        if WAYPOINT_INDICATORS:
            spawn_waypoint_indicators(world, waypoints, actor_list)

        while WAYPOINT_INDICATORS:
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

        if RANDOM_OLD: 
                all_waypoints = generate_all_waypoints_old(world, num_waypoints=12, separation=5.0)

        while POPULATE_HIGHWAY:
            populate_autopilot_cars(world, all_waypoints, actor_list, auto_list)

            inp = input("")


            for actor in actor_list:
                actor.destroy()

            actor_list = []  # List to keep track of spawned actors

            if inp == "y":
                break
        
        while CHANGE_LANE:
            command = input("Enter 'l' to change to the left lane, 'r' to change to the right lane, or 'q' to quit: ")
            if command == 'q':
                break
            elif command in ['l', 'r']:
                change_lane(actor, world, command)
            else:
                print("Invalid command.")

        
        # Spawn another vehicle ahead at spawn point 296
        # spawn_vehicle_ahead(world, spawn_points[296])

        # populate_autopilot_cars(world, waypoints, actor_list, auto_list, num_cars=10)

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

