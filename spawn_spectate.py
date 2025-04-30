import carla
import time

def main():
    """
    Main function to set the spectator's position and spawn a vehicle at a fixed location.
    """
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

        # Spawn a vehicle at the fixed location
        vehicle = spawn_vehicle(world, fixed_location)

        spawn_next_vehicle_ahead(world, vehicle, 10.0)

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
    """
    Sets the spectator's position behind and above a fixed location with a tilted-down view.

    Args:
        world (carla.World): The CARLA world instance.
        fixed_location (carla.Location): The fixed location to base the spectator's position on.
    """
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


def spawn_vehicle(world, fixed_location):
    """
    Spawns a vehicle at the fixed location.

    Args:
        world (carla.World): The CARLA world instance.
        fixed_location (carla.Location): The location to spawn the vehicle.

    Returns:
        carla.Actor: The spawned vehicle actor, or None if spawning failed.
    """
    # Get the blueprint library
    blueprint_library = world.get_blueprint_library()

    # Find a vehicle blueprint (e.g., Nissan Micra)
    vehicle_bp = blueprint_library.find('static.prop.vendingmachine')

    # Define the spawn transform
    spawn_transform = carla.Transform(
        location=fixed_location,
        rotation=carla.Rotation(pitch=0, yaw=0, roll=0)  # Default rotation
    )

    # Spawn the vehicle
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
    if vehicle:
        print("Vehicle spawned successfully at the fixed location.")
        return vehicle
    else:
        print("Failed to spawn vehicle at the fixed location.")
        return None

def spawn_next_vehicle_ahead(world, current_vehicle, distance=50.0):
    """
    Spawns a vehicle further ahead in the same lane as the current vehicle.

    Args:
        world (carla.World): The CARLA world instance.
        current_vehicle (carla.Actor): The current vehicle.
        distance (float): The distance ahead of the current vehicle to spawn the new vehicle.

    Returns:
        carla.Actor: The spawned vehicle actor, or None if spawning failed.
    """
    # Get the current vehicle's transform
    current_transform = current_vehicle.get_transform()

    # Get the waypoint of the current vehicle
    current_waypoint = world.get_map().get_waypoint(current_transform.location)

    # Get the waypoint further ahead in the same lane
    next_waypoints = current_waypoint.next(distance)  # Get waypoints at the specified distance
    if not next_waypoints:
        print("No waypoint found ahead.")
        return None

    next_waypoint = next_waypoints[0]  # Take the first waypoint

    # Ensure the new waypoint is in the same lane
    if next_waypoint.lane_id != current_waypoint.lane_id:
        print(f"Waypoint ahead is in a different lane (current lane: {current_waypoint.lane_id}, next lane: {next_waypoint.lane_id}).")
        return None

    # Define the spawn transform for the new vehicle
    spawn_transform = carla.Transform(
        location=next_waypoint.transform.location,
        rotation=next_waypoint.transform.rotation
    )

    # Raise the spawn location slightly above the ground
    spawn_transform.location.z += 0.5

    # Get the blueprint library
    blueprint_library = world.get_blueprint_library()

    # Find a vehicle blueprint (e.g., Nissan Micra)
    vehicle_bp = blueprint_library.find('vehicle.nissan.micra')

    # Try to spawn the vehicle
    new_vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
    if new_vehicle:
        print(f"Vehicle spawned successfully {distance} meters ahead in the same lane.")
        return new_vehicle
    else:
        print("Failed to spawn vehicle ahead.")
        return None


if __name__ == "__main__":
    main()

