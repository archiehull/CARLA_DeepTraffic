import carla

# Connect to the CARLA server
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

# Load the desired map (Town04)
world = client.load_world('Town04_OPT')

# Get the spawn points
spawn_points = world.get_map().get_spawn_points()
print(f"Number of spawn points: {len(spawn_points)}")

# Get the blueprint library
blueprint_library = world.get_blueprint_library()

# Find a vehicle blueprint
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

# Variable to track the last valid spawn index
last_spawn_index = None
current_spawn_index = 0

# Variable to track the currently spawned vehicle
current_vehicle = None

while True:
    try:
        # Ask the user for input
        user_input = input(f"Enter 'n' for next, 'd' for previous, an index (0 to {len(spawn_points) - 1}), or 'exit' to quit: ").strip().lower()

        # Check if the user wants to exit
        if user_input == "exit":
            print("Exiting...")
            break

        # Check if the user wants to go back to the previous spawn point
        if user_input == "d":
            if last_spawn_index is None or last_spawn_index <= 0:
                print("No previous spawn point to go back to.")
                continue
            current_spawn_index = last_spawn_index - 1
        elif user_input == "n" or user_input == "":
            if current_spawn_index >= len(spawn_points) - 1:
                print("No more spawn points to move to.")
                continue
            current_spawn_index += 1
        else:
            # Check if the input is a valid index number
            try:
                spawn_index = int(user_input)
                if spawn_index < 0 or spawn_index >= len(spawn_points):
                    print(f"Invalid index. Please enter a number between 0 and {len(spawn_points) - 1}.")
                    continue
                current_spawn_index = spawn_index
            except ValueError:
                print("Invalid input. Please enter 'n', 'd', an index number, or 'exit'.")
                continue

        # Destroy the current vehicle before spawning a new one
        if current_vehicle is not None:
            current_vehicle.destroy()
            current_vehicle = None

        # Get the selected spawn point
        spawn_point = spawn_points[current_spawn_index]
        print(f"Moving to spawn point {current_spawn_index + 1}/{len(spawn_points)}: {spawn_point.location}")

        # Spawn a vehicle at the spawn point
        current_vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if not current_vehicle:
            print(f"Failed to spawn vehicle at spawn point {current_spawn_index + 1}")
            continue

        # Update the last valid spawn index
        last_spawn_index = current_spawn_index

        # Move the spectator to a position behind and above the vehicle
        spectator = world.get_spectator()
        vehicle_transform = current_vehicle.get_transform()

        # Calculate the spectator's position behind the vehicle
        offset_distance = 10.0  # Distance behind the vehicle
        offset_height = 5.0     # Height above the vehicle
        spectator_location = carla.Location(
            x=vehicle_transform.location.x - offset_distance * vehicle_transform.get_forward_vector().x,
            y=vehicle_transform.location.y - offset_distance * vehicle_transform.get_forward_vector().y,
            z=vehicle_transform.location.z + offset_height
        )
        spectator_rotation = carla.Rotation(
            pitch=-15,  # Slightly tilted down
            yaw=vehicle_transform.rotation.yaw,  # Match the vehicle's yaw
            roll=0
        )
        spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))

    except ValueError:
        print("Invalid input. Please enter 'n', 'd', an index number, or 'exit'.")
    except KeyboardInterrupt:
        print("\nExiting...")
        break

# Destroy the last vehicle when exiting the program
if current_vehicle is not None:
    current_vehicle.destroy()





    '''
    Moving to spawn point 155/372: Location(x=-69.650841, y=37.353630, z=10.215619)
    Moving to spawn point 172/372: Location(x=-365.865570, y=33.573753, z=0.281942)
    312

    '''