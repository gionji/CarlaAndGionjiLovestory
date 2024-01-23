import carla
import numpy as np

# Connect to CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Get the world and map
world = client.get_world()
carla_map = world.get_map()

# Assuming you have a camera sensor
camera_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=-15.0))

camera_sensor = world.spawn_actor(camera_blueprint, camera_transform)
camera_sensor.listen(lambda image: process_image(image))

# Function to process the camera image
def process_image(image):
    # Extract pixel coordinates (u, v) from the image
    u, v = 100, 150  # Replace with your pixel coordinates

    # Obtain intrinsic parameters
    intrinsic_params = np.array([[image.fov, 0.0, image.width / 2],
                                [0.0, image.fov, image.height / 2],
                                [0.0, 0.0, 1.0]])

    # Obtain extrinsic parameters
    extrinsic_params = camera_sensor.get_transform()

    # Map pixel coordinates to camera coordinates
    pixel_to_camera = np.linalg.inv(intrinsic_params) @ np.array([u, v, 1])

    # Convert the NumPy array to a carla.Vector3D
    pixel_to_camera_carla = carla.Vector3D(*pixel_to_camera)

    # Map camera coordinates to world coordinates
    camera_to_world = extrinsic_params.transform(pixel_to_camera_carla)

    # Print the 3D position in the world
    print(f"World Coordinates (x, y, z): {camera_to_world}")


# Keep the script running to receive camera data
while True:
    pass
