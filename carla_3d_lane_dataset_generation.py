#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import math
import numpy as np
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, data, blend=False):
    if isinstance(data, carla.libcarla.Image):
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
    elif isinstance(data, np.ndarray):
        array = data
    else:
        raise ValueError("Unsupported data type for draw_image")

    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def depth_to_3d(depth_img, fx, fy, cx, cy):
    rows, cols = depth_img.shape
    y, x = np.meshgrid(range(rows), range(cols), indexing='ij')

    X = (x - cx) * depth_img / fx
    Y = (y - cy) * depth_img / fy
    Z = depth_img

    return X, Y, Z


def calculate_camera_parameters(width, height, fov_x, fov_y):
    fx = width / (2 * math.tan(math.radians(fov_x / 2)))
    fy = height / (2 * math.tan(math.radians(fov_y / 2)))
    
    cx = width / 2
    cy = height / 2
    
    return fx, fy, cx, cy


def encode_xyz_to_rgb(X, Y, Z):
    # Normalize X, Y, Z to the range [0, 1]
    normalized_X = (X - X.min()) / (X.max() - X.min())
    normalized_Y = (Y - Y.min()) / (Y.max() - Y.min())
    normalized_Z = (Z - Z.min()) / (Z.max() - Z.min())
    # Combine normalized X, Y, Z into a single RGB array
    encoded_image = np.stack([normalized_X, normalized_Y, normalized_Z], axis=-1)
    # Scale to 0-255 and convert to uint8
    encoded_image = (encoded_image * 255).astype(np.uint8)

    return encoded_image



def extract_lane_mask(segmentation_image, segmentation_class_id=6):
    if isinstance(segmentation_image, carla.libcarla.Image):
        # Convert the Image object to a NumPy array
        segmentation_array = np.frombuffer(segmentation_image.raw_data, dtype=np.uint8)
        segmentation_array = np.reshape(segmentation_array, (segmentation_image.height, segmentation_image.width, 4))
        segmentation_array = segmentation_array[:, :, :3]
    else:
        segmentation_array = segmentation_image

    # Assuming CityScapesPalette format, modify the values based on your segmentation map
    lane_segmentation_values = [segmentation_class_id]  # Replace with the actual values for road lanes

    # Check if the segmentation image is 1D
    if segmentation_array.ndim == 1:
        segmentation_array = segmentation_array.reshape((1, segmentation_array.shape[0]))

    # Create a binary mask for road lanes
    lane_mask = np.zeros_like(segmentation_array, dtype=np.uint8)
    for value in lane_segmentation_values:
        lane_mask += (segmentation_array == value)

    # Convert the mask to binary (1 for road lanes, 0 for other classes)
    lane_mask = (lane_mask > 0).astype(np.uint8) * 255

    #lane_mask = np.sum(lane_mask, axis=-1, keepdims=True)

    return lane_mask



def carla_image_to_numpy(carla_image):
    # Convert the Carla image to a NumPy array
    img_data = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
    img_data = np.reshape(img_data, (carla_image.height, carla_image.width, 4))

    # Remove the alpha channel to get a 3-channel RGB image
    rgb_image = img_data[:, :, :3]

    return rgb_image



import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go



def plot_colored_3d_points(XYZ, rgb_image, num_points_to_plot=None):
    """
    Plot a 3D scatter plot with colored points.

    Parameters:
        - XYZ: 3D array containing X, Y, and Z values.
        - rgb_image: RGB image for colors.
        - num_points_to_plot: Number of points to randomly select and plot.
                             If set to None, plot all points.
    """
    # Ensure all arrays have the same shape
    assert XYZ.shape[2] == 3, "Input array must have shape (N, M, 3)"

    # Flatten XYZ to 1D arrays
    x_points, y_points, z_points = XYZ[:, :, 0].flatten(), XYZ[:, :, 1].flatten(), XYZ[:, :, 2].flatten()

    # Convert RGBA to RGB
    rgb_image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2RGB)

    # Flatten the RGB image to 1D array
    rgb_flat = rgb_image_rgb.reshape(-1, 3)

    # Determine the number of points to plot
    if num_points_to_plot is None:
        num_points_to_plot = XYZ.size // 3

    # Generate random indices if not plotting all points
    random_indices = (
        np.random.choice(XYZ.size // 3, size=num_points_to_plot, replace=False)
        if num_points_to_plot is not None
        else np.arange(XYZ.size // 3)
    )

    # Extract points and corresponding colors
    selected_x = x_points[random_indices]
    selected_y = y_points[random_indices]
    selected_z = z_points[random_indices]
    selected_colors = rgb_flat[random_indices]

    # Convert RGB values to 'rgb(r, g, b)' format
    color_strings = [f'rgb({r},{g},{b})' for r, g, b in selected_colors]

    # Create a 3D scatter plot with colors
    fig = go.Figure(data=[go.Scatter3d(
        x=selected_x,
        y=selected_y,
        z=selected_z,
        mode='markers',
        marker=dict(
            size=8,
            color=color_strings,  # Use RGB values for color
            opacity=0.8
        )
    )])

    # Configure axis labels
    fig.update_layout(scene=dict(xaxis_title='X-axis', yaxis_title='Y-axis', zaxis_title='Z-axis'))

    # Update layout for better visualization
    fig.update_layout(scene=dict(aspectmode="data"))

    # Show the plot
    fig.show()



def convert_carla_depth_to_grayscale(depth_image):
    # Extract depth values
    depth_values = np.array(depth_image.raw_data).reshape((depth_image.height, depth_image.width, 4))[:, :, 0]
    # Normalize depth values to the range [0, 1]
    normalized_depth = (depth_values - depth_values.min()) / (depth_values.max() - depth_values.min())
    # Convert normalized depth values to grayscale
    grayscale_depth = (normalized_depth * 255).astype(np.uint8)

    return grayscale_depth



def apply_mask(XYZ, mask):
    """
    Apply a mask to the XYZ array.

    Parameters:
        - XYZ: 3D array containing X, Y, and Z values.
        - mask: Binary mask where True values indicate pixels to keep.

    Returns:
        - Masked XYZ array.
    """
    masked_XYZ = XYZ.copy()
    masked_XYZ[~mask] = [0, 0, 0]
    return masked_XYZ



def main():
    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        # Spawn the vehicle
        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.*')),
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)

        sensors_transform = carla.Transform(carla.Location(x=2.5, z=1.0), carla.Rotation(pitch=0))

        # Attach the cameras to the vehicle
        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            sensors_transform,
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        camera_depth = world.spawn_actor(
            blueprint_library.find('sensor.camera.depth'),
            sensors_transform,
            attach_to=vehicle)
        actor_list.append(camera_depth)

        camera_semseg = world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            sensors_transform,
            attach_to=vehicle)
        actor_list.append(camera_semseg)

        # Camera intrinsics
        width  = 800   # Replace with your image width in pixels
        height = 600   # Replace with your image height in pixels
        fov_x  = 60.0  # Replace with your horizontal FOV in degrees
        fov_y  = 45.0  # Replace with your vertical FOV in degrees
        fx, fy, cx, cy = calculate_camera_parameters(width, height, fov_x, fov_y)

        plot_3d_is_on = False

        # Specify the class ID you want to visualize
        class_id_to_visualize = 24

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_semseg, camera_depth, fps=30) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg, image_depth = sync_mode.tick(timeout=2.0)

                np_image_rgb = np.array(image_rgb.raw_data).reshape((height, width, 4))[:, :, :3]  
                np_image_rgb = cv2.cvtColor(np_image_rgb, cv2.COLOR_BGR2RGB)
                
                #convert to np_array
                grayscale_deep_np = convert_carla_depth_to_grayscale(image_depth)
                # Display segmentation mask after conversion
                np_image_semseg = np.array(image_semseg.raw_data).reshape((height, width, 4))[:, :, :3]

                # Create a binary mask for the specified class
                class_mask = np_image_semseg[:, :, 2] == class_id_to_visualize
                # Set pixels of other classes to black
                mask = np_image_semseg.copy()
                mask[~class_mask] = [0,  0, 0]
                mask[class_mask]  = [0, 255, 0]

                # Count the number of non-[0, 0, 0] elements in the mask array
                num_nonzero_mask_elements = np.count_nonzero(np.all(mask != [0, 0, 0], axis=-1))
                print("Number of non-[0, 0, 0] elements in mask:", num_nonzero_mask_elements)




                # Count the number of non-zero pixels in the modified mask
                num_nonzero_pixels = np.count_nonzero(class_mask)
                print("Number of non-zero pixels in the modified mask:", num_nonzero_pixels)

                # Create the 3d map
                X, Y, Z    = depth_to_3d(grayscale_deep_np, fx, fy, cx, cy)
                XYZ        = encode_xyz_to_rgb(X, Y, Z)
                masked_XYZ = apply_mask(XYZ, class_mask)

                # Count the number of non-[0, 0, 0] elements in the masked_XYZ array
                num_nonzero_elements = np.count_nonzero(np.all(masked_XYZ != [0, 0, 0], axis=-1))

                # Print the result
                print("Number of non-[0, 0, 0] elements in masked_XYZ:", num_nonzero_elements)
                print()

                # Draw the display with lane visualization
                #draw_image(display, mask, blend=True)
                draw_image(display, Y, blend=True)
                pygame.display.flip()
                
                if plot_3d_is_on:
                    plot_colored_3d_points(XYZ, np_image_rgb, 36000)
                    return
            
                # Choose the next waypoint and update the car location.
                waypoint = random.choice(waypoint.next(1.5))
                vehicle.set_transform(waypoint.transform)
    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')