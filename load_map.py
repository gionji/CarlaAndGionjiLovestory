import carla
import math
import random
import time
import queue
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import os
import string

'''
The layout of layered maps is the same as non-layered maps but it is 
possible to toggle off and on the layers of the map. There is a minimum 
layout that cannot be toggled off and consists of roads, sidewalks, traffic 
lights and traffic signs. Layered maps can be identified by the suffix _Opt, 
for example, Town01_Opt. With these maps it is possible to load and unload 
layers via the Python API
'''

def main():
    client = carla.Client('localhost', 2000)


    ### Map management
    # set the map name or get it from the running carla instance
    world = client.get_world()

    # get available maps in the distro
    available_maps = client.get_available_maps()
    print('Available maps:', available_maps)

    # get a random one
    random_map = random.choice( available_maps )

    # load it
    world = client.load_world( random_map ) 
    print('Loaded map:', world.get_map().name )

    # reload it
    world = client.reload_world()






    ### Layered maps ###


    object_categories = [ 'Buildings', 'Decals', 'Foliage', 'Ground', 
                        'ParkedVehicles', 'Particles', 'Props', 'StreetLights',
                        'Walls', 'All']
    
    # Toggle all buildings off
    world.unload_map_layer(carla.MapLayer.Buildings)

    time.sleep(2)

    # Toggle all buildings on   
    world.load_map_layer(carla.MapLayer.Buildings)


if __name__ == "__main__":
    main()