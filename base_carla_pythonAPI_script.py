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



def main():
    client = carla.Client('localhost', 2000)

    # set the map name or get it from the running carla instance
    world  = client.get_world()
    world = client.load_world('Town01')


if __name__ == "__main__":
    main()

