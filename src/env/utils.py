import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.transform import Rotation as R

def get_collider_points():
    diameter = size / float((res - 1))
    coverage = 0.5 * diameter
    points = []
    for y in range(layers): # height
        for z in range(res): # radius
            dist = z * coverage
            arc = 2 * np.pi * dist
            count = int(np.rint(arc / coverage))
            for x in range(count): # angle
                degrees = x / count * 2 * np.pi
                coords = [dist * np.cos(degrees), y * coverage, dist * np.sin(degrees)]
                points.append(coords)
    return np.array(points)


def rotation_matrix(forward, up):
    forward, up = np.float32(forward) / np.linalg.norm(forward), np.float32(up) / np.linalg.norm(forward)
    left = np.cross(forward, up)
    up = np.cross(left, forward)
    
    return np.array([forward, up, left])