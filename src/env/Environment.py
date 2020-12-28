import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.transform import Rotation as R
from src.env.Transform import Transform
from src.env import drawing

class Interactive(object):
    def __init__(self, obj, position, direction):
        self.points = obj[:,:3]
        self.occ = obj[:,3]
        self.position = position
        self.direction = direction
        self.up = np.array([0,1,0])
        
    def get(self, position, direction):
        points = self.points @ rotation_matrix(direction, self.up).T + position
        return np.concatenate([points, self.occ[:,None]],1)
        
class Box:
    def __init__(self, transform = Transform(), size = [1,1,1]):
        self.transform = transform
        self.size = size

    def get_def(self):
        return self.transform.apply([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])[:,[0,2,1]] 
    
    def collide(self, points, r):
        points = self.transform.inverse().apply(points)

        A = np.linalg.norm(np.maximum(-points, 0), axis = 1)
        B = np.linalg.norm(np.maximum(points - self.size, 0), axis = 1)
        d = 1 - np.minimum(np.maximum(A,B) / r,1)
        
        return d
    
class BoxEnv:
    def __init__(self, boxes, size = 4, layers = 9, res = 9):
        self.boxes = boxes
        self.size = size
        self.layers = layers
        self.res = res
        self.sensor_radius = 0.5 * size / float((res - 1))

        
    def collide_point(self, point):
        return (self.size - np.minimum(np.linalg.norm(np.maximum((p - X) * [[-1],[1]], 0), axis = 2).max(1), self.size)).min()
    
    def get_collider_points(self, position = [0,0], direction = [0,1]):
        position,direction = np.array(position), np.array(direction)
        diameter = self.size / float((self.res - 1))
        coverage = 0.5 * diameter
        points = []
        _x,_z = position
        _w = np.arctan2(direction[1], direction[0])
        
        for y in range(self.layers): # height
            for z in range(self.res): # radius
                dist = z * coverage
                arc = 2 * np.pi * dist
                count = int(np.rint(arc / coverage))
                for x in range(count): # angle
                    degrees = x / count * 2 * np.pi
                    coords = [_x + dist * np.cos(_w + degrees), y * coverage, _z + dist * np.sin(_w + degrees)]
                    points.append(coords)

        self.last = np.array(points)
        return np.array(points)

    def collide(self, skeleton):
        points = self.get_collider_points(skeleton.position, skeleton.direction)
        d = np.zeros(len(points))
        for box in self.boxes:
            d = np.maximum(d, box.collide(points, self.sensor_radius))
        return d

    def draw(self, ax = None):
        graphs = []
        for box in self.boxes:
            graphs += drawing.draw_cube(box.get_def(), ax)
        return graphs