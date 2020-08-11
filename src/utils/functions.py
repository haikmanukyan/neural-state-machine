import time
import numpy as np
from torch import nn

class Marker:
    def __init__(self, log = True):
        self.last = time.time()
        self.log = log
    def __call__(self, str = "", log = True):
        ctime = time.time()
        if log and self.log: print (str, ctime - self.last)
        self.last = ctime

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def gen_points(size, res, layers):
    diameter = size / float((res - 1))
    coverage = 0.5 * diameter
    # radius = 0.5 * np.sqrt(2) * coverage
    points = []

    for y in range(layers): # height
        for z in range(res): # radius
            dist = z * coverage
            arc = 2 * np.pi * dist
            count = int(np.rint(arc / coverage))

            for x in range(count): # angle
                degrees = x / count * 2 * np.pi
                points.append([dist * np.cos(degrees), y * coverage, dist * np.sin(degrees)])

    return np.array(points)

def str2list(str):
    return list(map(int,str.split(',')))

def getangle(u,v):
    c = np.einsum("ij,ij->i", u, v) / np.linalg.norm(u, axis = 1) / np.linalg.norm(v, axis = 1)
    return np.rad2deg(np.arccos(np.clip(c, -1, 1)))

def clipangle(x):
    return (x + 2 * np.pi) % (2 * np.pi)

def normalize(X, mean, std):
    return (X - mean) / (1e-5 + std)

def unnormalize(X, mean, std):
    return X * std + mean