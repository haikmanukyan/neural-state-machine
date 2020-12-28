import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from src.env.Environment import BoxEnv, Box
from src.env.Transform import Transform
from src.env.Skeleton import Skeleton
from src.env.Environment import Interactive
from src.data.ShapeManager import Data
from src.data.WorldFrame import WorldFrame

class World:
    def __init__(self, env, objects, skeleton = None):
        self.is_paused = False
        self.env = env
        self.obj_idx = -1
        self.objects = objects
        
        if skeleton is None:
            skeleton = Skeleton()
        self.skeleton = skeleton
        
    def update(self):
        if self.is_paused: return
        self.skeleton.update(self)

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False


if __name__ == "__main__":
    env = BoxEnv([])
    obj_list = []
    world = World(env, obj_list)
    world.skeleton.set_goal([5,0,5])

    # !!!!! NO DEMO !!!!