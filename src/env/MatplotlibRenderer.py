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
    def __init__(self, env, objects):
        self.is_paused = False
        self.env = env
        self.obj_idx = -1
        self.objects = objects
        
        self.ax = None
        self.reset()
        self.init()

    def press(self, event):
        if event.key == 'p':
            self.is_paused = not self.is_paused
        if event.key == 'r':
            self.reset()

    def reset(self):
        # if self.ax is not None: self.ax.clear()
        self.skeleton = Skeleton([0,0],[0, 1])
        self.skeleton.sense(self.env, self.objects)
    
    def init(self):
        fig = plt.figure(figsize = (12,12))
        fig.tight_layout()
        fig.canvas.mpl_connect('key_press_event', self.press)
    
        self.ax = fig.add_subplot(111,projection='3d')
        self.ax.set_xlim3d(-3,3)
        self.ax.set_ylim3d(-3,3)
        self.ax.set_zlim3d(0,6)
        self.ax.set_axis_off()
        
        self.frame = WorldFrame(self)
        
        xx,yy = np.mgrid[-5:5,-5:5]
        self.graph = [self.ax.plot_surface(xx,yy,0*xx,alpha = 0.1)]
        
        self.graph += self.frame.draw(self.ax)
        self.graph += self.env.draw(self.ax)

        
    def update(self):
        if self.is_paused: return
        self.skeleton.update(self.env, [])

    def update_frame(self, frame):
        if self.is_paused: return self.graph
        self.update()
        self.frame.update(self.graph)
        return self.graph
        
    def play(self):
        self.anim = anim.FuncAnimation(self.ax.figure, self.update_frame, interval = 1000 / 120., blit=False)

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False


if __name__ == "__main__":
    env = BoxEnv([])
    obj_list = []
    world = World(env, obj_list)

    world.skeleton.set_goal([5,0,5])

    world.play()
    world.pause()
    plt.show()