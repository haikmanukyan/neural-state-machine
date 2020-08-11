import sys
sys.path.append('.')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from src.data.SkeletonFrame import SkeletonFrame

class InteractiveAnimation:
    def __init__(self, skeleton):
        self.skeleton = skeleton
        self.initialize()

    def predict(self):
        self.output, self.display = self.predict_fn(self.input)

    def initialize(self):
        self.fig = plt.figure(figsize = (6,6))
        self.fig.tight_layout()

        ax = self.fig.add_subplot(111, projection="3d")
        size = 1.2
        ax.set_xlim3d(-size,size)
        ax.set_ylim3d(-size,size)
        ax.set_zlim3d(0,2 * size)

        # self.frame = self.skeleton.get_frame()
        self.frame = SkeletonFrame(self.skeleton.local)
        self.graph = self.frame.draw(ax)
        self.ax = ax
    
    def update(self, frame):
        self.skeleton.update(None)
        self.frame = SkeletonFrame(self.skeleton.local)
        # self.frame = self.skeleton.get_frame()
        self.frame.update(self.graph)

        return self.graph

    def play(self):
        self.anim = anim.FuncAnimation(self.fig, self.update, interval = 1000 / 120., blit=True)

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def save(self, path = 'data/animation.gif'):
        self.anim.save(path, writer = 'imagemagick', fps = 30)