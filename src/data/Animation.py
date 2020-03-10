import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from .InputFrame import InputFrame

class Animation:
    def __init__(self, data):
        self.frames = [InputFrame(x) for x in data]

    def draw(self):
        self.fig = plt.figure(figsize = (10,10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.set_xlim3d(-2,2)
        self.ax.set_ylim3d(-2,2)
        self.ax.set_zlim3d(0,3)
        
        self.graph = self.frames[0].draw(self.ax)

    def update(self, frame_idx):
        return self.frames[frame_idx].update(self.graph)

    def play(self):
        self.anim = anim.FuncAnimation(self.fig, self.update, len(self.frames), interval = 1000 / 120., blit=True)