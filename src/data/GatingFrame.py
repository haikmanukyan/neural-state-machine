import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D

class GatingFrame:
    def __init__(self, angle):
        # angular value
        self.angle = angle
        self.radius = 0.4
        self.size = 20
        self.projection = None

        
    def draw(self, ax):
        circle1 = plt.Circle((0, 0), self.radius, color='r', fill=False)
        ax.add_artist(circle1)
        
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        x = self.radius * np.cos(self.angle)
        y = self.radius * np.sin(self.angle)
        
        graph, = ax.plot([x], [y], markersize = self.size,  marker="o")

        return graph

    def update(self, graph):
        x = self.radius * np.cos(self.angle)
        y = self.radius * np.sin(self.angle)
        
        graph.set_data([x], [y])
        return [graph]