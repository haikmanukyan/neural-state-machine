import sys; sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from src.utils.functions import gen_points

class SkeletonFrame:
    def __init__(self, data, skeleton = None):
        # Structure
        # Frame (23 joints): 
        #   pos xyz | forw xyz | up xyz | vel xyz || 23 * 12 = 276
        # Trajectory (13 points):
        #   posx, posy, dirx, diry
        #   actiosn: idle, walk, run, carry, open, crouch, sit, climb
        # Goal
        #   pos xyz, direction xyz
        #   action: idle, walk, run, carry, open, crouch, sit
        # Environment (cylinder)
        #   occupancy
        # Interaction
        #   pos3d | occupancy
        # Gaiting
        #   ---------
        
        self.data = data
        self.skeleton = skeleton
        self.projection = "3d"

        self.bone_order = list(range(7)) + [5] + list(range(7,11)) \
                + list(range(10,6,-1)) + [5] + list(range(11,15)) \
                + list(range(14,10,-1)) + list(range(5,-1,-1)) \
                    + list(range(15,19)) + list(range(18,14,-1)) + [0] \
                        + list(range(19,23))

    def draw_points(self, ax, points, size = 5, alpha = 1., linestyle = "", linewidth = 3, marker = "o", c = None):
        return ax.plot(points[:,0],points[:,2],points[:,1], linestyle=linestyle, c = c, linewidth=linewidth, marker=marker, markersize = size, alpha = alpha)[0]
    
    def update_points(self, graph, points, size = 5):
        graph.set_data(points[:,0],points[:,2])
        graph.set_3d_properties(points[:,1])

    def draw_goals(self, ax):
        return self.draw_points(ax, self.data.goal_position, linestyle=None)
    def update_goals(self, graph):
        self.update_points(graph, self.data.goal_position)

    def draw_trajectory(self, ax):
        return ax.plot(
            self.data.trajectory_position[:,0],
            self.data.trajectory_position[:,1],
            self.data.trajectory_position[:,0] * 0,
            marker="o"
        )[0]
    def update_trajectory(self, graph):
        graph.set_data(
            self.data.trajectory_position[:,0],
            self.data.trajectory_position[:,1]
        )
        graph.set_3d_properties(
            self.data.trajectory_position[:,0] * 0,
        )

    def draw_interaction(self, ax):
        return self.draw_points(
            ax, 
            self.data.interaction_position[self.data.interaction_occupancy > 1e-5]
        )
    def update_interaction(self, graph):
        self.update_points(
            graph, 
            self.data.interaction_position[self.data.interaction_occupancy > 1e-5]
        )

    def draw_environment(self, ax):
        points = gen_points(4, 9, 9)
        return ax.scatter(points[:,0],points[:,2],points[:,1], s = 100 * self.data.environment, alpha = 0.5)

        # return self.draw_points(
        #     ax, 
        #     points, 
        #     alpha = 0.5, 
        #     size = self.data.environment
        # )

    def update_environment(self, graph):
        # points = gen_points(4, 9, 9)
        # points = points[self.data.environment > 1e-5]
        # self.update_points(graph, points)
        # graph.set_data(points[:,0],points[:,2])
        # graph.set_3d_properties(points[:,1])
        
        graph.set_sizes(100 * self.data.environment)

    def draw_character(self, ax):
        return self.draw_points(ax, self.data.bone_position[self.bone_order], linestyle="-")

    def update_character(self, graph):
        self.update_points(graph, self.data.bone_position[self.bone_order])

    def draw(self, ax):
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])

        graph = [
            self.draw_goals(ax),
            self.draw_trajectory(ax),
            self.draw_environment(ax),
            self.draw_interaction(ax),
            self.draw_character(ax)
        ]

        return graph

    def update(self, graph):
        func = [
            self.update_goals,
            self.update_trajectory,
            self.update_environment,
            self.update_interaction,
            self.update_character
        ]
        for f,x in zip(func, graph):
            f(x)
        return graph