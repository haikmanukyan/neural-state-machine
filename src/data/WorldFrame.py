import sys; sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from src.utils.functions import gen_points

class WorldFrame:
    def __init__(self, world):
        self.world = world
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

    def draw_goals(self, ax, goal_position):
        return self.draw_points(ax, goal_position, linestyle=None)
    def update_goals(self, graph, goal_position):
        self.update_points(graph, goal_position)

    def draw_trajectory(self, ax, trajectory_position):
        return ax.plot(
            trajectory_position[:,0],
            trajectory_position[:,1],
            trajectory_position[:,0] * 0,
            marker="o"
        )[0]
    def update_trajectory(self, graph, trajectory_position):
        graph.set_data(
            trajectory_position[:,0],
            trajectory_position[:,1]
        )
        graph.set_3d_properties(
            trajectory_position[:,0] * 0,
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

    def draw_environment(self, ax, env_points, collision):
        return ax.scatter(env_points[:,0],env_points[:,2],env_points[:,1], s = 100 * (collision), alpha = 0.5)
    def update_environment(self, graph, env_points, collision):
        graph._offsets3d = (env_points[:,0],env_points[:,2],env_points[:,1])
        graph.set_sizes(100 * (collision))
        

    def draw_character(self, ax, bone_position):
        return self.draw_points(ax, bone_position[self.bone_order], linestyle="-")
    def update_character(self, graph, bone_position):
        self.update_points(graph, bone_position[self.bone_order])

    def draw(self, ax):
        skeleton = self.world.skeleton
        data = skeleton.get_data()
        env_points = self.world.env.get_collider_points(skeleton.position, skeleton.direction)

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])

        graph = [
            self.draw_trajectory(ax, data.trajectory_position),
            self.draw_environment(ax, env_points, data.environment),
            # self.draw_interaction(ax, data.interac),
            self.draw_character(ax, data.bone_position),
            self.draw_goals(ax, data.goal_position)
        ]

        return graph

    def update(self, graph):
        skeleton = self.world.skeleton
        data = skeleton.get_data()
        env_points = self.world.env.get_collider_points(skeleton.position, skeleton.direction)
        # env_points = self.world.env.last

        self.update_trajectory(graph[1], data.trajectory_position)
        self.update_environment(graph[2], env_points, data.environment)
        # self.update_interaction()
        self.update_character(graph[3], data.bone_position)
        self.update_goals(graph[4], data.goal_position)
        
        return graph
