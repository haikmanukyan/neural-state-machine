import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D

def gen_points(size, res, layers):
    diameter = size / float((res - 1))
    coverage = 0.5 * diameter
    # radius = 0.5 * np.sqrt(2) * coverage
    points = []

    for y in range(layers):
        for z in range(res):
            dist = z * coverage
            arc = 2 * np.pi * dist
            count = int(np.rint(arc / coverage))

            for x in range(count):
                degrees = x / count * 2 * np.pi
                points.append([dist * np.cos(degrees), y * coverage, dist * np.sin(degrees)])

    return np.array(points)

class Frame(object):
    def draw_points(self, ax, points, size = 5, alpha = 1., linestyle = ""):
        return ax.plot(points[:,0],points[:,2],points[:,1], linestyle=linestyle, marker="o", markersize = size, alpha = alpha)[0]

    def update_points(self, graph, points):
        graph.set_data(points[:,0],points[:,2])
        graph.set_3d_properties(points[:,1])

    def draw_goals(self, ax):
        return self.draw_points(ax, self.goal_positions, linestyle=None)
    def update_goals(self, graph):
        self.update_points(graph, self.goal_positions)

    def draw_trajectory(self, ax):
        return ax.plot(
            self.trajectory_points[:,0],
            self.trajectory_points[:,1],
            np.zeros_like(self.trajectory_points[:,0]),
            marker="o"
        )[0]
    def update_trajectory(self, graph):
        graph.set_data(
            self.trajectory_points[:,0],
            self.trajectory_points[:,1]
        )
        graph.set_3d_properties(
            np.zeros_like(self.trajectory_points[:,0]),
        )

    def draw_interaction(self, ax):
        return self.draw_points(
            ax, 
            self.interaction_points[self.interaction_occupied > 0]
        )
    def update_interaction(self, graph):
        self.update_points(
            graph, 
            self.interaction_points[self.interaction_occupied > 0]
        )

    def draw_environment(self, ax):
        points = gen_points(4, 9, 9)
        points = points[self.environment > 0]
        return self.draw_points(
            ax, 
            points, 
            alpha = 0.1, 
            size = 15
        )
    def update_environment(self, graph):
        points = gen_points(4, 9, 9)
        points = points[self.environment > 0]
        self.update_points(graph, points)

    def draw_character(self, ax):
        return self.draw_points(ax, self.joint_positions)
    def update_character(self, graph):
        self.update_points(graph, self.joint_positions)

    def draw(self, ax):
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