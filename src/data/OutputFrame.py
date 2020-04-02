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

class OutputFrame:
    def __init__(self, data):
        # Structure
        # Position 
        #   starting frame: pos | forw | up | vel
        #   final frame: pos | forw | up | vel
        # Trajectory 
        #   starting frame (+6 frames): pos | forw | up | vel | action7
        #   final frame (-6 frames): pos | forw | up | vel
        # Goal (-13 frames)
        #   pos | direction | action
        # Contact
        #   5 bool
        # PhaseUpdate (7 points)
        #   value (angular)

        self.data = data

        self.n_bones = 23
        self.n_trajectory_points = 13
        self.n_actions = 8
        self.n_interaction = 512 # for later
        self.projection = "3d"

        # Bones in egocentric
        self.joint_data = np.split(data[:276].reshape(-1, 12), [3,6,9], 1)
        self.joint_positions, self.joint_forward, self.joint_up, self.joint_velocity = self.joint_data
        self.joint_inverse_positions = data[276:345].reshape(-1,3)
        
        # pos dir action
        self.trajectory_data = np.split(data[345:429].reshape(-1, 12), [2,4], 1)
        self.trajectory_points, self.trajectory_dir, self.trajectory_action = self.trajectory_data
        self.trajectory_inverse_points = data[429:457].reshape(-1,2)

        # Goal        
        self.goal_data = np.split(data[457:626].reshape(-1, 13), [3,6], 1)
        self.goal_positions,self.goal_directions,self.goal_actions = self.goal_data

        self.contact = data[626:631]
        self.phase_update = data[631:]

        self.bone_order = list(range(7)) + [5] + list(range(7,11)) \
                + list(range(10,6,-1)) + [5] + list(range(11,15)) \
                + list(range(14,10,-1)) + list(range(5,-1,-1)) \
                    + list(range(15,19)) + list(range(18,14,-1)) + [0] \
                        + list(range(19,23))

    def draw_points(self, ax, points, size = 5, alpha = 1., linestyle = "", linewidth = 3):
        return ax.plot(points[:,0],points[:,2],points[:,1], linestyle=linestyle, marker="o", linewidth = linewidth, markersize = size, alpha = alpha)[0]
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

    def draw_character(self, ax):
        return self.draw_points(ax, self.joint_positions[self.bone_order], linestyle="-")

    def update_character(self, graph):
        self.update_points(graph, self.joint_positions[self.bone_order])

    def draw(self, ax):
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])

        graph = [
            self.draw_goals(ax),
            self.draw_trajectory(ax),
            self.draw_character(ax)
        ]   

        return graph

    def update(self, graph):
        func = [
            self.update_goals,
            self.update_trajectory,
            self.update_character
        ]
        for f,x in zip(func, graph):
            f(x)
        return graph