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

class InputFrame:
    def __init__(self, data):
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

        self.n_bones = 23
        self.n_trajectory_points = 13
        self.n_actions = 8
        self.n_interaction = 512 # for later


        joint_data = np.split(data[:276].reshape(-1, 12), [3,6,9], 1)
        self.joint_positions, self.joint_forward, self.joint_up, self.joint_velocity = joint_data

        trajectory_data = np.split(data[276:432].reshape(-1, 12), [2,4], 1)
        self.trajectory_points, self.trajectory_dir, self.trajectory_action = trajectory_data
        
        goal_data = np.split(data[432:601].reshape(-1, 13), [3,6], 1) # 3d pos, dir + actions (-climb)
        self.goal_positions,self.goal_directions,self.goal_actions = goal_data
        
        interaction_data = np.split(data[2635:4683].reshape(-1,4), [3,], 1) # 3d points + label
        self.interaction_points, self.interaction_occupied = interaction_data
        self.interaction_occupied = self.interaction_occupied[:,0]
        
        self.environment = data[601:2635]
        self.gating = data[4683:]


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