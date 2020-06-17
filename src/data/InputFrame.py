import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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
        self.projection = "3d"


        self.joint_data = np.split(data[:276].reshape(-1, 12), [3,6,9], 1)

        self.joint_positions, self.joint_forward, self.joint_up, self.joint_velocity = self.joint_data

        self.trajectory_data = np.split(data[276:432].reshape(-1, 12), [2,4], 1)
        self.trajectory_points, self.trajectory_dir, self.trajectory_action = self.trajectory_data
        
        self.goal_data = np.split(data[432:601].reshape(-1, 13), [3,6], 1) # 3d pos, dir + actions (-climb)
        self.goal_positions,self.goal_directions,self.goal_actions = self.goal_data
        
        self.interaction_data = np.split(data[2635:4683].reshape(-1,4), [3,], 1) # 3d points + label
        self.interaction_points, self.interaction_occupied = self.interaction_data
        self.interaction_occupied = self.interaction_occupied[:,0]
        
        self.environment = data[601:2635]
        self.gating = data[4683:]

        # color = plt.cm.viridis(np.linspace(0, 1,len(self.joint_positions)))
        # fig, ax = plt.subplots()
        # ax.scatter(np.array(self.joint_positions[:,0]), np.array(self.joint_positions[:,1]), c = color)
        # for i in range(len(self.joint_positions)):
        #     ax.annotate(i, (self.joint_positions[i,0], self.joint_positions[i,1]))
        # plt.show()

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
            self.interaction_points[self.interaction_occupied > 1e-5]
        )
    def update_interaction(self, graph):
        self.update_points(
            graph, 
            self.interaction_points[self.interaction_occupied > 1e-5]
        )

    def draw_environment(self, ax):
        points = gen_points(4, 9, 9)
        points = points[self.environment > 0]
        return self.draw_points(
            ax, 
            points, 
            alpha = 0.5, 
            size = 15
        )
    def update_environment(self, graph):
        points = gen_points(4, 9, 9)
        points = points[self.environment > 1e-5]
        self.update_points(graph, points)

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