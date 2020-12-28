import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.env.utils import get_collider_points
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def draw_cube(cube_definition, ax = None):
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    faces.set_facecolor((0,0,1,0.1))

    graphs = [faces]
    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    graphs.append(ax.scatter(points[:,0], points[:,1], points[:,2], s=0))
    return graphs

def draw_points(points = None, env = None, ax = None, s = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
    
    if env is not None:
        ax.set_xlim3d(env.bounds[0][0], env.bounds[0][0] + env.bounds[0][1])
        ax.set_ylim3d(env.bounds[1][0], env.bounds[1][0] + env.bounds[1][1])
        ax.set_zlim3d(env.bounds[2][0], env.bounds[2][0] + env.bounds[2][1])

    X,Z,Y = points.T
    
    if s is None: ax.scatter(X,Y,Z, alpha = 0.5)
    else: ax.scatter(X,Y,Z, alpha = 0.5, s = s)
    return ax

def draw_env(env):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim3d(env.bounds[0][0], env.bounds[0][0] + env.bounds[0][1])
    ax.set_ylim3d(env.bounds[1][0], env.bounds[1][0] + env.bounds[1][1])
    ax.set_zlim3d(env.bounds[2][0], env.bounds[2][0] + env.bounds[2][1])
    
    points = np.array(np.where(env.map > 0)).T
    points = points / env.res * env.bounds[:,1] + env.bounds[:,0]
    return draw_points(points, ax = ax)
    
    
def draw_obj(obj, position = [0,0,0], direction = [1,0,0], ax = None):
    points = obj.get(position, direction)
    points = points[:,:3]
    return draw_points(points, ax = ax)

def draw_collider(X, env, ax = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    points = env.get_collider_points()
    # points = env_points[X > 0]
    return draw_points(points, ax = ax, s = 100 * X)