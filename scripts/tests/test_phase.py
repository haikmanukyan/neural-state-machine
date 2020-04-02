import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

'''
Used this to reverse engineer the gating data generation. Checking if the generated data matches the real data.
'''
np.set_printoptions(precision = 3, suppress = True)

def getangle(u,v):
    c = np.einsum("ij,ij->i", u, v) / np.linalg.norm(u, axis = 1) / np.linalg.norm(v, axis = 1)
    return np.rad2deg(np.arccos(np.clip(c, -1, 1)))

def get_phase(input_data):
    trajectory_data = input_data[276:432].reshape(13, 12)
    sign = 2 * trajectory_data[:, 4] - 1

    gating_data = input_data[4683:].reshape(-1,2)
    phase_sin = sign * gating_data[:,0].reshape(13, 29)[:,0]
    phase_cos = sign * gating_data[:,1].reshape(13, 29)[:,0]
    phase = np.arctan2(phase_sin, phase_cos)    

    return phase

def gen_gating_data(phase, trajectory_data, goal_data):
    root_pos = np.zeros((13, 3))
    root_pos[:,0] = trajectory_data[:,0]
    root_pos[:,2] = trajectory_data[:,1]
    root_dir = np.zeros((13,3))
    root_dir[:,0] = trajectory_data[:,2]
    root_dir[:,2] = trajectory_data[:,3]

    goal_pos = goal_data[:,:3]
    goal_pos[:,1] = 0
    goal_dir = goal_data[:,3:6]
    goal_dir[:, 1] = 0

    dist = np.linalg.norm(goal_pos - root_pos, axis = 1)[:,None]
    angle = getangle(goal_dir, root_dir)[:, None]

    trajectory_action = 2 * trajectory_data[:,4:] - 1
    goal_action = 2 * goal_data[:,6:] - 1
    goal_action = goal_action.repeat(3,1)
    goal_action[:,1::3] *= dist
    goal_action[:,2::3] *= angle

    phase = phase.repeat(29)
    X_ = np.concatenate([trajectory_action, goal_action], 1).flatten()
    gating_data = np.stack([X_ * np.sin(phase), X_ * np.cos(phase)], 1).flatten()
    
    return gating_data

def normalize(X, mean, std):
    return (X - mean) / (1e-5 + std)
    std[std == 0] = 1
    return (X - mean) / std

def unnormalize(X, mean, std):
    return X * std + mean

A = np.loadtxt('data/x.txt')
input_norm = np.load('data/input_norm.npy').astype(np.float32)
output_norm = np.load('data/output_norm.npy').astype(np.float32)

input_data = A[:5437]
output_data = A[5437:]

# input_data = normalize(input_data, input_norm[0], input_norm[1])
# output_data = normalize(output_data, output_norm[0], output_norm[1])

gating_data = input_data[4683:]
gating_data = normalize(input_data, input_norm[0], input_norm[1])[4683:]


trajectory_data = input_data[276:432].reshape(13, 12)
goal_data = input_data[432:601].reshape(13, 13)


phase = get_phase(input_data)
gating_data_new = gen_gating_data(phase, trajectory_data, goal_data)
gating_data_new = normalize(gating_data_new, input_norm[0][4683:], input_norm[1][4683:])

phase = phase.repeat(29)
Y = gating_data.reshape(-1, 2)
X__ = (Y[:,0] / np.sin(phase))

print (gating_data - gating_data_new)
