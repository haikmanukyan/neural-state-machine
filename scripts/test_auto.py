import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from src.data import Animation
from src.data import InputFrame, OutputFrame, GatingFrame
from src.nn.nets import NSM
import torch
import sys

np.set_printoptions(precision = 3, suppress = True)

def getangle(u,v):
    c = np.einsum("ij,ij->i", u, v) / np.linalg.norm(u, axis = 1) / np.linalg.norm(v, axis = 1)
    return np.rad2deg(np.arccos(np.clip(c, -1, 1)))

def clipangle(x):
    return (x + 2 * np.pi) % (2 * np.pi)

def get_phase(input_data):
    trajectory_data = input_data[276:432].reshape(13, 12)
    sign = 2 * trajectory_data[:, 4] - 1

    gating_data = input_data[4683:].reshape(-1,2)
    phase_sin = sign * gating_data[:,0].reshape(13, 29)[:,0]
    phase_cos = sign * gating_data[:,1].reshape(13, 29)[:,0]
    phase = clipangle(np.arctan2(phase_sin, phase_cos))

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

def get_sample(input_data, output_data):
    return [
        torch.from_numpy(input_data[:, 0:432]).cuda(),
        torch.from_numpy(input_data[:, 432:601]).cuda(),
        torch.from_numpy(input_data[:, 601:2635]).cuda(),
        torch.from_numpy(input_data[:, 2635:4683]).cuda(),
        torch.from_numpy(input_data[:, 4683:5437]).cuda(),
        torch.from_numpy(output_data).cuda()
    ]

def normalize(X, mean, std):
    return (X - mean) / (1e-5 + std)

def unnormalize(X, mean, std):
    return X * std + mean

def update_phase(phase, phase_update):
    phase_new = phase.copy()
    phase_new[:6] += 2 * np.pi * phase_update[0]
    
    phase_new[6:] = phase[6]
    phase_new[6:] += 2 * np.pi * phase_update

    return clipangle(phase_new)

if __name__ == "__main__":
    # start = np.random.randint(40) * 240
    start = 240 * 54
    size = 240 * 2

    # Loading
    data = np.load('data/test32.npy')[start:start+size]
    input_norm = np.load('data/input_norm.npy').astype(np.float32)
    output_norm = np.load('data/output_norm.npy').astype(np.float32)
    
    # Normalize
    input_data = data[:,:5437]
    input_data_normed = normalize(input_data, input_norm[0], input_norm[1])
    output_data = data[:,5437:]
    output_data_normed = normalize(output_data, output_norm[0], output_norm[1])
    
    # Placeholders for network generated data
    input_net = input_data_normed.copy()
    output_net = output_data_normed.copy()
    input_net_normed = input_data_normed.copy()
    output_net_normed = output_data_normed.copy()

    if len(sys.argv) > 1:
        model = torch.load(sys.argv[1]).cuda()
    else:
        model = torch.load('models/best4.pt').cuda()

    print (model)
    # Inference
    frame_data, goal_data, environment_data, interaction_data, gating_data, output_data_test = get_sample(input_data_normed, output_data_normed)
    phase_data = np.array([get_phase(x)[7] for x in input_data]) # Get the phase info
    phase_net = phase_data.copy() # Dummy for predicted phase
    gating_data_gen = input_data[:, 4683:5437].copy() # Dummy for generated gating data
    gating_data_gen_norm = input_data[:, 4683:5437].copy() # Dummy for generated gating data (normalized)

    for i in range(len(frame_data)):
        if i % 240 == 0:
            frame,goal,environment,interaction,gating = frame_data[i],goal_data[i],environment_data[i],interaction_data[i],gating_data[i]
            
            # Initialize some values            
            phase = get_phase(input_data[i])
            joints = frame[:276]
            trajectory = frame[276:].reshape(13, 12)

        else:
            # frame,goal,environment,interaction,gating = frame_data[i],goal_data[i],environment_data[i],interaction_data[i],gating_data[i]
            frame,goal,environment,interaction,gating = frame_data[i],goal_data[i],environment_data[i],interaction_data[i],gating_data[i]

            trajectory_update = output_torch[345:429].reshape(7, 12)
            trajectory[:-1] = trajectory[1:].clone()
            trajectory[-1] = trajectory_update[-1]
            trajectory[-7:-1] = 0.5 * trajectory[-7:-1] + 0.5 * trajectory_update[:-1]

            frame[:276] = output_torch[:276] # Updated joints
            # frame[276:] = trajectory.view(-1)
            # goal = output_normed[457:626]

            # Update the phase
            phase = update_phase(phase, 2 * output_net[i-1][631:639])
            
            # trajectory_gen_normed = input_data_normed[i][276:432] # Current input trajectory data normalized
            trajectory_gen_normed = trajectory.flatten().cpu().detach().numpy()
            trajectory_gen = unnormalize(trajectory_gen_normed, input_norm[0,276:432], input_norm[1,276:432]).reshape(13, 12)

            goal_gen_normed = input_data_normed[i][432:601]
            goal_gen = unnormalize(goal_gen_normed, input_norm[0,432:601], input_norm[1,432:601]).reshape(13, 13)

            gating_data_gen[i] = gen_gating_data(phase, trajectory_gen, goal_gen)
            gating_data_gen_norm[i] = normalize(gating_data_gen[i], input_norm[0,4683:5437], input_norm[1,4683:5437])
            
            gating = torch.from_numpy(gating_data_gen_norm[i]).cuda()

        output_torch = model(frame[None], goal[None], environment[None], interaction[None], gating[None])[0]
        
        input_net_normed[i] = torch.cat([frame, goal, environment, interaction, gating]).cpu().detach().numpy()
        input_net[i] = unnormalize(input_net_normed[i], input_norm[0], input_norm[1])

        output_net_normed[i] = output_torch.cpu().detach().numpy()
        output_net[i] = unnormalize(output_net_normed[i],output_norm[0],output_norm[1])
        
    input_data_gen = input_data.copy()
    input_data_gen[:,4683:5437] = gating_data_gen
    phase_data_gen = np.array([get_phase(x)[7] for x in input_data_gen]) # Look at the generated phase values


    print (phase_data.shape, input_data.shape)
    # Drawing
    anim = Animation()
    # anim.add_animation(input_data, InputFrame)
    # anim.add_animation(phase_data, GatingFrame)
    # anim.add_animation(input_data, InputFrame, "Input From Test Set")
    # anim.add_animation(output_data, OutputFrame, "Output From Test Set")
    # anim.add_animation(phase_data, GatingFrame, "Phase From Test Set")
    anim.add_animation(input_net, InputFrame, "Actual Input")
    anim.add_animation(output_net, OutputFrame, "Predicted Output")
    anim.add_animation(phase_data_gen, GatingFrame, "Predicted Phase")
    anim.draw(220)
    anim.play()
    
    anim.save("anim/phase_fast.gif")
    plt.show()