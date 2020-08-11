import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('.')

from src.data import Animation
from src.data import GatingFrame
from src.data.SkeletonFrame import SkeletonFrame
from src.data.ShapeManager import Data
from src.nn.nets import NSM
from src.utils import gating
import torch
import sys
import argparse

np.set_printoptions(precision = 3, suppress = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-n", type = int, default = 1, help = "Number of clips to visualize")
    parser.add_argument("--starting-clip", default = 0, type = int, help="Index of the first clip to visualize")
    parser.add_argument("--save", action="store_true", default=False, help = "Save the animation as a gif")
    parser.add_argument("--save-dir", type = str, default = "examples/animation.gif")
    parser.add_argument("--predict-phase", action="store_true", default=False, help = "Use the phase predicted by the network")
    parser.add_argument("--predict-trajectory", action="store_true", default=False, help = "Use thetrajectory predicted by the network")
    parser.add_argument("--model-name", type=str, default = 'models/best.pt', help = "The name of the model you want to test, just give the folder name ignoring the parenthesis")
    parser.add_argument("--phase-mult", type = float, default=1, help = "Multiply the speed of the phase update by this value. Works only when --predict-phase is on")

    parser.add_argument("--show-data", action="store_true", default=True, help = "Visualize the ground truth")
    parser.add_argument("--show-phase", action="store_true", default=False, help = "Visualize the phase")
    parser.add_argument("--show-input", action="store_true", default=False, help = "Visualize the actual input to the network")
    parser.add_argument("--hide-output", action="store_true", default=False, help = "Do not show the output frame")
    
    args = parser.parse_args()
    start = 240 * args.starting_clip
    size = 240 * args.n

    # Loading
    data = np.load('data/test16.npy').astype(np.float32)
    sequences = np.loadtxt('./data/TestSequences.txt')
    input_norm = np.load('data/input_norm.npy').astype(np.float32)
    output_norm = np.load('data/output_norm.npy').astype(np.float32)
    
    # Normalize
    input_data = Data(data[:,:input_norm.shape[1]], input_norm, "input", sequences)
    output_data = Data(data[:,input_norm.shape[1]:], output_norm, "output", sequences)

    clip_idx = 24
    clip = input_data(clip_idx)
    clip_out = output_data(clip_idx)

    current = clip[0]

    model = torch.load(args.model_name).cuda()
    model.input_shape = [419,156,2034,2048,650]
    
    frames = []
    predicted = Data(clip_out[0].data, output_norm, "output")
    
    for i in range(len(clip)):
        if i > 0:
            alpha = 0.

            current.bones = predicted.bones
            
            x = current.trajectory.reshape(13,-1)
            x -= x[7]
            x[:-1] = x[1:]
            x[6:] = predicted.trajectory.reshape(7,-1)

            current.trajectory = alpha * current.trajectory + (1 - alpha) * x.flatten()
            current.goal = alpha * current.goal + (1 - alpha) * predicted.goal

            new_phase = None
            new_phase = gating.get_phase(clip[i])
            gating.update_gating(current, predicted, new_phase)
            
            # Set from data

            # current.bones = clip[i].bones
            # current.trajectory = clip[i].trajectory
            current.goal = clip[i].goal
            current.environment = clip[i].environment
            current.interaction = clip[i].interaction
            # current.gating = clip[i].gating


        x = torch.from_numpy(current.normed()[None]).cuda()
        y = model(x).cpu().detach().numpy()[0]
        predicted.set_normed(y)
        frames.append(SkeletonFrame(current.copy()))
    
    data_frames = [SkeletonFrame(clip[i]) for i in range(len(clip))]
    anim = Animation()
    anim.add_frames(data_frames)
    anim.add_frames(frames)

    anim.draw(120)
    anim.play()

    plt.show()

'''
    # bones, trajectory, action
    # goal, goal_action
    # env
    # int
    # GATING = phase * (action, goal_action, ga ga)

    # -> normalize() after GATING
    # right before input!

    
    # Placeholders for network generated data
    input_net = input_data_normed.copy()
    output_net = output_data_normed.copy()
    input_net_normed = input_data_normed.copy()
    output_net_normed = output_data_normed.copy()

    model = torch.load(args.model_name).cuda()

    # Inference
    frame_data, goal_data, environment_data, interaction_data, gating_data, output_data_test = get_sample(input_data_normed, output_data_normed)
    phase_data = np.array([get_phase(x)[7] for x in input_data]) # Get the phase info
    phase_net = phase_data.copy() # Dummy for predicted phase
    gating_data_gen = input_data[:, IN-GA:IN].copy() # Dummy for generated gating data
    gating_data_gen_norm = input_data[:, IN-GA:IN].copy() # Dummy for generated gating data (normalized)

    for i in range(len(frame_data)):
        if i == 0: #i % 240 == 0:
            frame,goal,environment,interaction,gating = frame_data[i],goal_data[i],environment_data[i],interaction_data[i],gating_data[i]
            
            # Initialize some values            
            phase = get_phase(input_data[i])
            joints = frame[:276]
            trajectory = frame[276:].reshape(13, TT)

            gating__ = gen_gating_data(phase, trajectory.cpu().numpy(), goal.reshape(13,GG).cpu().numpy())
            # np.savetxt('data/sample.txt', input_data[i])
            # np.savetxt('data/sample_out.txt', output_data[i])
            # np.savetxt('data/sample_gen.txt', gating__)
            # # print (phase)
            # print (input_data)
            # input()

        elif True:
            frame,goal,environment,interaction,gating = frame_data[i],goal_data[i],environment_data[i],interaction_data[i],gating_data[i]
            
            # Update the frame
            frame[:276] = output_torch[:276]

            if args.predict_trajectory:
                trajectory_update = output_torch[B:B+T].reshape(7, TT)
                trajectory[:-1] = trajectory[1:].clone()
                trajectory[-1] = trajectory_update[-1]
                trajectory[-7:-1] = 0.5 * trajectory[-7:-1] + 0.5 * trajectory_update[:-1]
                frame[276:] = trajectory.view(-1)
            else:
                trajectory = frame[276:]
            
            # goal = output_normed[457:626]

            # Update the phase
            if args.predict_phase:            
                phase = update_phase(phase, args.phase_mult * output_net[i-1][631:639])
            
                trajectory_gen_normed = trajectory.flatten().cpu().detach().numpy()
                trajectory_gen = unnormalize(trajectory_gen_normed, input_norm[0,276:432], input_norm[1,276:432]).reshape(13, 12)

                goal_gen_normed = input_data_normed[i][432:601]
                goal_gen = unnormalize(goal_gen_normed, input_norm[0,432:601], input_norm[1,432:601]).reshape(13, 13)

                gating_data_gen[i] = gen_gating_data(phase, trajectory_gen, goal_gen)
                gating_data_gen_norm[i] = normalize(gating_data_gen[i], input_norm[0,4683:5437], input_norm[1,4683:5437])

                # gating = torch.from_numpy(gating_data_gen_norm[i]).cuda()
        else:
            pass
            # frame[:276] = output_torch[:276]
            frame[:276] = torch.from_numpy(unnormalize(output_net_normed[i][:276], *input_norm[:,:276])).cuda()
            # phase = update_phase(phase, args.phase_mult * output_net[i-1][631:639])
            # trajectory_gen_normed = trajectory.flatten().cpu().detach().numpy()
            # trajectory_gen = unnormalize(trajectory_gen_normed, input_norm[0,276:432], input_norm[1,276:432]).reshape(13, 12)
            # goal_gen_normed = input_data_normed[i][432:601]
            # goal_gen = unnormalize(goal_gen_normed, input_norm[0,432:601], input_norm[1,432:601]).reshape(13, 13)
            # gating_data_gen[i] = gen_gating_data(phase, trajectory_gen, goal_gen)
            # gating_data_gen_norm[i] = normalize(gating_data_gen[i], input_norm[0,4683:5437], input_norm[1,4683:5437])
            # gating = torch.from_numpy(gating_data_gen_norm[i]).cuda()

        output_torch = model(frame[None], goal[None], environment[None], interaction[None], gating[None])[0]
        
        input_net_normed[i] = torch.cat([frame, goal, environment, interaction, gating]).cpu().detach().numpy()
        input_net[i] = unnormalize(input_net_normed[i], input_norm[0], input_norm[1])
        # input_net[i] = input_net_normed[i]
        

        output_net_normed[i] = output_torch.cpu().detach().numpy()
        output_net[i] = unnormalize(output_net_normed[i],output_norm[0],output_norm[1])
        # output_net[i] = 10 * output_net_normed[i]
        
    input_data_gen = input_data.copy()
    input_data_gen[:,IN-GA:IN] = gating_data_gen
    phase_data_gen = np.array([get_phase(x)[7] for x in input_data_gen]) # Look at the generated phase values

    
    # Drawing
    anim = Animation()
    n = 0
    if args.show_data:
        if args.show_input:
            anim.add_animation(input_data, InputFrame, "Input Ground Truth")
            n += 1
        if not args.hide_output:
            anim.add_animation(output_data, OutputFrame, "Output Ground Truth")
            n += 1
        if args.show_phase:
            anim.add_animation(phase_data, GatingFrame, "Phase Ground Truth")
            n += 1

    if args.show_input:
        anim.add_animation(input_net, InputFrame, "Network Input")
        n += 1  
        pass
    if not args.hide_output:
        anim.add_animation(output_net, OutputFrame, "Predicted Output")
        n += 1
    if args.show_phase:
        anim.add_animation(phase_data_gen, GatingFrame, "Predicted Phase")
        n += 1
    

    w = np.ceil(np.sqrt(n))
    h = np.ceil(n / w)
    anim.draw(100 * h + 10 * w)
    anim.play()

    anim.dump()
    
    if args.save:
        anim.save(args.save_dir)
    plt.show()
'''