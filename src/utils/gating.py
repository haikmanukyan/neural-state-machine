import numpy as np
from src.utils.functions import *


def get_phase(frame):
    action = 2 * frame.trajectory_action.reshape(13,-1)[:,:1] - 1
    phase = frame.gating.reshape(13, -1, 2)[:,0,:] * action

    phase = np.arctan2(phase[:,0],phase[:,1])
    return phase

def update_gating(frame, output_frame, new_phase = None):
    
    if new_phase is None:
        phase = get_phase(frame)
        phase = update_phase(phase, output_frame.phase)
    else:
        phase = new_phase

    frame.gating = gen_gating_data(
        phase,
        frame.goal_position,
        frame.goal_direction,
        frame.goal_action,
        frame.trajectory_position,
        frame.trajectory_direction,
        frame.trajectory_action
    )

def get_phase_data(input_data):
    trajectory_data = input_data[B:B+T].reshape(13, TT)
    sign = 2 * trajectory_data[:, 4] - 1

    gating_data = input_data[IN-GA:].reshape(-1,2)
    phase_sin = sign * gating_data[:,0].reshape(13, GAGA)[:,0]
    phase_cos = sign * gating_data[:,1].reshape(13, GAGA)[:,0]
    phase = clipangle(np.arctan2(phase_sin, phase_cos))
    return phase

def update_phase(phase, phase_update):
    phase_new = phase.copy()
    phase_new[:6] += 2 * np.pi * phase_update[0]
    
    phase_new[6:] = phase[6]
    phase_new[6:] += 2 * np.pi * phase_update

    return clipangle(phase_new)

def gen_gating_data(
    phase,
    goal_pos,
    goal_dir,
    goal_action,
    pos,
    dir,
    action
    ):
    phase = phase.repeat(25)
    goal_action = goal_action.reshape(13,-1)
    action = action.reshape(13,-1)
    
    dist = np.linalg.norm(goal_pos[:,[0,2]] - pos, axis = 1)
    angle = getangle(goal_dir[:,[0,2]], dir)

    action = 2 * action - 1
    
    goal_action = 2 * goal_action.repeat(3,1) - 1
    goal_action[:,1::3] *= dist[:,None]
    goal_action[:,2::3] *= angle[:,None]

    X_ = np.concatenate([action, goal_action], 1).flatten()
    gating_data = np.stack([X_ * np.sin(phase), X_ * np.cos(phase)], 1).flatten()
    
    return gating_data

def gen_gating_data_(
    phase,
    goal_pos,
    goal_dir,
    goal_action,
    pos,
    dir,
    action
    ):
    phase = phase.repeat(action.shape[1] + 3 * goal_action.shape[1])

    dist = np.linalg.norm(goal_pos - pos, axis = 1)[:,None]
    angle = getangle(goal_dir, dir)[:, None]

    action = 2 * action[:,4:] - 1
    
    goal_action = 2 * goal_action.repeat(3,1) - 1
    goal_action[:,1::3] *= dist
    goal_action[:,2::3] *= angle

    X_ = np.concatenate([action, goal_action], 1).flatten()
    gating_data = np.stack([X_ * np.sin(phase), X_ * np.cos(phase)], 1).flatten()
    
    return gating_data