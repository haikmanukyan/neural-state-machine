import sys
sys.path.append('.')

import torch
import numpy as np
from src.nn.nets import NSM
from src.nn.controller import NSMController
from src.utils import gating
from src.data.SkeletonFrame import SkeletonFrame
from src.data.ShapeManager import Data
from src.env.Transform import Transform

initial_pose = np.load('data/initial_pose.npy')

def load_data():
    # Loading
    data = np.load('data/test16.npy').astype(np.float32)
    sequences = np.loadtxt('./data/TestSequences.txt')
    input_norm = np.load('data/input_norm.npy').astype(np.float32)
    output_norm = np.load('data/output_norm.npy').astype(np.float32)
    
    # Normalize
    input_data = Data(data[:,:input_norm.shape[1]], input_norm, "input", sequences)
    output_data = Data(data[:,input_norm.shape[1]:], output_norm, "output", sequences)

    return input_data, output_data


def init_from_clip(clip_idx = 1):
    input_data, output_data = load_data()
    return input_data(clip_idx)[0]

class UpdateData():
    def __init__(self, clip_idx = 1):
        input_data, output_data = load_data()
        self.frame_idx = 0
        self.clip = input_data(clip_idx)

    def __call__(self, skeleton):
        local = skeleton.local.copy()
        
        local.bones = self.clip[self.frame_idx].bones
        local.trajectory = self.clip[self.frame_idx].trajectory
        local.goal = self.clip[self.frame_idx].goal
        local.gating = self.clip[self.frame_idx].gating
        frame_idx += 1

        if self.frame_idx == len(self.clip):
            self.frame_idx = 0

        return local

class Skeleton:
    def __init__(self, position = [0,0], direction = [0,1], controller = None):
        self.position = np.float32(position)
        self.direction = np.float32(direction)
        
        if controller is None: controller = NSMController()
        self.controller = controller
        
        self.local = init_from_clip(1)

    def update_frame(self, predicted):
        alpha1 = 0.0
        alpha2 = 0.5

        self.local.bones = predicted.bones

        # Update Trajectory        
        x = self.local.trajectory.reshape(13,-1)
        x[:,:2] -= x[7,:2]
        x[:-1] = x[1:]
        x[6:] = predicted.trajectory.reshape(7,-1)
        self.local.trajectory = alpha1 * self.local.trajectory + (1 - alpha1) * x.flatten()

        # Update Goal
        self.local.goal = alpha2 * self.local.goal + (1 - alpha2) * predicted.goal

        # Update Phase
        gating.update_gating(self.local, predicted)
            

    def set_goal(self, goal, action = 0):
        self.goal_position = goal
        self.goal_action = action
        self.update_goal()
    
    def update_goal(self):
        if not hasattr(self,"goal_position"): return
        position = np.zeros(3)
        position[[0,2]] = self.position

        distance = np.linalg.norm(self.goal_position - position)

        goal_position = position + np.arange(0,1+1e-5,1/12)[:,None] * (self.goal_position - position)
        angle = np.rad2deg(np.arctan2(*self.direction))
        
        if self.goal_action in [5]:
            goal_direction = np.array([-1,0,0])
        else:
            goal_direction = self.goal_position - position

        goal_direction = goal_direction / np.linalg.norm(goal_direction)
        goal_angle = np.rad2deg(np.arctan2(*goal_direction[[0,2]]))
        goal_angle = (goal_angle - angle + 180) % 360 - 180
        goal_angle = np.deg2rad(goal_angle)

        
        goal_direction = np.array([np.sin(goal_angle),0,np.cos(goal_angle)])
        
        transform = Transform.from_rot_pos(
            [angle, 0, 0],
            [self.position[0],0,self.position[1]]
        ).inverse()

        rotation = Transform.from_euler([angle, 0, 0]).inverse()
        
        self.local.goal_position = transform.apply(goal_position)
        self.local.goal_direction = np.tile(goal_direction, [13,1])

        if self.goal_action in [1,2]:
            self.local.goal_action = np.tile(np.eye(6)[0], [13,1])
            
            if distance > 0.3:
                self.local.goal_action = np.tile(np.eye(6)[self.goal_action], [13,1])

        elif self.goal_action in [5]:          
            if distance > 0.3 and False:
                self.local.goal_action[-1] = np.eye(6)[self.goal_action]
                
            self.local.goal_action = np.tile(np.eye(6)[self.goal_action], [13,1])
    
    def update_pose(self, predicted):
        rpos = 13 / 60 * predicted.trajectory_position[1]
        rdir = predicted.trajectory_direction[1]
        
        dx = self.direction[::-1] * [1,-1]
        dy = self.direction
        npos = self.position + rpos[0] * dx + rpos[1] * dy

        angle = np.arctan2(*dy)
        rangle = np.arctan2(*rdir)

        nangle = angle + 13 / 60 * rangle
        ndir = np.array([np.sin(nangle), np.cos(nangle)])

        self.position = npos
        self.direction = ndir
        

    def get_data(self):
        angle = np.rad2deg(np.arctan2(*self.direction))

        A = Transform.from_vec([self.position[0],0,self.position[1]])
        B = Transform.from_euler([angle, 0, 0])

        transform  = A.mul(B)
        
        data = self.local.copy()
        data.bone_position = transform.apply(data.bone_position)
        data.goal_position = transform.apply(data.goal_position)

        data.trajectory_position = np.delete(
            transform.apply(
                np.insert(data.trajectory_position, 1, 0, axis = 1)
            ), 
            1, 
            axis = 1
        )
        self.transform = transform
        # env_points = env.get_collider_points(self) + [self.position[0], 0, self.position[1]]
        
        return data

    def update(self, world = None):
        if world is not None:
            self.local.environment = world.env.collide(self)

        predicted = self.controller(self)

        self.update_frame(predicted)
        self.update_pose(predicted)
        self.update_goal()
        

if __name__ == "__main__":
    # initial_pose = load_data()[0][0].bone_position
    # np.save('data/initial_pose', initial_pose)

    import matplotlib.pyplot as plt
    from src.data.InteractiveAnimation import InteractiveAnimation

    skel = Skeleton()
    anim = InteractiveAnimation(skel)
    anim.play()
    plt.show()