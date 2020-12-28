#!/home/hayk/.pyenv/shims/python

import sys
sys.path.append('.')

from src.env.Skeleton import initial_pose, load_data
from src.viz import GLWindow
import numpy as np
from src.gl import *
from src.data.ShapeManager import Data

frame_idx = 0

def draw():
    draw_ground(2)
    global frame_idx
    draw_skeleton(data[frame_idx].bone_position)
    frame_idx += 1
    if frame_idx == len(data):
        frame_idx = 0
    window.frame_idx = frame_idx

def update():
    # glRotate(1,1,1,1)
    pass

dims = 276
norm = np.load('./data/input_norm.npy')

if len(sys.argv) > 1:
    data = np.load(sys.argv[1])
else:
    data = np.load('./data/test_clip.npy')
data = Data(data, norm[:,:dims], "input")

window = GLWindow(draw, update)
window.run()