import sys
sys.path.append('.')

from src.env.Skeleton import initial_pose, load_data
from test import GLWindow
import numpy as np
from src.gl import *

frame_idx = 0

def draw():
    global frame_idx

    draw_skeleton(data[frame_idx].bone_position)
    frame_idx += 1

def update():
    # glRotate(1,1,1,1)
    pass


data,_ = load_data()

window = GLWindow(draw, update)
window.run()
