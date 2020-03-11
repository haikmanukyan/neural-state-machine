import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from .InputFrame import InputFrame
from .OutputFrame import OutputFrame

class Animation:
    def __init__(self, input_data, output_data):
        self.input_frames = [InputFrame(x) for x in input_data]
        self.output_frames = [OutputFrame(x) for x in output_data]

    def add_axis(self, fig, idx, frame):
        ax = fig.add_subplot(idx, projection='3d')
        ax.set_xlim3d(-2,2)
        ax.set_ylim3d(-2,2)
        ax.set_zlim3d(0,3)
        graph = frame.draw(ax)
        return graph

    def draw(self):
        self.fig = plt.figure(figsize = (18,8))
        
        self.input_graph = self.add_axis(self.fig, 121, self.input_frames[0])
        self.output_graph = self.add_axis(self.fig, 122, self.output_frames[0])

    def update(self, frame_idx):
        return self.input_frames[frame_idx].update(self.input_graph) + self.output_frames[frame_idx].update(self.output_graph)

    def play(self):
        self.anim = anim.FuncAnimation(self.fig, self.update, len(self.input_frames), interval = 1000 / 120., blit=True)

    def save(self):
        Writer = anim.writers['ffmpeg']
        writer = Writer(fps=120, metadata=dict(artist='Me'), bitrate=1800)
        self.anim.save('data/animation.mp4', writer = writer)