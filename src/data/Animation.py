import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D

class Animation:
    def __init__(self):
        self.animations = []
        self.titles = []

    def add_frames(self, frames, title = None):
        self.animations.append(frames)
        self.titles.append(title)
    
    def add_animation(self, data, frame_type, title = None):
        frames = [frame_type(x) for x in data]
        self.add_frames(frames, title)

    def add_axis(self, fig, idx, frame, title = None):
        if frame.projection == "3d":
            ax = fig.add_subplot(idx, projection="3d")
            size = 1.2
            ax.set_xlim3d(-size,size)
            ax.set_ylim3d(-size,size)
            ax.set_zlim3d(0,2 * size)

        else:
            ax = fig.add_subplot(idx)
            ax.axis('equal')
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)

        ax.axis('off')
        ax.set_title(title)

        graph = frame.draw(ax)
        return graph

    def draw(self, format = 220):
        h = int(format / 100) * 4
        w = int((format % 100) / 10) * 4
        self.fig = plt.figure(figsize = (w,h))
        self.fig.tight_layout()
        self.graphs = []

        for i, (animation, title) in enumerate(zip(self.animations, self.titles)):
            self.graphs.append(self.add_axis(self.fig, format + 1 + i, animation[0], title))

    def update(self, frame_idx):
        ret = []
        for animation, graph in zip(self.animations, self.graphs):
            ret += animation[frame_idx].update(graph)
        return ret
        # return self.input_frames[frame_idx].update(self.input_graph) + self.output_frames[frame_idx].update(self.output_graph)

    def play(self):
        self.anim = anim.FuncAnimation(self.fig, self.update, len(self.animations[0]), interval = 1000 / 120., blit=True)

    def dump(self, path = 'data/animation.txt'):
        pose_data = np.array([frame.joint_positions for frame in self.animations[0]])
        np.savetxt(path, pose_data.reshape(pose_data.shape[0],-1), "%.3f")

    def save(self, path = 'data/animation.gif'):
        # Writer = anim.writers['ffmpeg']
        # Writer = anim.writers['imagemagick']
        # writer = Writer(fps=120, metadata=dict(artist='Me'), bitrate=1800)
        # self.anim.save('data/animation.gif', writer = writer)
        self.anim.save(path, writer = 'imagemagick', fps = 30)