import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from src.data import Animation
from src.data import InputFrame


if __name__ == "__main__":
    start = 1000
    size = 720
    # Draw Animation
    data = np.array([np.load('../data/input_extracted/sample_%08d.npy' % i) for i in range(start, start + size)])
    sample = Animation(data)
    sample.draw()
    sample.play()
    
    plt.show()