import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from src.data import Animation
from src.data import InputFrame


if __name__ == "__main__":
    start = 15000
    size = 720
    # Draw Animation
    data = np.load('data/data32/chunk12.npy')[start:start+size]
    sample = Animation(data)
    sample.draw()
    sample.play()
    
    plt.show()