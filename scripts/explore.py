import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from src.data import Animation
from src.data import InputFrame
from src.nn.nets import NSM
import torch

def get_sample(input_data, output_data):
    return [
        torch.from_numpy(input_data[:, 0:432]).cuda(),
        torch.from_numpy(input_data[:, 432:601]).cuda(),
        torch.from_numpy(input_data[:, 601:2635]).cuda(),
        torch.from_numpy(input_data[:, 2635:4683]).cuda(),
        torch.from_numpy(input_data[:, 4683:5437]).cuda(),
        torch.from_numpy(output_data).cuda()
    ]

if __name__ == "__main__":
    start = 15000
    size = 720

    
    # Draw Animation
    data = np.load('data/data32/test/chunk1.npy')[start:start+size]
    input_data = data[:,:5437]
    output_data = data[:,5437:]

    model = torch.load('models/t15/model_5_0.pt').cuda()


    Fr, G, E, I, Ga, output = get_sample(input_data, output_data)

    output_data = model(Fr, G, E, I, Ga).cpu().detach().numpy()
    
    anim = Animation(input_data, output_data)
    anim.draw()
    anim.play()
    
    plt.show()
    # anim.save()