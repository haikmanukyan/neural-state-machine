import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from src.data import Animation
from src.data import InputFrame
from src.nn.nets import NSM
import torch
import sys

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
    data = np.load('data/chunks32/train/chunk5.npy')[start:start+size]
    input_norm = np.load('data/input_norm.npy').astype(np.float32)
    output_norm = np.load('data/output_norm.npy').astype(np.float32)

    input_data = data[:,:5437]
    output_data = data[:,5437:]
    
    input_normed = (input_data - input_norm[0]) / (1e-5 + input_norm[1])
    output_normed = (output_data - output_norm[0]) / (1e-5 + output_norm[1])

    # model = torch.load(f'models/t{sys.argv[1]}/model_{sys.argv[2]}.pt').cuda()
    model = torch.load(sys.argv[1]).cuda()


    Fr, G, E, I, Ga, output = get_sample(input_normed, output_data)

    output_normed = model(Fr, G, E, I, Ga).cpu().detach().numpy()
    output_data = output_normed * (1e-5 + output_norm[1]) + output_norm[0]
    
    anim = Animation(input_data, output_data)
    anim.draw()
    anim.play()
    
    plt.show()
    # anim.save()