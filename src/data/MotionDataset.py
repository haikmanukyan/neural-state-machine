from __future__ import print_function, division
import os
import time
import torch
import threading
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm

# Bones 276
# Trajectory 13 * (2 + 2 + 7)
# Goal 13 * (3 + 3 + 6)
# Environment 2034
# Interaction 2048
# Garing 650

A = 7
B = 276
T = 13 * (2 + 2 + A)
G = 13 * (3 + 3 + A - 1)
E = 2034
I = 2048
GA = 650

class MotionDataset(Dataset):
    """Motion dataset."""

    def __init__(self, path, input_shape, input_norm, output_norm):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_shape = input_shape
        
        data = np.load(path).astype(np.float32)
        self.input_data = data[:, :self.input_shape]
        self.output_data = data[:, self.input_shape:]

        # input_norm[input_norm == 0] = 1.
        self.input_data -= input_norm[0]
        self.input_data /= 1e-5 + input_norm[1]
        
        # output_norm[output_norm == 0] = 1.
        self.output_data -= output_norm[0]
        self.output_data /= 1e-5 + output_norm[1]


    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_data = self.input_data[idx]
        output_data = self.output_data[idx]
        
        sample = {
            'frame': torch.from_numpy(input_data[0:B + T]), # Joints + trajectory = Fr
            'goal': torch.from_numpy(input_data[B + T:B + T + G]), # Goal + Action = G
            'environment': torch.from_numpy(input_data[B+T+G:B+T+G+E]), # Environment = E
            'interaction': torch.from_numpy(input_data[B+T+G+E:B+T+G+E+I]), # INteraction = I
            'gating': torch.from_numpy(input_data[B+T+G+E+I:B+T+G+E+I+GA]), # Gating Network = Ga
            'output': torch.from_numpy(output_data)
        }

        return sample

if __name__ == "__main__":
    dataset = MotionDataset('data/dataset32.npy', input_shape = 5437)
    i = 0
    for x in dataset:
        tqdm.write ("%d %s" % (i, str(x['frame'].shape)))
        i += 1
        if i == 20: break