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
        
        data = np.load(path)
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
            'frame': torch.from_numpy(input_data[0:432]), # Joints + trajectory = Fr
            'goal': torch.from_numpy(input_data[432:601]), # Goal + Action = G
            'environment': torch.from_numpy(input_data[601:2635]), # Environment = E
            'interaction': torch.from_numpy(input_data[2635:4683]), # INteraction = I
            'gating': torch.from_numpy(input_data[4683:5437]), # Gating Network = Ga
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