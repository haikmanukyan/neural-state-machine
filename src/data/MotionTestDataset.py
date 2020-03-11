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


class MotionTestDataset(Dataset):
    """Motion dataset."""

    def __init__(self, root_dir, input_shape, num_samples):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.num_chunks = len(os.listdir(root_dir))
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.len = num_samples
        self.loaded_chunks = []

        for i in range(self.num_chunks):
            data = np.load(self.root_dir + '/chunk%d.npy' % i)
            self.loaded_chunks.append(data)

        self.loaded_chunks = np.concatenate(self.loaded_chunks)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        input_data = self.load_chunks[idx][:self.input_shape]
        output_data = self.load_chunks[idx][self.input_shape:]

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
    dataset = MotionTestDataset('data/data32/test', input_shape = 5437, num_samples = 741286)
    i = 0
    for x in dataset:
        tqdm.write ("%d %s" % (i, str(x['frame'].shape)))
        i += 1
        if i == 20: break