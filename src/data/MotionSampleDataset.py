from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm

class MotionSampleDataset(Dataset):
    """Motion dataset."""

    def __init__(self, root_dir, preload = False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.len = len(os.listdir(root_dir + '/input_extracted'))
        self.preload = preload
        self.root_dir = root_dir
        
        if preload:
            self.input_preloaded = []
            self.output_preloaded = []
            for idx in tqdm(range(self.len)):
                input_data = np.load(self.root_dir + '/input_extracted/sample_%08d.npy' % idx).astype(np.float32)
                output_data = np.load(self.root_dir + '/output_extracted/sample_%08d.npy' % idx).astype(np.float32)
            
                self.input_preloaded.append(input_data)
                self.output_preloaded.append(output_data)


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.preload:
            input_data = self.input_preloaded[idx]
            output_data = self.output_preloaded[idx]
        else:
            input_data = np.load(self.root_dir + '/input_extracted/sample_%08d.npy' % idx).astype(np.float32)
            output_data = np.load(self.root_dir + '/output_extracted/sample_%08d.npy' % idx).astype(np.float32)
        
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
    dataset = MotionSampleDataset('../../data')
    print (dataset[19])