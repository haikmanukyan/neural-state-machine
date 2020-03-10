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


class MotionChunkDataset(Dataset):
    """Motion dataset."""

    def __init__(self, root_dir, input_shape, num_samples, shuffle = True, num_chunks = 2):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.num_chunks = num_chunks
        self.total_chunks = len(os.listdir(root_dir))

        self.shuffle = shuffle
        self.chunks_loading = False
        self.len = num_samples
        self.setup_epoch()

    def setup_epoch(self):
        tqdm.write("Setting up new epoch")
        self.num_used_samples = 0
        self.current_chunk  = 0
        self.chunk_order = np.arange(self.total_chunks)
        if self.shuffle:
            np.random.shuffle(self.chunk_order)

        self.active_chunks = None
        self.loaded_chunks = []

        self.load_chunks()
        self.load_chunks()

    def load_chunks(self):
        tqdm.write ("Loading a new chunk. Wait = " + str(self.chunks_loading))
        while self.chunks_loading:
            time.sleep(1)
        
        if len(self.loaded_chunks) > 0:
            self.active_chunks = self.loaded_chunks
            self.num_used_samples_chunk = 0
            self.num_loaded_samples = len(self.active_chunks)
            self.sample_order = np.arange(self.num_loaded_samples)
            if self.shuffle:
                np.random.shuffle(self.sample_order)
                
        t = threading.Thread(target=self.load_chunks_thread)
        t.start()

    def load_chunks_thread(self):
        if self.current_chunk == self.total_chunks:
            tqdm.write ("Chunks depleted, end of epoch")
            return

        self.chunks_loading = True
        self.loaded_chunks = []

        for i in range(self.num_chunks):
            data = np.load(self.root_dir + '/chunk%d.npy' % (self.chunk_order[self.current_chunk] + 1))
            self.loaded_chunks.append(data)

            self.current_chunk += 1
            if self.current_chunk == self.total_chunks: break

        self.loaded_chunks = np.concatenate(self.loaded_chunks)
        self.chunks_loading = False
        tqdm.write ("New chunk Loaded")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.num_used_samples_chunk == self.num_loaded_samples:
            self.load_chunks()
        if self.num_used_samples == self.num_samples:
            self.setup_epoch()
        
        idx = self.sample_order[self.num_used_samples_chunk]
        
        input_data = self.active_chunks[idx][:self.input_shape]
        output_data = self.active_chunks[idx][self.input_shape:]

        sample = {
            'frame': torch.from_numpy(input_data[0:432]), # Joints + trajectory = Fr
            'goal': torch.from_numpy(input_data[432:601]), # Goal + Action = G
            'environment': torch.from_numpy(input_data[601:2635]), # Environment = E
            'interaction': torch.from_numpy(input_data[2635:4683]), # INteraction = I
            'gating': torch.from_numpy(input_data[4683:5437]), # Gating Network = Ga
            'output': torch.from_numpy(output_data)
        }

        self.num_used_samples += 1
        self.num_used_samples_chunk += 1
        return sample

if __name__ == "__main__":
    dataset = MotionChunkDataset('data/data32', input_shape = 5437, num_samples = 741286, shuffle = True)
    i = 0
    for x in dataset:
        i += 1
        tqdm.write ("%d %s" % (i, str(x['frame'].shape)))
        if i == 20: break