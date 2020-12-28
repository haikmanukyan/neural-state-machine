import sys
import os

sys.path.append('.')
DATA_DIR =  os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/data'

import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence

from src.data.SkeletonFrame import SkeletonFrame
from src.data.ShapeManager import Data

def clip_seq(seq, max_len = 500, min_len = 100):
    seq_ = []
    for x in seq:
        while len(x) > max_len:
            N = len(x)
            if N > max_len + min_len:
                seq_.append(x[:max_len])
                x = x[max_len:]
            else:
                seq_.append(x[:N//2])
                x = x[N//2:]
        seq_.append(x)
    return seq_

class NSMRawDataset:
    def __init__(self, clip = True, to_torch = False, only_test = False):        
        if not only_test:
            self.train_data = np.load(DATA_DIR + '/train16.npy')
        self.test_data = np.load(DATA_DIR + '/test16.npy')

        if to_torch:
            if not only_test:
                self.train_data = torch.from_numpy(self.train_data)
            self.test_data = torch.from_numpy(self.test_data)

        if clip:
            if not only_test:
                train_sequences = np.loadtxt(DATA_DIR + '/TrainSequencesClipped.txt', dtype = np.int32)
            test_sequences = np.loadtxt(DATA_DIR + '/TestSequencesClipped.txt', dtype = np.int32)
        else:
            if not only_test:
                train_sequences = np.loadtxt(DATA_DIR + '/TrainSequences.txt', dtype = np.int32)
            test_sequences = np.loadtxt(DATA_DIR + '/TestSequences.txt', dtype = np.int32)

        if not only_test:
            train_sequences = [np.where(train_sequences == i)[0] for i in np.unique(train_sequences)]
            self.train_clips = [self.train_data[i] for i in train_sequences]
        
        test_sequences = [np.where(test_sequences == i)[0] for i in np.unique(test_sequences)]
        self.test_clips = [self.test_data[i] for i in test_sequences]

        if False:
            train_sequences = clip_seq(train_sequences)
            test_sequences = clip_seq(test_sequences)

        self.input_norm = np.load(DATA_DIR + '/input_norm.npy')
        self.output_norm = np.load(DATA_DIR + '/output_norm.npy')

        self.input_mean, self.input_std = self.input_norm
        self.output_mean, self.output_std = self.output_norm

if __name__ == "__main__":
    print (NSMRawDataset())