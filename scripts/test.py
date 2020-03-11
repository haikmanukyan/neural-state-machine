# Make module importable
import sys
sys.path.append('.')

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from src.nn import MotionNet, GatingNet



X = torch.zeros([10,20,30])
Y = torch.zeros([10,30])

print (locals())

# print ((X * Y[:,None,:]).shape)




# n_experts = 10
# n_gating = 754
# input_shape = [432, 169, 2034, 2048]
# encoders_shape = [512,128,512,128]
# network_shape = [512,512,638]

# dataset = MotionDataset('data')
# sample = dataset[0]

# sample = map(lambda x: x.cuda(), sample.values())

# Fr,G,E,I,Ga,O = sample

# gatingnet = GatingNet(n_gating, 128, n_experts)
# motionnet = MotionNet(input_shape, encoders_shape, network_shape, n_experts)

# out = motionnet(Fr, G, I, E, gatingnet(Ga))

# print (motionnet)
