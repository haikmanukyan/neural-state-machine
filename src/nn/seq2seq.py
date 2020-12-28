import sys
sys.path.append('.')

import numpy as np
import torch
from torch import nn

from src.data.ShapeManager import Data
from src.data.SkeletonFrame import SkeletonFrame
from scripts.train_seq2seq import *
from src.data.Animation import Animation
import matplotlib.animation as anim

import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Seq2Seq:
    def __init__(self):
        self.MAX_LENGTH = 100
        self.input_size = 575
        self.encoder = torch.load('models/seq2seq/encoder.pt')
        self.decoder = torch.load('models/seq2seq/decoder.pt')

    def encode(self, input_tensor):
        encoder_hidden = self.encoder.initHidden()
        input_length = input_tensor.size(0)
        encoder_outputs = torch.zeros(MAX_LENGTH, self.encoder.hidden_size).cuda()

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
        return encoder_outputs, encoder_hidden

    def decode(self, decoder_input, decoder_hidden, encoder_outputs):
        decoder_outputs = torch.zeros(self.MAX_LENGTH, self.input_size).cuda()

        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output.squeeze().detach()
            decoder_outputs[di] = decoder_output

        return decoder_outputs
    
    def predict(self, input_data):
        input_tensor = torch.from_numpy(input_data.astype(np.float32)).cuda()
        encoder_outputs, encoder_hidden = self.encode(input_tensor)        
        prediction_data = self.decode(input_tensor[-1], encoder_hidden, encoder_outputs)
        return prediction_data.cpu().detach().numpy()
    
    def predict_next(self, input_data):
        pass