import sys; sys.path.append('.')

import os
from tqdm import tqdm 
import numpy as np
from itertools import chain
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import time
import math

from src.utils import *
from src.data import MotionDataset
from src.nn.nets import NSM
from numpy.random import randint

import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from src.data.ShapeManager import Data

MAX_LENGTH = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def Encoder(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ELU(),
        nn.Linear(out_dim, out_dim),
        nn.ELU(),
        nn.Linear(out_dim, out_dim)
    )

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = Encoder(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = Encoder(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

teacher_forcing_ratio = 0.5

def train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = input_tensor[-1]
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di][None])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output.squeeze()

            loss += criterion(decoder_output, target_tensor[di][None])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train(encoder, decoder, n_iters, print_every=100, plot_every=10, learning_rate=0.01):
    start_time = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())
    criterion = nn.MSELoss()
    
    try:
        for iter in range(1, n_iters + 1):
            idx = randint(len(input_data.clips))
            sequence = input_data(idx).data[:,:575]

            if len(sequence) > 200:
                start = randint(len(sequence) - 200)
                input_tensor = sequence[start:start+100]
                target_tensor = sequence[start+100:start+200]
            elif len(sequence) > 100:
                split = randint(len(sequence) - 100, 100)
                input_tensor = sequence[:split]
                target_tensor = sequence[split:]
            else:
                split = randint(len(sequence) // 3, 2 * len(sequence) // 3)
                input_tensor = sequence[:split]
                target_tensor = sequence[split:]
            ##

            input_tensor = torch.from_numpy(input_tensor.astype(np.float32)).cuda()
            target_tensor = torch.from_numpy(target_tensor.astype(np.float32)).cuda()

            loss = train_step(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter == 1:
                print ("INitiall", loss)
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%d %s (%d %d%%) %.4f' % (iter, timeSince(start_time, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
    except KeyboardInterrupt:
        print ("Stopped")

    showPlot(plot_losses)
    torch.save(encoder, "models/seq2seq/encoder.pt")
    torch.save(decoder, "models/seq2seq/decoder.pt")

if __name__ == "__main__":
    data = np.load('./data/train16.npy')
    sequences = np.loadtxt('./data/TrainSequences.txt')

    input_data = Data(data, np.load('./data/input_norm.npy'), "input", sequences)
    output_data = Data(data, np.load('./data/output_norm.npy'), "output", sequences)

    hidden_size = 256
    encoder1 = EncoderRNN(575, hidden_size).to(device)
    decoder1 = DecoderRNN(hidden_size, 575).to(device)

    train(encoder1, decoder1, 75000, print_every=10)