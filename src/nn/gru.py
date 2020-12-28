import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence

class GRUNet(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, n_layers):
        super(GRUNet, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dims, hidden_dims, n_layers)
        self.fc = nn.Linear(hidden_dims,output_dims)
        
    def forward(self, input, h = None):
        input_packed = pack_sequence(input, enforce_sorted = False).float()
        
        out_packed, h = self.gru(input_packed, h)
        out_padded, lengths = pad_packed_sequence(out_packed)
        out_padded = self.fc(out_padded)
        out = [x[:l] for x,l in zip(out_padded.transpose(1,0), lengths)]

        return out, h
    
    def initHidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_dims))