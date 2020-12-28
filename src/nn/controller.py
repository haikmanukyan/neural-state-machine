import sys
sys.path.append('.')

import torch
import numpy as np
from src.utils import gating
from src.env.Transform import Transform
from src.data.ShapeManager import Data

from src.nn.nets import NSM
from src.nn.seq2seq import *

class Seq2SeqController:
    def __init__(self):
        self.model = Seq2Seq()
        
        self.input_norm = np.load('data/input_norm.npy').astype(np.float32)
        self.output_norm = np.load('data/output_norm.npy').astype(np.float32)

        self.x = None

    def __call__(self, skeleton):
        # x = torch.from_numpy(skeleton.local.normed()[None]).cuda()
        
        # if self.X is None:
        #     self.X = x

        # y = self.model(x).cpu().detach().numpy()[0]
        
        x = skeleton.local.normed()[:575]
        if self.x is None:
            self.x = np.zeros((100,len(x)))
            self.x[:] = x
        self.x[:-1] = self.x[1:]
        self.x[-1] = x

        prediction_normed = self.model.predict(self.x)

        # prediction_normed = self.model.predict(input_numpy[:100, :575])

        prediction = skeleton.local.copy()
        prediction = Data(np.zeros(575), self.input_norm[:,:575], "input")
        prediction.set_normed(prediction_normed[0])

        return prediction

class NSMController:
    def __init__(self):
        self.model = torch.load('./models/best.pt').cuda()
        self.model.input_shape = [419,156,2034,2048,650]

        self.input_norm = np.load('data/input_norm.npy').astype(np.float32)
        self.output_norm = np.load('data/output_norm.npy').astype(np.float32)

    def __call__(self, skeleton):
        predicted = Data(np.zeros_like(self.output_norm[0]), self.output_norm, "output")

        x = torch.from_numpy(skeleton.local.normed()[None]).cuda()
        y = self.model(x).cpu().detach().numpy()[0]
        predicted.set_normed(y)

        return predicted
        
if __name__ == "__main__":
    seq = Seq2SeqController()