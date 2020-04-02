import time
from torch import nn

class Marker:
    def __init__(self, log = True):
        self.last = time.time()
        self.log = log
    def __call__(self, str = "", log = True):
        ctime = time.time()
        if log and self.log: print (str, ctime - self.last)
        self.last = ctime

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def str2list(str):
    return list(map(int,str.split(',')))