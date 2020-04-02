import numpy as np
from tqdm import tqdm 

X = np.load('data/dataset32.npy')
N = int(len(X) / 240) * 240
print (N, len(X))
X = X[:N]
X = X.reshape(int(N / 240), 240, -1)
np.random.shuffle(X)
X = X.reshape(N, -1)

Y = X[:100*240]
X = X[100*240:]

np.save('data/train32.npy', X)
np.save('data/test32.npy', Y)