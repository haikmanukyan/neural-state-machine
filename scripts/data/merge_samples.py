import numpy as np
import os
from tqdm import tqdm

n = len(os.listdir('data/input_samples'))
p_in = 'data/input_samples/sample_%08d.npy'
p_out = 'data/output_samples/sample_%08d.npy'

X = []

for i in tqdm(range(n)):
    x = np.load(p_in % i).astype(np.float32)
    y = np.load(p_out % i).astype(np.float32)
    X.append(np.concatenate([x,y]))

X = np.array(X)
np.save('data/dataset32.npy', X)
print (X.shape, X.dtype)