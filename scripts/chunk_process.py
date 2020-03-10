import numpy as np
from tqdm import tqdm 

for i in range(1,24):
    print ("Chunk %d of 23" % i)
    X = np.load('data/input/chunk%d.npy' % i)
    Y = np.load('data/output/chunk%d.npy' % i)

    X = np.concatenate([X,Y],1)
    np.save('data/input16/chunk%d.npy' % i, X.astype(np.float16))
