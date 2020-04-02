import numpy as np
from tqdm import tqdm 

for i in range(1,24):
    print ("Chunk %d of 23" % i)
    X = np.load('data/output/chunk%d.npy' % i)
    for j,x in enumerate(tqdm(X)):
        np.save("data/output_extracted/sample%d_%d" % (i,j), x)

