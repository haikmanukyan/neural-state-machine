import numpy as np
import os
from tqdm import tqdm

chunk_size = 28511
n_chunks = 26

dataset_path = '/media/hayk/STORAGE/dev/repos/AI4Animation/AI4Animation/SIGGRAPH_Asia_2019/Export/samples/'
output_path = 'data/data32/'

n_samples = len(os.listdir(dataset_path))
sample_order = np.arange(n_samples)
np.random.shuffle(sample_order)

i = 0
chunk_idx = 0
X = []
for idx in tqdm(sample_order):
    x = np.load(dataset_path + 'sample_%08d.npy' % idx)
    X.append(x)
    i += 1
    if i == chunk_size:
        X = np.stack(X).astype(np.float32)
        np.save(output_path + "chunk_%02d" % chunk_idx, X)
        X = []
        i = 0
        chunk_idx += 1