import os
from tqdm import tqdm
import numpy as np

chunk_size = 32768

input_dir = "../data/input_extracted/"
output_dir = "../data/output_extracted/"
idx = 0

for i in range(1,24):
    for j in tqdm(range(chunk_size)):
        input_path = input_dir + "sample%d_%d.npy" % (i,j)
        output_path = output_dir + "sample%d_%d.npy" % (i,j)
        
        if os.path.exists(input_path):
            os.rename(input_path, input_dir + "sample_%08d.npy" % idx)
            os.rename(output_path, output_dir + "sample_%08d.npy" % idx)

            idx += 1