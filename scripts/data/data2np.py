import numpy as np
from tqdm import tqdm


input_txt = 'path/to/Input.txt'
output_txt = 'path/to/Output.txt'
save_path = 'data/samples/'

f_in = open(input_txt)
f_out = open(output_txt)

for i,(x,y) in tqdm(enumerate(zip(f_in, f_out))):
    x = np.array(x.split(' ')).astype(np.float32)
    y = np.array(y.split(' ')).astype(np.float32)

    np.save(save_path + "sample_%08d.npy" % i, np.concatenate([x,y]))
