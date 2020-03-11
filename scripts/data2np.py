import numpy as np
from tqdm import tqdm


# InputNorm = np.loadtxt('../Export/InputNorm.txt')
# np.save("data/input_norm.npy", InputNorm)
# OutputNorm = np.loadtxt('../Export/OutputNorm.txt')
# np.save("data/output_norm.npy", OutputNorm)

input_txt = '/media/hayk/STORAGE/dev/repos/AI4Animation/AI4Animation/SIGGRAPH_Asia_2019/Export/Input.txt'
output_txt = '/media/hayk/STORAGE/dev/repos/AI4Animation/AI4Animation/SIGGRAPH_Asia_2019/Export/Output.txt'
save_path = '/media/hayk/STORAGE/dev/repos/AI4Animation/AI4Animation/SIGGRAPH_Asia_2019/Export/samples/'

f_in = open(input_txt)
f_out = open(output_txt)

for i,(x,y) in tqdm(enumerate(zip(f_in, f_out))):
    x = np.array(x.split(' ')).astype(np.float32)
    y = np.array(y.split(' ')).astype(np.float32)

    np.save(save_path + "sample_%08d.npy" % i, np.concatenate([x,y]))
