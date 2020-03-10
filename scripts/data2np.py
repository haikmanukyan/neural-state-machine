import numpy as np
from tqdm import tqdm

def process(path, save_path, chunk_size = 32768):
    with open(path) as f:
        i = 1
        x = "start"

        while x != "":
            X = []
            for _ in tqdm(range(chunk_size)):
                x = f.readline()
                if x == "": break
                X.append(list(map(float, x.split(' '))))
            
            np.save(save_path + "/chunk%d" % i, np.array(X))
            print ("Chunk {i} of 25 complete")
            i += 1
                

# InputNorm = np.loadtxt('../Export/InputNorm.txt')
# np.save("data/input_norm.npy", InputNorm)
# OutputNorm = np.loadtxt('../Export/OutputNorm.txt')
# np.save("data/output_norm.npy", OutputNorm)

process('../Export/Input.txt', 'data/input')
process('../Export/Output.txt', 'data/output') 