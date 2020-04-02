import numpy as np
from tqdm import tqdm


# InputNorm = np.loadtxt('../Export/InputNorm.txt')
# np.save("data/input_norm.npy", InputNorm)
# OutputNorm = np.loadtxt('../Export/OutputNorm.txt')
# np.save("data/output_norm.npy", OutputNorm)

input_txt = '/media/hayk/STORAGE/dev/repos/AI4Animation/AI4Animation/SIGGRAPH_Asia_2019/Export/Input.txt'
output_txt = '/media/hayk/STORAGE/dev/repos/AI4Animation/AI4Animation/SIGGRAPH_Asia_2019/Export/Output.txt'
save_path = 'data/new32/'

f_in = open(input_txt)
f_out = open(output_txt)
n_samples = 741286

chunk_size = 28511
n_chunks = 26

chunk_size = 4073
n_chunks = 182

sample_order = np.arange(n_samples)
np.random.shuffle(sample_order)
chunks = np.split(sample_order, n_chunks)

chunk_idx = 0
i = 0

# for chunk in chunks:
#     chunk = np.sort(chunk)
#     X,Y = [],[]
    
#     for i,x in tqdm(enumerate(f_in)):
#         if i == chunk[0]:
#             chunk = chunk[1:]
#             X.append(i)

#     print (X)
#     break
# print (len(f_in.readline()))
# print (len(f_in.readline()))
# print (len(f_in.readline()))
# print (len(f_in.readline()))
# print (len(f_in.readline()))


# for idx in sample_order:
#     # f_in.seek(idx)
#     # f_out.seek(idx)

#     x = f_in.readline()
#     y = f_out.readline()

#     # print (y)
#     X.append(list(map(float, x.split(' '))))
#     Y.append(list(map(float, y.split(' '))))

#     i += 1

#     print (np.concatenate([X,Y], 1).shape)

#     if i == chunk_size:
#         # np.save(save_path + "/chunk_%d" % i, np.array(X))
#         pass

#     break

# # print (f_in.)
# # print (np.linalg.norm(X[0]))


# import numpy as np
# from tqdm import tqdm

# def process(path, save_path, chunk_size = 32768):
#     with open(path) as f:
#         i = 1
#         x = "start"

#         while x != "":
#             X = []
#             for _ in tqdm(range(chunk_size)):
#                 x = f.readline()
#                 if x == "": break
#                 X.append(list(map(float, x.split(' '))))
            
#             np.save(save_path + "/chunk%d" % i, np.array(X))
#             print ("Chunk {i} of 25 complete")
#             i += 1
                

# # InputNorm = np.loadtxt('../Export/InputNorm.txt')
# # np.save("data/input_norm.npy", InputNorm)
# # OutputNorm = np.loadtxt('../Export/OutputNorm.txt')
# # np.save("data/output_norm.npy", OutputNorm)

# process('../Export', 'data/newdata32')