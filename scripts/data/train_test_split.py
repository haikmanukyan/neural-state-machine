import numpy as np
from tqdm import tqdm 

seq = np.loadtxt('./data/Sequences.txt', np.int32)
test_ratio = 0.1
n_seq = max(seq)
test_seq = np.random.choice(n_seq, int(0.1 * n_seq), False)
test_seq.sort()

test_ind = np.concatenate([np.where(seq == x)[0] for x in test_seq])


X = np.load('data/data16.npy')
test_mask = np.zeros(len(X), np.bool)
test_mask[test_ind] = True
train_mask = ~test_mask
print (sum(train_mask), sum(test_mask))

np.save('data/train16.npy', X[train_mask])
np.save('data/test16.npy', X[test_mask])

# N = int(len(X) / 240) * 240
# print (N, len(X))
# X = X[:N]
# X = X.reshape(int(N / 240), 240, -1)
# np.random.shuffle(X)
# X = X.reshape(N, -1)

# Y = X[:100*240]
# X = X[100*240:]

# np.save('data/train16.npy', X)
# np.save('data/test16.npy', Y)