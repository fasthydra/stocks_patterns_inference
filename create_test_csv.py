import numpy as np

src_file_path = "data/sber_clst_2021_1_1.30_10.npy"
test_file_path = "test_inference.csv"

array = np.load(src_file_path)
array = array[:min(len(array), 10)]
array = np.reshape(array, (array.shape[0], -1))
with open(test_file_path, 'w') as f:
    np.savetxt(f, array, delimiter=',')