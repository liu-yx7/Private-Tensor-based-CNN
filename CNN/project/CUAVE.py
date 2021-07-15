import scipy.io as sio
import numpy as np
import pprint

np.set_printoptions(threshold=np.inf)
load_path = r'D:\EEE\Dataset\CUAVE\g01_aligned.mat'
load_data = sio.loadmat(load_path)
print('长度： ', len(load_data['labels'][0]))

# print(type(load_data))
print(load_data.keys())
# print(load_data['__header__'])
# print(load_data['video'], load_data['video'].shape)
# print(load_data['audioIndexed'], load_data['audioIndexed'].shape)
# print(load_data['mfccs'], load_data['mfccs'].shape)
pprint.pprint((load_data['labels'], load_data['labels'].shape))
# print(load_data['frameNumbers'], load_data['frameNumbers'].shape)
# print(load_data['fps'], load_data['fps'].shape)
# print(load_data['fs'], load_data['fs'].shape)