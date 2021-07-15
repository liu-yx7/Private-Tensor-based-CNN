import numpy as np
import tensorflow as tf
import sklearn
from sklearn.preprocessing import StandardScaler

'''读取'''
file_name = r'D:\EEE\Dataset\tensor_input\g1_nocoding_zip.npz'
data = np.load(file_name)
print(type(data['tensors']), type(data['labels']))

tensors = data['tensors']
labels = data['labels']
print(type(tensors), type(labels))
print(tensors.dtype, labels.dtype)
print(tensors.shape, labels.shape)
print(tensors[5])
'''归一化'''
scaler = StandardScaler()
print(tensors.reshape(-1, 1).shape)
tensors_scaled = scaler.fit_transform(tensors.reshape(-1, 1)).reshape(-1, 1, 13, 15, 10, 2)
print(tensors_scaled.shape)
print(tensors_scaled[5])
# dataset = tf.data.Dataset.from_tensor_slices((data['tensors'], data['labels']))
# dataset = dataset.shuffle(200)
#
#
# for ten, lab in dataset.take(5):
#     print(f'tensor: {ten.shape}, label: {lab.shape}')
#     print(lab)
