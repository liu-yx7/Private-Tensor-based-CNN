import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import os
import sys
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

print(tf.__version__)

# 导入数据集
print('开始导入')
'''pandas'''
# file_name = r'D:\EEE\Dataset\tensor_input\g1_50.csv'
# df = pd.read_csv(file_name)
# labels = df.pop('labels')
# print(df['tensor'].values.dtype)
# print(labels.values.dtype)
# dataset = tf.data.Dataset.from_tensor_slices((df['tensor'].values, labels.values))


'''numpy'''
# tensors=np_tensors, labels=np_labels
file_name = r'D:\EEE\Dataset\tensor_input\g1_nocoding_zip.npz'
data = np.load(file_name)
print(type(data['tensors']), type(data['labels']))

tensors = data['tensors']
labels = data['labels']
print(tensors.shape, labels.shape)

# 归一化
scaler = StandardScaler()
tensors = scaler.fit_transform(tensors.reshape(-1, 1)).reshape(-1, 1, 13, 15, 10, 2)
print(tensors.shape)

''' 划分训练集，验证集，测试集 比例：6:2:2'''

index = []  # index[数据集索引][0：训练集 1：测试集]
sss1 = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

for train_index, test_index in sss1.split(tensors, labels):
    print("TRAIN:", train_index, "TEST:", test_index)
    index.append([train_index, test_index])
print(len(index))

# 拆分出测试集
for each in index:
    print(type(each))
    x_train_all, x_test = tensors[each[0]], tensors[each[1]]
    y_train_all, y_test = labels[each[0]], labels[each[1]]

    print(type(x_train_all), type(y_train_all))
    print(x_train_all.shape, y_train_all.shape)
    print(x_test.shape, y_test.shape)
    # 拆分出训练集， 验证集
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    for train_ind, valid_ind in sss2.split(x_train_all, y_train_all):
        x_train, x_valid = x_train_all[train_ind], x_train_all[valid_ind]
        y_train, y_valid = y_train_all[train_ind], y_train_all[valid_ind]

        print(x_train.shape, x_valid.shape, x_test.shape)
        print(y_train.shape, y_valid.shape, y_test.shape)


# x_train_all, x_test, y_train_all, y_test = train_test_split(tensors, labels, test_size=0.1, random_state=0)
# print(x_train_all.shape, y_train_all.shape)
# print(x_test.shape, y_test.shape)
# x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.1, random_state=0)


# 转化为tf -- dataset
# dataset = tf.data.Dataset.from_tensor_slices((data['tensors'], data['labels']))
# dataset = dataset.shuffle(200)
#
#
# for ten, lab in dataset.take(5):
#     print(f'tensor: {ten.shape}, label: {lab.shape}')
#     print(lab)
