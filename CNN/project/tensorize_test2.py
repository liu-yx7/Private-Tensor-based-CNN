import scipy.io as sio
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import time

np.set_printoptions(threshold=np.inf)  # 显示全部数据

# 文件路径
dir_path = r'D:\EEE\Dataset\CUAVE_sub'
file_name = os.listdir(dir_path)
file_path = []
for i in range(0, len(file_name)):
    file_path.append(os.path.join(dir_path, file_name[i]))
# print(file_path)

tensor = []
label = []
# data = {'tensor': defusion_matrix, 'labels': label}

# 每个文件：读取&融合
for each_file_path in file_path:
    # 读取数据
    load_data = sio.loadmat(each_file_path)
    # print(load_data.keys())
    data_length = 50
    # len(load_data['labels'][0])  # 1000左右
    print(data_length)

    # 单个文件内：拼接单位长度上 video1， video2， mcff
    for index in range(data_length):
        video_sub1 = load_data['video'][0, 0][:, :, index]
        video_sub2 = load_data['video'][0, 1][:, :, index]
        video_sub = np.stack((video_sub1, video_sub2), axis=2)
        # print(video_sub.shape)
        mcff_sub = load_data['mfccs'][:, index]
        # print(mcff_sub.shape)
        label_sub = load_data['labels'][:, index]

        # 张量外积
        defusion_matrix = np.zeros([13, 75, 50, 2])  # [a, b, c, d]

        for a in range(defusion_matrix.shape[0]):
            for b in range(defusion_matrix.shape[1]):
                for c in range(defusion_matrix.shape[2]):
                    for d in range(defusion_matrix.shape[3]):
                        defusion_matrix[a, b, c, d] = video_sub[b, c, d] * mcff_sub[a]
        # print(defusion_matrix.shape)

        # 写入总data数据集

        tensor.append(defusion_matrix.reshape((1, 13, 75, 50, 2)))
        label.append(label_sub)

    # label one-hot 编码
    '''tf'''
    # label = tf.one_hot(label, depth=4).numpy().reshape([data_length, 4, 1])
    '''sklearn '''
    # lb = sklearn.preprocessing.LabelBinarizer()
    # lb.fit(label)
    # label = lb.transform(label)
    print(label.shape, type(label))
    # label = label.reshape([data_length, 4, 1])
    # print(label.shape, type(label))
    print(label[40])







