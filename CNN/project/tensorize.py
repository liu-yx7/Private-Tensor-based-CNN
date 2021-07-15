import scipy.io as sio
import pandas as pd
import numpy as np
import tensorly as tl
import os
import tensorflow as tf
import time


def HOSVD(matrix):
    """HOSVD 降阶"""
    # # 展开
    unfold_a = tl.unfold(matrix, mode=0)
    # print(unfold_a.shape)
    unfold_b = tl.unfold(matrix, mode=1)
    # print(unfold_b.shape)
    unfold_c = tl.unfold(matrix, mode=2)
    # print(unfold_c.shape)
    unfold_d = tl.unfold(matrix, mode=3)
    # print(unfold_d.shape)
    # print(unfold_a.dtype, unfold_b.dtype, unfold_c.dtype, unfold_d.dtype)

    # SVD
    s_a, u_a, v_a = tf.linalg.svd(unfold_a)
    # print(u_a.dtype)
    # print(u_a)
    # print(u_a.shape)
    s_b, u_b, v_b = tf.linalg.svd(unfold_b)
    # print(u_b.shape)
    s_c, u_c, v_c = tf.linalg.svd(unfold_c)
    # print(u_c.shape)
    s_d, u_d, v_d = tf.linalg.svd(unfold_d)
    # print(u_d.shape)

    # 截断
    # a, b, c, d = 13, 75, 50, 2
    u_b = u_b[:, 0:15]
    u_c = u_c[:, 0:10]
    # print(u_b.shape, u_c.shape)

    # 提取
    core_matrix_a = tl.fold(np.dot(np.transpose(u_a), unfold_a), 0, shape=(13, 75, 50, 2))
    core_matrix_b = tl.fold(np.dot(np.transpose(u_b), tl.unfold(core_matrix_a, 1)), 1, shape=(13, 15, 50, 2))
    core_matrix_c = tl.fold(np.dot(np.transpose(u_c), tl.unfold(core_matrix_b, 2)), 2, shape=(13, 15, 10, 2))
    core_matrix_d = tl.fold(np.dot(np.transpose(u_d), tl.unfold(core_matrix_c, 3)), 3, shape=(13, 15, 10, 2))

    # print(core_matrix_d.shape, core_matrix_d.dtype, type(core_matrix_d))
    core_matrix = core_matrix_d.reshape((1, 13, 15, 10, 2))

    return core_matrix


print(tf.__version__)

np.set_printoptions(threshold=np.inf)  # 显示全部数据
start = time.process_time()

'''读取数据集'''

# 生成文件路径
dir_path = r'D:\EEE\Dataset\CUAVE_sub\sub5'  # 修正
file_name = os.listdir(dir_path)
file_path = []
for i in range(0, len(file_name)):
    file_path.append(os.path.join(dir_path, file_name[i]))
# print(file_path)

# 总列表
tensor = []
label = []
length = []
# data = {'tensors': tensor, 'labels': label}

# 遍历每个文件：读取&融合
for each_file_path in file_path:
    # 读取数据
    load_data = sio.loadmat(each_file_path)
    # print(load_data.keys())
    data_length = len(load_data['labels'][0])  # 1000左右
    length.append(data_length)
    print(data_length)

    # 遍历单个文件内坐标：拼接单位长度上 video1， video2， mcff
    for index in range(data_length):

        video_sub1 = load_data['video'][0, 0][:, :, index]
        video_sub2 = load_data['video'][0, 1][:, :, index]
        video_sub = np.stack((video_sub1, video_sub2), axis=2)
        # print(video_sub.shape)
        mcff_sub = load_data['mfccs'][:, index]
        label_sub = load_data['labels'][:, index]
        # print(mcff_sub.shape)

        # 张量外积
        defusion_matrix = np.zeros([13, 75, 50, 2])  # [a, b, c, d]

        for a in range(defusion_matrix.shape[0]):
            for b in range(defusion_matrix.shape[1]):
                for c in range(defusion_matrix.shape[2]):
                    for d in range(defusion_matrix.shape[3]):
                        defusion_matrix[a, b, c, d] = video_sub[b, c, d] * mcff_sub[a]
        # print(defusion_matrix.shape)

        # 降阶
        reduced_matrix = HOSVD(defusion_matrix)
        # print(reduced_matrix.shape)

        # 写入总data数据集
        tensor.append(reduced_matrix)
        label.append(label_sub)
        # data['tensors'].append(defusion_matrix)
        # data['labels'].append(label_sub)

print(type(tensor), type(label))
print(len(tensor), len(label))


''' 写入npz文件'''
# numpy 方法
print('开始写入')

np_tensor = np.asarray(tensor)
np_label = np.asarray(label)
print(np_tensor.shape, np_label.shape)
print(np_tensor.dtype, np_label.dtype)

save_name = r'sub5.npz'  # 修正
save_dir = r'D:\EEE\Dataset\CUAVE_input'
save_path = os.path.join(save_dir, save_name)
np.savez(save_path, tensors=np_tensor, labels=np_label)

len_txt = os.path.join(save_dir, f'sub5_len.txt')  # 修正
with open(len_txt, 'w') as f:
    f.write('长度: ' + str(length))
end = time.process_time()

print(f'写入时间: {end - start} Seconds')
