import scipy.io as sio
import tensorly as tl
import numpy as np
import tensorflow as tf

# np.set_printoptions(threshold=np.inf)
load_path = r'D:\EEE\Dataset\CUAVE\g01_aligned.mat'
load_data = sio.loadmat(load_path)
print('长度： ', len(load_data['labels'][0]))

index = 5

video_sub1 = load_data['video'][0, 0][:, :, index]
video_sub2 = load_data['video'][0, 1][:, :, index]
video_sub = np.stack((video_sub1, video_sub2), axis=2)
print(video_sub.shape)
print(video_sub.dtype, type(video_sub))

mcff_sub = load_data['mfccs'][:, index]
# print(mcff_sub.shape)
print(mcff_sub.dtype, type(mcff_sub))
label_sub = load_data['labels'][:, index]
# print(label_sub.shape)
print(label_sub.dtype, type(label_sub))

# # 张量外积
a, b, c, d = 13, 75, 50, 2
defusion_matrix = np.zeros([a, b, c, d])  # [a, b, c, d]  b, c 降阶

for a in range(defusion_matrix.shape[0]):
    for b in range(defusion_matrix.shape[1]):
        for c in range(defusion_matrix.shape[2]):
            for d in range(defusion_matrix.shape[3]):
                defusion_matrix[a, b, c, d] = video_sub[b, c, d] * mcff_sub[a]

print(defusion_matrix.shape)
print(defusion_matrix.dtype, type(defusion_matrix))

'''HOSVD 降阶'''
# # 展开
unfold_a = tl.unfold(defusion_matrix, mode=0)
print(unfold_a.shape)
unfold_b = tl.unfold(defusion_matrix, mode=1)
print(unfold_b.shape)
unfold_c = tl.unfold(defusion_matrix, mode=2)
print(unfold_c.shape)
unfold_d = tl.unfold(defusion_matrix, mode=3)
print(unfold_d.shape)

print(unfold_a.dtype, unfold_b.dtype, unfold_c.dtype, unfold_d.dtype)

# SVD
S_a, U_a, V_a = tf.linalg.svd(unfold_a)
print(U_a.dtype)
print(U_a)
print(U_a.shape)
S_b, U_b, V_b = tf.linalg.svd(unfold_b)
print(U_b.shape)
S_c, U_c, V_c = tf.linalg.svd(unfold_c)
print(U_c.shape)
S_d, U_d, V_d = tf.linalg.svd(unfold_d)
print(U_d.shape)

# 截断
# a, b, c, d = 13, 75, 50, 2
U_b = U_b[:, 0:15]
U_c = U_c[:, 0:10]
print(U_b.shape, U_c.shape)

# 提取
core_matrix_a = tl.fold(np.dot(np.transpose(U_a), unfold_a), 0, shape=(13, 75, 50, 2))
core_matrix_b = tl.fold(np.dot(np.transpose(U_b), tl.unfold(core_matrix_a, 1)), 1, shape=(13, 15, 50, 2))
core_matrix_c = tl.fold(np.dot(np.transpose(U_c), tl.unfold(core_matrix_b, 2)), 2, shape=(13, 15, 10, 2))
core_matrix_d = tl.fold(np.dot(np.transpose(U_d), tl.unfold(core_matrix_c, 3)), 3, shape=(13, 15, 10, 2))

print(core_matrix_d.shape, core_matrix_d.dtype, type(core_matrix_d))
core_matrix = core_matrix_d.reshape((1, 13, 15, 10, 2))
print(core_matrix.shape, core_matrix.dtype, type(core_matrix))