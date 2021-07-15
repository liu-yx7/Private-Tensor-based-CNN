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


# plt 学习曲线
def learn_curve(history):
    pd.DataFrame(history.history).plot()
    plt.grid(1)
    plt.gca().set_ylim(0, 3)
    plt.show()


print(tf.__version__)

''' 神经网络结构'''

# 函数式API

input = layers.Input(shape=(13, 15, 10, 2))
conv1 = layers.Conv3D(32, kernel_size=2, activation='relu', data_format="channels_last")(input)
conv2 = layers.Conv3D(32, kernel_size=2, activation='relu', data_format="channels_last")(conv1)
pool1 = layers.AveragePooling3D(pool_size=2)(conv2)
flat1 = layers.Flatten()(pool1)
dens1 = layers.Dense(64, activation='relu')(flat1)
output = layers.Dense(4, activation='softmax')(dens1)

# one-hot
# model = tf.keras.models.Model(inputs=[input], outputs=[tf.reshape(output, (4, 1))])
# 无编码
model = tf.keras.models.Model(inputs=[input], outputs=[output])

''' 编译模型 '''
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
model.summary()

''' 导入数据集'''

print('开始导入')
file_name = r'D:\EEE\Dataset\tensor_input\g1_nocoding_zip.npz'
data = np.load(file_name)

if len(data['tensors']) == len(data['labels']):
    print(len(data['tensors']))
else:
    print('数据，标签长度不一致！')

tensors = data['tensors']
labels = data['labels']

# 归一化
scaler = StandardScaler()
tensors = scaler.fit_transform(tensors.reshape(-1, 1)).reshape(-1, 1, 13, 15, 10, 2)
print(tensors.shape)

# one-hot 编码

''' 划分训练集，验证集，测试集 '''
# 比例：6:2:2
index = []  # index[数据集索引][0：训练集 1：测试集]
sss1 = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

for train_index, test_index in sss1.split(tensors, labels):
    index.append([train_index, test_index])

print(len(index))

counter = 1  # 计数器
# 拆分出测试集
for each in index:

    x_train_all, x_test = tensors[each[0]], tensors[each[1]]
    y_train_all, y_test = labels[each[0]], labels[each[1]]

    # 拆分出训练集， 验证集
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    for train_ind, valid_ind in sss2.split(x_train_all, y_train_all):
        x_train, x_valid = x_train_all[train_ind], x_train_all[valid_ind]
        y_train, y_valid = y_train_all[train_ind], y_train_all[valid_ind]

    print(x_train.shape, x_valid.shape, x_test.shape)
    print(y_train.shape, y_valid.shape, y_test.shape)

    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    ''' 训练模型  history返回训练数据结果 '''

    # callback 需定义验证集
    logdir = r'D:\EEE\Dataset\history\sub1'
    output_dir = os.path.join(logdir, f'epoch{counter}')
    output_file = os.path.join(logdir, f'DCCM_epoch{counter}.h5')
    callbacks = [
        tf.keras.callbacks.TensorBoard(output_dir),
        tf.keras.callbacks.ModelCheckpoint(output_file, save_best_only=1),
        tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
    ]

    # 训练
    history = model.fit(dataset_train, epochs=3, validation_data=dataset_valid, callbacks=callbacks)

    ''' 预测'''
    score = model.evaluate(dataset_test)

    acc_txt = os.path.join(output_dir, f'epoch{counter}_acc.txt')
    with open(acc_txt, 'w') as f:
        f.write('loss & acc: ' + str(score))

    counter += 1
