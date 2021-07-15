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


''' 导入数据集'''

print('开始导入')
file_name = r'D:\EEE\Dataset\CUAVE_input\sub1.npz'
data = np.load(file_name)

# 检测长度
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

'''划分训练集，验证集，测试集'''
# train:test = 7:3
index = []  # index[数据集索引][0：训练集 1：测试集]
sss1 = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=0)

for train_index, test_index in sss1.split(tensors, labels):
    index.append([train_index, test_index])
print(len(index))

acc = []

for each in index:

    x_train, x_test = tensors[each[0]], tensors[each[1]]
    y_train, y_test = labels[each[0]], labels[each[1]]

    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # 取前三组
    for ten, lab in dataset_train.take(3):
        print(f'tensors: {ten.shape}, labels: {lab}')
    for ten, lab in dataset_test.take(3):
        print(f'tensors: {ten.shape}, labels: {lab}')

    ''' 神经网络结构'''
    # VGG16: 5conv(2,2,3,3,3)+maxpooling >> 3dense(4096,4096,1000 softmax)
    VGG16 = [
        layers.Conv3D(64, kernel_size=3, activation='relu', padding='same', input_shape=(13, 15, 10, 2)),
        layers.Conv3D(64, kernel_size=3, activation='relu', padding='same'),
        # layers.MaxPool3D(pool_size=2),
        layers.Conv3D(128, kernel_size=3, activation='relu', padding='same'),
        layers.Conv3D(128, kernel_size=3, activation='relu', padding='same'),
        # layers.MaxPool3D(pool_size=2),
        layers.Conv3D(256, kernel_size=3, activation='relu', padding='same'),
        layers.Conv3D(256, kernel_size=3, activation='relu', padding='same'),
        layers.Conv3D(256, kernel_size=3, activation='relu', padding='same'),
        # layers.MaxPool3D(pool_size=2),
        layers.Conv3D(512, kernel_size=3, activation='relu', padding='same'),
        layers.Conv3D(512, kernel_size=3, activation='relu', padding='same'),
        layers.Conv3D(512, kernel_size=3, activation='relu', padding='same'),
        # layers.MaxPool3D(pool_size=2),
        layers.Conv3D(512, kernel_size=3, activation='relu', padding='same'),
        layers.Conv3D(512, kernel_size=3, activation='relu', padding='same'),
        layers.Conv3D(512, kernel_size=3, activation='relu', padding='same'),
        # layers.MaxPool3D(pool_size=2),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(4, activation='softmax')
    ]

    model = tf.keras.Sequential(VGG16)

    ''' 编译模型 '''
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    model.summary()

    ''' 训练模型  history返回训练数据结果 '''
    history = model.fit(dataset_train, epochs=5)
    learn_curve(history)

    ''' 预测'''
    score = model.evaluate(x_test, y_test)
    acc.append(score)
    print(score)


save_name = r'D:\EEE\Dataset\history\VGG16_sub1_acc.txt'  # 修正
with open(save_name, 'w') as f:
    f.write('loss & acc: ' + str(acc))


# 函数式API
#
# input = layers.Input(shape=(13, 15, 10, 2))
# conv1 = layers.Conv3D(32, kernel_size=2, activation='relu', data_format="channels_last")(input)
# conv2 = layers.Conv3D(32, kernel_size=2, activation='relu', data_format="channels_last")(conv1)
# pool1 = layers.AveragePooling3D(pool_size=2)(conv2)
# flat1 = layers.Flatten()(pool1)
# dens1 = layers.Dense(64, activation='relu')(flat1)
# output = layers.Dense(4, activation='softmax')(dens1)

# one-hot
# model = tf.keras.models.Model(inputs=[input], outputs=[tf.reshape(output, (4, 1))])
# 无编码
# model = tf.keras.models.Model(inputs=[input], outputs=[output])





''' 训练模型  history返回训练数据结果 '''
# callback 需定义验证集
# logdir = r'D:\EEE\project_img'
# output_file = os.path.join(logdir, 'DCCM.h5')
# callbacks = [
#     tf.keras.callbacks.TensorBoard(logdir),
#     tf.keras.callbacks.ModelCheckpoint(output_file, save_best_only=1),
#     tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
# ]

# history = model.fit(dataset, epochs=3)
# learn_curve(history)


''' 预测'''
# score = model.evaluate(x_test, y_test)
# print(score)
