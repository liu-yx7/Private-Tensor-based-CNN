import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import os
import sys
from tensorflow.keras import layers


# 显示数据集
def show_image(img_arr):
    plt.imshow(img_arr, cmap='binary')
    plt.show()


# plt 学习曲线
def learn_curve(history):
    pd.DataFrame(history.history).plot()
    plt.grid(1)
    plt.gca().set_ylim(0, 1)
    plt.show()


print(tf.__version__)
# 导入数据集 minist （60000，28,28）
(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_valid = x_train_all[5000:], x_train_all[:5000]  # 拆分验证集
y_train, y_valid = y_train_all[5000:], y_train_all[:5000]
print(np.shape(x_train))
# 数据集归一化 （对于高斯分布）

# show_image(x_train[0])

# 构建层， 层堆叠
# cnn = [
#     layers.Flatten(input_shape=[28, 28]),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ]

model = tf.keras.Sequential()

model.add(layers.Flatten(input_shape=[28, 28]))
for each in range(10):  # BN层放置问题：激活函授之前&之后
    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
model.add(layers.Dense(10, activation='softmax'))

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
model.summary()

# 训练模型 history返回训练数据结果
# callback 定义： tesnorboard
logdir = r'D:\EEE\project_img'
output_file = os.path.join(logdir, 'minist_model.h5')
callbacks = [
    tf.keras.callbacks.TensorBoard(logdir),
    tf.keras.callbacks.ModelCheckpoint(output_file, save_best_only=1),
    # tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.01)
]
history = model.fit(x_train, y_train, batch_size=512, epochs=1, validation_data=(x_valid, y_valid), callbacks=callbacks)

# learn_curve(history)
# 预测
score = model.evaluate(x_test, y_test)

counter = 1
logdir = r'D:\EEE\Dataset\CUAVE_sub\sub1\history'
acc_txt = os.path.join(logdir, f'epoch{counter}_acc.txt')
with open(acc_txt, 'w') as f:
    f.write('loss & acc: ' + str(score))
print(score)
