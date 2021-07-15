import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import os
import sys
from tensorflow.keras import layers

print(tf.__version__)


# 显示数据集
def show_image(img_arr):
    plt.imshow(img_arr, cmap='binary')
    plt.show()


# plt 学习曲线
def learn_curve(history):
    pd.DataFrame(history.history).plot()
    plt.grid(1)
    plt.gca().set_ylim(0, 3)
    plt.show()


# 导入数据集 minist （60000，28,28）
(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_valid = x_train_all[5000:], x_train_all[:5000]  # 拆分验证集
y_train, y_valid = y_train_all[5000:], y_train_all[:5000]

print(np.shape(x_train), np.shape(y_train))
x_train = x_train.reshape(x_train.shape[0], 784)  # 展平数据
x_valid = x_valid.reshape(x_valid.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

# 函数式API搭建神经网络：定义层输入输出，合并层，固化模型
input = tf.keras.layers.Input(shape=x_train.shape[1:])
hidden1 = tf.keras.layers.Dense(50, activation='relu')(input)
hidden2 = tf.keras.layers.Dense(50, activation='relu')(hidden1)

contact = tf.keras.layers.concatenate([input, hidden2])
output = tf.keras.layers.Dense(10, activation='softmax')(contact)

model = tf.keras.models.Model(inputs=[input], outputs=[output.reshape()])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
model.summary()

# 训练模型 history返回训练数据结果
# callback 定义： tesnorboard
logdir = r'D:\EEE\project img'
output_file = os.path.join(logdir, 'minist_modle.h5')
callbacks = [
    # tf.keras.callbacks.TensorBoard(logdir),
    # tf.keras.callbacks.ModelCheckpoint(output_file, save_best_only=1),
    # tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.01)
]
history = model.fit(x_train, y_train, batch_size=512, epochs=10, validation_data=(x_valid, y_valid))
learn_curve(history)

# 预测
score = model.evaluate(x_test, y_test)
print(score)