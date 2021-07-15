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
print(np.shape(x_train), np.shape(x_valid), np.shape(x_test))


# 封装keras模型，转换为sklearn模型
def build_model(hidden_layers=1,
                layer_size=30,
                learning_rate=3e-3):
    model = tf.keras.models.Sequential()

    model.add(layers.Dense(layer_size, activation='relu',
                           input_shape=x_train.shape[1:]))
    for each in range(hidden_layers):
        model.add(layers.Dense(layer_size, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

    return model


skl_model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_model)

# 定义超参数网格
from scipy.stats import reciprocal

param_distribution = {
    'hidden_layers': [1, 2],
    'layer_size': np.arange(1, 5).tolist(),  # 字典列表，用.tolist()。 list()引起报错！！
    'learning_rate': reciprocal.rvs(1e-4, 1e-3, size=10).tolist()  # scipy中分布： f(x) = 1/(x*log(b/a))  a <= x <= b
}

from sklearn.model_selection import RandomizedSearchCV

random_search_cv = RandomizedSearchCV(skl_model,
                                      param_distribution,
                                      n_iter=10)
random_search_cv.fit(x_train, y_train, epochs=2, validation_data=(x_valid, y_valid))

print(random_search_cv.best_params_)
print(random_search_cv.best_score_)
print(random_search_cv.best_estimator_)

# 导出最优模型，测试集验证
model = random_search_cv.best_estimator_.model
model.evaluate(x_test, y_test)
