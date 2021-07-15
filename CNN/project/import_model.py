import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import os
import sys
from tensorflow.keras import layers


print(tf.__version__)
# 导入数据集 minist （60000，28,28）
(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_valid = x_train_all[5000:], x_train_all[:5000]  # 拆分验证集
y_train, y_valid = y_train_all[5000:], y_train_all[:5000]
print(np.shape(x_train_all))
print(np.shape(y_train_all))

print(np.shape(x_train))
print(np.shape(y_train))

print(y_train[2], type(y_train), type(y_train[2]))


model_name = r'D:\EEE\projectimg\minist_modle.h5'
new_model = tf.keras.models.load_model(model_name)
new_model.summary()

# Evaluate the model
loss, acc = new_model.evaluate(x_valid, y_valid)
print(f"Restored model, accuracy: {acc}, loss: {loss}")








