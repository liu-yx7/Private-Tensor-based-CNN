import numpy as np


# 定义张量卷积
def TC(input, kernel):
    # 数据类型？？ ndarray，tensor

    # input(C): h1*h2*..*hn    kernel(B): l1*l2*..*ln   hn >= ln
    h = input.shape
    l = kernel.shape

    # 读取阶数， 判断是否一致
    assert len(h) != len(l), "卷积对象阶数不同"

    order = len(h)
    k = []

    # 输出维度：
    for each in range(order):
        k[each] = h[each] - l[each] + 1

    result = np.zeros(k)
    # if 选择， 阶数依据
    if order == 4:
        pass

    if order == 5:
        pass
