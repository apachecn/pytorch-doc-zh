# Warm-up: numpy

> 译者：[@yongjay13](https://github.com/yongjay13)、[@speedmancs](https://github.com/speedmancs)
> 
> 校对者：[@bringtree](https://github.com/bringtree) 

本例中的神经网络有一个隐藏层, 后接ReLU激活层, 并且不带偏置参数. 训练时使用欧几里得误差来学习从x到y的映射.

我们只用到了numpy, 完全手写实现神经网络, 包括前向计算, 误差计算和后向传播.

numpy的数组类型是一种通用的N维数组; 它没有内置深度学习的函数, 既不知道怎么求导, 也没有计算图的概念, 只能做一些通用的数值计算.

```py
import numpy as np

# N 是一个batch的样本数量; D_in是输入维度;
# H 是隐藏层向量的维度; D_out是输出维度.
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机的输入输出数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# 随机初始化权重参数
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # 前向计算, 算出y的预测值
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # 计算并打印误差值
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # 在反向传播中, 计算出误差关于w1和w2的导数
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # 更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

```
