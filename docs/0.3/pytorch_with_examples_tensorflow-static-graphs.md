# TensorFlow: 静态图

> 译者：[@yongjay13](https://github.com/yongjay13)、[@speedmancs](https://github.com/speedmancs)
> 
> 校对者：[@bringtree](https://github.com/bringtree) 

本例中的全连接神经网络有一个隐藏层, 后接ReLU激活层, 并且不带偏置参数. 训练时通过最小化欧式距离的平方, 来学习从x到y的映射.

在实现中, 我们会用基本的TensorFlow操作来建立一个计算图, 随后多次执行这个图来训练网络.

TensorFlow和PyTorch有一个很大的区别, 就是TensorFlow用的是静态计算图, 而PyTorch则用动态计算图.

用TensorFlow我们先建立计算图, 然后在多次执行过程中, 计算图固定不变.

```py
import tensorflow as tf
import numpy as np

# 首先我们设置计算图:

# N 批量大小; D_in是输入尺寸;
# H是隐藏尺寸; D_out是输出尺寸.
N, D_in, H, D_out = 64, 1000, 100, 10

# 为输入数据和目标数据创建占位符;
# 当我们执行图时,这些将被填充真实的数据.
x = tf.placeholder(tf.float32, shape=(None, D_in))
y = tf.placeholder(tf.float32, shape=(None, D_out))

# 为权重创建变量并用随机数据初始化它们.
# 一个TensorFlow变量在图的执行中保持其值.
w1 = tf.Variable(tf.random_normal((D_in, H)))
w2 = tf.Variable(tf.random_normal((H, D_out)))

# 正向传递:使用TensorFlow Tensors上的运算来计算预测的y.
# 请注意此代码实际上并未执行任何数字操作;
# 它只是设置我们稍后将执行的计算图.
h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

# 使用TensorFlow张量上的操作计算损失
loss = tf.reduce_sum((y - y_pred) ** 2.0)

# 计算相对于w1和w2的损失梯度.
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# 使用梯度下降更新权重.
# 要实际更新权重,我们需要在执行图时评估new_w1和new_w2.
# 请注意,在TensorFlow中,更新权值的行为是计算图的一部分
# 在PyTorch中,这发生在计算图之外.
learning_rate = 1e-6
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# 现在我们已经构建了计算图,所以我们输入一个TensorFlow会话来实际执行图.
with tf.Session() as sess:
    # 运行一次图形初始化变量w1和w2.
    sess.run(tf.global_variables_initializer())

    # 创建包含输入x和目标y的实际数据的numpy数组
    x_value = np.random.randn(N, D_in)
    y_value = np.random.randn(N, D_out)
    for _ in range(500):
        # 多次执行图. 每次执行时,
        # 我们都想将x_value绑定到x,将y_value绑定到y,用feed_dict参数指定.
        # 每次我们执行图时,我们都想计算损失值new_w1 和 new_w2;
        # 这些张量的值作为numpy数组返回.
        loss_value, _, _ = sess.run([loss, new_w1, new_w2],
                                    feed_dict={x: x_value, y: y_value})
        print(loss_value)

```
