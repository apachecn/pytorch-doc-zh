# -*- coding: utf-8 -*-
"""
PyTorch: Variables and autograd
-------------------------------

A fully-connected ReLU network with one hidden layer and no biases, trained to
predict y from x by minimizing squared Euclidean distance.

This implementation computes the forward pass using operations on PyTorch
Variables, and uses PyTorch autograd to compute gradients.

A PyTorch Variable is a wrapper around a PyTorch Tensor, and represents a node
in a computational graph. If x is a Variable then x.data is a Tensor giving its
value, and x.grad is another Variable holding the gradient of x with respect to
some scalar value.

PyTorch Variables have the same API as PyTorch tensors: (almost) any operation
you can do on a Tensor you can also do on a Variable; the difference is that
autograd allows you to automatically compute gradients.
"""
import torch
from torch.autograd import Variable

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # 取消注释以在GPU上运行

# N 批量大小; D_in是输入尺寸;
# H是隐藏尺寸; D_out是输出尺寸.
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机张量来保存输入和输出,并将它们包装在变量中.
# 设置requires_grad = False表示我们不需要计算渐变
# 关于这些变量在落后的过程中.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# 为权重创建随机张量,并将其包装在变量中.
# 设置requires_grad = True表示我们想要用来计算渐变
# 在落后的过程中尊重这些变量.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 正向传递:使用变量上的运算来计算预测的y; 这些
    # 与我们用于计算使用张量的正向传递完全相同,
    # 但我们不需要保留对中间值的引用,
    # 因为我们没有实现向后传递.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # 使用变量上的操作计算和打印损失.
    # 现在损失是形状变量 (1,) 并且 loss.data 是形状的张量
    # (1,); loss.data[0] 是持有损失的标量值.
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data[0])

    # 使用autograd来计算反向传递. 
    # 该调用将使用requires_grad = True来计算相对于所有变量的损失梯度.
    # 在这次调用之后 w1.grad 和 w2.grad 将是变量
    # 它们分别相对于w1和w2保存损失的梯度.
    loss.backward()

    # 使用渐变下降更新权重; w1.data 和 w2.data 是张量,
    # w1.grad 和 w2.grad 是变量并且 w1.grad.data 和 w2.grad.data 
    # 是张量.
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # 更新权重后手动将梯度归零
    w1.grad.data.zero_()
    w2.grad.data.zero_()
