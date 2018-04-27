# -*- coding: utf-8 -*-
"""
PyTorch: Custom nn Modules
--------------------------

A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.

This implementation defines the model as a custom Module subclass. Whenever you
want a model more complex than a simple sequence of existing Modules you will
need to define your model this way.
"""
import torch
from torch.autograd import Variable


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        在构造函数中,我们实例化两个nn.Linear模块并将它们分配为成员变量.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        在forward函数中,我们接受一个变量的输入数据,我们必须返回一个变量的输出数据.
        我们可以使用构造函数中定义的模块以及变量上的任意运算符.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

# N 是一个batch的样本数量; D_in是输入维度;
# H 是隐藏层向量的维度; D_out是输出维度.
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机张量来保存输入和输出,并将它们包装在变量中. 
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# 通过实例化上面定义的类来构建我们的模型
model = TwoLayerNet(D_in, H, D_out)

# 构建我们的损失函数和优化器. 
# 对SGD构造函数中的model.parameters()的调用将包含作为模型成员的两个nn.Linear模块的可学习参数.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # 正向传递：通过将x传递给模型来计算预测的y
    y_pred = model(x)

    # 计算和打印损失
    loss = criterion(y_pred, y)
    print(t, loss.data[0])

    # 零梯度执行反向传递并更新权重.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
