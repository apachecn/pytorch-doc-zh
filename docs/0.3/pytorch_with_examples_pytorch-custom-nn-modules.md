# PyTorch: 定制化nn模块

> 译者：[@yongjay13](https://github.com/yongjay13)、[@speedmancs](https://github.com/speedmancs)
> 
> 校对者：[@bringtree](https://github.com/bringtree) 

本例中的全连接神经网络有一个隐藏层, 后接ReLU激活层, 并且不带偏置参数. 训练时通过最小化欧式距离的平方, 来学习从x到y的映射.

在实现中我们将定义一个定制化的模块子类. 如果已有模块串起来不能满足你的复杂需求, 那么你就能以这种方式来定义自己的模块。

```py
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

# N 批量大小; D_in是输入尺寸;
# H是隐藏尺寸; D_out是输出尺寸.
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

    # 梯度置零, 执行反向传递并更新权重.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

```
