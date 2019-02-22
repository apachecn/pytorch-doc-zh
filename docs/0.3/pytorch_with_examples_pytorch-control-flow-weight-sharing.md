# PyTorch: 动态控制流程 + 权重共享

> 译者：[@yongjay13](https://github.com/yongjay13)、[@speedmancs](https://github.com/speedmancs)
> 
> 校对者：[@bringtree](https://github.com/bringtree) 

为了展示PyTorch的动态图的强大, 我们实现了一个非常奇异的模型: 一个全连接的ReLU激活的神经网络, 每次前向计算时都随机选一个1到4之间的数字n, 然后接下来就有n层隐藏层, 每个隐藏层的连接权重共享.

```py
import random
import torch
from torch.autograd import Variable

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
 在构造函数中,我们构造了三个nn.Linear实例,我们将在正向传递中使用它们.
 """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
 对于模型的正向通道,我们随机选择0,1,2或3,
 并重复使用多次计算隐藏层表示的middle_linear模块.

 由于每个正向通道都会生成一个动态计算图,因此在定义模型的正向通道时,
 我们可以使用普通的Python控制流操作符(如循环或条件语句).

 在这里我们也看到,定义计算图时多次重复使用相同模块是完全安全的.
 这是Lua Torch的一大改进,每个模块只能使用一次.
 """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred

# N 批量大小; D_in是输入尺寸;
# H是隐藏尺寸; D_out是输出尺寸.
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机张量来保存输入和输出,并将它们包装在变量中.
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# 通过实例化上面定义的类来构建我们的模型
model = DynamicNet(D_in, H, D_out)

# 构建我们的损失函数和优化器.
# 用随机梯度下降训练这个奇怪的模型非常困难,所以我们使用动量
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
    # 正向传递:通过将x传递给模型来计算预测的y
    y_pred = model(x)

    # 计算和打印损失
    loss = criterion(y_pred, y)
    print(t, loss.data[0])

    # 零梯度执行反向传递并更新权重.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

```
