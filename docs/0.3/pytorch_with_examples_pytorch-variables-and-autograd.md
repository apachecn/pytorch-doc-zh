# PyTorch: 变量和autograd

> 译者：[@yongjay13](https://github.com/yongjay13)、[@speedmancs](https://github.com/speedmancs)
> 
> 校对者：[@bringtree](https://github.com/bringtree) 

本例中的全连接神经网络有一个隐藏层, 后接ReLU激活层, 并且不带偏置参数. 训练时通过最小化欧式距离的平方, 来学习从x到y的映射.

在实现中, 我们将使用PyTorch变量的函数来进行前向计算, 并用PyTorch的autograd计算梯度

PyTorch变量是PyTorch张量的封装, 表示计算图中的一个节点. 如果x是变量, 那么x.data就是 表示其值的张量, 而x.grad则是另一个变量, 其中包含某个标量关于x的梯度.

PyTorch变量的API和张量是一样的: 几乎所有Tensor上能做的操作, 你在变量上也可以调用. 区别 在于用变量时, autograd可以自动计算梯度.

```py
import torch
from torch.autograd import Variable

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # 取消注释以在GPU上运行

# N 批量大小; D_in是输入尺寸;
# H是隐藏尺寸; D_out是输出尺寸.
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机张量来保存输入和输出,并将它们包装在变量中.
# 设置requires_grad = False, 因为在后向传播时, 我们并不需要计算关于这些变量的梯度
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# 为权重创建随机张量,并将其包装在变量中.
# 设置requires_grad = True, 因为在后向传播时, 我们需要计算关于这些变量的梯度
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

    # 使用梯度下降更新权重; w1.data 和 w2.data 是张量,
    # w1.grad 和 w2.grad 是变量并且 w1.grad.data 和 w2.grad.data
    # 是张量.
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # 更新权重后手动将梯度归零
    w1.grad.data.zero_()
    w2.grad.data.zero_()

```
