# 通过示例学习 PyTorch

> 译者：[runzhi214](https://github.com/runzhi214)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/Learning_PyTorch/learning_pytorch_with_examples.md/>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/pytorch_with_examples.html>

**作者**：[Justin Johnson](https://github.com/jcjohnson/pytorch-examples)

> 注意:
> 这是我们老版本PyTorch的教程。你可以在[学习基本知识](../Introduction_to_PyTorch/learn_the_basics.md)中查看我们最新的初学者内容

本教程通过独立的示例介绍 [PyTorch](https://github.com/pytorch/pytorch) 的基本概念。

PyTorch 的核心是提供两个主要功能：

*   n 维张量，类似于 NumPy，但可以在 GPU 上运行
*   用于构建和训练神经网络的自动微分

我们将使用将三阶多项式拟合`y = sin(x)`的问题作为运行示例。 该网络将具有四个参数，并且将通过使网络输出与实际输出之间的欧几里德距离最小化来进行梯度下降训练，以适应随机数据。

注意

您可以在[本页](#示例)浏览各个示例。

## 张量

### 预热：NumPy

在介绍 PyTorch 之前，我们将首先使用 numpy 实现网络。

Numpy 提供了一个 n 维数组对象，以及许多用于操纵这些数组的函数。 Numpy 是用于科学计算的通用框架。 它对计算图，深度学习或梯度一无所知。 但是，通过使用 numpy 操作手动实现网络的前向和后向传递，我们可以轻松地使用 numpy 使三阶多项式拟合正弦函数：

```py
# -*- coding: utf-8 -*-
import numpy as np
import math

# 创建随机输入值和输出数据
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# 随机初始化权重
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # 前向传递: 计算y的预测值
    # y = a + b x + c x^2 + d x^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # 计算并输出损失
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # 反向传播来计算相对于损失的a, b, c, d的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # 更新权重
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')

```

### PyTorch：张量

Numpy 是一个很棒的框架，但是它不能利用 GPU 来加速其数值计算。 对于现代深度神经网络，GPU 通常会提供 [50 倍或更高](https://github.com/jcjohnson/cnn-benchmarks)的加速，因此遗憾的是，numpy 不足以实现现代深度学习。

在这里，我们介绍最基本的 PyTorch 概念：**张量**。 PyTorch 张量在概念上与 numpy 数组相同：张量是 n 维数组，PyTorch 提供了许多在这些张量上进行操作的函数。 在幕后，张量可以跟踪计算图和梯度，但它们也可用作科学计算的通用工具。

与 numpy 不同，PyTorch 张量可以利用 GPU 加速其数字计算。 要在 GPU 上运行 PyTorch 张量，您只需要指定正确的设备即可。

在这里，我们使用 PyTorch 张量将三阶多项式拟合为正弦函数。 像上面的 numpy 示例一样，我们需要手动实现通过网络的正向和反向传递：

```py
# -*- coding: utf-8 -*-

import torch
import math

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # 取消注释这一行来在GPU上运行

# 创建随机输入值和输出数据
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# 随机初始化权重
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # 前向传递: 计算y的预测值
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # 计算并输出损失
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # 反向传播来计算相对于损失的a, b, c, d的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # 用梯度下降来更新权重
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')

```

## Autograd

### PyTorch：张量和 Autograd

在上述示例中，我们必须手动实现神经网络的前向和后向传递。 对于小型的两层网络，手动实现反向传递并不是什么大问题，但是对于大型的复杂网络来说，可以很快变得非常麻烦。

幸运的是，我们可以使用[自动微分](https://en.wikipedia.org/wiki/Automatic_differentiation)来自动计算神经网络中的反向传递。 PyTorch 中的 **Autograd** 包正是提供了此功能。 使用 Autograd 时，网络的正向传播将定义**计算图**； 图中的节点为张量，边为从输入张量产生输出张量的函数。 然后通过该图进行反向传播，可以轻松计算梯度。

这听起来很复杂，在实践中非常简单。 每个张量代表计算图中的一个节点。 如果`x`是具有`x.requires_grad=True`的张量，则`x.grad`是另一个张量，其保持`x`相对于某个标量值的梯度。

在这里，我们使用 PyTorch 张量和 Autograd 来实现我们的正弦波与三阶多项式示例； 现在我们不再需要通过网络手动实现反向传递：

```py
# -*- coding: utf-8 -*-
import torch
import math

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")  # 取消注释这一行来在GPU上运行

# 创建存储输入值和输出值的张量
# 默认情况下，requires_grad=False，意味着我们不需要在反向传递过程中计算相对于这些张量的梯度
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# 创建随机权重张量。对于一个三阶多项式，我们需要4个权重值
# y = a + b x + c x^2 + d x^3
# 设置requires_grad=True 意味着我们想要在反向产地过程中
# 计算相对于这些张量的梯度。
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    # 前向传递: 通过对张量进行运算来计算y的预测值
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # 通过对张量运算计算并输出损失
    # 现在损失值是一个 (1,) 形状的张量
    # loss.item() 获取loss中持有的标量值
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # 用autograd来计算反向传递。这个调用会计算相对于所有带requires_grad=True张量的损失梯度
    # 在这次调用之后，a.grad, b.grad. c.grad and d.grad会分别成为持有相对于
    # a, b, c, d的损失梯度的张量
    loss.backward()

    # 手动用梯度下降更新权重。代码包裹在torch.no_grad()中
    # 因为权重值具有requires_grad=True，但是在autograd中我们不需要跟踪这个
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # 在权重更新之后手动清零梯度
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')

```

### PyTorch：定义新的 Autograd 函数

在幕后，每个原始的 Autograd 运算符实际上都是在张量上运行的两个函数。 **正向**函数从输入张量计算输出张量。 **反向**函数接收相对于某个标量值的输出张量的梯度，并计算相对于相同标量值的输入张量的梯度。

在 PyTorch 中，我们可以通过定义`torch.autograd.Function`的子类并实现`forward`和`backward`函数来轻松定义自己的 Autograd 运算符。 然后，我们可以通过构造实例并像调用函数一样调用新的 Autograd 运算符，并传递包含输入数据的张量。

在此示例中，我们将模型定义为 $y = a + bP_3(c + dx)$ 而不是 $y = a + bx + cx^2 + dx^3$ ，其中 $P_3(x) = \frac{1}{2}(5x^3 - 3x)$ 是三次的[勒让德多项式](https://en.wikipedia.org/wiki/Legendre_polynomials)。 我们编写了自己的自定义 Autograd 函数来计算 $P_3$ 的前进和后退，并使用它来实现我们的模型：

```py
# -*- coding: utf-8 -*-
import torch
import math

class LegendrePolynomial3(torch.autograd.Function):
    """
    我们可以通过子类化torch.autograd.Function
    并实现对张量操作的forward and backward 方法
    来实现我们自己自定义的Function
    """

    @staticmethod
    def forward(ctx, input):
        """
        在前向传递中我们接收一个包含输入值的张量并返回一个包含输出值的张量。
        ctx一个用来为反向计算缓存信息的上下文对象。你可以使用 ctx.save_for_backward 方法
        来缓存任意对象给反向传递中使用。
        """
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        在反向传递中，我们接收一个包含相对于输出值的损失梯度的张量，
        我们需要计算相对于输入值的损失梯度
        """
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 - 1)

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")  # 取消注释这一行来在GPU上运行

# 创建存储输入值和输出值的张量
# 默认情况下，requires_grad=False，意味着我们不需要在反向传递过程中计算相对于这些张量的梯度
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# 创建随机权重张量。在这个例子中，我们需要4个权重值
# y = a + b * P3(c + d * x)，这些权重值需要被初始化为
# 离正确结果不能太远的值来保证收敛
# 设置requires_grad=True 意味着我们想要在反向产地过程中
# 计算相对于这些张量的梯度。
a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

learning_rate = 5e-6
for t in range(2000):
    # 我们用 Function.apply 方法来应用我们的函数。我们为它设置'P3'的别名。
    P3 = LegendrePolynomial3.apply

    # 前向传递: 运算出预测值y_pred；我们用我们自定义的
    # autograd运算来计算P3
    y_pred = a + b * P3(c + d * x)

    # 计算并输出损失
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # 用 autograd 来计算反向传递
    loss.backward()

    # 用梯度下降来更新权重
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # 在权重更新之后手动清零梯度
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)')

```

## `nn`模块

### PyTorch：`nn`

计算图和 Autograd 是定义复杂运算符并自动求导的非常强大的范例。 但是对于大型神经网络，原始的 Autograd 可能会太低级。

在构建神经网络时，我们经常想到将计算安排在**层**中，其中某些层具有**可学习的参数**，这些参数会在学习期间进行优化。

在 TensorFlow 中，像 [Keras](https://github.com/fchollet/keras) ， [TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) 和 [TFLearn](http://tflearn.org/) 之类的包在原始计算图上提供了更高层次的抽象，可用于构建神经网络。

在 PyTorch 中，`nn`包也达到了相同的目的。 `nn`包定义了一组**模块**，它们大致等效于神经网络层。 模块接收输入张量并计算输出张量，但也可以保持内部状态，例如包含可学习参数的张量。 `nn`包还定义了一组有用的损失函数，这些函数通常在训练神经网络时使用。

在此示例中，我们使用`nn`包来实现我们的多项式模型网络：

```py
# -*- coding: utf-8 -*-
import torch
import math

# 创建张量来存储输入值和输出
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 就这个例子来说，输出值y是(x, x^2, x^3)的一个线性函数，所以
# 我们把它看成一个神经网络的线性层。让我们准备好这个张量 (x, x^2, x^3)
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# 在上面的代码中，x.unsqueeze(-1) 的形状为(2000, 1), p的形状为(3,),
# 在这种情况下，会应用广播(broadcasting)语义来获得一个形状为(2000, 3)的张量

# 使用nn包来将我们的模型定义为一个包含许多层的序列。nn.Sequential
# 是一个容纳其它模块，并按顺序应用这些模块来产生输出值的模块。
# 这个线性模块对输入值使用线性函数来计算输出值，并将自身的权重和偏差值持有在内部张量中。
# 展平层(Flatten layer)将线性层的输出值展平为1维张量来匹配 `y`的形状
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

# nn包还包含了流行的损失函数的定义。在这个例子中我们将会使用
# 均方误差 Mean Squared Error (MSE)作为我们的损失函数
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(2000):

    # 前向传递: 将x传递给模型来计算y预测值。模块(Module)类型对象重载了 __call__ 运算符
    # 让你能够像函数一样调用它们。当你可以把输入数据的张量在调用Module的时候传递，
    # 调用结果将产生一个输出数据的张量
    y_pred = model(xx)

    # 计算和输出损失值。我们传递包含y预测值的张量和y真实值的张量给损失函数
    # 损失函数返回一个包含损失值的张量
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 在运行反向传递前清零梯度
    model.zero_grad()

    # 反向传递: 计算模型所有的可学习参数的相关损失梯度。在内部，每个模块的参数都是在
    # requires_grad=True的张量中存储的，所以这次调用会计算模型中所有可学习参数的梯度
    loss.backward()

    # 用梯度下降来更新权重。每个参数都是一个张量，
    # 所以我们像之前做的那样访问它们的梯度。
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# 你可以像访问列表的第一个元素一样访问 `model` 的第一层
linear_layer = model[0]

# 对于线性层它的参数是以 `weight` 和 `bias` 的形式存储的。
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

```

### PyTorch：`optim`

到目前为止，我们已经通过使用`torch.no_grad()`手动更改持有可学习参数的张量来更新模型的权重。 对于像随机梯度下降这样的简单优化算法来说，这并不是一个巨大的负担，但是在实践中，我们经常使用更复杂的优化器（例如 AdaGrad，RMSProp，Adam 等）来训练神经网络。

PyTorch 中的`optim`包抽象了优化算法的思想，并提供了常用优化算法的实现。

在此示例中，我们将使用`nn`包像以前一样定义我们的模型，但是我们将使用`optim`包提供的 RMSprop 算法来优化模型：

```py
# -*- coding: utf-8 -*-
import torch
import math

# 创建张量来存储输入值和输出
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 准备输入张量 (x, x^2, x^3)
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# 用nn包来定义模型和损失函数
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use RMSprop; the optim package contains many other
# optimization algorithms. The first argument to the RMSprop constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
for t in range(2000):
    # 前向传递: 将x传递给模型来计算y预测值。
    y_pred = model(xx)

    # 计算和输出损失值。
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 在向后传递之前，使用优化器对象来清零所有它将更新的变量（也就是模型的可学习权重）的梯度。
    # 这是因为默认情况下，当`backward()`方法被调用的时候偶，梯度是在缓冲区中累计的（也就是说，并不是被覆写了）。
    # 更详细的内容请查看torch.autograd.backward的文档
    optimizer.zero_grad()

    # 反向传递: 计算相对于模型参数的的损失梯度
    loss.backward()

    # 调用优化器(Optimizer)的step函数来更新参数
    optimizer.step()

linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

```

### PyTorch：自定义`nn`模块

有时，您将需要指定比一系列现有模块更复杂的模型。 对于这些情况，您可以通过子类化`nn.Module`并定义一个`forward`来定义自己的模块，并定义一个能够接收输入张量、然后用其他模块或者其他autograd运算操作、生成输出值张量的`forward`方法。

在此示例中，我们将三阶多项式实现为自定义`Module`子类：

```py
# -*- coding: utf-8 -*-
import torch
import math

class Polynomial3(torch.nn.Module):
    def __init__(self):
        """
        我们在构造器中实例化四个参数，并以成员参数赋值
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        在前向函数中我们接收一个输入数据的张量、且必须返回一个输出数据的张量。
        我们可以使用定义在构造器的模块或者任何想用的张量运算符。
        """
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def string(self):
        """
        就像是Python中的任何类一样，你也可以自定义PyTorch模块的方法
        """
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

# 创建张量来存储输入值和输出
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 实例化上面的类型来构建模型
model = Polynomial3()

# 构建我们的损失函数和优化器。在随机梯度下降（优化器）的构造器中调用model.parameters()方法
# 将容纳作为模型成员存在的nn.Linear模块的可学习参数。
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(2000):
    # 前向传递: 将x传递给模型来计算y的预测值
    y_pred = model(x)

    # 计算并输出损失
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 清空梯度、执行反向传递、更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')

```

### PyTorch：控制流 + 权重共享

作为动态图和权重共享的示例，我们实现了一个非常奇怪的模型：一个三阶多项式，在每个正向传播中选择 3 到 5 之间的一个随机数，并使用该阶数，多次使用相同的权重重复计算四和五阶。

对于此模型，我们可以使用常规的 Python 流控制来实现循环，并且可以通过在定义正向传播时简单地多次重复使用相同的参数来实现权重共享。

我们可以轻松地将此模型实现为`Module`子类：

```py
# -*- coding: utf-8 -*-
import random
import torch
import math

class DynamicNet(torch.nn.Module):
    def __init__(self):
        """
        我们在构造器中实例化五个参数，并以成员参数赋值
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        对于模型的前向传递，我们会随机选择四阶多项式或者五阶多项式，并重用e参数来计算该阶的贡献值

        因为每次前向传递会构建一个动态的计算图，我们可以在定义模型的前向传递时
        使用正常的Python流控制运算符（像循环或者条件语句）。

        这里我们还能看到在定义一个计算图的时候偶重用同一个参数许多次是完全安全的。
        """
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x ** exp
        return y

    def string(self):
        """
        就像是Python中的任何类一样，你也可以自定义PyTorch模块的方法
        """
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 ? + {self.e.item()} x^5 ?'

# 创建张量来存储输入值和输出
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 实例化上面的类型来构建模型
model = DynamicNet()

# 构建我们的损失函数和优化器。用香草随机梯度下降来训练这个奇怪的模型有点困难,
# 所以我们使用动量(momentum)参数
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
for t in range(30000):
    # 前向传递: 将x传递给模型来计算y的预测值
    y_pred = model(x)

    # 计算并输出损失
    loss = criterion(y_pred, y)
    if t % 2000 == 1999:
        print(t, loss.item())

    # 清空梯度、执行反向传递、更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')

```

## 示例

您可以在此处浏览以上示例。

### 张量

### Autograd

### `nn`模块
