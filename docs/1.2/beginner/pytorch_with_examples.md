# 用例子学习 PyTorch

> **作者** ：[Justin Johnson](https://github.com/jcjohnson/pytorch-examples)
>
> 校正：[宁采晨](https://github.com/yangkae)

本教程通过独立的示例介绍了[PyTorch](https://github.com/pytorch/pytorch)的基本概念 。

PyTorch的核心是提供两个主要功能：

  * n维张量，类似于numpy，但可以在GPU上运行
  * 自动区分以构建和训练神经网络

我们将使用完全连接的ReLU网络作为我们的运行示例。该网络将具有单个隐藏层，并且将通过最小化网络输出与真实输出之间的欧几里德距离来进行梯度下降训练，以适应随机数据。

- 注意

  您可以在本页结尾浏览各个示例。

[TOC]



## 张量

### Warm-up：Numpy

在介绍PyTorch之前，我们将首先使用numpy实现网络。

Numpy提供了一个n维数组对象，以及许多用于操纵这些数组的函数。Numpy是用于科学计算的通用框架；它对计算图，深度学习或梯度一无所知。然而，我们可以很容易地使用NumPy，手动实现网络的前向和反向传播，来拟合随机数据：

```python
# -*- coding: utf-8 -*-
import numpy as np

# N是批尺寸参数；D_in是输入维度
# H是隐藏层维度；D_out是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生随机输入和输出数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# 随机初始化权重
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # 前向传播：计算预测值y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # 计算并显示loss(损失）
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # 反向传播，计算w1、w2对loss的梯度
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


###  PyTorch：张量

Numpy是一个伟大的框架，但它不能利用GPU来加速其数值计算。对于现代的深层神经网络，GPU通常提供的[50倍以上](https://github.com/jcjohnson/cnn-benchmarks)的加速，仅凭numpy不足以实现现代深度学习。

在这里，我们介绍最基本的PyTorch概念：张量(**Tensor**）。PyTorch张量在概念上与numpy数组相同：张量是n维数组，而PyTorch提供了很多函数操作这些tensor。张量可以跟踪计算图和渐变，它们也可用作科学计算的通用工具。

与numpy不同，PyTorch张量可以利用GPU加速其数字计算。要在GPU上运行PyTorch Tensor，只需将其转换为新的数据类型。

这里我们利用PyTorch的tensor在随机数据上训练一个两层的网络。和前面NumPy的例子类似，我们使用PyTorch的tensor，手动在网络中实现前向传播和反向传播：

```python
# -*- coding: utf-8 -*-

import torch


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N是批尺寸大小； D_in 是输入维度；
# H 是隐藏层维度； D_out 是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生随机输入和输出数据
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 随机初始化权重
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # 前向传播：计算预测值y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # 计算并输出loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # 反向传播，计算w1、w2对loss的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 使用梯度下降更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```




##  自动求导

###  PyTorch：张量和自动求导

在以上示例中，我们必须手动实现神经网络的前向和后向传递。对于小型的两层网络而言，手动实现反向传递并不重要，但对于大型的复杂网络而言，这变得非常麻烦。

幸运的是，我们可以使用[自动微分](https://en.wikipedia.org/wiki/Automatic_differentiation) 来自动计算神经网络中的反向传播。**PyTorch**中的 **autograd**软件包提供了这个功能。使用autograd时，您的网络正向传递将定义一个 **计算图**；图中的节点为张量，图中的边为从输入张量产生输出张量的函数。通过该图进行反向传播，可以轻松计算梯度。

这听起来很复杂，在实践中非常简单。每个张量代表计算图中的一个节点。如果 `x`是一个张量，并且有 `x.requires_grad=True`，那么`x.grad`就是另一个张量，代表着`x`相对于某个标量值的梯度。

在这里，我们使用PyTorch张量和autograd来实现我们的两层网络。现在我们不再需要手动实现网络的反向传播：

```python
# -*- coding: utf-8 -*-
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N是批尺寸大小；D_in是输入维度；
# H是隐藏层维度；D_out是输出维度 
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生随机输入和输出数据，将requires_grad置为False，意味着我们不需要在反向传播时候计算这些值的梯度
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 产生随机权重tensor，将requires_grad设置为True，意味着我们希望在反向传播时候计算这些值的梯度
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
	# 前向传播：使用tensor的操作计算预测值y。
    # 由于w1和w2有requires_grad=True,涉及这些张量的操作将让PyTorch构建计算图，从而允许自动计算梯度。
    # 由于我们不再手工实现反向传播，所以不需要保留中间值的引用。
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # 计算并输出loss
    # loss是一个形状为(1,)的张量
    # loss.item()是这个张量对应的python数值
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # 使用autograd计算反向传播,这个调用将计算loss对所有requires_grad=True的tensor的梯度。
    # 这次调用后，w1.grad和w2.grad将分别是loss对w1和w2的梯度张量。
    loss.backward()

    # 使用梯度下降更新权重。对于这一步，我们只想对w1和w2的值进行原地改变；不想为更新阶段构建计算图，
    # 所以我们使用torch.no_grad()上下文管理器防止PyTorch为更新构建计算图
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 反向传播之后手动将梯度置零
        w1.grad.zero_()
        w2.grad.zero_()
```

### PyTorch：定义新的自动求导函数

在底层，每一个原始的自动求导运算实际上是两个在Tensor上运行的函数。其中，**forward**函数计算从输入Tensors获得的输出Tensors。而**backward**函数接收输出Tensors对于某个标量值的梯度，并且计算输入Tensors相对于该相同标量值的梯度。

在PyTorch中，我们可以很容易地通过定义`torch.autograd.Function`的子类并实现`forward`和`backward`函数，来定义自己的自动求导运算。然后，我们可以通过构造实例并像调用函数一样调用它，并传递包含输入数据的张量。

这个例子中，我们自定义一个自动求导函数来展示ReLU的非线性。并用它实现我们的两层网络：  

```python
# -*- coding: utf-8 -*-
import torch


class MyReLU(torch.autograd.Function):
    """
    我们可以通过建立torch.autograd的子类来实现我们自定义的autograd函数，并完成张量的正向和反向传播。
    """

    @staticmethod
    def forward(ctx, input):
        """
        在前向传播中，我们收到包含输入和返回的张量包含输出的张量。 
        ctx是可以使用的上下文对象存储信息以进行向后计算。 
        您可以使用ctx.save_for_backward方法缓存任意对象，以便反向传播使用。
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
		在反向传播中，我们接收到上下文对象和一个张量，其包含了相对于正向传播过程中产生的输出的损失的梯度。
        我们可以从上下文对象中检索缓存的数据，并且必须计算并返回与正向传播的输入相关的损失的梯度。
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N是批尺寸大小； D_in 是输入维度；
# H 是隐藏层维度； D_out 是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生输入和输出的随机张量
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 产生随机权重的张量
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 为了使用我们的方法，我们调用Function.apply方法。 我们将其命名为“ relu”。
    relu = MyReLU.apply

    # 正向传播：使用张量上的操作来计算输出值y;
    # 我们使用自定义的自动求导操作来计算 RELU.
    y_pred = relu(x.mm(w1)).mm(w2)

    # 计算并输出loss
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # 使用autograd计算反向传播过程。
    loss.backward()

    # 用梯度下降更新权重
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 在反向传播之后手动清零梯度
        w1.grad.zero_()
        w2.grad.zero_()
```


###  TensorFlow：静态图

PyTorch自动求导看起来很像TensorFlow：在两个框架中我们都定义了一个计算图，并使用自动微分来计算梯度。两者之间的最大区别是TensorFlow的计算图是**静态的，**而PyTorch使用 **动态**计算图。

在TensorFlow中，我们一次定义了计算图，然后一遍又一遍地执行相同的图，可能将不同的输入数据提供给该图。在PyTorch中，每个前向传递都定义一个新的计算图。

静态图的好处在于您可以预先优化图。例如，框架可能决定融合某些图形操作以提高效率，或者想出一种在多个GPU或许多机器之间分布图形的策略。如果您重复用同一张图，那么随着一遍一遍地重复运行同一张图，可以分摊这种潜在的昂贵的前期优化。

静态图和动态图不同的一个方面是控制流。对于某些模型，我们可能希望对每个数据点执行不同的计算。例如，对于每个数据点，循环网络可能会展开不同数量的时间步长；此展开可以实现为循环。对于静态图，循环构造必须是图的一部分；因此，TensorFlow提供了诸如`tf.scan`将循环嵌入到图中的运算符。使用动态图，情况更简单：由于我们为每个示例动态生成图，因此可以使用常规命令流控制来执行针对每个输入而不同的计算。

与上面的PyTorch autograd示例形成对比，这里我们使用TensorFlow来拟合一个简单的两层网络：

```python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# 首先我们建立计算图:

# N是批大小；D是输入维度；
# H是隐藏层维度；D_out是输出维度。
N, D_in, H, D_out = 64, 1000, 100, 10

# 为输入和目标数据创建placeholder
# 当执行计算图时，他们将会被真实的数据填充
x = tf.placeholder(tf.float32, shape=(None, D_in))
y = tf.placeholder(tf.float32, shape=(None, D_out))

# 为权重创建Variable并用随机数据初始化
# TensorFlow的Variable在执行计算图时不会改变
w1 = tf.Variable(tf.random_normal((D_in, H)))
w2 = tf.Variable(tf.random_normal((H, D_out)))

# 前向传播：使用TensorFlow的张量运算计算预测值y
# 注意这段代码实际上不执行任何数值运算
# 它只是建立了我们稍后将执行的计算图
h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

# 使用TensorFlow的张量运算损失(loss）
loss = tf.reduce_sum((y - y_pred) ** 2.0)

# 计算loss对于w1和w2的梯度
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# 使用梯度下降更新权重。为了实际更新权重，我们需要在执行计算图时计算new_w1和new_w2
# 注意，在TensorFlow中，更新权重值的行为是计算图的一部分
# 但在PyTorch中，这发生在计算图形之外
learning_rate = 1e-6
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# 现在我们搭建好了计算图，所以我们开始一个TensorFlow的会话(session）来实际执行计算图
with tf.Session() as sess:
    # 运行一次计算图来初始化变量w1和w2
    sess.run(tf.global_variables_initializer())

    # 创建numpy数组来存储输入x和目标y的实际数据
    x_value = np.random.randn(N, D_in)
    y_value = np.random.randn(N, D_out)
    for t in range(500):
        # 多次运行计算图。每次执行时，我们都用feed_dict参数
        # 将x_value绑定到x，将y_value绑定到y
        # 每次执行图形时我们都要计算损失、new_w1和new_w2
        # 这些张量的值以numpy数组的形式返回
        loss_value, _, _ = sess.run([loss, new_w1, new_w2],
                                    feed_dict={x: x_value, y: y_value})
        if t % 100 == 99:
            print(t, loss_value)
```


##  nn 模块

###  PyTorch：nn 

计算图和autograd是定义复杂运算符并自动采用导数的非常强大的范例。但是对于大型神经网络，原始的autograd可能会有点太低了。

在构建神经网络时，我们经常考虑将计算分为几层，其中一些层具有可学习的参数 ，这些参数将在学习过程中进行优化。

在TensorFlow中，诸如[Keras](https://github.com/fchollet/keras)， [TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)和[TFLearn之](http://tflearn.org/)类的软件包在原始计算图上提供了更高级别的抽象接口，这些封装对构建神经网络很有用。

在PyTorch中，该`nn`程序包达到了相同的目的。该`nn` 包定义了一组**Modules**，它们大致等效于神经网络层。模块接收输入张量并计算输出张量，但也可以保持内部状态，例如包含可学习参数的张量。该`nn`软件包还定义了一组有用的损失函数，这些函数通常在训练神经网络时使用。

在此示例中，我们使用该`nn`包来实现我们的两层网络：

```python
# -*- coding: utf-8 -*-
import torch

# N是批大小；D是输入维度
# H是隐藏层维度；D_out是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生输入和输出随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 使用nn包将我们的模型定义为一系列的层
# nn.Sequential是包含其他模块的模块，并按顺序应用这些模块来产生其输出
# 每个线性模块使用线性函数从输入计算输出，并保存其内部的权重和偏差张量
# 在构造模型之后，我们使用.to()方法将其移动到所需的设备
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# nn包还包含常用的损失函数的定义
# 在这种情况下，我们将使用平均平方误差(MSE)作为我们的损失函数
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    # 前向传播：通过向模型传入x计算预测的y
    # 模块对象重载了__call__运算符，所以可以像函数那样调用它们
    # 这么做相当于向模块传入了一个张量，然后它返回了一个输出张量
    y_pred = model(x)

    # 计算并打印损失。我们传递包含y的预测值和真实值的张量，损失函数返回包含损失的张量
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 反向传播之前清零梯度
    model.zero_grad()

    # 反向传播：计算模型的损失对所有可学习参数的梯度
    # 在内部，每个模块的参数存储在requires_grad=True的张量中
    # 因此这个调用将计算模型中所有可学习参数的梯度
    loss.backward()

    # 使用梯度下降更新权重
    # 每个参数都是张量，所以我们可以像我们以前那样可以得到它的数值和梯度
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
```


###  PyTorch：optim

到现在为止，我们已经通过手动更改持有可学习参数的张量来更新模型的权重(使用`torch.no_grad()` 或`.data`避免在autograd中跟踪历史记录）。对于像随机梯度下降这样的简单优化算法来说，这并不是一个沉重的负担，但是在实践中，我们经常使用更复杂的优化器(例如AdaGrad，RMSProp，Adam等）来训练神经网络。

PyTorch中的软件包`optim`抽象了优化算法的思想，并提供了常用优化算法的实现。

在此示例中，我们将像之前一样使用`nn`包来定义模型，但是用`optim`包提供的Adam算法来优化模型：   

```python
# -*- coding: utf-8 -*-
import torch

# N是批大小；D是输入维度
# H是隐藏层维度；D_out是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生随机输入和输出张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 使用nn包定义模型和损失函数
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# 使用optim包定义优化器(Optimizer）。Optimizer将会为我们更新模型的权重
# 这里我们使用Adam优化方法；optim包还包含了许多别的优化算法
# Adam构造函数的第一个参数告诉优化器应该更新哪些张量
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # 前向传播：通过像模型输入x计算预测的y
    y_pred = model(x)

    # 计算并输出loss
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 在反向传播之前，使用optimizer将它要更新的所有张量的梯度清零(这些张量是模型可学习的权重)。
    # 这是因为默认情况下，每当调用.backward(）时，渐变都会累积在缓冲区中(即不会被覆盖）
    # 有关更多详细信息，请查看torch.autograd.backward的文档。
    optimizer.zero_grad()

    # 反向传播：根据模型的参数计算loss的梯度
    loss.backward()

    # 调用Optimizer的step函数使它所有参数更新
    optimizer.step()
```

### PyTorch：自定义nn模块

有时，您将需要指定比一系列现有模块更复杂的模型。在这些情况下，您可以通过继承`nn.Module`和定义一个`forward`来定义自己的模型，这个`forward`模块可以使用其他模块或在Tensors上的其他自动求导运算来接收输入Tensors并生成输出Tensors。

在此示例中，我们将使用自定义的Module子类构建两层网络：

```python
# -*- coding: utf-8 -*-
import torch


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        在构造函数中，我们实例化了两个nn.Linear模块，并将它们作为成员变量。
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        在前向传播的函数中，我们接收一个输入的张量，也必须返回一个输出张量。
        我们可以使用构造函数中定义的模块以及张量上的任意的(可微分的）操作。
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# N是批大小； D_in 是输入维度；
# H 是隐藏层维度； D_out 是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生输入和输出的随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 通过实例化上面定义的类来构建我们的模型
model = TwoLayerNet(D_in, H, D_out)

# 构造损失函数和优化器
# SGD构造函数中对model.parameters()的调用
# 将包含模型的一部分，即两个nn.Linear模块的可学习参数
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # 前向传播：通过向模型传递x计算预测值y
    y_pred = model(x)

    # 计算并输出loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 清零梯度，反向传播，更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```


###  PyTorch：控制流+权重共享

作为动态图和权重共享的示例，我们实现了一个非常奇怪的模型：一个完全连接的ReLU网络，该网络在每个前向传递中选择1到4之间的随机数作为隐藏层的层数，多次重复使用相同的权重计算最里面的隐藏层。

对于此模型，我们可以使用常规的Python流控制来实现循环，并且可以通过在定义前向传递时简单地多次重复使用同一模块来实现最内层之间的权重共享。

我们利用Mudule的子类很容易实现这个模型：

    # -*- coding: utf-8 -*-
    import random
    import torch
    
    
    class DynamicNet(torch.nn.Module):
        def __init__(self, D_in, H, D_out):
            """
            在构造函数中，我们构造了三个nn.Linear实例，它们将在前向传播时被使用。
            """
            super(DynamicNet, self).__init__()
            self.input_linear = torch.nn.Linear(D_in, H)
            self.middle_linear = torch.nn.Linear(H, H)
            self.output_linear = torch.nn.Linear(H, D_out)
    
        def forward(self, x):
            """
            对于模型的前向传播，我们随机选择0、1、2、3，并重用了多次计算隐藏层的middle_linear模块。
            由于每个前向传播构建一个动态计算图，
            我们可以在定义模型的前向传播时使用常规Python控制流运算符，如循环或条件语句。
            在这里，我们还看到，在定义计算图形时多次重用同一个模块是完全安全的。
            这是Lua Torch的一大改进，因为Lua Torch中每个模块只能使用一次。
            """
            h_relu = self.input_linear(x).clamp(min=0)
            for _ in range(random.randint(0, 3)):
                h_relu = self.middle_linear(h_relu).clamp(min=0)
            y_pred = self.output_linear(h_relu)
            return y_pred
    
    
    # N是批大小；D是输入维度
    # H是隐藏层维度；D_out是输出维度
    N, D_in, H, D_out = 64, 1000, 100, 10
    
    # 产生输入和输出随机张量
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    
    # 实例化上面定义的类来构造我们的模型
    model = DynamicNet(D_in, H, D_out)
    
    # 构造我们的损失函数(loss function）和优化器(Optimizer）
    # 用平凡的随机梯度下降训练这个奇怪的模型是困难的，所以我们使用了momentum方法
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    for t in range(500):
        # 前向传播：通过向模型传入x计算预测的y
        y_pred = model(x)
    
        # 计算并输出损失loss
        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())
    
        # 清零梯度，反向传播，更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


## 例子

你可以在此处浏览以上示例。

### 张量

[Warm-up：Numpy](https://pytorch.org/tutorials/beginner/examples_tensor/two_layer_net_numpy.html#sphx-glr-beginner-examples-tensor-two-layer-net-numpy-py)

[PyTorch：张量](https://pytorch.org/tutorials/beginner/examples_tensor/two_layer_net_tensor.html#sphx-glr-beginner-examples-tensor-two-layer-net-tensor-py)

### Autograd

[PyTorch：张量和自动求导](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_autograd.html#sphx-glr-beginner-examples-autograd-two-layer-net-autograd-py)

[PyTorch：定义新的自动求导函数](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html#sphx-glr-beginner-examples-autograd-two-layer-net-custom-function-py)

[TensorFlow：静态图](https://pytorch.org/tutorials/beginner/examples_autograd/tf_two_layer_net.html#sphx-glr-beginner-examples-autograd-tf-two-layer-net-py)

### nn模块

[PyTorch：nn](https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_nn.html#sphx-glr-beginner-examples-nn-two-layer-net-nn-py) 

[PyTorch：optim](https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_optim.html#sphx-glr-beginner-examples-nn-two-layer-net-optim-py)

[PyTorch：自定义nn模块](https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html#sphx-glr-beginner-examples-nn-two-layer-net-module-py)

[PyTorch：控制流+权重共享](https://pytorch.org/tutorials/beginner/examples_nn/dynamic_net.html#sphx-glr-beginner-examples-nn-dynamic-net-py)