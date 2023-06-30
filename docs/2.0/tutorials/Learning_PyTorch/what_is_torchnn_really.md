# `torch.nn`究竟是什么？

> 译者：[runzhi214](https://github.com/runzhi214)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/Learning_PyTorch/what_is_torchnn_really.md/>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/nn_tutorial.html>

**作者**：Jeremy Howard, [fast.ai](https://www.fast.ai/). Thanks to Rachel Thomas and Francisco Ingham.

我们推荐把这个教程作为Notebook(.ipynb)而不是脚本来运行。点击页面顶部的[链接](https://pytorch.org/tutorials/_downloads/d9398fce39ca80dc4bb8b8ea55b575a8/nn_tutorial.ipynb)来下载Notebook。

PyTorch提供设计地很优雅的模块和类，比如[torch.nn](), [torch.optim](), [Dataset]() 和[DataLoader]()来帮助你创建和训练神经网络。为了能完全发挥它们的能力并自定义它们来解决你的问题，你需要切实的理解他们在做什么。为了让你由浅入深的理解，我们会在MNIST数据集上首先训练基本的神经网络，暂时还不使用任何这些模型的特性; 我们一开始只使用PyTorch张量的最基本的功能。然后，我们会逐渐地从这些特性中每次增加一个，展示每一样究竟做了什么，以及它是如何让代码更加准确或者更加灵活的。

**在这个教程中，我们假设你已经安装了PyTorch，并且已经熟悉了张量的基本运算。**（如果你对NumPy的数组操作很熟悉，那你会发现这里使用的PyTorch的张量操作几乎一样）

## MNIST 数据准备

我们将使用经典的[MNIST](http://deeplearning.net/data/mnist/)数据集，它由许多手写的0～9之间的数字的黑白的图像组成。

我们将使用[pathlib库](https://docs.python.org/3/library/pathlib.html)来处理路径问题，并使用[requests库](http://docs.python-requests.org/en/master/)来下载数据集。我们只会在使用到它们的时候引入，所以你可以清晰地看到每一步究竟使用了什么。

```py
from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
```

该数据集采取numpy数组的格式，并且使用对象序列化（python特有的序列话数据的格式）来存储。

```py
import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
```

每张图片的尺寸是28x28，作为长度为784的行（28x28展平为784）存储。让我们看一张图片，首先我们需要将它重塑为2维。

```py
from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# ``pyplot.show()`` only if not on Colab
try:
    import google.colab
except ImportError:
    pyplot.show()
print(x_train.shape)
```

![]()

Out:

```py
(50000, 784)
```

PyTorch使用`torch.tensor`而不是numpy数组，所以我们需要转换数据。

```py
import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())
```

Out:

```py
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]]) tensor([5, 0, 4,  ..., 8, 4, 8])
torch.Size([50000, 784])
tensor(0) tensor(9)
```

## 神经网络从零开始(不用`torch.nn`)

让我们首先只用PyTorch张量操作来创建一个模型。我们假设你已经对神经网络的基本概念很熟悉了（如果你不熟悉，可以在[course.fast.ai](https://course.fast.ai/)学习一下）。

PyTorch提供创建填充了随机数或者零的张量的方法，我们将用这些方法来创建一个简单线性模型的对应权重（张量）和偏差值（张量）。它们只是很普通的张量，只有一点特殊：我们告诉PyTorch这些张量需要梯度。这会让PyTorch记录所有在张量上的运算，以便于它能够在反向传播的过程中自动计算梯度。

对于权重，我们在初始化**后**设置`requires_grad`,因为我们并不希望初始化的这一步也加入梯度（请注意: 在PyTorch中一个`_`标志着运算是原位(in-place)发生的）。

> 注意:
> 我们用 [ Xavier 初始化](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) 来初始化权重（即乘以`1/sqrt(n)`）。

```py
import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)
```

感谢PyTorch自动计算梯度的能力，我们可以使用任何Python函数作为模型！所以让我们写一个普通的矩阵乘法和广播加法来创建一个简单的线性模型。我们还需要一个激活函数，所以我们写`log_softmax`并用它做激活函数。请记住：尽管PyTorch提供了许多事先写好的损失函数、激活函数，你仍然可以用普通python写一个自己自定义的。PyTorch甚至会为你的函数自动创建更快的GPU代码或者向量化的CPU代码。

```py
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)
```

在上面这段代码中，`@`代表矩阵乘法运算。我们在一批数据（在这个例子中是64张图片）上调用我们的函数。这是一次**前向传递**。注意：由于我们是从随机权重开始的，在这个阶段我们的预测值不会比随机数好多少。

```py
bs = 64  # 批大小

xb = x_train[0:bs]  # x中的一个微批
preds = model(xb)  # 预测
preds[0], preds.shape
print(preds[0], preds.shape)
```

Out:

```py
tensor([-2.5452, -2.0790, -2.1832, -2.6221, -2.3670, -2.3854, -2.9432, -2.4391,
        -1.8657, -2.0355], grad_fn=<SelectBackward0>) torch.Size([64, 10])
```

如你所见，`preds`张量不仅包含张量值，还包含了一个梯度函数。后面我们会用这个来做反向传播。让我们实现一个负对数似然来用做损失函数（我们仍然只用普通的Python）：

```py
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll
```

让我们用我们的随机模型来检查一下损失值，以便观察在一次反向传播传递后模型是否改善了。

```py
yb = y_train[0:bs]
print(loss_func(preds, yb))
```

Out:

```py
tensor(2.4020, grad_fn=<NegBackward0>)
```

让我们再实现一个函数来计算模型的准确率。对于每个预测值，如果最大值的索引匹配目标值，那么预测就是准确的（即索引0-9分别代表预测数字为0-9的对应值）。

```py
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()
```

让我们检查一下我们的随机模型的准确率，以便我们观察随着损失值改善了我们的准确率是否也改善了。

```py
print(accuracy(preds, yb))
```

Out:

```py
tensor(0.0938)
```

现在我们运行一个训练循环。在每次迭代中，我们将会:

- 选取一个微批的数据（数量为`bs`）
- 用模型来做预测
- 计算损失值
- `loss.backward()`更新模型的梯度，在这个例子中，意味着更新`weights`和`bias`。

现在我们用这些梯度来更新权重(`weights`)和偏差值(`bias`)。我在`torch.no_grad()`上下文管理器中这么做，因为我们不希望为下一次梯度计算记录这些动作。你可以在[这里](https://pytorch.org/docs/stable/notes/autograd.html)阅读更多关于PyTorch的Autograd如何记录运算。

然后我们将梯度设置为零，给下一轮循环做好准备。否则，我们的梯度会记录所有发生的运算的合计（也就是说，`loss.backward()` 会对存储的值**累加**梯度，而不是替代）。

> 提示:
> 你可以用标准Python调试器来逐步执行PyTorch代码，让你在每一步检查不同变量的值。取消注释下面的`set_trace()`来尝试一下。

```py
from IPython.core.debugger import set_trace

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        # set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
```

写完了: 我们已经从零开始创建、训练了一个微型的神经网络（在这个例子中，一个逻辑回归，因为我们没有使用隐藏层）。

让我们检查损值和精确度，并和我们之前获取的值对比一下。我们预期损失值会下降、精确度会提升。事实证明确实如此:

```py
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
```

Out:

```py
tensor(0.0813, grad_fn=<NegBackward0>) tensor(1.)
```

## 使用`torch.nn.functional`

现在我们要重构我们的代码，让它做和之前一样的事情，只不过我们开始利用PyTorch的`nn`包来让代码更加精准、更加灵活。从这里出发的每一步，我们都会让我们的代码变得或多或少: 更短、更易读，并且/或者 更灵活。

第一步也是最简单的一步是通过把我们手写的激活函数和损失函数用`torch.nn.functional`(通常)中的函数替代，来让代码更短。
