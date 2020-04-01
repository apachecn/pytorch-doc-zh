# *torch.nn* 到底是什么？

> 译者：[lhc741](https://github.com/lhc741)
>
> 校验：[DrDavidS](https://github.com/DrDavidS)

**作者**：Jeremy Howard，[fast.ai](https://www.fast.ai/)。感谢Rachel Thomas和Francisco Ingham的帮助和支持。

我们推荐使用notebook来运行这个教程，而不是脚本，点击[这里](https://pytorch.org/tutorials/beginner/nn_tutorial.html#sphx-glr-download-beginner-nn-tutorial-py)下载notebook(.ipynb)文件。

Pytorch提供了[torch.nn](https://pytorch.org/docs/stable/nn.html)、[torch.optim](https://pytorch.org/docs/stable/optim.html)、[Dataset](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset)和[DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)这些设计优雅的模块和类以帮助使用者创建和训练神经网络。
为了最大化利用这些模块和类的功能，并使用它们做出适用于你所研究问题的模型，你需要真正理解他们是如何工作的。
为了做到这一点，我们首先基于MNIST数据集训练一个没有任何特征的简单神经网络。
最开始我们只会用到PyTorch中最基本的tensor功能，然后我们将会逐渐的从`torch.nn`，`torch.optim`，`Dataset`，`DataLoader`中选择一个特征加入到模型中，来展示新加入的特征会对模型产生什么样的效果，以及它是如何使模型变得更简洁或更灵活。

**在这个教程中，我们假设你已经安装好了PyTorch，并且已经熟悉了基本的tensor运算。**(如果你熟悉Numpy的数组运算，你将会发现这里用到的PyTorch tensor运算和numpy几乎是一样的)

## MNIST数据安装

我们将要使用经典的[MNIST](http://deeplearning.net/data/mnist/)数据集，这个数据集由手写数字(0到9）的黑白图片组成。

我们将使用[pathlib](https://docs.python.org/3/library/pathlib.html)来处理文件路径的相关操作(python3中的一个标准库），使用[request](http://docs.python-requests.org/en/master/)来下载数据集。
我们只会在用到相关库的时候进行引用，这样你就可以明确在每个操作中用到了哪些库。

```py
from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
```

该数据集采用numpy数组格式，并已使用pickle存储，pickle是一个用来把数据序列化为python特定格式的库。

```py
import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
```

每一幅图像都是28 x 28的，并被拉平成长度为784(=28x28)的一行。
我们以其中一个为例展示一下，首先需要将这个一行的数据重新变形为一个2d的数据。

```py
%matplotlib inline
from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)
```

![image](https://pytorch.org/tutorials/_images/sphx_glr_nn_tutorial_001.png)

输出：

```py
(50000, 784)
```

PyTorch使用`torch.tensor`，而不是numpy数组，所以我们需要将数据转换。

```py
import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())
```

输出：

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

## 神经网络从零开始(不使用torch.nn）

我们先来建立一个只使用PyTorch张量运算的模型。
我们假设你已经熟悉神经网络的基础。(如果你还不熟悉，可以访问[course.fast.ai](https://course.fast.ai/)进行学习）。

PyTorch提供创建随机数填充或全零填充张量的方法，我们使用该方法初始化一个简单线性模型的权重和偏置。
这两个都是普通的张量，但它们有一个特殊的附加条件：设置需要计算梯度的参数为True。这样PyTorch就会记录所有与这个张量相关的运算，使其能在反向传播阶段自动计算梯度。

对于weights而言，由于我们希望初始化张量过程中存在梯度，所以我们在初始化**之后**设置`requires_grad`。(注意：尾缀为`_`的方法在PyTorch中表示这个操作会被立即被执行。）

> - 注意：
>
> 我们以[Xavier初始化](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)方法(每个元素都除以1/sqrt(n)）为例来对权重进行初始化。

```py
import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)
```

多亏了PyTorch具有自动梯度计算功能，我们可以使用Python中任何标准函数(或者可调用对象）来创建模型！
因此，让我们编写一个普通的矩阵乘法和广播加法建立一个简单的线性模型。
我们还需要一个激活函数，所以我们编写并使用一个log_softmax函数。
请记住：尽管Pytorch提供了许多预先编写好的损失函数、激活函数等等，你仍然可以使用纯python轻松实现你自己的函数。
Pytorch甚至可以自动地为你的函数创建快速的GPU代码或向量化的CPU代码。

```py
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)
```

在上面的一段代码中，`@`表示点积运算符。我们将调用我们的函数计算一个批次的数据(本例中为64幅图像）。
这是一次模型前向传递的过程。
请注意，因为我们使用了随机数来初始化权重，所以在这个阶段我们的预测值并不会比随机的更好。

```py
bs = 64  # 一批数据个数

xb = x_train[0:bs]  # 从x获取一小批数据
preds = model(xb)  # 预测值
preds[0], preds.shape
print(preds[0], preds.shape)
```

输出：

```py
tensor([-2.5125, -2.3091, -2.0139, -1.8648, -2.7132, -2.5598, -2.3423, -1.8809,
        -2.5860, -2.7542], grad_fn=<SelectBackward>) torch.Size([64, 10])
```

可以从上面的结果不难看出，张量`preds`不仅包括了张量值，还包括了梯度函数。这个梯度函数我们可以在后面的反向传播阶段用到。

下面我们来实现一个负的对数似然函数(Negative log-likehood）作为损失函数(同样也使用纯python实现）：

```py
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll
```

让我们查看下随机模型的损失值，这样我们就可以确认在执行反向传播的步骤后，模型的预测效果是否有了改进。

```py
yb = y_train[0:bs]
print(loss_func(preds, yb))
```

输出：

```py
tensor(2.3860, grad_fn=<NegBackward>)
```

我们再来实现一个用来计算模型准确率的函数。对于每次预测，我们规定如果预测结果中概率最大的数字和图片实际对应的数字是相同的，那么这次预测就是正确的。

```py
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()
```

我们先来看一下被随机初始化的模型的准确率，这样我们就可以看到损失值降低的时候准确率是否提高了。

```py
print(accuracy(preds, yb))
```

输出：

```py
tensor(0.1094)
```

现在我们可以运行一个完整的训练步骤了，每次迭代，我们会进行以下几个操作：

- 从全部数据中选择一小批数据(大小为`bs`）
- 使用模型进行预测
- 计算当前预测的损失值
- 使用`loss.backward()`更新模型中的梯度，在这个例子中，更新的是`weights`和`bias`

现在，我们来利用计算出的梯度对权值和偏置项进行更新，因为我们不希望这一步的操作被用于下一次迭代的梯度计算，所以我们在`torch.no_grad()`这个上下文管理器中完成。想要了解更多关于PyTorch Autograd是如何记录操作的，可以点击[这里](https://pytorch.org/docs/stable/notes/autograd.html)。

接下来，我们将梯度设置为0，来为下一次循环做准备。否则我们的梯度将会记录所有已经执行过的运算(如，`loss.backward()`会将梯度变化值直接与变量已有值进行累加，而不是替换变量原有的值）。


> 小贴士
>
> 您可以使用标准python调试器对PyTorch代码进行单步调试，从而在每一步检查不同的变量值。取消下面的`set_trace()`注释，来尝试该功能。

```py
from IPython.core.debugger import set_trace

lr = 0.5  # 学习率
epochs = 2  # 训练的轮数

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
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

目前为止，我们已经从零开始完成了建立和训练一个最小的神经网络(因为我们建立的logistic回归模型不包含隐层）！

现在，我们来看一下模型的损失值和准确率，并于我们之前输出的值进行比较。结果正如我们预期的，损失值下降，准确率提高。

```py
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
```

输出：

```py
tensor(0.0831, grad_fn=<NegBackward>) tensor(1.)
```

## torch.nn.functional的使用

现在，我们要对前面的代码进行重构，使代码在完成相同功能的同时，用PyTorch的`nn`来使代码变得更加简洁和灵活。
从现在开始，接下来的每一步我们都会使代码变得更短，更好理解或更灵活。

要进行的第一步也是最简单的一步，是使用`torch.nn.functional`(通过会在引用时用F表示）中的函数替换我们自己的激活函数和损失函数使代码变得更短。
这个模块包含了`torch.nn`库中的所有函数(这个库的其它部分是各种类），所以在这个模块中还会找到其它便于建立神经网络的函数，比如池化函数。(模块中还包含卷积函数，线性函数等等，不过在后面的内容中我们会看到，这些操作使用库中的其它部分会更好。）

如果你使用负对数似然损失和对数柔性最大值(softmax)激活函数，PyTorch有一个结合了这两个函数的简单函数`F.cross_entropy`供你使用，这样我们就可以删掉模型中的激活函数。

```py
import torch.nn.functional as F

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias
```

注意在`model`函数中我们不再调用`log_softmax`。现在我们来确认一下损失值和准确率与之前相同。

```py
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
```

输出：

```py
tensor(0.0822, grad_fn=<NllLossBackward>) tensor(1.)
```

## 使用nn.Module进行重构
接下来，我们将会用到`nn.Model`和`nn.Parameter`来完成一个更加清晰简洁的训练循环。我们继承`nn.Module`(它是一个能够跟踪状态的类)。在这个例子中，我们想要新建一个类，实现存储权重，偏置和前向传播步骤中所有用到方法。`nn.Module`包含了许多属性和方法(比如`.parameters()`和`.zero_grad()`），我们会在后面用到。

> 注意
> 
> `nn.Module`(M大写）是一个PyTorch中特有的概念，它是一个会经常用到的类。不要和Python中[module](https://docs.python.org/3/tutorial/modules.html)(`m`小写）混淆，module是一个可以被引入的Python代码文件。

```py
from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias
```

既然现在我们要使用一个对象而不是函数，我们要先对模型进行实例化。

```py
model = Mnist_Logistic()
```

现在我们可以像之前那样计算损失值了。注意`nn.Module`对象的使用方式很像函数(例如它们是*可调用的*），但是PyTorch将会自动调用我们的`forward`函数

```py
print(loss_func(model(xb), yb))
```

输出：

```py
tensor(2.2001, grad_fn=<NllLossBackward>)
```

之前在每个训练循环中，我们通过变量名对每个变量的值进行更新，并手动的将每个变量的梯度置为0，像这样：

```py
with torch.no_grad():
    weights -= weights.grad * lr
    bias -= bias.grad * lr
    weights.grad.zero_()
    bias.grad.zero_()
```

现在我们可以利用`model.parameters()`和`model.zero_grad()`(这两个都是PyTorch定义在`nn.Module`中的）使这些步骤变得更加简洁并且更不容易忘记更新部分参数，尤其是模型很复杂的情况：

```py
with torch.no_grad():
    for p in model.parameters(): p -= p.grad * lr
    model.zero_grad()
```

下面我们把训练循环封装进`fit`函数中，这样就能在后面再次运行。

```py
def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

fit()
```

我们来再次检查一下我们的损失值是否下降。

```py
print(loss_func(model(xb), yb))
```

输出：

```py
tensor(0.0812, grad_fn=<NllLossBackward>)
```

## 使用nn.Linear进行重构
我们继续对代码进行重构。我们将用PyTorch中的[nn.Linear](https://pytorch.org/docs/stable/nn.html#linear-layers)代替手动定义和初始化`self.weights`和`self.bias`以及计算`xb @ self.weights + self.bias`, 因为`nn.Linear`可以完成这些操作。
PyTorch中预设了很多类型的神经网络层，使用它们可以极大的简化我们的代码，通常还会带来速度上的提升。

```py
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)
```

我们初始化模型并像之前那样计算损失值：

```py
model = Mnist_Logistic()
print(loss_func(model(xb), yb))
```

输出：

```py
tensor(2.2731, grad_fn=<NllLossBackward>)
```

我们仍然可以像之前那样使用`fit`函数：

```py
fit()

print(loss_func(model(xb), yb))
```

输出：

```py
tensor(0.0817, grad_fn=<NllLossBackward>)
```

## 使用optim进行重构
PyTorch还有一个包含很多优化算法的包————`torch.optim`。我们可以使用优化器中的`step`方法执行前向传播过程中的步骤来替换手动更新每个参数。

这个方法将允许我们替换之前手动编写的优化步骤：

```py
with torch.no_grad():
    for p in model.parameters(): p -= p.grad * lr
    model.zero_grad()
```

替换后如下：

```py
opt.step()
opt.zero_grad()
```

(`optim.zero_grad()`将梯度重置为0，我们需要在计算下一次梯度之前调用它）

```py
from torch import optim
```

我们将建立模型和优化器的步骤定义为一个小函数方便将来复用。

```py
def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()
print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
```

输出：

```py
tensor(2.2795, grad_fn=<NllLossBackward>)
tensor(0.0813, grad_fn=<NllLossBackward>)
```

## 使用Dataset进行重构
Pytorch包含一个Dataset抽象类。Dataset可以是任何东西，但它始终包含一个`__len__`函数(通过Python中的标准函数`len`调用）和一个用来索引到内容中的`__getitem__`函数。
[这篇教程](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)以创建`Dataset`的自定义子类`FacialLandmarkDataset`为例进行介绍。

PyTorch中的[TensorDataset](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#TensorDataset)是一个封装了张量的Dataset。通过定义长度和索引的方式，是我们可以对张量的第一维进行迭代，索引和切片。这将使我们在训练中，获取同一行中的自变量和因变量更加容易。

```py
from torch.utils.data import TensorDataset
```

可以把`x_train`和`y_train`中的数据合并成一个简单的`TensorDataset`，这样就可以方便的进行迭代和切片操作。

```py
train_ds = TensorDataset(x_train, y_train)
```

之前，我们不得不分别对x和y的值进行迭代循环。

```py
xb = x_train[start_i:end_i]
yb = y_train[start_i:end_i]
```

现在我们可以将这两步合二为一了。

```py
xb,yb = train_ds[i*bs : i*bs+bs]
```

```py
model, opt = get_model()

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs: i * bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
```

输出：

```py
tensor(0.0825, grad_fn=<NllLossBackward>)
```

## 使用DataLoader进行重构
PyTorch的`DataLoader`负责批量数据管理，你可以使用任意的`Dataset`创建一个`DataLoader`。`DataLoader`使得对批量数据的迭代更容易。`DataLoader`自动的为我们提供每一小批量的数据来代替切片的方式`train_ds[i*bs : i*bs+bs]`。

```py
from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)
```

之前我们像下面这样按批(xb,yb)对数据进行迭代：

```py
for i in range((n-1)//bs + 1):
    xb,yb = train_ds[i*bs : i*bs+bs]
    pred = model(xb)
```

现在我们的循环变得更加简洁，因为使用了data loader来自动获取数据。

```py
for xb,yb in train_dl:
    pred = model(xb)
```

```py
model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
```

输出：

```py
tensor(0.0823, grad_fn=<NllLossBackward>)
```

多亏PyTorch中的`nn.Module`，`nn.Parameter`，`Dataset`和`DataLoader`，我们的训练代码变得非常简洁易懂。下面我们来试着增加一些用于提高模型效率所必需的的基本特征。

## 增加验证集
 在第一部分，我们仅仅是试着为我们的训练集构建一个合理的训练步骤，但实际上，我们**始终**应该有一个[验证集](https://www.fast.ai/2017/11/13/validation-sets/)来确认模型是否过拟合。
 
 打乱训练数据的顺序通常是避免不同批数据中存在相关性和过拟合的[重要步骤](https://www.quora.com/Does-the-order-of-training-data-matter-when-training-neural-networks)。另一方面，无论是否打乱顺序计算出的验证集损失值都是一样的。鉴于打乱顺序还会消耗额外的时间，所以打乱验证集数据是没有任何意义的。
 
 我们在验证集上用到的每批数据的数量是训练集的两倍，这是因为在验证集上不需要进行反向传播，这样就会占用较小的内存(因为它并不需要储存梯度）。我们利用了这一点，使用了更大的batchsize，更快的计算出了损失值。

```py
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
```

我们将会在每轮(epoch)结束后计算并输出验证集上的损失值。

(注意：在训练前我们总是会调用`model.train()`函数，在推断之前调用`model.eval()`函数，因为这些会被`nn.BatchNorm2d`，`nn.Dropout`等层使用，确保在不同阶段的准确性。）

```py
model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))
```

输出：

```py
0 tensor(0.3039)
1 tensor(0.2842)
```

## 编写fit()和get_data()函数
现在我们来重构一下我们自己的函数。
我们在计算训练集和验证集上的损失值时执行了差不多的过程两次，因此我们将这部分代码提炼成一个函数`loss_batch`，用来计算每个批的损失值。

我们为训练集传递一个优化器参数来执行反向传播。对于验证集我们不传优化器参数，这样就不会执行反向传播。

```py
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)
```

`fit`执行了训练模型的必要操作，并在每轮(epoch)结束后计算模型在训练集和测试集上的损失。

```py
import numpy as np

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)
```

`get_data`返回训练集和验证集需要使用到的dataloaders。

```py
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )
```

现在，我们只需要三行代码就可以获取数据、拟合模型了。

```py
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
    model, opt = get_model()
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)

```

输出：

```py
0 0.3368099178314209
1 0.28037907302379605
```

现在你能用这三行代码训练各种各样的模型。我们来看一下能否用它们来训练一个卷积神经网络(CNN）吧！

## 应用到卷积神经网络

我们现在将要创建一个包含三个卷积层的神经网络。因为前面章节中没有一个函数涉及到模型的具体形式，所以我们不需要对它们进行任何修改就可以训练一个卷积神经网络。

我们将会使用PyTorch中预先定义好的[Conv2d](https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d)类作为我们的卷积层。我们定义一个有三个卷积层的卷积神经网络。每个卷积层之后会执行ReLu。在最后，我们会执行一个平均池化操作。
(注意：`view`是PyTorch版的numpy `reshape`）

```py
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

lr = 0.1
```

[Momentum](https://cs231n.github.io/neural-networks-3/#sgd)是随机梯度下降的一个变型，它将前面步骤的更新也考虑在内，通常能够加快训练速度。

```py
model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

输出：

```py
0 0.3590330636262894
1 0.24007404916286468
```

## nn.Sequential
`torch.nn`中还有另一个类可以方便的用来简化我们的代码：[Sequential](https://pytorch.org/docs/stable/nn.html#torch.nn.Sequential)。一个`Sequential`对象可以序列化运行它包含的模块。这是一个更简单的搭建神经网络的方式。

想要充分利用这一优势，我们要能够使用给定的函数轻松的定义一个**自定义层**。比如说，PyTorch中没有`view`层，我们需要为我们的网络定义一个。
`Lambda`函数将会创建一个层，并在后面使用`Sequential`定义神经网络的时候用到。

```py
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)
```

使用`Sequential`创建模型非常简单：

```py
model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

输出：

```py
0 0.340390869307518
1 0.2571977460026741
```

## 封装 DataLoader
我们的卷积神经网络已经非常简洁了，但是它只能运行在MNIST数据集上，原因如下：
- 它假定输入是长度为28\*28的向量
- 它假定卷积神经网络最终输出是大小为4\*4的网格(因为这是平均值池化操作时我们使用的核大小）

让我们摆脱这两种假定，这样我们的模型就可以运行在任意的2d单通道图像上。
首先，我们可以删除最初的Lambda层，并将数据预处理放在一个生成器中。

```py
def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
```

接下来，我们用`nn.AdaptiveAvgPool2d`替换`nn.AvgPool2d`，这个函数允许我们定义期望*输出*张量的大小，而不是定义已有*输入*的大小。
这样我们的模型就可以处理任意大小的输入了。

```py
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
```

我们来试一下新的模型：

```py
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

输出：

```py
0 0.44035838775634767
1 0.2957034994959831
```

## 使用你的GPU
如果你有幸拥有支持CUDA的GPU(你可以租一个，大部分云服务提供商的价格使0.5$/每小时），那你可以用GPU来加速你的代码。
首先检查一下的GPU是否可以被PyTorch调用：

```py
print(torch.cuda.is_available())
```

输出：

```py
True
```

接下来，新建一个设备对象：

```py
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
```

然后更新一下`preprocess`函数将批运算移到GPU上计算

```py
def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
```

最后，我们可以把模型移动到GPU上。

```py
model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
```

你现在应该能发现运算变快了。

```py
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

输出：

```py
0 0.2397482816696167
1 0.2180072937965393
```

## 总结
现在我们有一个用PyTorch构建的通用数据管道和训练循环可以用来训练很多类型的模型。
想要知道训练一个模型有多么简单，可以参照`mnist_sample`这个例子。

当然了，你可能还想要在模型中加入很多其它的东西，比如数据扩充，超参数调整，训练监控，迁移学习等。这些特性可以在fastai库中获取到，该库使用与本教程中介绍的相同设计方法开发的，为想要扩展模型的学习者提供了合理的后续步骤。

在教程的开始部分，我们说了要通过例子对`torch.nn`，`torch.optim`，`Dataset`和`DataLoader`进行说明。
现在我们来总结一下，我们都讲了些什么：
- **torch.nn**
    - `Module`：创建一个可调用的，其表现类似于函数，但又可以包含状态(比如神经网络层的权重）的对象。该对象知道它包含的`Parameter`(s），并可以将梯度置为0，以及对梯度进行循环以更新权重等。
    - `Parameter`：是一个对张量的封装，它告诉`Module`在反向传播阶段更新权重。只有设置了*requires_grad*属性的张量会被更新。
    - `functional`：一个包含了梯度函数、损失函数等以及一些无状态的层，如卷积层和线性层的模块(通常使用`F`作为导入的别名）。
- `torch.optim`：包含了优化器，比如在反向阶段更新`Parameter`中权重的`SGD`。
- `Dataset`：一个抽象接口，包含了`__len__`和`__getitem__`，还包含了PyTorch提供的类，如`TensorDataset`。
- `DataLoader`：接受任意的`Dataset`并生成一个可以批量返回数据的迭代器(iterator )。
