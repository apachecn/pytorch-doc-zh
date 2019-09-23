# torch.nn 到底是什么？

杰里米·霍华德，[ fast.ai [HTG1。由于雷切尔·托马斯和弗朗西斯科英厄姆。](https://www.fast.ai)

我们建议运行本教程为笔记本电脑，而不是一个脚本。要下载笔记本（.ipynb）文件，请点击页面顶部的链接。

PyTorch提供了优雅的设计模块和类[ torch.nn ](https://pytorch.org/docs/stable/nn.html)，[
torch.optim
](https://pytorch.org/docs/stable/optim.html)，[数据集](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset)和[的DataLoader
](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)，以帮助您创建和火车神经网络。为了充分利用他们的权力和它们进行自定义你的问题，你需要真正了解他们在做什么。为了开发这样的认识，我们将第一列火车在MNIST数据，而无需使用来自这些模型的任何功能设置基本的神经网络;我们最初将只使用最基本的PyTorch张量的功能。然后，我们将逐步增加一个功能可以从`
torch.nn`，`torch.optim`，`数据集 `或`的DataLoader
`的时间，正好显示每一块做什么，以及它如何使代码或者更简洁，更灵活。

**本教程假设你已经安装PyTorch，并熟悉操作张的基础。** （如果你熟悉numpy的数组操作，你会发现这里使用几乎相同的PyTorch张量操作）。

## MNIST数据建立

我们将使用经典[ MNIST ](http://deeplearning.net/data/mnist/)数据集，它由手绘数字黑色和白色图像（0至9）的。

我们将使用[ pathlib
](https://docs.python.org/3/library/pathlib.html)与路径处理（Python的3标准库的一部分），并使用[请求](http://docs.python-
requests.org/en/master/)下载数据集。当我们使用它们，我们将只导入模块，这样你就可以清楚地看到什么是在每个点被使用。

    
    
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
    

此数据集是在numpy的阵列格式，以及使用泡菜，用于序列数据的特定蟒格式已经被存储。

    
    
    import pickle
    import gzip
    
    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
            ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    

每个图像是28×28，并且被存储为784长度（= 28x28）的扁平行。让我们来看看一个[]我们需要重塑它首先到2d。

    
    
    from matplotlib import pyplot
    import numpy as np
    
    pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
    print(x_train.shape)
    

![../_images/sphx_glr_nn_tutorial_001.png](../_images/sphx_glr_nn_tutorial_001.png)

日期：

    
    
    (50000, 784)
    

PyTorch使用`torch.tensor`，而不是numpy的阵列，因此我们需要我们的数据转换。

    
    
    import torch
    
    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    n, c = x_train.shape
    x_train, x_train.shape, y_train.min(), y_train.max()
    print(x_train, y_train)
    print(x_train.shape)
    print(y_train.min(), y_train.max())
    

Out:

    
    
    tensor([[0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.],
            ...,
            [0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.]]) tensor([5, 0, 4,  ..., 8, 4, 8])
    torch.Size([50000, 784])
    tensor(0) tensor(9)
    

## 从头（无torch.nn）神经网络

让我们先创建一个使用无非是PyTorch张量操作的模式。我们假设你已经熟悉了神经网络的基本知识。 （如果你没有，你可以在[ course.fast.ai
](https://course.fast.ai)学习他们）。

PyTorch提供方法来创建随机或零填充的张量，我们将用它来创建我们的权重和偏见一个简单的线性模型。这些只是普通的张量，一个非常特殊的另外：我们告诉PyTorch，他们需要一个梯度。这将导致PyTorch记录所有对张进行操作的，所以它可以反向传播
_自动_ 在计算梯度！

对于权重，我们设定`requires_grad`**后** 初始化，因为我们不希望包含在梯度一步。 （请注意，一个trailling `_
`在PyTorch表示该操作是在就地进行。）

Note

我们正在与[泽维尔初始化](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)这里初始化权重（通过用1
/ SQRT（N乘以））。

    
    
    import math
    
    weights = torch.randn(784, 10) / math.sqrt(784)
    weights.requires_grad_()
    bias = torch.zeros(10, requires_grad=True)
    

由于PyTorch的自动计算梯度，我们可以使用任何标准的Python函数（或可调用对象）作为模型的能力！所以让我们只写一个简单的矩阵乘法和广播除了创建一个简单的线性模型。我们还需要一个激活的功能，所以我们写
log_softmax 并使用它。记住：虽然PyTorch提供了大量的预先书面挂失功能，激活功能，等等，你可以很容易地编写自己的使用普通蟒蛇。
PyTorch甚至会为你的函数自动创建快速的GPU或CPU矢量代码。

    
    
    def log_softmax(x):
        return x - x.exp().sum(-1).log().unsqueeze(-1)
    
    def model(xb):
        return log_softmax(xb @ weights + bias)
    

另外，在上述中，`@`代表点积操作。我们将调用数据的一个批次（在这种情况下，64个图像）我们的函数。这是一个 _直传_
。请注意，我们的预测会不会比随机更好的在这个阶段，因为我们开始与随机权。

    
    
    bs = 64  # batch size
    
    xb = x_train[0:bs]  # a mini-batch from x
    preds = model(xb)  # predictions
    preds[0], preds.shape
    print(preds[0], preds.shape)
    

Out:

    
    
    tensor([-2.4595, -1.9240, -1.9316, -2.6839, -2.5857, -1.9705, -2.4925, -2.4569,
            -2.2955, -2.6387], grad_fn=<SelectBackward>) torch.Size([64, 10])
    

正如你看到的，`preds`张量不仅包含了张量的值，也是一个渐变的功能。我们将利用这个做以后backprop。

让我们来实现负对数似然的损失函数（同样，我们可以用标准的Python）使用方法：

    
    
    def nll(input, target):
        return -input[range(target.shape[0]), target].mean()
    
    loss_func = nll
    

让我们来看看我们的损失与我们的随机模型，所以我们可以看到，如果我们提高后backprop后通过。

    
    
    yb = y_train[0:bs]
    print(loss_func(preds, yb))
    

Out:

    
    
    tensor(2.3762, grad_fn=<NegBackward>)
    

我们还要实现一个函数来计算我们模型的准确性。对于每一个预测，如果与最大值的指标目标值相匹配，那么预测是正确的。

    
    
    def accuracy(out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()
    

让我们来看看我们的随机模型的准确性，所以我们可以看到，如果我们的准确度随着我们的损失得到改善。

    
    
    print(accuracy(preds, yb))
    

Out:

    
    
    tensor(0.0625)
    

现在，我们可以运行一个训练循环。对于每次迭代，我们将：

  * 选择迷你一批数据（大小`BS的 `）
  * 使用模型进行预测
  * 计算损失
  * `loss.backward（） `更新模型的梯度，在这种情况下，`权重 `和`偏压 `。

我们现在使用这些梯度更新权重和偏见。我们这样做的`torch.no_grad内（）
`上下文管理，因为我们不希望被记录为我们的下一个梯度的计算这些操作。你可以阅读更多关于PyTorch的Autograd如何记录操作[此处[HTG5。](https://pytorch.org/docs/stable/notes/autograd.html)

然后，我们设置渐变到零，让我们准备下一个循环。否则，我们的梯度将记录这一切已经发生的业务流水账（即`loss.backward（） `_增加_
梯度到任何已存储的，而不是取代它们）。

小费

您可以使用标准的Python调试器分步PyTorch代码，让您在每一步检查各种变量值。取消注释`set_trace（） `下面尝试一下。

    
    
    from IPython.core.debugger import set_trace
    
    lr = 0.5  # learning rate
    epochs = 2  # how many epochs to train for
    
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
    

就是这样：我们创建和培养了极少的神经网络（在这种情况下，逻辑回归，因为我们没有隐藏层）完全从头开始！

让我们来看看损失，准确性和比较那些我们前面了。我们预计，损失将有所减少和准确性都有所增加，并且他们有。

    
    
    print(loss_func(model(xb), yb), accuracy(model(xb), yb))
    

Out:

    
    
    tensor(0.0824, grad_fn=<NegBackward>) tensor(1.)
    

## 使用torch.nn.functional

现在，我们将重构我们的代码，以便它像以前一样做同样的事情，只有我们将开始利用PyTorch的`NN
`类，使其更加简洁和灵活。在这里，从每一步，我们应该使我们的代码的一个或多个：更短，更容易理解，和/或更灵活。

第一和最容易的步骤是使我们的代码越短由与那些从`代替我们的手写激活和损耗函数torch.nn.functional`（其通常导入到命名空间`F
[HTG7按照惯例]）。该模块包含在`torch.nn
`库中的所有功能（而库的其它部分包含类）。除了各种各样的损失和激活功能，你会也在这里找到有关创建神经网络，如池功能的一些方便的功能。
（也有做卷积，线性层等功能，但正如我们所看到的，这些通常是使用该库的其他部分更好地处理。）`

如果您在使用负对数似然损失和日志SOFTMAX激活，然后Pytorch提供了一个单一的功能`F.cross_entropy
`，结合了两个。因此，我们甚至可以删除从我们的模型激活功能。

    
    
    import torch.nn.functional as F
    
    loss_func = F.cross_entropy
    
    def model(xb):
        return xb @ weights + bias
    

请注意，我们不再调用`log_softmax`中的`模型 `功能。让我们确认，我们的损失和准确度都和以前一样：

    
    
    print(loss_func(model(xb), yb), accuracy(model(xb), yb))
    

Out:

    
    
    tensor(0.0824, grad_fn=<NllLossBackward>) tensor(1.)
    

## 使用重构nn.Module

接下来，我们将使用`nn.Module`和`nn.Parameter`，更清晰和更简洁的训练循环。我们继承`nn.Module
`（这本身是一类，并能跟踪状态）。在这种情况下，我们要创建一个保存我们的砝码，偏置和方法向前台阶的类。 `nn.Module`具有许多属性和方法（如`
.parameters的（） `和`.zero_grad （） `），我们将使用。

Note

`nn.Module`（大写的M）是PyTorch具体的概念，是我们将要使用大量的类。 `nn.Module
`不与的Python的概念混淆（小写`M
`）[模块](https://docs.python.org/3/tutorial/modules.html)这是Python代码的文件可以导入。

    
    
    from torch import nn
    
    class Mnist_Logistic(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
            self.bias = nn.Parameter(torch.zeros(10))
    
        def forward(self, xb):
            return xb @ self.weights + self.bias
    

因为现在我们使用的对象，而不是只使用一个功能，我们首先必须初始化我们的模型：

    
    
    model = Mnist_Logistic()
    

现在，我们可以计算出之前以相同方式的损失。需要注意的是`nn.Module`使用的对象，就好像它们是函数（即它们 _调用_
），但幕后Pytorch会打电话给我们的`向前 `自动方法。

    
    
    print(loss_func(model(xb), yb))
    

Out:

    
    
    tensor(2.2882, grad_fn=<NllLossBackward>)
    

以前我们的训练循环，我们必须更新值按名称各参数，并分别手动清零每个参数的梯度，这样的：

    
    
    with torch.no_grad():
        weights -= weights.grad * lr
        bias -= bias.grad * lr
        weights.grad.zero_()
        bias.grad.zero_()
    

现在，我们可以利用model.parameters（）和model.zero_grad（），以使这些步骤更简洁，不容易（这两者都是由PyTorch为`
nn.Module`中定义）遗忘我们的一些参数，特别是如果我们有一个更复杂的模型的错误：

    
    
    with torch.no_grad():
        for p in model.parameters(): p -= p.grad * lr
        model.zero_grad()
    

我们将包裹我们的小训练循环在`适合 `功能，所以我们可以稍后再运行它。

    
    
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
    

让我们仔细检查，我们的损失已经下降：

    
    
    print(loss_func(model(xb), yb))
    

Out:

    
    
    tensor(0.0795, grad_fn=<NllLossBackward>)
    

## 使用重构nn.Linear

我们继续重构我们的代码。代替手动定义和初始化`self.weights`和`self.bias`，并计算`XB  @
self.weights  +  self.bias`，我们将改用Pytorch类[ nn.Linear
](https://pytorch.org/docs/stable/nn.html#linear-layers)为线性层，它确实所有的我们。
Pytorch有许多类型的预定义层，可以大大简化我们的代码，而且往往使得它更快了。

    
    
    class Mnist_Logistic(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(784, 10)
    
        def forward(self, xb):
            return self.lin(xb)
    

我们实例模型和计算以同样的方式和以前的损失：

    
    
    model = Mnist_Logistic()
    print(loss_func(model(xb), yb))
    

Out:

    
    
    tensor(2.3549, grad_fn=<NllLossBackward>)
    

我们仍然能够像以前一样使用我们同样`适合 `方法。

    
    
    fit()
    
    print(loss_func(model(xb), yb))
    

Out:

    
    
    tensor(0.0820, grad_fn=<NllLossBackward>)
    

## 使用重构的Optim

Pytorch还具有与各种优化算法的软件包，`torch.optim`。我们可以使用`步骤
`方法从我们的优化器作为一个前步骤，而不是手动更新各参数。

这将让我们取代我们以前的手工编码的优化步骤：

    
    
    with torch.no_grad():
        for p in model.parameters(): p -= p.grad * lr
        model.zero_grad()
    

而是只需使用：

    
    
    opt.step()
    opt.zero_grad()
    

（`optim.zero_grad（） `复位梯度为0，我们需要计算用于下一minibatch梯度之前调用它。）

    
    
    from torch import optim
    

我们将定义一个小功能来创建我们的模型和优化，所以我们可以在未来重复使用。

    
    
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
    

Out:

    
    
    tensor(2.3540, grad_fn=<NllLossBackward>)
    tensor(0.0828, grad_fn=<NllLossBackward>)
    

## 使用重构数据集

PyTorch有一个抽象的DataSet类。数据集可以是任何具有`__len__`函数（由Python的标准`LEN`函数调用）和`
__getitem__`用作索引的一种方式进去。
[本教程](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)遍历创建自定义`
FacialLandmarkDataset`类作为`数据集 `的子类的一个很好的例子。

PyTorch的[ TensorDataset
](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#TensorDataset)是一个数据集包装张量。通过定义索引的长度和方式，这也为我们提供了一种方式来遍历，指数，并沿张量的第一个维度切片。这将使它更容易访问两个自变量和因变量在同一行，因为我们训练。

    
    
    from torch.utils.data import TensorDataset
    

既`x_train`和`y_train`可以以组合的单`TensorDataset`，这将更容易遍历，切片。

    
    
    train_ds = TensorDataset(x_train, y_train)
    

以前，我们必须通过x的minibatches和y值分别进行迭代：

    
    
    xb = x_train[start_i:end_i]
    yb = y_train[start_i:end_i]
    

现在，我们可以做这两个步骤一起：

    
    
    xb,yb = train_ds[i*bs : i*bs+bs]
    
    
    
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
    

Out:

    
    
    tensor(0.0823, grad_fn=<NllLossBackward>)
    

## 使用重构的DataLoader

Pytorch的`的DataLoader`负责管理批次。您可以创建一个`的DataLoader`从任何`数据集 [HTG11。 `
的DataLoader`可以更容易地在批次迭代。而不必使用`train_ds [I * BS  ： 我* BS + BS]
`时，的DataLoader给我们每个自动minibatch。`

    
    
    from torch.utils.data import DataLoader
    
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs)
    

此前，我们的循环遍历批次（XB，YB）是这样的：

    
    
    for i in range((n-1)//bs + 1):
        xb,yb = train_ds[i*bs : i*bs+bs]
        pred = model(xb)
    

现在，我们的循环是更清洁，如（XB，YB）会自动从数据加载器加载：

    
    
    for xb,yb in train_dl:
        pred = model(xb)
    
    
    
    model, opt = get_model()
    
    for epoch in range(epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)
    
            loss.backward()
            opt.step()
            opt.zero_grad()
    
    print(loss_func(model(xb), yb))
    

Out:

    
    
    tensor(0.0807, grad_fn=<NllLossBackward>)
    

由于Pytorch的`nn.Module`，`nn.Parameter`，`数据集 `，和`的DataLoader
`，我们的训练循环现显着更小，更容易理解。现在，让我们尝试添加必要在实践中创造effecive模型的基本特征。

## 添加验证

在第1节，我们只是想获得一个合理的训练循环建立对我们的训练数据的使用。在现实中，你 **总是**
也应该有一个[验证设置](https://www.fast.ai/2017/11/13/validation-sets/)，以确定如果你过度拟合。

洗牌训练数据是[重要](https://www.quora.com/Does-the-order-of-training-data-matter-when-
training-neural-
networks)为了防止批料和过拟合之间的相关性。在另一方面，确认损失将是相同的，我们是否洗牌验证设置与否。由于洗牌需要额外的时间，这是没有意义的洗牌验证数据。

我们将使用的批次数量为验证集是两倍，对于训练集。这是因为验证集不需要反向传播并且因此需要较少的存储器（它并不需要存储的梯度）。我们利用这一点来使用更大的批量大小和更迅速地计算损失。

    
    
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    
    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
    

我们将计算，并在每个时代的结束打印确认损失。

（请注意，我们随时调用`model.train（） `训练前，和`model.eval（） `推理之前，因为这些都是通过层如`
nn.BatchNorm2d`和`nn.Dropout`，以确保这些不同的阶段适当的行为使用）。

    
    
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
    

Out:

    
    
    0 tensor(0.3543)
    1 tensor(0.4185)
    

## 创建拟合（）和GET_DATA（）

现在，我们会尽自己的一点点重构。因为我们经历了类似的过程计算训练集和验证集既损失的两倍，让我们做的是为自己的功能，`loss_batch
`，其计算为一个损失批量。

我们通过一个优化在训练集，并用它来执行backprop。对于验证集，我们没有通过优化，因此该方法不执行backprop。

    
    
    def loss_batch(model, loss_func, xb, yb, opt=None):
        loss = loss_func(model(xb), yb)
    
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()
    
        return loss.item(), len(xb)
    

`适合 `运行必要的操作来训练我们的模型和计算每个时期的训练和验证的损失。

    
    
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
    

`GET_DATA`返回dataloaders用于训练和验证集。

    
    
    def get_data(train_ds, valid_ds, bs):
        return (
            DataLoader(train_ds, batch_size=bs, shuffle=True),
            DataLoader(valid_ds, batch_size=bs * 2),
        )
    

现在，我们整个获取数据装载机和拟合模型的过程可在3行代码运行：

    
    
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
    model, opt = get_model()
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)
    

Out:

    
    
    0 0.29517731761932375
    1 0.2856491837501526
    

您可以使用这些基本的3行代码，培养各种各样的模型。让我们来看看，如果我们可以用它们来训练卷积神经网络（CNN）！

## 切换到CNN

现在，我们要建立我们的神经网络有三个卷积层。因为没有上一节中的功能，承担有关模型的形式什么，我们就可以用它们来训练CNN不作任何修改。

我们将使用Pytorch的预定义[ Conv2d
](https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d)类作为我们的卷积层。我们定义了一个CNN
3个卷积层。每一圈之后是RELU。最后，我们执行的平均池。 （请注意，`视图 `时PyTorch的版本numpy的年代`重塑 `）

    
    
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
    

[动量](https://cs231n.github.io/neural-
networks-3/#sgd)是采用以前的更新考虑以及和通常会导致更快的训练上随机梯度下降的变化。

    
    
    model = Mnist_CNN()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)
    

Out:

    
    
    0 0.35516392378807066
    1 0.25097596280574797
    

## nn.Sequential

`torch.nn`还有一个方便的类，我们可以用它来简单的我们的代码：[顺序[HTG5。 A `序贯
`对象运行的每个包含在其内的模块的，以顺序的方式。这是我们写的神经网络的一个简单的方法。](https://pytorch.org/docs/stable/nn.html#torch.nn.Sequential)

要充分利用这一点，我们需要能够轻松地从一个给定函数定义 **自定义层[HTG1。例如，PyTorch没有一个 查看层，我们需要创建一个为我们的网络。 `
LAMBDA`将创建一个层，其限定与`序贯 `网络时，我们可以再使用。**

    
    
    class Lambda(nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func
    
        def forward(self, x):
            return self.func(x)
    
    
    def preprocess(x):
        return x.view(-1, 1, 28, 28)
    

与`序贯 `创建的模型只是：

    
    
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
    

Out:

    
    
    0 0.4354481062412262
    1 0.23530314621925355
    

## 包裹的DataLoader

Our CNN is fairly concise, but it only works with MNIST, because:

    

  * 它假定输入是一个28 * 28长向量
  * 它假定最终CNN格大小为4×4（因为这是平均

我们使用的池内核大小）

让我们摆脱这两个假设，因此我们的模型与任何2D单通道形象工程。首先，我们可以删除初始LAMBDA层但移动数据预处理成发生器：

    
    
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
    

接下来，我们可以替换`nn.AvgPool2d`与`nn.AdaptiveAvgPool2d`，这使我们能够限定 _输出的大小_
张量我们想要的，而不是 _输入_ 张量，我们有。其结果是，我们的模型将与任何大小的输入工作。

    
    
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
    

让我们来尝试一下：

    
    
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)
    

Out:

    
    
    0 0.427955148935318
    1 0.2865892390727997
    

## 使用你的GPU

如果你足够幸运，有机会获得一个支持CUDA的GPU（你可以租一个大约$ 0.50
/小时，大多数云供应商），你可以用它来加速你的代码。首先检查你的GPU在Pytorch工作：

    
    
    print(torch.cuda.is_available())
    

Out:

    
    
    True
    

然后为它创建一个设备对象：

    
    
    dev = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    

让我们更新`预处理 `以分批转移到GPU：

    
    
    def preprocess(x, y):
        return x.view(-1, 1, 28, 28).to(dev), y.to(dev)
    
    
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
    train_dl = WrappedDataLoader(train_dl, preprocess)
    valid_dl = WrappedDataLoader(valid_dl, preprocess)
    

最后，我们可以把我们的模型转移到GPU。

    
    
    model.to(dev)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    

你会发现，现在运行得更快：

    
    
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)
    

Out:

    
    
    0 0.1994757580757141
    1 0.1848785598754883
    

## 关闭的想法

我们现在有一个通用数据管道和训练循环，您可以使用训练多种类型的使用Pytorch模型。看样板可以训练多么简单，现在是，看看在 mnist_sample
样本笔记本。

当然，也有很多事情你需要添加，比如数据扩充，超参数调整，监控培训，迁移学习，等等。这些功能在fastai库，它已使用本教程中相同的设计方法研制，对从业人员提供希望进一步采取他们的模型的下一步可用。

我们承诺在本教程的开始，我们会通过例子来说明每种`torch.nn`，`torch.optim`，`数据集 `和`的DataLoader
`。因此，让我们总结一下，我们看到：

>   * **torch.nn**

>     * `模块 `：创建一个可调用的，其行为类似的功能，但也可以包含状态（如神经网络层的权重）它知道什么`参数
[HTG14（S）它包含并可以通过它们更新权重为零，他们的所有梯度，循环等HTG15]

>     * `参数 `：用于张量，告诉一个包装一`模块 `，它具有需要backprop期间更新权重。只用 requires_grad
属性集张量被更新

>     * `官能 `：一个模块（通常导入到`F`按照惯例命名空间），其包含激活的功能，损失函数，等等，以及层的非状态版本，如卷积和线性层。

> `

>   * `torch.optim`：包含优化如`SGD`，它更新的权重的`参数 `在向后步骤

>   * `数据集 `：对象的抽象接口与`__len__`状语从句一个`__getitem__`，包括设置有Pytorch类，如`
TensorDataset`

>   * `的的DataLoader`：取任何`数据集 `并且创建返回数据的批量的迭代器。

>

**脚本的总运行时间：** （1分钟2.848秒）

[`Download Python source code:
nn_tutorial.py`](../_downloads/a6246751179fbfb7cad9222ef1c16617/nn_tutorial.py)

[`Download Jupyter notebook:
nn_tutorial.ipynb`](../_downloads/5ddab57bb7482fbcc76722617dd47324/nn_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](../intermediate/torchvision_tutorial.html "TorchVision Object
Detection Finetuning Tutorial") [![](../_static/images/chevron-right-
orange.svg) Previous](saving_loading_models.html "Saving and Loading Models")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * 什么是 torch.nn  _真的_ ？ 
    * MNIST数据建立
    * 从头神经网（无torch.nn）
    * 使用torch.nn.functional 
    * 使用nn.Module重构
    * 使用nn.Linear重构
    * 使用重构的Optim 
    * 使用数据集重构
    * 使用的DataLoader重构
    * 添加验证
    * 创建拟合（）和GET_DATA（）
    * 转为CNN 
    * nn.Sequential 
    * 包装纸的DataLoader 
    * 使用你的GPU 
    * 合的想法

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)

## 文件

对于PyTorch访问完整的开发文档

[View Docs](https://pytorch.org/docs/stable/index.html)

## 教程

获取详细的教程，对于初学者和高级开发者

[View Tutorials](https://pytorch.org/tutorials)

## 资源

查找开发资源，并得到回答您的问题

[View Resources](https://pytorch.org/resources)

[](https://pytorch.org/)

  * [ PyTorch ](https://pytorch.org/)
  * [入门](https://pytorch.org/get-started)
  * [特点](https://pytorch.org/features)
  * [生态系统](https://pytorch.org/ecosystem)
  * [博客](https://pytorch.org/blog/)
  * [资源](https://pytorch.org/resources)

  * [支持](https://pytorch.org/support)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [讨论](https://discuss.pytorch.org)
  * [ Github的问题](https://github.com/pytorch/pytorch/issues)
  * [松弛](https://pytorch.slack.com)
  * [贡献](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md)

  * 跟着我们
  * 邮箱地址

[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)

  * 入门
  * 特点
  * 生态系统
  * [博客](https://pytorch.org/blog/)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [资源](https://pytorch.org/resources)
  * [ Github的](https://github.com/pytorch/pytorch)

