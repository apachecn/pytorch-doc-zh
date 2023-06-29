# 自动微分运算-`TORCH.AUTOGRAD`

> 译者：[runzhi214](https://github.com/runzhi214)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/Introduction_to_PyTorch/automatic_differentation_with_torch_autograd/>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html>

在训练神经网络的时候，最常用的算法就是反向传播算法。在这个算法中，模型参数根据针对每个给定参数的损失函数的梯度来调整。

为了计算这些梯度，PyTorch有一个内置的微分运算引擎叫`torch.autograd`。它支持对与任何计算图自动计算梯度。

考虑一个最简单的单层神经网络，它有输入值`x`、参数`x`和`b`、和一些损失函数。它可以在PyTorch中这么定义：

```py
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```

## 张量、函数和计算图

这个代码会定义下面的计算图:

[图片]

在这个网络中，`w`和`b`都是我们需要优化的参数。因此，我们需要能够对这些变量分别计算损失函数的梯度。为了这么做，我们设置这些张量的`requires_grad`属性。

> 注意:
> 你可以在创建张量的时候就设置`requires_grad`的值、或者在创建之后用`x.requires_grad_(True)`方法来设置。

我们对张量应用来创建计算图的函数事实上是一个`Function`类的对象。这个对象知道如何*前向*地计算函数，以及如何在*向后传播的步骤中*计算导数。反向传播函数的一个引用保存在张量的`grad_fn`的属性中。你可以在[文档](https://pytorch.org/docs/stable/autograd.html#function)中找到更多关于`Function`的信息。

```py
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
```

Out:

```py
Gradient function for z = <AddBackward0 object at 0x7f9615a14580>
Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x7f9615a14bb0>
```

## 计算梯度

