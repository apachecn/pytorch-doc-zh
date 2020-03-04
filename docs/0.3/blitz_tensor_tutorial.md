# PyTorch 是什么？

> 译者：[@小王子](https://github.com/VPrincekin)
> 
> 校对者：[@李子文](https://github.com/liziwenzzzz)

它是一个基于 Python 的科学计算包, 其主要是为了解决两类场景:

*   NumPy 的替代品, 以使用 GPU 的强大加速功能
*   一个深度学习研究平台, 提供最大的灵活性和速度

## 新手入门

### Tensors(张量）

Tensors 与 NumPy 的 ndarrays 非常相似, 除此之外还可以在 GPU 上使用张量来加速计算.

```py
from __future__ import print_function
import torch

```

构建一个 5x3 的矩阵, 未初始化的:

```py
x = torch.Tensor(5, 3)
print(x)

```

构建一个随机初始化的矩阵:

```py
x = torch.rand(5, 3)
print(x)

```

获得 size:

```py
print(x.size())

```

注解：

`torch.Size` 实际上是一个 tuple(元组）, 所以它支持所有 tuple(元组）的操作.

### 操作

针对操作有许多语法. 在下面的例子中, 我们来看看加法运算.

加法: 语法 1

```py
y = torch.rand(5, 3)
print(x + y)

```

加法: 语法 2

```py
print(torch.add(x, y))

```

加法: 提供一个输出 tensor 作为参数

```py
result = torch.Tensor(5, 3)
torch.add(x, y, out = result)
print(result)

```

加法: in-place(就地操作）

```py
# adds x to y
y.add_(x)
print(y)

```

注解：

任何改变张量的操作方法都是以后缀 `_` 结尾的. 例如: `x.copy_(y)`, `x.t_()`, 将改变张量 `x`.

你可以用类似Numpy的索引来处理所有的张量！

```py
print(x[:, 1])

```

改变大小: 如果你想要去改变tensor的大小, 可以使用 `torch.view`:

```py
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

```

**稍候阅读:**

> 100+ Tensor 操作, 包括换位, 索引, 切片, 数学运算, 线性代数, 随机数, 等等, 都在 [这里](http://pytorch.apachecn.org/cn/docs/0.3.0/torch.html) 有描述.

## NumPy Bridge

将一个 Torch Tensor 转换为 NumPy 数组, 反之亦然.

Torch Tensor 和 NumPy 数组将会共享它们的实际的内存位置, 改变一个另一个也会跟着改变.

### 转换一个 Torch Tensor 为 NumPy 数组

```py
a = torch.ones(5)
print(a)

```

```py
b = a.numpy()
print(b)

```

查看 numpy 数组是如何改变的.

```py
a.add_(1)
print(a)
print(b)

```

### 转换 NumPy 数组为 Torch Tensor

看看改变 np 数组之后 Torch Tensor 是如何自动改变的

```py
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out = a)
print(a)
print(b)

```

除了 CharTensor 之外, CPU 上的所有 Tensor 都支持与Numpy进行互相转换

## CUDA Tensors

可以使用 `.cuda` 方法将 Tensors 在GPU上运行.

```py
# 只要在  CUDA 是可用的情况下, 我们可以运行这段代码
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y

```
