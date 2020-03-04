# 什么是PyTorch？

> 译者：[bat67](https://github.com/bat67)
>
> 校对者：[FontTian](https://github.com/fonttian)

**作者**： [Soumith Chintala](http://soumith.ch/)

PyTorch是一个基于python的科学计算包，主要针对两类人群：

* 作为NumPy的替代品，可以利用GPU的性能进行计算
* 作为一个高灵活性、速度快的深度学习平台

## 入门

### 张量

`Tensor`(张量）类似于`NumPy`的`ndarray`，但还可以在GPU上使用来加速计算。

```python
from __future__ import print_function
import torch
```


创建一个没有初始化的5*3矩阵：

```python
x = torch.empty(5, 3)
print(x)
```

输出：

```python
tensor([[2.2391e-19, 4.5869e-41, 1.4191e-17],
        [4.5869e-41, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00]])
```

创建一个随机初始化矩阵：

```python
x = torch.rand(5, 3)
print(x)
```

输出：

```python
tensor([[0.5307, 0.9752, 0.5376],
        [0.2789, 0.7219, 0.1254],
        [0.6700, 0.6100, 0.3484],
        [0.0922, 0.0779, 0.2446],
        [0.2967, 0.9481, 0.1311]])
```

构造一个填满`0`且数据类型为`long`的矩阵:

```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
```

输出：

```python
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
```

直接从数据构造张量：

```python
x = torch.tensor([5.5, 3])
print(x)
```

输出：
```python
tensor([5.5000, 3.0000])
```

或者根据已有的tensor建立新的tensor。除非用户提供新的值，否则这些方法将重用输入张量的属性，例如dtype等：

```python
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # 重载 dtype!
print(x)                                      # 结果size一致
```

输出：

```python
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[ 1.6040, -0.6769,  0.0555],
        [ 0.6273,  0.7683, -0.2838],
        [-0.7159, -0.5566, -0.2020],
        [ 0.6266,  0.3566,  1.4497],
        [-0.8092, -0.6741,  0.0406]])
```

获取张量的形状：

```python
print(x.size())
```

输出：

```python
torch.Size([5, 3])
```

> **注意**：
>
> `torch.Size`本质上还是`tuple`，所以支持tuple的一切操作。



### 运算

一种运算有多种语法。在下面的示例中，我们将研究加法运算。

加法：形式一

```python
y = torch.rand(5, 3)
print(x + y)
```

输出：

```python
tensor([[ 2.5541,  0.0943,  0.9835],
        [ 1.4911,  1.3117,  0.5220],
        [-0.0078, -0.1161,  0.6687],
        [ 0.8176,  1.1179,  1.9194],
        [-0.3251, -0.2236,  0.7653]])
```


加法：形式二

```python
print(torch.add(x, y))
```

输出：

```python
tensor([[ 2.5541,  0.0943,  0.9835],
        [ 1.4911,  1.3117,  0.5220],
        [-0.0078, -0.1161,  0.6687],
        [ 0.8176,  1.1179,  1.9194],
        [-0.3251, -0.2236,  0.7653]])
```

加法：给定一个输出张量作为参数

```python
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
```

输出：

```python
tensor([[ 2.5541,  0.0943,  0.9835],
        [ 1.4911,  1.3117,  0.5220],
        [-0.0078, -0.1161,  0.6687],
        [ 0.8176,  1.1179,  1.9194],
        [-0.3251, -0.2236,  0.7653]])
```

加法：原位/原地操作(in-place）

```python
# adds x to y
y.add_(x)
print(y)
```

输出：

```python
tensor([[ 2.5541,  0.0943,  0.9835],
        [ 1.4911,  1.3117,  0.5220],
        [-0.0078, -0.1161,  0.6687],
        [ 0.8176,  1.1179,  1.9194],
        [-0.3251, -0.2236,  0.7653]])
```

>注意：
>
>任何一个in-place改变张量的操作后面都固定一个`_`。例如`x.copy_(y)`、`x.t_()`将更改x


也可以使用像标准的NumPy一样的各种索引操作：

```python
print(x[:, 1])
```

输出：

```python
tensor([-0.6769,  0.7683, -0.5566,  0.3566, -0.6741])
```


改变形状：如果想改变形状，可以使用`torch.view`

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
```

输出：

```python
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

如果是仅包含一个元素的tensor，可以使用`.item()`来得到对应的python数值

```python
x = torch.randn(1)
print(x)
print(x.item())
```

输出：
```python
tensor([0.0445])
0.0445479191839695
```

>后续阅读：
>
>超过100种tensor的运算操作，包括转置，索引，切片，数学运算，
线性代数，随机数等，具体访问[这里](https://pytorch.org/docs/stable/torch.html)

## 桥接 NumPy

将一个Torch张量转换为一个NumPy数组是轻而易举的事情，反之亦然。

Torch张量和NumPy数组将共享它们的底层内存位置，因此当一个改变时,另外也会改变。

### 将torch的Tensor转化为NumPy数组

输入：

```python
a = torch.ones(5)
print(a)
```

输出：

```python
tensor([1., 1., 1., 1., 1.])
```

输入：

```python
b = a.numpy()
print(b)
```

输出：

```python
[1. 1. 1. 1. 1.]
```


看NumPy数组是如何改变里面的值的：

```python
a.add_(1)
print(a)
print(b)
```

输出：

```python
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
```


### 将NumPy数组转化为Torch张量

看改变NumPy数组是如何自动改变Torch张量的：

```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
```

输出：

```python
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

CPU上的所有张量(CharTensor除外)都支持与Numpy的相互转换。

## CUDA上的张量

张量可以使用`.to`方法移动到任何设备(device）上：

```python
# 当GPU可用时,我们可以运行以下代码
# 我们将使用`torch.device`来将tensor移入和移出GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
    x = x.to(device)                       # 或者使用`.to("cuda")`方法
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype
```

输出：

```python
tensor([1.0445], device='cuda:0')
tensor([1.0445], dtype=torch.float64)
```
