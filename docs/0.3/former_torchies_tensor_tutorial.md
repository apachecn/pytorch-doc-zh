# Tensors

> 译者：@unknown
> 
> 校对者：[@bringtree](https://github.com/bringtree)

Tensors 在 PyTorch 中的操作方式 与 Torch 几乎完全相同.

用未初始化的内存创建一个大小为 (5 x 7) 的 tensor:

```py
import torch
a = torch.FloatTensor(5, 7)

```

用 mean=0, var=1 的正态分布随机初始化一个tensor:

```py
a = torch.randn(5, 7)
print(a)
print(a.size())

```

注解：

`torch.Size` 实际上是一个 tuple, 因此它支持相同的操作

## Inplace / Out-of-place

第一个不同点在于 tensor 上的所有操作, 如果想要在 tensor 自身上进行的操作 (in-place) 就要加上一个 `_` 作为后缀. 例如, `add` 是一个 out-of-place 的 version ,而 `add_` 是一个 in-place 的 version .

```py
a.fill_(3.5)
# a 的值现在变为 3.5

b = a.add(4.0)
# a 的值仍然是 3.5
# 返回的值 3.5 + 4.0 = 7.5 将作为b的值.

print(a, b)

```

还有一些像 `narrow` 的操作是没有 in-place version , 所以也就不存在 `.narrow_` . 同样的, 也有像 `fill_` 的一些操作没有 out-of-place version . 因此, `.fill` 也同样不存在.

## Zero Indexing (零索引)

Tensors 是 zero-indexed (索引从零开始)这是另外一个不同点. (在 lua 中, tensors 是 one-indexed (索引从一开始))

```py
b = a[0, 3]  # 从 a 中选择第一行第四列的值.

```

Tensors 也可以用 Python 的切片索引

```py
b = a[:, 3:5]  # 从 a 中选择所有行中第四列和第五列的值.

```

## No camel casing

接下来一个小的不同是所有的函数都不是 camelCase 了. 例如 `indexAdd` 现在被称为 `index_add_`

```py
x = torch.ones(5, 5)
print(x)

```

```py
z = torch.Tensor(5, 2)
z[:, 0] = 10
z[:, 1] = 100
print(z)

```

```py
x.index_add_(1, torch.LongTensor([4, 0]), z)
print(x)

```

## Numpy Bridge

将 torch Tensor 转换为一个 numpy array, 反之亦然. Torch Tensor 和 numpy array 将会共用底层的内存, 改变其中一个, 另外一个也会随之改变.

### 将 torch Tensor 转换为 numpy Array

```py
a = torch.ones(5)
print(a)

```

```py
b = a.numpy()
print(b)

```

```py
a.add_(1)
print(a)
print(b)    # 看一下 numpy array 值的变化

```

### 将 numpy Array 转换为 torch Tensor

```py
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)  # 看一下通过改变 np array 来自动的改变 torch Tensor

```

除了 CharTensor 之外, 所有 CPU 上的 Tensors 支持转变为 NumPy 并且 转换回来.

## CUDA Tensors

CUDA Tensors 在 pytorch 中非常好用, 并且一个 CUDA tensor 从 CPU 转换到 GPU 仍将保持它底层的类型.

```py
# 让我们在 CUDA 可用的时候运行这个单元
if torch.cuda.is_available():
    # 创建一个 LongTensor 并且将其转换使用 GPU
    # 的 torch.cuda.LongTensor 类型
    a = torch.LongTensor(10).fill_(3).cuda()
    print(type(a))
    b = a.cpu()
    # 将它转换到 CPU
    # 类型变回 torch.LongTensor

```
