# Tensors
> 译者：[Daydaylight](https://github.com/Daydaylight)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/basics/tensors_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html>

Tensors是一种特殊的数据结构，与数组和矩阵非常相似。在 PyTorch 中，我们使用tensors对模型的输入和输出以及模型的参数进行编码。



Tensors 类似于 [NumPy’s](https://numpy.org/) 的 ndarrays，只不过tensors可以在 GPU 或其他硬件加速器上运行。事实上，tensors和NumPy 数组通常可以共享相同的底层内存，这样就不需要复制数据了(参见使用 NumPy 的 Bridge)。
Tensors也针对自动微分进行了优化(我们将在后面的 [Autograd](autogradqs_tutorial.html)部分中看到更多)。如果您熟悉 ndarray，那么您就会熟悉 Tensor API。如果不熟悉，那么请跟我来！
```py
import torch
import numpy as np
```
## 初始化Tensor
Tensors可以用不同的方式初始化。看看下面的例子:

**Directly from data**

Tensors可以直接从数据中创建。数据类型可以自动推断。
```py
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```
**From a NumPy array**

Tensors可以从 NumPy 数组创建(反之亦然——参见`bridge-to-np-label`)。 
```py
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

**From another tensor:**

新tensor保留参数tensor的属性(形状、数据类型) ，除非显式重写。 
```py
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
```
输出：





```py
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.8823, 0.9150],
        [0.3829, 0.9593]])
```
**With random or constant values:**

``shape`` 是tensor维数的元组。在下面的函数中，它决定了输出tensor的维数。

```py
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```
输出：
```py
Random Tensor:
 tensor([[0.3904, 0.6009, 0.2566],
        [0.7936, 0.9408, 0.1332]])

Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```
## Tensor的属性
Tensor 属性描述了它们的形状、数据类型和存储它们的设备。
```py
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```
输出：
```py
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```
## Tensors的操作


有超过100种tensor操作，包括算术，线性代数，矩阵操作(转置，索引，切片) ，抽样等在[这里](https://pytorch.org/docs/stable/torch.html)全面描述。

这些操作都可以在 GPU 上运行(通常比在 CPU 上运行的速度更快)。如果你使用 Colab，通过Runtime > Change runtime type > GPU来分配一个GPU。

默认情况下，tensors是在CPU上创建的。我们需要使用``.to``方法显式地将tensors移动到GPU上（在检查GPU的可用性之后）。请记住，在不同的设备上复制大型的tensors，在时间和内存上都是很昂贵的!


```py
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
```


尝试列表中的一些操作。如果您熟悉 NumPy API，您会发现使用 Tensor API 简直易如反掌。


**标准的类似numpy的索引和分片：**
```py
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
```

输出：
```py
First row: tensor([1., 1., 1., 1.])
First column: tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```
**连接 tensors** 您可以使用 ``torch.cat`` 将一系列张量沿着给定的维数连接起来。另请参见 [torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html)，它是另一个张量连接运算符，与 ``torch.cat`` 略有不同。



```py
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```
输出：
```py
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```

**算术运算**



```py
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```
输出：
```py

tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

**单个元素的 tensors** 如果你有一个单元tensors，例如通过将tensors的所有值聚合成一个值，你可以使用 ``item()``将它转换成 Python 数值:













```py
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```
输出：
```py
12.0 <class 'float'>
```

**就地操作**将结果存储到操作数中的操作被称为就地操作。它们用后缀``_``来表示。例如：``x.copy_(y)``, ``x.t_()``, 将改变 ``x``。













```py
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
```
输出：
```py
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])     
```
<div class="alert alert-info"><h4>注意</h4><p>就地操作可以节省一些内存，但是在计算导数时可能会出现问题，因为会立即丢失历史记录。因此，不鼓励使用它们。</p></div>




## 与 NumPy 的桥梁
CPU 和 NumPy 数组上的Tensors都可以共享它们的底层内存位置，更改其中一个将更改另一个。
### Tensor 到 NumPy 数组




```py
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```
输出：
```py
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
```
tensor的变化反映在 NumPy 数组中。

```py
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```
输出：
```py
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
```

### NumPy 数组 到 Tensor 




```py
n = np.ones(5)
t = torch.from_numpy(n)
```
输出：
```py
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
```
NumPy 数组中的更改反映在tensor中。
```py
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```
输出：
```py
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
```
脚本的总运行时间: (0分钟1.118秒)

