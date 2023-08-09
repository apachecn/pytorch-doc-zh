# PyTorch 张量入门

> 译者：[Fadegentle](https://github.com/Fadegentle)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/introyt/tensors_deeper_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html>

请跟随下面的视频或在 [youtube](https://www.youtube.com/watch?v=r7QDUPb2dCM) 上观看。

<iframe width="560" height="315" src="https://www.youtube.com/embed/r7QDUPb2dCM" title="Introduction to PyTorch Tensors" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

张量是 PyTorch 的核心数据抽象。本互动笔记本将深入介绍 `torch.Tensor` 类。

首先，让我们导入 PyTorch 模块。另外因为某些示例需要，我们还要添加 Python 的数学模块。

```python
import torch
import math
```

## 创建张量
创建张量最简单的方法是调用 `torch.empty()`：

```python
x = torch.empty(3, 4)
print(type(x))
print(x)
```

输出：
```shell
<class 'torch.Tensor'>
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])
```

让我们来解读一下刚才的操作：

- 我们使用 `torch` 模块附带的众多工厂方法之一创建了一个张量。
- 该张量本身是二维的，有 3 行 4 列。
- 返回对象的类型是 `torch.Tensor`，它是 `torch.FloatTensor` 的别名；默认情况下，PyTorch 张量使用 32 位浮点数填充。（下面将详细介绍数据类型。）
- 在打印张量时，您可能会看到一些随机的数值。`torrent.empty()` 调用为张量分配了内存，但没用任何值对其初始化——所以您看到的是分配时内存中的内容。

关于张量及其维数和术语的简要说明：

- 有时，您会看到一个一维张量被称为向量。
- 同样，二维张量通常被称为矩阵。
- 超过两个维度的情况一般都被称为张量。

在大多数情况下，您会想要用一些值来初始化您的张量。常见的情况有全 0、全 1 或随机值，`torch` 模块为这些情况都提供了工厂方法：

```python
zeros = torch.zeros(2, 3)
print(zeros)

ones = torch.ones(2, 3)
print(ones)

torch.manual_seed(1729)
random = torch.rand(2, 3)
print(random)
```

输出：
```shell
tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[0.3126, 0.3791, 0.3087],
        [0.0736, 0.4216, 0.0691]])
```

这些工厂方法都如您所愿——我们得到了一个全是 0 的张量，另一个全是 1 的张量，还有一个包含 0 和 1 之间随机值的张量。

### 随机张量和种子设置
说到随机张量，您是否注意到紧接在它之前调用了 `torch.manual_seed()` ？将张量初始化为随机值是常见的做法，比如模型的学习权重，但在某些情况下（尤其是在研究环境中）您可能希望确保您的结果是可重现的。手动设置随机数生成器的种子就能做到这个，让我们仔细看一下：

```python
torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print(random1)

random2 = torch.rand(2, 3)
print(random2)

torch.manual_seed(1729)
random3 = torch.rand(2, 3)
print(random3)

random4 = torch.rand(2, 3)
print(random4)
```

输出：
```shell
tensor([[0.3126, 0.3791, 0.3087],
        [0.0736, 0.4216, 0.0691]])
tensor([[0.2332, 0.4047, 0.2162],
        [0.9927, 0.4128, 0.5938]])
tensor([[0.3126, 0.3791, 0.3087],
        [0.0736, 0.4216, 0.0691]])
tensor([[0.2332, 0.4047, 0.2162],
        [0.9927, 0.4128, 0.5938]])
```

您应该在上面看到，`random1` 和 `random3` 携带着相同的值，`random2` 和 `random4` 也是如此。手动设置随机数生成器的种子会重置它，因此一般而言，相同随机数的相同计算应有相同的结果。

有关更多信息，请参阅 [PyTorch 文档关于可重现性的部分](https://pytorch.org/docs/stable/notes/randomness.html)。

### 张量形状
通常，当您对两个或多个张量执行操作时，它们需要具有相同的形状——也就是说，维数相同和每个维度中的单元数相同。为此，我们有 `torch.*_like()` 方法：

```python
x = torch.empty(2, 2, 3)
print(x.shape)
print(x)

empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)
```

输出：
```shell
torch.Size([2, 2, 3])
tensor([[[ 1.3323e-33,  0.0000e+00,  1.2565e-33],
         [ 0.0000e+00, -6.9300e-03, -2.9693e-02]],

        [[-4.2094e-02,  2.6203e-02,  6.7262e-44],
         [ 0.0000e+00,  6.7262e-44,  0.0000e+00]]])
torch.Size([2, 2, 3])
tensor([[[ 6.0476e-35,  0.0000e+00, -9.5918e-01],
         [ 4.5559e-41,  4.4842e-44,  0.0000e+00]],

        [[ 8.9683e-44,  0.0000e+00,  1.3039e-33],
         [ 0.0000e+00,  1.1351e-43,  0.0000e+00]]])
torch.Size([2, 2, 3])
tensor([[[0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.]]])
torch.Size([2, 2, 3])
tensor([[[1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.]]])
torch.Size([2, 2, 3])
tensor([[[0.6128, 0.1519, 0.0453],
         [0.5035, 0.9978, 0.3884]],

        [[0.6929, 0.1703, 0.1384],
         [0.4759, 0.7481, 0.0361]]])
```

上面代码单元中的第一个新功能是在张量上使用 `.shape` 属性。该属性包含一个列表，其中包含张量各维度的范围——在我们的例子中，`x` 是一个形状为 2 x 2 x 3 的三维张量。

下面，我们将调用 `.empty_like()`、`.zeros_like()`、`.one_like()` 和 `.rand_like()` 方法。通过使用 `.shape` 属性，我们可以验证这些方法返回的张量的维数和范围都是相同的。

创建张量的最后一种方法是直接从 PyTorch 集合中指定数据：

```python
some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print(some_constants)

some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print(some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print(more_integers)
```

输出：
```shell
tensor([[3.1416, 2.7183],
        [1.6180, 0.0073]])
tensor([ 2,  3,  5,  7, 11, 13, 17, 19])
tensor([[2, 4, 6],
        [3, 6, 9]])
```

如果数据已经存在于 Python 元组或列表中，使用 `torch.tensor()` 是创建张量最简单的方法。如上所示，嵌套集合将产生多维张量。

!!! note "注意"
    `torch.tensor()` 会创建一个数据副本。

### 张量数据类型
设置张量的数据类型有几种方法：

```python
a = torch.ones((2, 3), dtype=torch.int16)
print(a)

b = torch.rand((2, 3), dtype=torch.float64) * 20.
print(b)

c = b.to(torch.int32)
print(c)
```

输出：
```shell
tensor([[1, 1, 1],
        [1, 1, 1]], dtype=torch.int16)
tensor([[ 0.9956,  1.4148,  5.8364],
        [11.2406, 11.2083, 11.6692]], dtype=torch.float64)
tensor([[ 0,  1,  5],
        [11, 11, 11]], dtype=torch.int32)
```

设置张量底层数据类型的最简单方法是在创建时使用可选的参数。在上述单元格的第一行，我们为张量 `a` 设置了 `dtype=torch.int16`。当我们打印 `a` 时，可以看到它被填满了 `1`，而不是 `1.0`——这是 Python 的小提示，表示这是整数类型而非浮点数。

在打印 `a` 时还需要注意的一点是，与将 `dtype` 保持默认值（32 位浮点数）不同，打印张量还会指定其 `dtype`。

您可能也发现了，我们从指定张量的形状为一系列整数参数，变成了将这些参数分组为一个元组。严格来说，这并不是必须的（ PyTorch 会将一系列初始的、无标签的整数参数作为张量形状），但在添加可选参数时，这可以使您的意图便于理解。

另一种设置数据类型的方法是使用 `.to()` 方法。在上面的单元格中，我们按照通常的方式创建了一个随机浮点张量 `b`。在随后，我们使用 `.to()` 方法将 `b` 转换为 32 位整数来创建 `c`。请注意，`c` 包含的所有值与 `b` 相同，但被截断为整数。

可用数据类型包括：

- `torch.bool`
- `torch.int8`
- `torch.uint8`
- `torch.int16`
- `torch.int32`
- `torch.int64`
- `torch.half`
- `torch.float`
- `torch.double`
- `torch.bfloat`

## 用 PyTorch 张量进行数学和逻辑运算
现在您已经了解了一些创建张量的方法，那么您可以用它们做什么呢？

首先，让我们先看一下基本的算术运算，以及张量如何与简单的标量进行交互：

```python
ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos ** 2
sqrt2s = twos ** 0.5

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)
```

输出：
```shell
tensor([[1., 1.],
        [1., 1.]])
tensor([[2., 2.],
        [2., 2.]])
tensor([[3., 3.],
        [3., 3.]])
tensor([[4., 4.],
        [4., 4.]])
tensor([[1.4142, 1.4142],
        [1.4142, 1.4142]])
```

如您在上面看到的，张量与标量之间的算术运算，比如加法、减法、乘法、除法和指数运算，会作用于张量的每个元素。因为此类运算的输出将是一个张量，您可以按通用运算符优先规则将它们链接在一起，就像我们在创建 `threes` 的那一行中所示。

类似的操作在两个张量之间也如您所愿：

```python
powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print(powers2)

fives = ones + fours
print(fives)

dozens = threes * fours
print(dozens)
```

输出：
```shell
tensor([[ 2.,  4.],
        [ 8., 16.]])
tensor([[5., 5.],
        [5., 5.]])
tensor([[12., 12.],
        [12., 12.]])
```

这里需要注意的是，上面代码单元格中的所有张量都具有相同的形状。如果我们尝试在形状不同的张量上执行二元运算，会发生什么呢？

!!! note "注意"
    以下单元格会抛出运行时错误。是故意为之。

```python
a = torch.rand(2, 3)
b = torch.rand(3, 2)

print(a * b)
```

通常，您不能以这种方式操作不同形状的张量，即使像上面的单元格中，张量具有相同数量元素的情况，也不行。

### 简述：张量广播

!!! note "注意"
    如果您熟悉 NumPy ndarrays 中的广播语义，您会发现相同的规则也适用于这里。

相同形状规则的例外是张量广播。以下是一个示例：


```python
rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)

print(rand)
print(doubled)
```

输出：
```shell
tensor([[0.6146, 0.5999, 0.5013, 0.9397],
        [0.8656, 0.5207, 0.6865, 0.3614]])
tensor([[1.2291, 1.1998, 1.0026, 1.8793],
        [1.7312, 1.0413, 1.3730, 0.7228]])
```

这里有什么诀窍？我们是如何将一个 2x4 张量乘以一个 1x4 张量的呢？

广播是在形状相似的张量之间进行运算的一种方式。在上面的例子中，单行四列张量与双行四列张量的两行相乘。

这是深度学习中的一个重要操作。常见的例子是将一个学习权重张量与一批输入张量相乘，分别对批次中的每个实例进行运算，然后返回一个形状相同的张量——就像我们上面的 (2, 4) * (1, 4) 例子一样，返回一个形状为 (2, 4) 的张量。

广播的规则如下：

- 每个张量必须至少有一个维度，不能是空张量。
- 比较两个张量的维数大小，从最后一个到第一个：
    - 每个维度必须相等，或
    - 其中一个维的大小必须为 1，或
    - 维度在一个张量中不存在

当然，形状相同的张量是“可广播”的，这在前面已经提到过。

下面是一些符合上述规则的可广播示例：

```python
a =     torch.ones(4, 3, 2)

b = a * torch.rand(   3, 2) # 3rd & 2nd dims identical to a, dim 1 absent
print(b)

c = a * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
print(c)

d = a * torch.rand(   1, 2) # 3rd dim identical to a, 2nd dim = 1
print(d)
```

输出：
```shell
tensor([[[0.6493, 0.2633],
         [0.4762, 0.0548],
         [0.2024, 0.5731]],

        [[0.6493, 0.2633],
         [0.4762, 0.0548],
         [0.2024, 0.5731]],

        [[0.6493, 0.2633],
         [0.4762, 0.0548],
         [0.2024, 0.5731]],

        [[0.6493, 0.2633],
         [0.4762, 0.0548],
         [0.2024, 0.5731]]])
tensor([[[0.7191, 0.7191],
         [0.4067, 0.4067],
         [0.7301, 0.7301]],

        [[0.7191, 0.7191],
         [0.4067, 0.4067],
         [0.7301, 0.7301]],

        [[0.7191, 0.7191],
         [0.4067, 0.4067],
         [0.7301, 0.7301]],

        [[0.7191, 0.7191],
         [0.4067, 0.4067],
         [0.7301, 0.7301]]])
tensor([[[0.6276, 0.7357],
         [0.6276, 0.7357],
         [0.6276, 0.7357]],

        [[0.6276, 0.7357],
         [0.6276, 0.7357],
         [0.6276, 0.7357]],

        [[0.6276, 0.7357],
         [0.6276, 0.7357],
         [0.6276, 0.7357]],

        [[0.6276, 0.7357],
         [0.6276, 0.7357],
         [0.6276, 0.7357]]])
```

仔细观察上面每个张量的值：

- 创建张量 `b` 的乘法操作是在张量 `a` 的每个“层”上进行广播的。
- 对于 `c`，操作在 `a` 的每个层和每一行上进行广播——每个由 3 个元素组成的列是相同的。
- 对于 `d`，我们将它改变了——现在每一行在层和列之间都是相同的。

有关广播的更多信息，请参阅 [PyTorch 的相关文档](https://pytorch.org/docs/stable/notes/broadcasting.html)。

以下是一些尝试进行广播的示例，这些示例会失败：

!!! note "注意"
    以下单元格会抛出运行时错误。是故意为之。

```python
a =     torch.ones(4, 3, 2)

b = a * torch.rand(4, 3)    # dimensions must match last-to-first

c = a * torch.rand(   2, 3) # both 3rd & 2nd dims different

d = a * torch.rand((0, ))   # can't broadcast with an empty tensor
```

### 更多张量数学运算
PyTorch 张量上有三百多种可以执行的操作。

以下是一小部分主要类别操作的示例：

```python
# common functions
a = torch.rand(2, 4) * 2 - 1
print('Common functions:')
print(torch.abs(a))
print(torch.ceil(a))
print(torch.floor(a))
print(torch.clamp(a, -0.5, 0.5))

# trigonometric functions and their inverses
angles = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
sines = torch.sin(angles)
inverses = torch.asin(sines)
print('\nSine and arcsine:')
print(angles)
print(sines)
print(inverses)

# bitwise operations
print('\nBitwise XOR:')
b = torch.tensor([1, 5, 11])
c = torch.tensor([2, 7, 10])
print(torch.bitwise_xor(b, c))

# comparisons:
print('\nBroadcasted, element-wise equality comparison:')
d = torch.tensor([[1., 2.], [3., 4.]])
e = torch.ones(1, 2)  # many comparison ops support broadcasting!
print(torch.eq(d, e)) # returns a tensor of type bool

# reductions:
print('\nReduction ops:')
print(torch.max(d))        # returns a single-element tensor
print(torch.max(d).item()) # extracts the value from the returned tensor
print(torch.mean(d))       # average
print(torch.std(d))        # standard deviation
print(torch.prod(d))       # product of all numbers
print(torch.unique(torch.tensor([1, 2, 1, 2, 1, 2]))) # filter unique elements

# vector and linear algebra operations
v1 = torch.tensor([1., 0., 0.])         # x unit vector
v2 = torch.tensor([0., 1., 0.])         # y unit vector
m1 = torch.rand(2, 2)                   # random matrix
m2 = torch.tensor([[3., 0.], [0., 3.]]) # three times identity matrix

print('\nVectors & Matrices:')
print(torch.cross(v2, v1)) # negative of z unit vector (v1 x v2 == -v2 x v1)
print(m1)
m3 = torch.matmul(m1, m2)
print(m3)                  # 3 times m1
print(torch.svd(m3))       # singular value decomposition
```

输出：
```shell
Common functions:
tensor([[0.9238, 0.5724, 0.0791, 0.2629],
        [0.1986, 0.4439, 0.6434, 0.4776]])
tensor([[-0., -0., 1., -0.],
        [-0., 1., 1., -0.]])
tensor([[-1., -1.,  0., -1.],
        [-1.,  0.,  0., -1.]])
tensor([[-0.5000, -0.5000,  0.0791, -0.2629],
        [-0.1986,  0.4439,  0.5000, -0.4776]])

Sine and arcsine:
tensor([0.0000, 0.7854, 1.5708, 2.3562])
tensor([0.0000, 0.7071, 1.0000, 0.7071])
tensor([0.0000, 0.7854, 1.5708, 0.7854])

Bitwise XOR:
tensor([3, 2, 1])

Broadcasted, element-wise equality comparison:
tensor([[ True, False],
        [False, False]])

Reduction ops:
tensor(4.)
4.0
tensor(2.5000)
tensor(1.2910)
tensor(24.)
tensor([1, 2])

Vectors & Matrices:
tensor([ 0.,  0., -1.])
tensor([[0.7375, 0.8328],
        [0.8444, 0.2941]])
tensor([[2.2125, 2.4985],
        [2.5332, 0.8822]])
torch.return_types.svd(
U=tensor([[-0.7889, -0.6145],
        [-0.6145,  0.7889]]),
S=tensor([4.1498, 1.0548]),
V=tensor([[-0.7957,  0.6056],
        [-0.6056, -0.7957]]))
```

这只是一小部分运算示例。如需了解更多详情和所有数学函数，请参阅[文档](https://pytorch.org/docs/stable/torch.html#math-operations)。

### 原地修改张量
大多数对张量的二进制运算都会返回第三个新张量。当我们说 `c = a * b`（其中 `a` 和 `b` 都是张量）时，新的张量 `c` 将占据一个与其他张量不同的内存区域。

不过，有时您可能希望原地改变一个张量——例如，如果您正在进行元素计算，您可以丢弃中间值。为此，大多数数学函数都有一个带下划线 (`_`) 的版本，可以原地改变张量。

例如：

```python
a = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('a:')
print(a)
print(torch.sin(a))   # this operation creates a new tensor in memory
print(a)              # a has not changed

b = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('\nb:')
print(b)
print(torch.sin_(b))  # note the underscore
print(b)              # b has changed
```

输出：
```shell
a:
tensor([0.0000, 0.7854, 1.5708, 2.3562])
tensor([0.0000, 0.7071, 1.0000, 0.7071])
tensor([0.0000, 0.7854, 1.5708, 2.3562])

b:
tensor([0.0000, 0.7854, 1.5708, 2.3562])
tensor([0.0000, 0.7071, 1.0000, 0.7071])
tensor([0.0000, 0.7071, 1.0000, 0.7071])
```

对于算术运算，也有类似的函数：

```python
a = torch.ones(2, 2)
b = torch.rand(2, 2)

print('Before:')
print(a)
print(b)
print('\nAfter adding:')
print(a.add_(b))
print(a)
print(b)
print('\nAfter multiplying')
print(b.mul_(b))
print(b)
```

输出：
```shell
Before:
tensor([[1., 1.],
        [1., 1.]])
tensor([[0.3788, 0.4567],
        [0.0649, 0.6677]])

After adding:
tensor([[1.3788, 1.4567],
        [1.0649, 1.6677]])
tensor([[1.3788, 1.4567],
        [1.0649, 1.6677]])
tensor([[0.3788, 0.4567],
        [0.0649, 0.6677]])

After multiplying
tensor([[0.1435, 0.2086],
        [0.0042, 0.4459]])
tensor([[0.1435, 0.2086],
        [0.0042, 0.4459]])
```

请注意，这些原地运算函数是 `torch.Tensor` 对象上的方法，而不是像许多其他函数（如 `torch.sin()`）那样附加到 `torch` 模块上。从 `a.add_(b)` 可以看出，调用的张量会原地发生变化。

还有一种方法可以将计算结果放入现有的已分配张量中。到目前为止，我们看到的许多方法和函数（包括创建方法）都有一个 `out` 参数，可以指定一个张量来接收输出结果。如果 `out` 张量的形状和 `dtype` 正确，就无需分配新的内存：

```python
a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.zeros(2, 2)
old_id = id(c)

print(c)
d = torch.matmul(a, b, out=c)
print(c)                # contents of c have changed

assert c is d           # test c & d are same object, not just containing equal values
assert id(c) == old_id  # make sure that our new c is the same object as the old one

torch.rand(2, 2, out=c) # works for creation too!
print(c)                # c has changed again
assert id(c) == old_id  # still the same object!
```

输出：
```shell
tensor([[0., 0.],
        [0., 0.]])
tensor([[0.3653, 0.8699],
        [0.2364, 0.3604]])
tensor([[0.0776, 0.4004],
        [0.9877, 0.0352]])
```

## 复制张量
与 Python 中的对象一样，将张量赋值给变量会使变量成为张量的标签，而不会复制它。例如：

```python
a = torch.ones(2, 2)
b = a

a[0][1] = 561  # we change a...
print(b)       # ...and b is also altered
```

输出：
```shell
tensor([[  1., 561.],
        [  1.,   1.]])
```

但是，如果您需要一个单独的数据副本来处理数据，该怎么办呢？clone()方法就能满足您的需求：

```python
a = torch.ones(2, 2)
b = a.clone()

assert b is not a      # different objects in memory...
print(torch.eq(a, b))  # ...but still with the same contents!

a[0][1] = 561          # a changes...
print(b)               # ...but b is still all ones
```

输出：
```shell
tensor([[True, True],
        [True, True]])
tensor([[1., 1.],
        [1., 1.]])
```

**使用 `clone()` 时需要注意一件重要的事情**。如果源张量启用了自动梯度，那么克隆也会启用。**关于自动梯度的视频将对此进行更深入的介绍**，但如果您想了解更多细节，请继续往下看。

_在大多情况下_，这就是您想要的。例如，如果模型的 `forward()` 方法有多个计算路径，而原始张量和克隆张量都对模型的输出有贡献，那么要实现模型学习，就需要同时打开两个张量的自动梯度。如果源张量启用了自动梯度（如果它是一组学习权重或从涉及权重的计算中导出，则通常会启用），那么就会得到想要的结果。

_另一方面_，如果您正在进行的计算中，原始张量或其克隆都不需要跟踪梯度，那么只要源张量关闭了自动梯度，就可以正常运行。

_不过还有第三种情况_,想象一下：您在模型的 `forward()` 函数中执行计算，默认情况下梯度都是打开的，但您想在中途取出一些值来生成一些指标。在这种情况下，您不希望源张量的克隆副本跟踪梯度——关闭自动梯度的历史跟踪可以提高性能。为此，您可以在源张量上使用 `.detach()` 方法：

```python
a = torch.rand(2, 2, requires_grad=True) # turn on autograd
print(a)

b = a.clone()
print(b)

c = a.detach().clone()
print(c)

print(a)
```

输出：
```shell
tensor([[0.0905, 0.4485],
        [0.8740, 0.2526]], requires_grad=True)
tensor([[0.0905, 0.4485],
        [0.8740, 0.2526]], grad_fn=<CloneBackward0>)
tensor([[0.0905, 0.4485],
        [0.8740, 0.2526]])
tensor([[0.0905, 0.4485],
        [0.8740, 0.2526]], requires_grad=True)
```

发生了什么？

- 我们创建了一个开启 `requirements_grad=True` 的 `a`。**我们还没有涉及这个可选参数，但会在自动梯度单元中涉及。**
- 当我们打印 `a` 时，它会告诉我们属性 `requires_grad=True` ——这意味着自动梯度和计算历史跟踪已打开。
- 我们拷贝 `a` 并标记为 `b`。当我们打印 `b` 时，可以看到它正在跟踪计算历史记录——它继承了 `a` 的自动梯度设置，并添加到了计算历史记录中。
- 我们将 `a` 复制到 `c` 中，但要先调用 `detach()`。
- 在打印 `c` 时，我们没有看到计算历史，也没有看到 `requires_grad=True`。

`detach()` 方法 _将张量从其计算历史中分离出来_。它说："不管接下来要做什么，都要像关闭自动梯度一样"。我们可以看到，当我们在最后再次打印 `a` 时，它保留了 `requires_grad=True` 属性。

## 转用 GPU
PyTorch 的主要优势之一是其在兼容 CUDA 的 Nvidia GPU 上的强大加速能力。 CUDA 是计算统一设备架构（Compute Unified Device Architecture）的缩写，是 Nvidia 的并行计算平台。到目前为止，我们所做的一切都是在 CPU 上完成的。我们该如何使用更快的硬件呢？

首先，我们应该使用 `is_available()` 方法检查 GPU 是否可用。

!!! note "注意"
    如果您没有安装与 CUDA 兼容的 GPU 和 CUDA 驱动程序，本节中的可执行单元将无法执行任何与 GPU 相关的代码。

```python
if torch.cuda.is_available():
    print('We have a GPU!')
else:
    print('Sorry, CPU only.')
```

输出：
```shell
We have a GPU!
```

一旦确定有一个或多个 GPU 可用，我们就需要将数据放到 GPU 可以看到的地方。CPU 在计算机 RAM 中对数据进行计算。而 GPU 则连接有专用内存。每当要在设备上执行计算时，必须将计算所需的所有数据移动到该设备可访问的内存中。（通俗地说，"将数据移至 GPU 可访问的内存 "简称为 "将数据移至 GPU"）。

将数据转移到目标设备上的方法有多种，可以在创建时进行：

```python
if torch.cuda.is_available():
    gpu_rand = torch.rand(2, 2, device='cuda')
    print(gpu_rand)
else:
    print('Sorry, CPU only.')
```

输出：
```shell
tensor([[0.3344, 0.2640],
        [0.2119, 0.0582]], device='cuda:0')
```

默认情况下，新张量是在 CPU 上创建的，因此我们必须使用可选的 `device` 参数来指定何时在 GPU 上创建张量。您可以看到，当我们打印新张量时，PyTorch 会告诉我们它在哪个设备上（如果它不在 CPU 上）。

您可以使用 `torch.cuda.device_count()` 来查询 GPU 的数量。如果有多个 GPU，可以通过索引指定：`device='cuda:0'`、`device='cuda:1'` 等。

在编码实践中，使用字符串常量指定设备是非常不稳健的。在理想情况下，无论您使用的是 CPU 还是 GPU 硬件，您的代码都应该能稳定运行。为此，您可以创建一个设备句柄，将其传递给您的张量，而非字符串：

```python
if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

x = torch.rand(2, 2, device=my_device)
print(x)
```

输出：
```shell
Device: cuda
tensor([[0.0024, 0.6778],
        [0.2441, 0.6812]], device='cuda:0')
```

如果在一个设备上已有一个张量，可以使用 `to()` 方法将其移动到另一个设备上。下面的代码在 CPU 上创建了一个张量，并将其移动到上一单元中获得的设备句柄。

```python
y = torch.rand(2, 2)
y = y.to(my_device)
```

要知道，想进行涉及两个或多个张量的计算，所有张量必须在同一设备上。无论您是否拥有 GPU 设备，以下代码都会在运行时出错：

```python
x = torch.rand(2, 2)
y = torch.rand(2, 2, device='gpu')
z = x + y  # exception will be thrown
```

## 操作张量形状
有时，您需要改变张量的形状。下面，我们将介绍几种常见情况，以及如何处理它们。

### 更改维数
您可能需要更改维数的一种情况是，向模型传递单个输入实例。PyTorch 模型通常需要成批的输入。

例如，想象一下有一个模型可以在 3 x 226 x 226 图像（一个具有 3 个颜色通道的 226 像素正方形）上工作。当您加载并转换它时，您会得到一个形状为 (`3`, `226`, `226`) 的张量。而您的模型则希望输入形状为（`N`，`3`，`226`，`226`）的张量，其中 `N` 是批次中图像的数量。那么，如何制作一个批次的图像呢？

```python
a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)

print(a.shape)
print(b.shape)
torch.Size([3, 226, 226])
torch.Size([1, 3, 226, 226])
```

`unsqueeze()` 方法添加了一个范围为 1 的维度，`unsqueeze(0)` 则将其添加为一个新的第零维度——现在您有了一批 1 的维度！

所以这就是 _维度扩展（ unsqueezing ）_？维度压缩（ squeezing ）又是什么意思？实际上，范围为 1 的任何维度都 _不会_ 改变张量中元素的数量。

```python
c = torch.rand(1, 1, 1, 1, 1)
print(c)
```

输出：
```shell
tensor([[[[[0.2347]]]]])
```

继续上面的例子，假设模型的输出是每个输入的 20 元素的向量。那么您就会期望输出的形状是 (`N`, `20`)，其中 `N` 是输入批次中实例的数量。这意味着，对于单输入批次而言，我们将得到一个形状为（`1`，`20`）的输出。

如果您想用该输出进行一些非批处理计算——只期望得到一个 20 元素的向量，该怎么办呢？

```python
a = torch.rand(1, 20)
print(a.shape)
print(a)

b = a.squeeze(0)
print(b.shape)
print(b)

c = torch.rand(2, 2)
print(c.shape)

d = c.squeeze(0)
print(d.shape)
```

输出：
```shell
torch.Size([1, 20])
tensor([[0.1899, 0.4067, 0.1519, 0.1506, 0.9585, 0.7756, 0.8973, 0.4929, 0.2367,
         0.8194, 0.4509, 0.2690, 0.8381, 0.8207, 0.6818, 0.5057, 0.9335, 0.9769,
         0.2792, 0.3277]])
torch.Size([20])
tensor([0.1899, 0.4067, 0.1519, 0.1506, 0.9585, 0.7756, 0.8973, 0.4929, 0.2367,
        0.8194, 0.4509, 0.2690, 0.8381, 0.8207, 0.6818, 0.5057, 0.9335, 0.9769,
        0.2792, 0.3277])
torch.Size([2, 2])
torch.Size([2, 2])
```
如果仔细观察上面单元格的输出，就会发现由于多了一个维度，打印 `a` 显示了一组“额外”的方括号 `[]`。

您只能 `squeeze()` 范围为 1 的维数。请看上图，我们试图在 `c` 中压缩一个尺寸为 2 的维度，结果得到的形状和开始时一样。对 `squeeze()` 和 `unsqueeze()` 的调用只能作用于范围为 1 的维度，否则会改变张量中元素的数量。

`unsqueeze() `的另一个用途是简化广播。回顾上面的示例，我们有如下代码：

```python
a =     torch.ones(4, 3, 2)

c = a * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
print(c)
```

这样做的效果是在 0 维和 2 维上广播操作，导致随机的 3 x 1 张量与 `a` 中每一列的 3 元素相乘。

如果随机向量只是 3 元素向量呢？我们将无法广播，因为根据广播规则，最终维数将不匹配。这时 `unsqueeze()` 就派上用场了：

```python
a = torch.ones(4, 3, 2)
b = torch.rand(   3)     # trying to multiply a * b will give a runtime error
c = b.unsqueeze(1)       # change to a 2-dimensional tensor, adding new dim at the end
print(c.shape)
print(a * c)             # broadcasting works again!
```

输出：
```shell
torch.Size([3, 1])
tensor([[[0.1891, 0.1891],
         [0.3952, 0.3952],
         [0.9176, 0.9176]],

        [[0.1891, 0.1891],
         [0.3952, 0.3952],
         [0.9176, 0.9176]],

        [[0.1891, 0.1891],
         [0.3952, 0.3952],
         [0.9176, 0.9176]],

        [[0.1891, 0.1891],
         [0.3952, 0.3952],
         [0.9176, 0.9176]]])
```

`squeeze()` 和 `unsqueeze()` 也有原地版，`squeeze_()` 和 `unsqueeze_()`：

```python
batch_me = torch.rand(3, 226, 226)
print(batch_me.shape)
batch_me.unsqueeze_(0)
print(batch_me.shape)
```

输出：
```shell
torch.Size([3, 226, 226])
torch.Size([1, 3, 226, 226])
```

有时，您会希望更彻底地改变张量的形状，同时仍保留元素的数量和内容。其中一种情况发生在模型的卷积层和线性层之间的接口处——这在图像分类模型中很常见。卷积核会产生一个形状特征 _x 宽 x 高_ 的输出张量，但接下来的线性层希望得到一个一维的输入。`reshape()` 可以帮您做到这一点，前提是您想要的维数与输入张量的元素数相同：

```python
output3d = torch.rand(6, 20, 20)
print(output3d.shape)

input1d = output3d.reshape(6 * 20 * 20)
print(input1d.shape)

# can also call it as a method on the torch module:
print(torch.reshape(output3d, (6 * 20 * 20,)).shape)
```

输出：
```shell
torch.Size([6, 20, 20])
torch.Size([2400])
torch.Size([2400])
```

!!! note "注意"
    上面单元格中最后一行的 `(6 * 20 * 20,)` 参数是因为 PyTorch 在指定张量形状时期望一个**元组**——但当方法的第一个参数是形状时，它允许我们欺骗一下，只使用一系列整数。在这里，我们必须添加括号和逗号，告诉方法这是一个单元素元组。

如果可以，`reshape()` 会返回要更改的张量的视图——也就是说，一个单独的张量对象将查看同样的底层内存。这一点很重要：这意味着对源张量所做的任何更改都会反映在该张量的视图中，除非您 `clone()` 它。

在某些情况下，`reshape()` 必须返回一个携带数据副本的张量，这超出了本介绍的范围。更多信息，请参阅[文档](https://pytorch.org/docs/stable/torch.html#torch.reshape)。

## NumPy 桥接
在上面关于广播的章节中，我们提到 PyTorch 的广播语义与 NumPy 的广播语义兼容，但 PyTorch 和 NumPy 之间的联系远不止于此。

如果您现有的 ML 或科学代码中的数据存储在 NumPy ndarrays 中，您可能希望用 PyTorch tensors 来表达相同的数据，无论是利用 PyTorch 的 GPU 加速，还是利用它构建 ML 模型的高效抽象。在 ndarrays 和 PyTorch tensors 之间切换非常简单：

```python
import numpy as np

numpy_array = np.ones((2, 3))
print(numpy_array)

pytorch_tensor = torch.from_numpy(numpy_array)
print(pytorch_tensor)
```

输出：
```shell
[[1. 1. 1.]
 [1. 1. 1.]]
tensor([[1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
```

PyTorch 创建了一个与 NumPy 数组形状、数据相同的张量，甚至保留了 NumPy 的默认 64 位浮点数据类型。

这种转换也可以很容易地反过来：

```python
pytorch_rand = torch.rand(2, 3)
print(pytorch_rand)

numpy_rand = pytorch_rand.numpy()
print(numpy_rand)
```

输出：
```shell
tensor([[0.8716, 0.2459, 0.3499],
        [0.2853, 0.9091, 0.5695]])
[[0.87163675 0.2458961  0.34993553]
 [0.2853077  0.90905803 0.5695162 ]]
```

要知道，这些转换后的对象与其源对象使用的是相同的底层内存，也就是说，其中一个对象的变化会反映在另一个对象上：

```python
numpy_array[1, 1] = 23
print(pytorch_tensor)

pytorch_rand[1, 1] = 17
print(numpy_rand)
```

输出：
```shell
tensor([[ 1.,  1.,  1.],
        [ 1., 23.,  1.]], dtype=torch.float64)
[[ 0.87163675  0.2458961   0.34993553]
 [ 0.2853077  17.          0.5695162 ]]
```