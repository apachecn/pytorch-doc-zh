# Tensor(张量）的属性

> 译者：[阿远](https://github.com/yuange250)

每个 `torch.Tensor` 对象都有以下几个属性： [`torch.dtype`](#torch.torch.dtype "torch.torch.dtype"), [`torch.device`](#torch.torch.device "torch.torch.device")， 和 [`torch.layout`](#torch.torch.layout "torch.torch.layout")。

## torch.dtype

```py
class torch.dtype
```

[`torch.dtype`](#torch.torch.dtype "torch.torch.dtype") 属性标识了 [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor")的数据类型。PyTorch 有八种不同的数据类型：

| Data type | dtype | Tensor types |
| --- | --- | --- |
| 32-bit floating point | `torch.float32` or `torch.float` | `torch.*.FloatTensor` |
| 64-bit floating point | `torch.float64` or `torch.double` | `torch.*.DoubleTensor` |
| 16-bit floating point | `torch.float16` or `torch.half` | `torch.*.HalfTensor` |
| 8-bit integer (unsigned) | `torch.uint8` | `torch.*.ByteTensor` |
| 8-bit integer (signed) | `torch.int8` | `torch.*.CharTensor` |
| 16-bit integer (signed) | `torch.int16` or `torch.short` | `torch.*.ShortTensor` |
| 32-bit integer (signed) | `torch.int32` or `torch.int` | `torch.*.IntTensor` |
| 64-bit integer (signed) | `torch.int64` or `torch.long` | `torch.*.LongTensor` |

## torch.device

```py
class torch.device
```

[`torch.device`](#torch.torch.device "torch.torch.device") 属性标识了[`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor")对象在创建之后所存储在的设备名称，而在对象创建之前此属性标识了即将为此对象申请存储空间的设备名称。

[`torch.device`](#torch.torch.device "torch.torch.device") 包含了两种设备类型 (`'cpu'` 或者 `'cuda'`) ，分别标识将Tensor对象储存于cpu内存或者gpu内存中，同时支持指定设备编号，比如多张gpu，可以通过gpu编号指定某一块gpu。 如果没有指定设备编号，则默认将对象存储于current_device()当前设备中； 举个例子， 一个[`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor") 对象构造函数中的设备字段如果填写`'cuda'`，那等价于填写了`'cuda:X'`，其中X是函数 [`torch.cuda.current_device()`](cuda.html#torch.cuda.current_device "torch.cuda.current_device")的返回值。

在[`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor")对象创建之后，可以通过访问[`Tensor.device`](tensors.html#torch.Tensor.device "torch.Tensor.device")属性实时访问当前对象所存储在的设备名称。

[`torch.device`](#torch.torch.device "torch.torch.device") 对象支持使用字符串或者字符串加设备编号这两种方式来创建：

通过字符串创建：

```py
>>> torch.device('cuda:0')
device(type='cuda', index=0)  # 编号为0的cuda设备

>>> torch.device('cpu')  # cpu内存
device(type='cpu')

>>> torch.device('cuda')  # 当前cuda设备
device(type='cuda')

```

通过字符串加设备编号创建：

```py
>>> torch.device('cuda', 0)
device(type='cuda', index=0)

>>> torch.device('cpu', 0)
device(type='cpu', index=0)

```

Note

当[`torch.device`](#torch.torch.device "torch.torch.device")作为函数的参数的时候， 可以直接用字符串替换。 这样有助于加快代码创建原型的速度。

```py
>>> # 一个接受torch.device对象为参数的函数例子
>>> cuda1 = torch.device('cuda:1')
>>> torch.randn((2,3), device=cuda1)

```

```py
>>> # 可以用一个字符串替换掉torch.device对象，一样的效果
>>> torch.randn((2,3), 'cuda:1')

```

Note

由于一些历史遗留问题, device对象还可以仅通过一个设备编号来创建，这些设备编号对应的都是相应的cuda设备。 这正好对应了 [`Tensor.get_device()`](tensors.html#torch.Tensor.get_device "torch.Tensor.get_device")函数, 这个仅支持cuda Tensor的函数返回的就是当前tensor所在的cuda设备编号，cpu Tensor不支持这个函数。

```py
>>> torch.device(1)
device(type='cuda', index=1)

```

Note

接受device参数的函数同时也可以接受一个正确格式的字符串或者正确代表设备编号的数字(数字这个是历史遗留问题）作为参数，以下的操作是等价的：

```py
>>> torch.randn((2,3), device=torch.device('cuda:1'))
>>> torch.randn((2,3), device='cuda:1')
>>> torch.randn((2,3), device=1)  # 历史遗留做法

```

## torch.layout

```py
class torch.layout
```

[`torch.layout`](#torch.torch.layout "torch.torch.layout") 属性标识了[`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor") 在内存中的布局模式。 现在， 我们支持了两种内存布局模式 `torch.strided` (dense Tensors) 和尚处试验阶段的`torch.sparse_coo` (sparse COO Tensors， 一种经典的稀疏矩阵存储方式).

`torch.strided` 跨步存储代表了密集张量的存储布局方式，当然也是最常用最经典的一种布局方式。 每一个strided tensor都有一个与之相连的`torch.Storage`对象, 这个对象存储着tensor的数据. 这些Storage对象为tensor提供了一种多维的， [跨步的(strided)](https://en.wikipedia.org/wiki/Stride_of_an_array)数据视图. 这一视图中的strides是一个interger整形列表：这个列表的主要作用是给出当前张量的各个维度的所占内存大小，严格的定义就是，strides中的第k个元素代表了在第k维度下，从一个元素跳转到下一个元素所需要跨越的内存大小。 跨步这个概念有助于提高多种张量运算的效率。

例子:

```py
>>> x = torch.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
>>> x.stride() 
(5, 1)     # 此时在这个二维张量中，在第0维度下，从一个元素到下一个元素需要跨越的内存大小是5，比如x[0] 到x[1]需要跨越x[0]这5个元素, 在第1维度下，是1，如x[0, 0]到x[0, 1]需要跨越1个元素

>>> x.t().stride()
(1, 5)

```

更多关于 `torch.sparse_coo` tensors的信息, 请看[torch.sparse](sparse.html#sparse-docs).

