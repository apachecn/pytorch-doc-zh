
```
# Tensor Attributes

Each `torch.Tensor` has a [`torch.dtype`](#torch.torch.dtype "torch.torch.dtype"), [`torch.device`](#torch.torch.device "torch.torch.device"), and [`torch.layout`](#torch.torch.layout "torch.torch.layout").
```
# Tensor（张量）的属性
> 译者：[阿远](https://github.com/yuange250)

每个 `torch.Tensor` 对象都有以下几个属性： [`torch.dtype`](#torch.torch.dtype "torch.torch.dtype"), [`torch.device`](#torch.torch.device "torch.torch.device"), 和 [`torch.layout`](#torch.torch.layout "torch.torch.layout").

```
## torch.dtype

py
class torch.dtype

A [`torch.dtype`](#torch.torch.dtype "torch.torch.dtype") is an object that represents the data type of a [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor"). PyTorch has eight different data types:
```
## torch.dtype

```py
class torch.dtype
```

[`torch.dtype`](#torch.torch.dtype "torch.torch.dtype") 是一个标识 [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor")的数据类型的属性. PyTorch 有八种不同的数据类型:

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

```
## torch.device

py
class torch.device

A [`torch.device`](#torch.torch.device "torch.torch.device") is an object representing the device on which a [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor") is or will be allocated.

The [`torch.device`](#torch.torch.device "torch.torch.device") contains a device type (`'cpu'` or `'cuda'`) and optional device ordinal for the device type. If the device ordinal is not present, this represents the current device for the device type; e.g. a [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor") constructed with device `'cuda'` is equivalent to `'cuda:X'` where X is the result of [`torch.cuda.current_device()`](cuda.html#torch.cuda.current_device "torch.cuda.current_device").

A [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor")’s device can be accessed via the [`Tensor.device`](tensors.html#torch.Tensor.device "torch.Tensor.device") property.

A [`torch.device`](#torch.torch.device "torch.torch.device") can be constructed via a string or via a string and device ordinal
```
## torch.device

```py
class torch.device
```

[`torch.device`](#torch.torch.device "torch.torch.device") 属性标识了[`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor")对象在创建之后所存储在的设备名称，而在对象创建之前此属性标识了即将为此对象申请存储空间的设备名称.

[`torch.device`](#torch.torch.device "torch.torch.device") 包含了两种设备类型 (`'cpu'` 或者 `'cuda'`) ，分别标识将Tensor对象储存于cpu内存或者gpu内存中，同时支持指定设备编号，比如多张gpu，可以通过gpu编号指定某一块gpu. 如果没有指定设备编号，则默认将对象存储于current_device()当前设备中; 举个例子， 一个[`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor") 对象构造函数中的设备字段如果填写`'cuda'`，那等价于填写了`'cuda:X'`，其中X是函数 [`torch.cuda.current_device()`](cuda.html#torch.cuda.current_device "torch.cuda.current_device")的返回值.

A [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor")’s device can be accessed via the [`Tensor.device`](tensors.html#torch.Tensor.device "torch.Tensor.device") property.

A [`torch.device`](#torch.torch.device "torch.torch.device") can be constructed via a string or via a string and device ordinal

Via a string:

```py
>>> torch.device('cuda:0')
device(type='cuda', index=0)

>>> torch.device('cpu')
device(type='cpu')

>>> torch.device('cuda')  # current cuda device
device(type='cuda')

```

Via a string and device ordinal:

```py
>>> torch.device('cuda', 0)
device(type='cuda', index=0)

>>> torch.device('cpu', 0)
device(type='cpu', index=0)

```

Note

The [`torch.device`](#torch.torch.device "torch.torch.device") argument in functions can generally be substituted with a string. This allows for fast prototyping of code.

```py
>>> # Example of a function that takes in a torch.device
>>> cuda1 = torch.device('cuda:1')
>>> torch.randn((2,3), device=cuda1)

```

```py
>>> # You can substitute the torch.device with a string
>>> torch.randn((2,3), 'cuda:1')

```

Note

For legacy reasons, a device can be constructed via a single device ordinal, which is treated as a cuda device. This matches [`Tensor.get_device()`](tensors.html#torch.Tensor.get_device "torch.Tensor.get_device"), which returns an ordinal for cuda tensors and is not supported for cpu tensors.

```py
>>> torch.device(1)
device(type='cuda', index=1)

```

Note

Methods which take a device will generally accept a (properly formatted) string or (legacy) integer device ordinal, i.e. the following are all equivalent:

```py
>>> torch.randn((2,3), device=torch.device('cuda:1'))
>>> torch.randn((2,3), device='cuda:1')
>>> torch.randn((2,3), device=1)  # legacy

```

## torch.layout

```py
class torch.layout
```

A [`torch.layout`](#torch.torch.layout "torch.torch.layout") is an object that represents the memory layout of a [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor"). Currently, we support `torch.strided` (dense Tensors) and have experimental support for `torch.sparse_coo` (sparse COO Tensors).

`torch.strided` represents dense Tensors and is the memory layout that is most commonly used. Each strided tensor has an associated `torch.Storage`, which holds its data. These tensors provide multi-dimensional, [strided](https://en.wikipedia.org/wiki/Stride_of_an_array) view of a storage. Strides are a list of integers: the k-th stride represents the jump in the memory necessary to go from one element to the next one in the k-th dimension of the Tensor. This concept makes it possible to perform many tensor operations efficiently.

Example:

```py
>>> x = torch.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
>>> x.stride()
(5, 1)

>>> x.t().stride()
(1, 5)

```

For more information on `torch.sparse_coo` tensors, see [torch.sparse](sparse.html#sparse-docs).

