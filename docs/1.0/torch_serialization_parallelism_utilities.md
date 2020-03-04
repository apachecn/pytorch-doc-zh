

## 序列化

> 译者：[ApacheCN](https://github.com/apachecn)

```py
torch.save(obj, f, pickle_module=<module 'pickle' from '/scratch/rzou/pt/release-env/lib/python3.7/pickle.py'>, pickle_protocol=2)
```

将对象保存到磁盘文件。

另请参阅：[保存模型的推荐方法](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/notes/serialization.html#recommend-saving-models)

参数：

*   **obj** - 保存对象
*   **f** - 类似文件的对象(必须实现写入和刷新）或包含文件名的字符串
*   **pickle_module** - 用于腌制元数据和对象的模块
*   **pickle_protocol** - 可以指定覆盖默认协议

警告

如果您使用的是Python 2，则torch.save不支持将StringIO.StringIO作为有效的类文件对象。这是因为write方法应该返回写入的字节数; StringIO.write(）不会这样做。

请使用像io.BytesIO这样的东西。

例

```py
>>> # Save to file
>>> x = torch.tensor([0, 1, 2, 3, 4])
>>> torch.save(x, 'tensor.pt')
>>> # Save to io.BytesIO buffer
>>> buffer = io.BytesIO()
>>> torch.save(x, buffer)

```

```py
torch.load(f, map_location=None, pickle_module=<module 'pickle' from '/scratch/rzou/pt/release-env/lib/python3.7/pickle.py'>)
```

从文件加载用 [`torch.save()`](#torch.save "torch.save") 保存的对象。

[`torch.load()`](#torch.load "torch.load") 使用Python的unpickling设施，但特别是处理作为张量传感器的存储器。它们首先在CPU上反序列化，然后移动到它们保存的设备上。如果此操作失败(例如，因为运行时系统没有某些设备），则会引发异常。但是，可以使用`map_location`参数将存储重新映射到另一组设备。

如果`map_location`是可调用的，则每个序列化存储器将调用一次，其中包含两个参数：存储和位置。存储参数将是驻留在CPU上的存储的初始反序列化。每个序列化存储都有一个与之关联的位置标记，用于标识从中保存的设备，此标记是传递给map_location的第二个参数。内置位置标签是CPU张量的`‘cpu’`和CUDA张量的`‘cuda:device_id’`(例如`‘cuda:2’`）。 `map_location`应返回None或存储。如果`map_location`返回存储，它将用作最终反序列化对象，已移动到正确的设备。否则， [![](/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/b69f1ef0735e18ff4ee132790112ce0d.jpg)](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/b69f1ef0735e18ff4ee132790112ce0d.jpg) 将回退到默认行为，就像未指定`map_location`一样。

如果`map_location`是一个字符串，它应该是一个设备标签，应该加载所有张量。

否则，如果`map_location`是一个dict，它将用于将文件(键）中出现的位置标记重新映射到指定存储位置(值）的位置标记。

用户扩展可以使用`register_package`注册自己的位置标记以及标记和反序列化方法。

Parameters:

*   **f** - 类文件对象(必须实现read，readline，tell和seek），或包含文件名的字符串
*   **map_location** - 一个函数，torch.device，string或dict，指定如何重新映射存储位置
*   **pickle_module** - 用于取消元数据和对象取消的模块(必须与用于序列化文件的pickle_module相匹配）

注意

当您在包含GPU张量的文件上调用 [`torch.load()`](#torch.load "torch.load") 时，默认情况下这些张量将被加载到GPU。在加载模型检查点时，可以调用`torch.load(.., map_location=’cpu’)`然后调用`load_state_dict()`以避免GPU RAM激增。

Example

```py
>>> torch.load('tensors.pt')
# Load all tensors onto the CPU
>>> torch.load('tensors.pt', map_location=torch.device('cpu'))
# Load all tensors onto the CPU, using a function
>>> torch.load('tensors.pt', map_location=lambda storage, loc: storage)
# Load all tensors onto GPU 1
>>> torch.load('tensors.pt', map_location=lambda storage, loc: storage.cuda(1))
# Map tensors from GPU 1 to GPU 0
>>> torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})
# Load tensor from io.BytesIO object
>>> with open('tensor.pt') as f:
 buffer = io.BytesIO(f.read())
>>> torch.load(buffer)

```

## 排比

```py
torch.get_num_threads() → int
```

获取用于并行化CPU操作的OpenMP线程数

```py
torch.set_num_threads(int)
```

设置用于并行化CPU操作的OpenMP线程数

## 在本地禁用渐变计算

上下文管理器`torch.no_grad()`，`torch.enable_grad()`和`torch.set_grad_enabled()`有助于本地禁用和启用梯度计算。有关其用法的更多详细信息，请参见[本地禁用梯度计算](/apachecn/pytorch-doc-zh/blob/master/docs/1.0/autograd.html#locally-disable-grad)。

例子：

```py
>>> x = torch.zeros(1, requires_grad=True)
>>> with torch.no_grad():
...     y = x * 2
>>> y.requires_grad
False

>>> is_train = False
>>> with torch.set_grad_enabled(is_train):
...     y = x * 2
>>> y.requires_grad
False

>>> torch.set_grad_enabled(True)  # this can also be used as a function
>>> y = x * 2
>>> y.requires_grad
True

>>> torch.set_grad_enabled(False)
>>> y = x * 2
>>> y.requires_grad
False

```

## 公用事业

```py
torch.compiled_with_cxx11_abi()
```

返回是否使用_GLIBCXX_USE_CXX11_ABI = 1构建PyTorch

