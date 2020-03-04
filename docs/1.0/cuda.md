

# torch.cuda

> 译者：[bdqfork](https://github.com/bdqfork)

这个包添加了对CUDA张量类型的支持，它实现了与CPU张量同样的功能，但是它使用GPU进计算。

它是懒加载的，所以你可以随时导入它，并使用 [`is_available()`](#torch.cuda.is_available "torch.cuda.is_available") 来决定是否让你的系统支持CUDA。

[CUDA semantics](notes/cuda.html#cuda-semantics) 有关于使用CUDA更详细的信息。

```py
torch.cuda.current_blas_handle()
```

返回一个cublasHandle_t指针给当前的cuBLAS处理。

```py
torch.cuda.current_device()
```

返回当前选择地设备索引。

```py
torch.cuda.current_stream()
```

返回当前选择地 [`Stream`](#torch.cuda.Stream "torch.cuda.Stream")。

```py
class torch.cuda.device(device)
```

Context-manager 用来改变选择的设备。

| 参数: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _或者_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 要选择的设备索引。如果这个参数是负数或者是 `None`，那么它不会起任何作用。 |
| --- | --- |

```py
torch.cuda.device_count()
```

返回可用的GPU数量。

```py
torch.cuda.device_ctx_manager
```

 [`torch.cuda.device`](#torch.cuda.device "torch.cuda.device") 的别名。

```py
class torch.cuda.device_of(obj)
```

Context-manager 将当前的设备改变成传入的对象。.

你可以使用张量或者存储作为参数。如果传入的对象没有分配在GPU上，这个操作是无效的。

| 参数: | **obj** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor") _或者_ _Storage_) – 分配在已选择的设备上的对象。|
| --- | --- |

```py
torch.cuda.empty_cache()
```

释放缓存分配器当前持有的所有未占用的缓存显存，使其可以用在其他GPU应用且可以在 `nvidia-smi`可视化。

注意

[`empty_cache()`](#torch.cuda.empty_cache "torch.cuda.empty_cache") 并不会增加PyTorch可以使用的GPU显存的大小。 查看 [显存管理](notes/cuda.html#cuda-memory-management) 来获取更多的GPU显存管理的信息。

```py
torch.cuda.get_device_capability(device)
```

获取一个设备的cuda容量。

| 参数: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _或者_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ 可选的) – 需要返回容量的设备。如果这个参数传入的是负数，那么这个方法不会起任何作用。如果[`device`](#torch.cuda.device "torch.cuda.device")是`None`(默认值），会通过 [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device")传入当前设备。 |
| --- | --- |
| 返回: | 设备的最大和最小的cuda容量。 |
| --- | --- |
| 返回 类型: | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")([int](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) |
| --- | --- |

```py
torch.cuda.get_device_name(device)
```

获取设备名称。

| 参数: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _或者_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) – 需要返回名称的设备。如果参数是负数，那么将不起作用。如果[`device`](#torch.cuda.device "torch.cuda.device")是`None`(默认值），会通过 [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device")传入当前设备。 |
| --- | --- |

```py
torch.cuda.init()
```

初始化PyTorch的CUDA状态。如果你通过C API与PyTorch进行交互，你可能需要显式调用这个方法。只有CUDA的初始化完成，CUDA的功能才会绑定到Python。用户一般不应该需要这个，因为所有PyTorch的CUDA方法都会自动在需要的时候初始化CUDA。

如果CUDA的状态已经初始化了，将不起任何作用。

```py
torch.cuda.is_available()
```

返回一个bool值，表示当前CUDA是否可用。

```py
torch.cuda.max_memory_allocated(device=None)
```

返回给定设备的张量的最大GPU显存使用量(以字节为单位）。

| 参数: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – 选择的设备。如果 [`device`](#torch.cuda.device "torch.cuda.device") 是`None`(默认的），将返回 [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device")返回的当前设备的数据。 |
| --- | --- |

注意

查看 [显存管理](notes/cuda.html#cuda-memory-management) 部分了解更多关于GPU显存管理部分的详细信息。

```py
torch.cuda.max_memory_cached(device=None)
```

返回给定设备的缓存分配器管理的最大GPU显存(以字节为单位）。

| 参数: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _或者_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) – 选择的设备。如果 [`device`](#torch.cuda.device "torch.cuda.device") 是`None`(默认的），将返回 [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device")返回的当前设备的数据。|
| --- | --- |

注意

查看 [显存管理](notes/cuda.html#cuda-memory-management) 部分了解更多关于GPU显存管理部分的详细信息。

```py
torch.cuda.memory_allocated(device=None)
```

返回给定设备的当前GPU显存使用量(以字节为单位）。

| 参数: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _或者_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) – 选择的设备。如果 [`device`](#torch.cuda.device "torch.cuda.device") 是`None`(默认的），将返回 [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device")返回的当前设备的数据。 |
| --- | --- |

注意

这可能比 `nvidia-smi` 显示的数量少，因为一些没有使用的显存会被缓存分配器持有，且一些上下文需要在GPU中创建。查看 [显存管理](notes/cuda.html#cuda-memory-management) 部分了解更多关于GPU显存管理部分的详细信息。

```py
torch.cuda.memory_cached(device=None)
```

返回由缓存分配器管理的当前GPU显存(以字节为单位）。

| 参数: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _或者_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) – 选择的设备。如果 [`device`](#torch.cuda.device "torch.cuda.device") 是`None`(默认的），将返回 [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device")返回的当前设备的数据。|
| --- | --- |

注意

查看 [显存管理](notes/cuda.html#cuda-memory-management) 部分了解更多关于GPU显存管理部分的详细信息。

```py
torch.cuda.set_device(device)
```

设置当前设备。

不鼓励使用此功能以支持 [`device`](#torch.cuda.device "torch.cuda.device").。在多数情况下，最好使用`CUDA_VISIBLE_DEVICES`环境变量。

| 参数: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _或者_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 选择的设备。如果参数是负数，将不会起任何作用。 |
| --- | --- |

```py
torch.cuda.stream(stream)
```

给定流的上下文管理器。

所有CUDA在上下文中排队的内核将会被添加到选择的流中。

| 参数: | **stream** ([_Stream_](#torch.cuda.Stream "torch.cuda.Stream")) – 选择的流。如果为`None`，这个管理器将不起任何作用。|
| --- | --- |

注意

流是针对每个设备的，这个方法只更改当前选择设备的“当前流”。选择一个不同的设备流是不允许的。

```py
torch.cuda.synchronize()
```

等待所有当前设备的所有流完成。

## 随机数生成器

```py
torch.cuda.get_rng_state(device=-1)
```

以ByteTensor的形式返回当前GPU的随机数生成器的状态。

| 参数: | **device** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) – 需要返回RNG状态的目标设备。默认：-1 (例如，使用当前设备)。 |
| --- | --- |

警告

此函数会立即初始化CUDA。

```py
torch.cuda.set_rng_state(new_state, device=-1)
```

设置当前GPU的随机数生成器状态。

| 参数: | **new_state** ([_torch.ByteTensor_](tensors.html#torch.ByteTensor "torch.ByteTensor")) – 目标状态 |
| --- | --- |

```py
torch.cuda.manual_seed(seed)
```

设置为当前GPU生成随机数的种子。如果CUDA不可用，可以安全地调用此函数；在这种情况下，它将被静默地忽略。

| 参数: | **seed** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 目标种子。 |
| --- | --- |

警告

如果您使用的是多GPU模型，那么这个函数不具有确定性。设置用于在所有GPU上生成随机数的种子，使用 [`manual_seed_all()`](#torch.cuda.manual_seed_all "torch.cuda.manual_seed_all").

```py
torch.cuda.manual_seed_all(seed)
```

设置用于在所有GPU上生成随机数的种子。 如果CUDA不可用，可以安全地调用此函数；在这种情况下，它将被静默地忽略。

| 参数: | **seed** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 目标种子。|
| --- | --- |

```py
torch.cuda.seed()
```

将用于生成随机数的种子设置为当前GPU的随机数。 如果CUDA不可用，可以安全地调用此函数；在这种情况下，它将被静默地忽略。

警告

如果您使用的是多GPU模型，此函数将只初始化一个GPU上的种子。在所有GPU上将用于生成随机数的种子设置为随机数， 使用 [`seed_all()`](#torch.cuda.seed_all "torch.cuda.seed_all").

```py
torch.cuda.seed_all()
```

在所有GPU上将用于生成随机数的种子设置为随机数。 如果CUDA不可用，可以安全地调用此函数；在这种情况下，它将被静默地忽略。

```py
torch.cuda.initial_seed()
```

返回当前GPU的当前随机种子。

警告

此函数会立即初始化CUDA。

## 通信集合

```py
torch.cuda.comm.broadcast(tensor, devices)
```

将张量广播到多个GPU。

| 参数: | 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 需要广播的张量。
*   **devices** (_Iterable_) – 一个要被广播的可迭代的张量集合。注意，它应该是这样的形式 (src, dst1, dst2, …)，其中第一个元素是广播的源设备。

| 返回: | 一个包含`tensor`副本的元组，放置在与设备索引相对应的设备上。|
| --- | --- |

```py
torch.cuda.comm.broadcast_coalesced(tensors, devices, buffer_size=10485760)
```

将序列张量广播到指定的GPU。 首先将小型张量合并到缓冲区中以减少同步次数。

| 参数: | 

*   **tensors** (_sequence_) – 要被广播的张量。
*   **devices** (_Iterable_) – 一个要被广播的可迭代的张量集合。注意，它应该是这样的形式 (src, dst1, dst2, …)，其中第一个元素是广播的源设备。
*   **buffer_size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 用于合并的缓冲区的最大大小


| 返回: | 一个包含`tensor`副本的元组，放置在与设备索引相对应的设备上。 |
| --- | --- |

```py
torch.cuda.comm.reduce_add(inputs, destination=None)
```

从多个GPU上对张量进行求和。

所有输入必须有相同的形状。

| 参数: | 

*   **inputs** (_Iterable__[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]_) – 一个可迭代的要添加的张量集合。
*   **destination** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) – 输出所在的设备。(默认值: 当前设备)。


| 返回: | 一个包含按元素相加的所有输入的和的张量，在 `destination` 设备上。 |
| --- | --- |

```py
torch.cuda.comm.scatter(tensor, devices, chunk_sizes=None, dim=0, streams=None)
```

将张量分散在多个GPU上。

| 参数: | 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 要分散的张量.
*   **devices** (_Iterable__[_[_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_]_) – 可迭代的数字集合，指明在哪个设备上的张量要被分散。
*   **chunk_sizes** (_Iterable__[_[_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_]__,_ _可选的_) – 每个设备上放置的块的大小。它应该和`devices`的长度相等，并相加等于`tensor.size(dim)`。如果没有指定，张量将会被分散成相同的块。
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) – 分块张量所在的维度。


| 返回: | 一个包含`tensor`块的元组，分散在给定的`devices`上。 |
| --- | --- |

```py
torch.cuda.comm.gather(tensors, dim=0, destination=None)
```

从多个GPU收集张量。

在所有维度中与`dim`不同的张量尺寸必须匹配。

| 参数: | 

*   **tensors** (_Iterable__[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]_) – 可迭代的张量集合。
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 纬度，张量将会在这个维度上被连接。
*   **destination** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) – 输出设备(-1 表示 CPU, 默认值: 当前设备)


| 返回: | 在`destination` 设备上的张量，这是沿着`dim`连接张量的结果。 |
| --- | --- |

## 流和事件

```py
class torch.cuda.Stream
```

围绕CUDA流的包装器。

CUDA流是属于特定设备的线性执行序列，独立于其他流。 查看 [CUDA semantics](notes/cuda.html#cuda-semantics) 获取更详细的信息。

参数:

*   **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _或者_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) – 要在其上分配流的设备。 如果 [`device`](#torch.cuda.device "torch.cuda.device") 为`None`(默认值）或负整数，则将使用当前设备。
*   **priority** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) – 流的优先级。数字越小，优先级越高。

```py
query()
```

检查提交的所有工作是否已完成。

| 返回: | 一个布尔值，表示此流中的所有内核是否都已完成。|
| --- | --- |

```py
record_event(event=None)
```

记录一个事件。

| 参数: | **event** ([_Event_](#torch.cuda.Event "torch.cuda.Event")_,_ _可选的_) – 需要记录的事件。如果没有给出，将分配一个新的。 |
| --- | --- |
| 返回: | 记录的事件。 |
| --- | --- |

```py
synchronize()
```

等待此流中的所有内核完成。

注意

这是一个围绕 `cudaStreamSynchronize()`的包装： 查看 [CUDA 文档](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html) 获取更详细的信息。

```py
wait_event(event)
```

使提交给流的所有未来工作等待事件。

| 参数: | **event** ([_Event_](#torch.cuda.Event "torch.cuda.Event")) – 需要等待的事件。 |
| --- | --- |

注意

这是一个围绕 `cudaStreamWaitEvent()`的包装： 查看 [CUDA 文档](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html) 获取更详细的信息。

此函数返回时无需等待`event`： 只有未来的操作受到影响。

```py
wait_stream(stream)
```

与另一个流同步。

提交给此流的所有未来工作将等到所有内核在呼叫完成时提交给给定流。

| 参数: | **stream** ([_Stream_](#torch.cuda.Stream "torch.cuda.Stream")) – 要同步的流。 |
| --- | --- |

注意

此函数返回时不等待`stream`中当前排队的内核 ： 只有未来的操作受到影响。

```py
class torch.cuda.Event(enable_timing=False, blocking=False, interprocess=False, _handle=None)
```

围绕CUDA事件的包装。

参数:

*   **enable_timing** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 表示事件是否应该测量时间(默认值：`False`）
*   **blocking** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果是`True`， [`wait()`](#torch.cuda.Event.wait "torch.cuda.Event.wait") 将会阻塞 (默认值: `False`)
*   **interprocess** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果是 `True`，事件将会在进程中共享 (默认值: `False`)

```py
elapsed_time(end_event)
```

返回记录事件之前经过的时间。

```py
ipc_handle()
```

返回此事件的IPC句柄。

```py
query()
```

检测事件是否被记录。

| 返回: | 一个布尔值，表示事件是否被记录。|
| --- | --- |

```py
record(stream=None)
```

记录给定流的一个事件。

```py
synchronize()
```

和一个事件同步。

```py
wait(stream=None)
```

使给定的流等待一个事件。

## 显存管理

```py
torch.cuda.empty_cache()
```

释放当前由缓存分配器保存的所有未占用的缓存显存，以便可以在其他GPU应用程序中使用这些缓存并在`nvidia-smi`中可见。

注意

[`empty_cache()`](#torch.cuda.empty_cache "torch.cuda.empty_cache") 不会增加PyTorch可用的GPU显存量。 查看 [显存管理](notes/cuda.html#cuda-memory-management) 以了解更多GPU显存管理的详细信息。

```py
torch.cuda.memory_allocated(device=None)
```

返回给定设备的当前GPU显存使用量(以字节为单位）。

| 参数: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _或者_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) – 选定的设备。如果 [`device`](#torch.cuda.device "torch.cuda.device") 是`None`(默认的），将返回 [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device")返回的当前设备的数据。 |
| --- | --- |

注意

这可能比 `nvidia-smi` 显示的数量少，因为一些没有使用的显存会被缓存分配器持有，且一些上下文需要在GPU中创建。查看 [显存管理](notes/cuda.html#cuda-memory-management) 部分了解更多关于GPU显存管理部分的详细信息。

```py
torch.cuda.max_memory_allocated(device=None)
```

返回给定设备的张量的最大GPU显存使用量(以字节为单位）。

| 参数: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _或者_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) –  选择的设备。如果 [`device`](#torch.cuda.device "torch.cuda.device") 是`None`(默认的），将返回 [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device")返回的当前设备的数据。  |
| --- | --- |

注意

查看 [显存管理](notes/cuda.html#cuda-memory-management) 部分了解更多关于GPU显存管理部分的详细信息。

```py
torch.cuda.memory_cached(device=None)
```

返回由缓存分配器管理的当前GPU显存(以字节为单位）。

| 参数: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) – 选择的设备。如果 [`device`](#torch.cuda.device "torch.cuda.device") 是`None`(默认的），将返回 [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device")返回的当前设备的数据。 |
| --- | --- |

注意

查看 [显存管理](notes/cuda.html#cuda-memory-management) 部分了解更多关于GPU显存管理部分的详细信息。

```py
torch.cuda.max_memory_cached(device=None)
```

返回给定设备的缓存分配器管理的最大GPU显存(以字节为单位）。

| 参数: | **device** ([_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device") _或者_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) – 选择的设备。如果 [`device`](#torch.cuda.device "torch.cuda.device") 是`None`(默认的），将返回 [`current_device()`](#torch.cuda.current_device "torch.cuda.current_device")返回的当前设备的数据。|
| --- | --- |

注意

查看 [显存管理](notes/cuda.html#cuda-memory-management) 部分了解更多关于GPU显存管理部分的详细信息。

## NVIDIA Tools Extension (NVTX)

```py
torch.cuda.nvtx.mark(msg)
```

描述某个时刻发生的瞬时事件。

| 参数: | **msg** (_string_) – 与时间相关的ASCII信息。 |
| --- | --- |

```py
torch.cuda.nvtx.range_push(msg)
```

将范围推到嵌套范围跨度的堆栈上。 返回启动范围的从零开始的深度。

| 参数: | **msg** (_string_) – 与时间相关的ASCII信息。 |
| --- | --- |

```py
torch.cuda.nvtx.range_pop()
```

从一堆嵌套范围跨度中弹出一个范围。 返回结束范围的从零开始的深度。

