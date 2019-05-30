# torch.cuda

> 译者：[@谈笑风生](https://github.com/zhu1040028623)
> 
> 校对者：[@smilesboy](https://github.com/smilesboy)

这个包增加了对 CUDA tensor (张量) 类型的支持,利用 GPUs 计算实现了与 CPU tensors 相同的类型.

这个是 lazily initialized (懒加载,延迟加载), 所以你可以一直导入它,并且可以用 `is_available()` 来判断 你的系统是否支持 CUDA.

[CUDA 语义](notes/cuda.html#cuda-semantics) 有更多关于使用 CUDA 的细节.

```py
torch.cuda.current_blas_handle()
```

返回指向当前 cuBLAS 句柄的 cublasHandle_t 指针

```py
torch.cuda.current_device()
```

返回当前选择的设备的索引.

```py
torch.cuda.current_stream()
```

返回当前选择的 `Stream` .

```py
class torch.cuda.device(idx)
```

更改选定设备的上下文管理器.

参数：`idx (int)` – 选择设备编号. 如果参数无效,则是无效操作.


```py
torch.cuda.device_count()
```

返回可用的 GPU 数量.

```py
torch.cuda.device_ctx_manager
```

`device`的别名。

```py
class torch.cuda.device_of(obj)
```

将当前设备更改为给定对象的上下文管理器.

可以使用张量和存储作为参数,如果给定的对象不是在 GPU 上分配的,这是一个无效操作.

参数：`obj (Tensor 或 Storage)` – 在选定设备上分配的对象.


```py
torch.cuda.empty_cache()
```

释放当前由缓存持有的所有未占用缓存内存分配器,以便可以在其他GPU应用程序中使用并在 `nvidia-smi` 中可见.

```py
torch.cuda.get_device_capability(device)
```

获取设备的 CUDA 算力.

参数：`device (int)` – 返回设备名, 参数无效时, 方法失效.

返回值：设备的主次要 CUDA 算力.

返回类型：`tuple(int, int)`

```py
torch.cuda.get_device_name(device)
```

获取设备名.

参数：`device (int)` – 返回设备名. 参数无效时,则是无效操作.


```py
torch.cuda.is_available()
```

返回一个 bool 值表示 CUDA 目前是否可用.

```py
torch.cuda.set_device(device)
```

设置当前设备.

不鼓励使用这个函数 `device` . 在大多数情况下,最好使用 `CUDA_VISIBLE_DEVICES` 环境变量.

参数：`device (int)` – 选择设备. 参数无效时,则是无效操作.


```py
torch.cuda.stream(*args, **kwds)
```

选择给定流的上下文管理器.

在选定的流上, 所有的CUDA内核在其上下文内排队.

参数：`stream (Stream)` – 选择流. 如果是 `None` , 管理器无效.


```py
torch.cuda.synchronize()
```

等待当前设备上所有流中的所有内核完成.

## Random Number Generator

```py
torch.cuda.get_rng_state(device=-1)
```

将当前 GPU 的随机数生成器状态作为 ByteTensor 返回.

参数：`device (int, 可选)` – 设备的 RNG 状态. Default: -1 (i.e., 使用当前设备).


警告：

函数需要提前初始化 CUDA .

```py
torch.cuda.set_rng_state(new_state, device=-1)
```

设置当前 GPU 的随机数发生器状态.

参数：`new_state (torch.ByteTensor)` – 所需的状态


```py
torch.cuda.manual_seed(seed)
```

设置用于当前 GPU 生成随机数的种子. 如果 CUDA 不可用,调用这个函数是安全的;在这种情况下,它将被忽略.

参数：`seed (int 或 long)` – 所需的种子.


警告：

如果您正在使用多 GPU 模型,则此功能不足以获得确定性. seef作用于所有 GPUs , 使用 `manual_seed_all()` .

```py
torch.cuda.manual_seed_all(seed)
```

设置在所有 GPU 上生成随机数的种子. 如果 CUDA 不可用, 调用此函数是安全的; 这种情况下,会被忽略.

参数：`seed (int 或 long)` – 所需的种子.


```py
torch.cuda.seed()
```

将用于生成随机数的种子设置为当前 GPU 的随机数. 如果 CUDA 不可用,则调用此函数是安全的. 在那种情况下,会被忽略.

警告：

如果您正在使用多 GPU 模型, 则此功能不足以获得确定性. seef作用于所有 GPUs , 使用 `seed_all()`.

```py
torch.cuda.seed_all()
```

在所有 GPU 上将用于生成随机数的种子设置为随机数. 如果 CUDA 不可用,则调用此函数是安全的. 在那种情况下,会被忽略.

```py
torch.cuda.initial_seed()
```

返回当前 GPU 的当前随机种子.

警告：

函数提前初始化 CUDA .

## Communication collectives

```py
torch.cuda.comm.broadcast(tensor, devices)
```

将张量广播给多个 GPU .

参数：

*   `tensor (Tensor)` – 需要广播的张量.
*   `devices (Iterable)` – 在一个可迭代设备中广播. 请注意, 它应该像 (src, dst1, dst2, …), 其中的第一个元素是来至其广播的源设备.


返回值：一个元组, 包含 `tensor` 副本,放置在与设备的索引相对应的 `设备` 上.


```py
torch.cuda.comm.reduce_add(inputs, destination=None)
```

从多个 GPU 中收集张量.

所有的输入应该有匹配的 shapes (形状).

参数：

*   `inputs (Iterable[Tensor])` – 添加一个可迭代的张量.
*   `destination (int, 可选)` – 放置输出的设备 (默认: 当前设备).


返回值：包含所有输入的元素和的张量, 存放在 `destination(目标)` 设备.


```py
torch.cuda.comm.scatter(tensor, devices, chunk_sizes=None, dim=0, streams=None)
```

分散张量到多个 GPU.

参数：

*   `tensor (Tensor)` – 需要分散的张量.
*   `devices (Iterable[int])` – 整数的迭代,指定张量应分散在哪些设备之间.
*   `chunk_sizes (Iterable[int], 可选)` – 要放在每个设备上的块的大小. 应该匹配 `设备` 长度和 `tensor.size(dim)` 的和. 如果未指定,张量将被划分成相等的块.
*   `dim (int, 可选)` – 分块张量沿着的维度


返回值：一个元组包含 `tensor` 块, 传递给 `devices` .


```py
torch.cuda.comm.gather(tensors, dim=0, destination=None)
```

从多个 GPU 收集张量.

张量尺寸在不同于 `dim` 的维度上都应该匹配.

参数：

*   `tensors (Iterable[Tensor])` – 张量集合的迭代器.
*   `dim (int)` – 张量被连接的维度.
*   `destination (int, 可选)` – 输出设备 (-1 代表 CPU, 默认: 当前设备)


返回值：一个位于 `目标` 设备上的张量, 将 `tensors` 沿着 `dim` 连接起来的结果.


## Streams and events

```py
class torch.cuda.Stream
```

CUDA 流的包装.

参数：

*   `device (int, 可选)` – 分配流的设备.
*   `priority (int, 可选)` – 流的优先级. 较低的数字代表较高的优先级.



```py
query()
```

检查事件是否已被记录.

返回值：一个 BOOL 值, 指示事件是否已被记录.


```py
record_event(event=None)
```

记录一个事件.

参数：`event (Event, 可选)` – 要记录的事件.如果没有给出,将分配一个新的.

返回值：记录的事件.


```py
synchronize()
```

等待流中的所有内核完成.

```py
wait_event(event)
```

将所有未来的工作提交到流等待事件.

参数：`event (Event)` – 等待的事件.


```py
wait_stream(stream)
```

与另一个流同步.

提交到此流的所有未来工作将等待直到所有核心在调用完成时提交给给定的流.

参数：`stream (Stream)` – 同步流.


```py
class torch.cuda.Event(enable_timing=False, blocking=False, interprocess=False, _handle=None)
```

CUDA 事件包装器.

参数：

*   `enable_timing (bool)` – 指示事件是否应测量时间 (默认: `False`)
*   `blocking (bool)` – 如果 `True`, `wait()` 将阻塞 (默认: `False` )
*   `interprocess (bool)` – 如果 `True`, 事件可以在进程之间共享 (默认: `False`)



```py
elapsed_time(end_event)
```

返回记录事件之前所经过的时间.

```py
ipc_handle()
```

返回此事件的 IPC 句柄.

```py
query()
```

检查事件是否已记录.

返回值：一个 BOOL 值, 指示事件是否已被记录.


```py
record(stream=None)
```

记录给定流中的事件.

```py
synchronize()
```

与事件同步.

```py
wait(stream=None)
```

使给定流等待事件发生.

## Memory management

```py
torch.cuda.empty_cache()
```

释放当前由缓存持有的所有未占用缓存内存分配器,以便可以在其他GPU应用程序中使用并在 `nvidia-smi` 中可见.

## NVIDIA Tools Extension (NVTX)

```py
torch.cuda.nvtx.mark(msg)
```

描述在某个时刻发生的瞬间事件.

参数：`msg (string)` – 事件(用 ASCII 编码表示).


```py
torch.cuda.nvtx.range_push(msg)
```

设置一个固定范围的堆栈,返回的堆栈范围深度从0开始.

参数：`msg (string)` – 范围(用 ASCII 编码设置)


```py
torch.cuda.nvtx.range_pop()
```

弹出一个固定范围的堆栈,返回的堆栈范围深度从0结束.