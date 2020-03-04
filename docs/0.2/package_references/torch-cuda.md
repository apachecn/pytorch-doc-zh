# torch.cuda
该包增加了对CUDA张量类型的支持，实现了与CPU张量相同的功能，但使用GPU进行计算。

它是懒惰的初始化，所以你可以随时导入它，并使用`is_available()`来确定系统是否支持CUDA。

[CUDA语义](../notes/cuda.md)中有关于使用CUDA的更多细节。

```python
torch.cuda.current_blas_handle()
```
返回cublasHandle_t指针，指向当前cuBLAS句柄

```python
torch.cuda.current_device()
```
返回当前所选设备的索引。

```python
torch.cuda.current_stream()
```
返回一个当前所选的`Stream`

```python
class torch.cuda.device(idx)
```
上下文管理器，可以更改所选设备。

**参数：**
- **idx** (*int*) – 设备索引选择。如果这个参数是负的，则是无效操作。

```python
torch.cuda.device_count()
```
返回可得到的GPU数量。

```python
class torch.cuda.device_of(obj)
```

将当前设备更改为给定对象的上下文管理器。

可以使用张量和存储作为参数。如果给定的对象不是在GPU上分配的，这是一个无效操作。

**参数：**
- **obj** (*Tensor* or *Storage*) – 在选定设备上分配的对象。

```python
torch.cuda.is_available()
```
返回一个bool值，指示CUDA当前是否可用。

```python
torch.cuda.set_device(device)
```
设置当前设备。

不鼓励使用此函数来设置。在大多数情况下，最好使用`CUDA_VISIBLE_DEVICES`环境变量。

**参数：**
- **device** (*int*) – 所选设备。如果此参数为负，则此函数是无效操作。

```python
torch.cuda.stream(stream)
```
选择给定流的上下文管理器。

在其上下文中排队的所有CUDA核心将在所选流上入队。

**参数：**
- **stream** (*Stream*) – 所选流。如果是`None`，则这个管理器是无效的。

```python
torch.cuda.synchronize()
```
等待当前设备上所有流中的所有核心完成。

## 交流集
```python
torch.cuda.comm.broadcast(tensor, devices)
```
向一些GPU广播张量。

**参数：**
- **tensor** (*Tensor*) – 将要广播的张量
- **devices** (*Iterable*) – 一个可以广播的设备的迭代。注意，它的形式应该像(src，dst1，dst2，...），其第一个元素是广播来源的设备。

**返回：** 一个包含张量副本的元组，放置在与设备的索引相对应的设备上。

```python
torch.cuda.comm.reduce_add(inputs, destination=None)
```
将来自多个GPU的张量相加。

所有输入应具有匹配的形状。

**参数：**
- **inputs** (*Iterable[Tensor]*) – 要相加张量的迭代
- **destination** (*int*, optional) – 将放置输出的设备(默认值：当前设备）。

**返回：** 一个包含放置在`destination`设备上的所有输入的元素总和的张量。

```python
torch.cuda.comm.scatter(tensor, devices, chunk_sizes=None, dim=0, streams=None)
```
打散横跨多个GPU的张量。

**参数：**
- **tensor** (*Tensor*) – 要分散的张量
- **devices** (*Iterable[int]*) – int的迭代，指定哪些设备应该分散张量。
- **chunk_sizes** (*Iterable[int]*, optional) – 要放置在每个设备上的块大小。它应该匹配`devices`的长度并且总和为`tensor.size(dim)`。 如果没有指定，张量将被分成相等的块。
- **dim** (*int*, optional) – 沿着这个维度来chunk张量

**返回：** 包含`tensor`块的元组，分布在给定的`devices`上。

```python
torch.cuda.comm.gather(tensors, dim=0, destination=None)
```
从多个GPU收集张量。

张量尺寸在不同于`dim`的所有维度上都应该匹配。

**参数：**
- **tensors** (*Iterable[Tensor]*) – 要收集的张量的迭代。
- **dim** (*int*) – 沿着此维度张量将被连接。
- **destination** (*int*, optional) – 输出设备(-1表示CPU，默认值：当前设备）。

**返回：** 一个张量位于`destination`设备上，这是沿着`dim`连接`tensors`的结果。

## 流和事件
```python
class torch.cuda.Stream
```
CUDA流的包装。

**参数：**
- **device** (*int*, optional) – 分配流的设备。
- **priority** (*int*, optional) – 流的优先级。较低的数字代表较高的优先级。

  > query()

  检查所有提交的工作是否已经完成。

  **返回：** 一个布尔值，表示此流中的所有核心是否完成。

  > record_event(event=None)

  记录一个事件。

  **参数：** **event** (*Event*, optional) – 要记录的事件。如果没有给出，将分配一个新的。
  **返回：** 记录的事件。

  > synchronize()

  等待此流中的所有核心完成。

  > wait_event(event)

  将所有未来的工作提交到流等待事件。

  **参数：** **event** (*Event*) – 等待的事件

  > wait_stream(stream)

  与另一个流同步。

  提交到此流的所有未来工作将等待直到所有核心在调用完成时提交给给定的流。

```python
class torch.cuda.Event(enable_timing=False, blocking=False, interprocess=False, _handle=None)
```

CUDA事件的包装。

**参数：**
- **enable_timing** (*bool*) – 指示事件是否应该测量时间(默认值：False）
- **blocking** (*bool*) – 如果为true，`wait()`将被阻塞(默认值：False）
- **interprocess** (*bool*) – 如果为true，则可以在进程之间共享事件(默认值：False）

  > elapsed_time(end_event)

  返回事件记录之前经过的时间。

  > ipc_handle()

  返回此事件的IPC句柄。

  > query()

  检查事件是否已被记录。

  **返回：** 一个布尔值，指示事件是否已被记录。

  > record(stream=None)

  记录给定流的事件。

  > synchronize()

  与事件同步。

  > wait(stream=None)

  使给定的流等待事件。
