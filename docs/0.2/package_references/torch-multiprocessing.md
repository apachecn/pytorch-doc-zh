# torch.multiprocessing
封装了`multiprocessing`模块。用于在相同数据的不同进程中共享视图。

一旦张量或者存储被移动到共享单元(见`share_memory_()`),它可以不需要任何其他复制操作的发送到其他的进程中。

这个API与原始模型完全兼容，为了让张量通过队列或者其他机制共享，移动到内存中，我们可以

由原来的`import multiprocessing`改为`import torch.multiprocessing`。

由于API的相似性，我们没有记录这个软件包的大部分内容，我们建议您参考原始模块的非常好的文档。

**`warning：`**
如果主要的进程突然退出(例如，因为输入信号)，Python中的`multiprocessing`有时会不能清理他的子节点。

这是一个已知的警告，所以如果您在中断解释器后看到任何资源泄漏，这可能意味着这刚刚发生在您身上。

## Strategy management
```python
torch.multiprocessing.get_all_sharing_strategies()
```
返回一组由当前系统所支持的共享策略

```python
torch.multiprocessing.get_sharing_strategy()
```
返回当前策略共享CPU中的张量。

```python
torch.multiprocessing.set_sharing_strategy(new_strategy)
```
设置共享CPU张量的策略

参数: new_strategy(str)-被选中策略的名字。应当是`get_all_sharing_strategies()`中值当中的一个。

## Sharing CUDA tensors
共享CUDA张量进程只支持Python3，使用`spawn`或者`forkserver`开始方法。

Python2中的`multiprocessing`只能使用`fork`创建子进程，并且不被CUDA支持。

**`warning：`**
CUDA API要求导出到其他进程的分配一直保持有效，只要它们被使用。

你应该小心，确保您共享的CUDA张量不要超出范围。

这不应该是共享模型参数的问题，但传递其他类型的数据应该小心。请注意，此限制不适用于共享CPU内存。

## Sharing strategies
本节简要概述了不同的共享策略如何工作。

请注意，它仅适用于CPU张量 - CUDA张量将始终使用CUDA API，因为它们是唯一的共享方式。

### File descriptor-`file_descripor`
**`NOTE：`**
这是默认策略(除了不支持的MacOS和OS X）。

此策略将使用文件描述符作为共享内存句柄。当存储被移动到共享内存中，一个由`shm_open`获得的文件描述符被缓存，

并且当它将被发送到其他进程时，文件描述符将被传送(例如通过UNIX套接字）。

接收者也将缓存文件描述符，并且`mmap`它，以获得对存储数据的共享视图。

请注意，如果要共享很多张量，则此策略将保留大量文件描述符。

如果你的系统对打开的文件描述符数量有限制，并且无法提高，你应该使用`file_system`策略。

### File system -file_system
这个策略将提供文件名称给`shm_open`去定义共享内存区域。

该策略不需要缓存从其获得的文件描述符的优点，但是容易发生共享内存泄漏。

该文件创建后不能被删除，因为其他进程需要访问它以打开其视图。

如果进程崩溃或死机，并且不能调用存储析构函数，则文件将保留在系统中。

这是非常严重的，因为它们在系统重新启动之前不断使用内存，或者手动释放它们。

为了记录共享内存文件泄露数量，`torch.multiprocessing`将产生一个守护进程叫做`torch_shm_manager`

将自己与当前进程组隔离，并且将跟踪所有共享内存分配。一旦连接到它的所有进程退出，

它将等待一会儿，以确保不会有新的连接，并且将遍历该组分配的所有共享内存文件。

如果发现它们中的任何一个仍然存在，它们将被释放。我们已经测试了这种方法，并且它已被证明对于各种故障都是稳健的。

如果你的系统有足够高的限制，并且`file_descriptor`是被支持的策略，我们不建议切换到这个。
