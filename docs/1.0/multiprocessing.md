

# 多进程包 - torch.multiprocessing

torch.multiprocessing 封装了原生的 [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "(in Python v3.7)") 模块。它注册了自定义的reducer，通过使用共享内存，让不同的进程可以以共享的方式访问相同的数据。当张量或存储数据移动到 shared_memory (详见 [`share_memory_()`](tensors.html#torch.Tensor.share_memory_ "torch.Tensor.share_memory_")) 时，它们无需任何拷贝即可被其他进程处理。

API 和原有模块100%兼容 —— 只需把 `import multiprocessing` 修改为 `import torch.multiprocessing`，就可以通过队列或者其他机制的共享，把所有的张量移动到共享内存中。

由于API非常相似，我们不会对这个包的大部分内容提供文档，建议参考原模块的文档。

警告

如果主进程异常退出（比如，由于收到一个信号），Python的 `multiprocessing` 有时无法清理它的子进程。这是一个已知的问题，所以如果在中断解释器时，发现存在资源泄露，很有可能就是遇到了这个问题。

## 策略管理

```py
torch.multiprocessing.get_all_sharing_strategies()
```

返回当前系统支持的共享策略的元组。

```py
torch.multiprocessing.get_sharing_strategy()
```

返回共享的 CPU 张量的当前策略。

```py
torch.multiprocessing.set_sharing_strategy(new_strategy)
```

设置共享 CPU 张量的策略。

| 参数: | **new_strategy** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")) – 选择的策略名字。应该是 [`get_all_sharing_strategies()`](#torch.multiprocessing.get_all_sharing_strategies "torch.multiprocessing.get_all_sharing_strategies") 返回的名字之一。 |
| --- | --- |

## 共享 CUDA 张量

进程间共享 CUDA 张量只支持 Python 3, 使用 `spawn` 或者 `forkserver` 开始调用方法。Python 2 中的[`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "(in Python v3.7)") 仅使用 `fork` 创建子进程，但 CUDA 运行时对此不支持。

警告

CUDA API 要求为其他进程的显存分配，只要这些进程还在使用，它们就一直有效。你需要注意并确保共享的 CUDA 张量除非在必要的情况下，否则不要离开使用的范围。对于共享模型参数来说，这不是问题，但传递其他类型的数据时就需要非常小心。注意共享 CPU 内存没有这个限制。

## 共享策略

本节对不同的共享策略的工作原理进行简单的总结。注意本节只涉及到 CPU 张量——CUDA 张量需要调用 CUDA API，这是它们可以被共享的唯一方法。

### 文件描述符 - `file_descriptor`

注意

这是默认的策略（macOS 和 OS X 除外，因为它们不支持）。

这个策略将使用文件描述符来操作共享内存。当一个存储移动到共享内存时，使用`shm_open`获得的文件描述符缓存这个对象，当要传递到其他进程中时，这个文件描述符就会被传递过去（比如，通过UNIX socket）。接收的进程也会缓存这个文件描述符，并使用`mmap`来获得存储数据的共享内容。

注意如果存在大量的共享张量，这个策略将会在大部分时间里保持大量的文件描述符。如果你的系统限制打开的文件描述符的数量比较少，你将无法创建这些文件描述符，此时你应该使用`file_system`策略。

### 文件系统 - `file_system`

这个策略将使用传递给`shm_open`的文件名来标识共享内存区域。它的好处是无需实现对获取的文件描述符缓存，但同时它容易泄露共享的内存。文件在创建后无法立即删除，因为其他的进程需要打开它来获取内容。如果进程异常崩溃，或者被杀死，从而无法调用存储的析构函数，那么文件将会一直存在系统中。这个问题非常严重，因为这些文件在系统重启前或者被手动释放前，它们会一直占用内存。

为了解决共享内存文件泄露的问题，[`torch.multiprocessing`](#module-torch.multiprocessing "torch.multiprocessing") 将创建一个叫做`torch_shm_manager`的守护进程，它会从当前的进程组中独立出来，并保持追踪所有的共享内存分配情况。当与它连接的所有进程退出了，它将会等待一会儿，以保证不会有新的连接，并在这个进程组分配的所有共享内存文件上迭代。如果发现依旧存在共享内存文件，将会释放它们。我们已经测试了这个方法，证明它在各种失败的情况下都可以自动完成这个工作。即便如此，如果你的系统有足够多的文件限制数据，同时也支持`file_descriptor`策略，我们不建议切换到这个策略。

## 创建子进程

注意

仅 Python &gt;= 3.4 可用。

该方法依赖于Python `multiprocessing` 包中的 `spawn` 方法。

通过创建`Process`实例，并调用`join`等待它们完成，就可以创建一些子进程来完成一些要执行的函数。这个方法在处理单一子进程时没有问题，但在处理多进程时，会有潜在的问题。

也就是说，按顺序执行子进程表明它们将会按顺序终止。如果它们不是这样，而且第一个进程并没有终止，那么进程的终止竟会无法被感知。同时，也没有原生的工具用于错误传递。

下面的`spawn`函数解决了这些问题，在无序终止的情况下，可以处理错误传递，并将在检测到进程中的错误时主动终止进程。

```py
torch.multiprocessing.spawn(fn, args=(), nprocs=1, join=True, daemon=False)
```

创建 `nprocs` 个使用 `args` 参数运行的 `fn` 函数。

如果其中有一个进程以非零的状态码退出，其他的进程将会被杀死，并抛出一个终止原因的异常。在子进程中捕获异常的情况下，父进程抛出的异常中将会包括该异常和它的回溯信息。

| 参数: | 

*   **fn** (_function_) –

    创建进程的入口所调用的函数。这个函数必须在模块的顶部定义，这样才能通过pickle化并创建。这是multiprocessing的一个要求。

    这个函数以`fn(i, *args)`的方式调用，其中`i`是进程的索引，`args`是传入的参数元组。

*   **args** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")) – 传入 `fn` 的参数。
*   **nprocs** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 创建进程的数量。
*   **join** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 所有进程间进行阻塞式调用。
*   **daemon** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 创建守护进程的标志。如果设置为True，守护进程将被创建。

 |
| --- | --- |
| 返回: | None if `join` is `True`, [`SpawnContext`](#torch.multiprocessing.SpawnContext "torch.multiprocessing.SpawnContext") if `join` is `False` |
| --- | --- |

```py
class torch.multiprocessing.SpawnContext
```

当使用`join=False`调用 [`spawn()`](#torch.multiprocessing.spawn "torch.multiprocessing.spawn") 的返回结果。

```py
join(timeout=None)
```

尝试在当前创建的进程上下文中执行一个或者多个进程。如果它们中有一个以非零状态退出，这个函数会杀死剩余的进程，并抛出第一个进程退出原因的异常。

如果所有进程已经执行成功则返回`True`，如果有多个进程需要执行则返回`False`。

| 参数: | **timeout** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – 在放弃等待前的等待时长。 |
| --- | --- |

