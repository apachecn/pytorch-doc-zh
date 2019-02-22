

# 多进程包 - torch.multiprocessing

> 译者：[hijkzzz](https://github.com/hijkzzz)


torch.multiprocessing 是一个本地 [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "(in Python v3.7)") 模块的包装. 它注册了自定义的reducers, 并使用共享内存为不同的进程在同一份数据上提供共享的视图. 一旦 tensor/storage 被移动到共享内存 (见 [`share_memory_()`](tensors.html#torch.Tensor.share_memory_ "torch.Tensor.share_memory_")), 将其发送到任何进程不会造成拷贝开销.

此 API 100% 兼容原生模块 - 所以足以将 `import multiprocessing` 改成 `import torch.multiprocessing` 使得所有的 tensors 通过队列发送或者使用其它共享机制, 移动到共享内存.

因为 APIs 的相似性, 我们没有为此包提供足够的文档, 所以推荐参考非常优秀的原生进程模块文档.

警告

如果主进程意外退出 (比如 因为一个信号的到来), Python’s `multiprocessing` 有时候会无法请理它的子进程. 这是一个众所周知的警告, 因此，如果你在中断解释器后发现任何资源泄漏，这可能意味着你刚刚发生了这种情况.

## 策略管理

```py
torch.multiprocessing.get_all_sharing_strategies()
```

返回当前系统支持的共享策略的集合.

```py
torch.multiprocessing.get_sharing_strategy()
```

返回当前的 CPU tensors 共享策略.

```py
torch.multiprocessing.set_sharing_strategy(new_strategy)
```

设置一个新的 CPU tensors 共享策略.

| 参数: | **new_strategy** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")) – 选定策略的名字. 必须是 [`get_all_sharing_strategies()`](#torch.multiprocessing.get_all_sharing_strategies "torch.multiprocessing.get_all_sharing_strategies") 的返回值中的一个. |
| --- | --- |

## 共享 CUDA tensors

在进程间共享 CUDA tensors 仅仅在 Python 3 中被支持, 使用 `spawn` 或者 `forkserver` 启动方法. [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "(in Python v3.7)") 在 Python 2 中只能使用 `fork` 创建新进程, 然而 CUDA 运行时不支持它.

警告

CUDA API要求导出到其他进程的分配只要被其他进程使用就保持有效. 您应该小心，并确保共享的CUDA tensor在必要时不会超出范围. 共享模型参数不应该是一个问题，但是传递其他类型的数据应该小心。注意，此限制不适用于共享CPU内存.

## 共享策略

本节简要概述不同的共享策略是如何工作的。注意，它只适用于CPU tensor——CUDA tensor总是使用CUDA API，因为这是它们可以共享的唯一方式。

### 文件描述符 - `file_descriptor`

注意

这是默认策略(macOS和OS X因为不支持除外)

该策略将使用文件描述符作为共享内存句柄。每当一个存储被移动到共享内存时，从`shm open`获得的文件描述符就会被对象缓存，当它被发送到其他进程时，文件描述符就会被传输(例如通过UNIX套接字)到它。接收者还将缓存文件描述符并`mmap`它，以获得存储数据上的共享视图。

请注意，如果共享了很多tensor，那么这种策略将在大多数情况下打开大量的文件描述符。如果您的系统对打开的文件描述符的数量限制很低，并且您不能提高它们的数量，那么您应该使用`file_system`策略。

### 文件系统 - `file_system`

该策略将使用指定给`shm open`的文件名来标识共享内存区域。这样做的好处是不需要实现缓存从中获得的文件描述符，但同时容易导致共享内存泄漏。文件不能在创建之后立即删除，因为其他进程需要访问它来打开它们的视图。如果进程致命地崩溃或被杀死，并且不调用存储析构函数，那么文件将保留在系统中。这是非常严重的，因为它们会一直使用内存，直到系统重新启动，或者重新手动释放。

为了解决共享内存文件泄漏的问题，`torch.multiprocessing`将生成一个名为`torch_shm_manager`的守护进程，它将自己与当前进程组隔离，并跟踪所有共享内存分配。连接到它的所有进程退出后，它将等待一段时间以确保没有新的连接，并将遍历组分配的所有共享内存文件。如果它发现其中任何一个仍然存在，就会解除它们的分配。我们对这种方法进行了测试，证明它对各种故障都具有鲁棒性。 不过，如果您的系统有足够高的限制，并且`file_descriptor`是受支持的策略，我们不建议切换到这个策略。

## Spawning 子线程

注意

仅支持 Python &gt;= 3.4.

依赖于 `spawn` 启动方法(在 Python 的 `multiprocessing` 包中)。

通过创建`进程`实例并调用join来等待它们完成，可以生成大量子进程来执行某些功能。这种方法在处理单个子进程时工作得很好，但在处理多个进程时可能会出现问题。

也就是说，顺序连接进程意味着它们将顺序终止。如果没有，并且第一个进程没有终止，那么进程终止将不被注意。 此外，没有用于错误传播的本地工具.

下面的`spawn`函数解决了这些问题，并负责错误传播、无序终止，并在检测到其中一个错误时主动终止进程.

```py
torch.multiprocessing.spawn(fn, args=(), nprocs=1, join=True, daemon=False)
```

Spawns `nprocs` 进程运行 `fn` 使用参数 `args`.

如果其中一个进程以非零退出状态退出，则会杀死其余进程，并引发异常，导致终止。在子进程中捕获异常的情况下，将转发该异常，并将其跟踪包含在父进程中引发的异常中。

参数: 

*   **fn** (_function_) –

    函数被称为派生进程的入口点。必须在模块的顶层定义此函数，以便对其进行pickle和派生。这是多进程强加的要求。

    该函数称为`fn(i， *args)`，其中`i`是进程索引，`args`是传递的参数元组。

*   **args** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")) – 传递给 `fn` 的参数.
*   **nprocs** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 派生的进程数.
*   **join** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 执行一个阻塞的join对于所有进程.
*   **daemon** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 派生进程守护进程标志。如果设置为True，将创建守护进程.


| 返回值: | None 如果 `join` 是 `True`, [`SpawnContext`](#torch.multiprocessing.SpawnContext "torch.multiprocessing.SpawnContext") 如果 `join` 是 `False` |
| --- | --- |

```py
class torch.multiprocessing.SpawnContext
```

由 [`spawn()`](#torch.multiprocessing.spawn "torch.multiprocessing.spawn") 返回, 当 `join=False`.

```py
join(timeout=None)
```

尝试连接此派生上下文中的一个或多个进程。如果其中一个进程以非零退出状态退出，则此函数将杀死其余进程，并引发异常，导致第一个进程退出。

返回 `True`如果所有进程正常退出, `False` 如果有更多的进程需要 join.

| Parameters: | **timeout** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – 放弃等待的最长时间. |
| --- | --- |

