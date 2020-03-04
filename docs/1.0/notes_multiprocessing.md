

# 多进程最佳实践

> 译者：[cvley](https://github.com/cvley)

[`torch.multiprocessing`](../multiprocessing.html#module-torch.multiprocessing "torch.multiprocessing") 是 Python 的 [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "(in Python v3.7)") 的直接替代模块。它支持完全相同的操作，但进行了扩展，这样所有的张量就可以通过一个 [`multiprocessing.Queue`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue "(in Python v3.7)") 进行传递，将数据移动到共享内存并只将句柄传递到另一个进程。

注意

当一个 [`Tensor`](../tensors.html#torch.Tensor "torch.Tensor") 传递到另一个进程时，[`Tensor`](../tensors.html#torch.Tensor "torch.Tensor") 的数据是共享的。如果 [`torch.Tensor.grad`](../autograd.html#torch.Tensor.grad "torch.Tensor.grad") 不是 `None`, 也会被共享。在一个没有 [`torch.Tensor.grad`](../autograd.html#torch.Tensor.grad "torch.Tensor.grad") 域的 [`Tensor`](../tensors.html#torch.Tensor "torch.Tensor") 被送到其他进程时，一个标准的进程专用的 `.grad` [`Tensor`](../tensors.html#torch.Tensor "torch.Tensor") 会被创建，而它在所有的进程中不会自动被共享，与 [`Tensor`](../tensors.html#torch.Tensor "torch.Tensor") 数据的共享方式不同。

这就允许实现各种训练方法，比如 Hogwild、A3C，或者其他那些需要异步操作的方法。

## 共享 CUDA 张量

进程间共享 CUDA 张量仅支持 Python 3，使用的是 `spawn` 或者 `forkserver` 启动方法。Python 2 中的 [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "(in Python v3.7)") 仅使用 `fork` 来创建子进程，而 CUDA 运行时不支持该方法。

警告

CUDA API 需要分配给其他进程的显存在它们还在使用的情况下一直有效。你需要仔细确保共享的 CUDA 张量若非必须，不会超出使用范围。这对于共享模型参数不会是一个问题，但传递其他类型的数据时需要谨慎。注意该限制并不适用于共享 CPU 内存。

也可以参考：[使用 nn.DataParallel 替代 multiprocessing](cuda.html#cuda-nn-dataparallel-instead)

## 最佳实践和提示

### 避免和处理死锁

当创建一个新进程时，很多情况会发生，最常见的就是后台线程间的死锁。如果任何一个线程有锁的状态或者引入了一个模块，然后调用了`fork`，子进程很有可能处于中断状态，并以另外的方式死锁或者失败。注意即使你没这么做，Python 内建的库也有可能这么做——无需舍近求远，[`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "(in Python v3.7)")即是如此。[`multiprocessing.Queue`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue "(in Python v3.7)") 实际上是一个非常复杂的类，可以创建多个线程用于串行、发送和接收对象，它们也会出现前面提到的问题。如果你发现自己遇到了这种情况，尝试使用 `multiprocessing.queues.SimpleQueue`，它不会使用额外的线程。

我们在尽最大努力为你化繁为简，确保不会发生死锁的情况，但有时也会出现失控的情况。如果你遇到任何暂时无法解决的问题，可以在论坛上求助，我们将会研究是否可以修复。

### 通过 Queue 传递重用缓存

记住每次将一个 [`Tensor`](../tensors.html#torch.Tensor "torch.Tensor") 放进一个 [`multiprocessing.Queue`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue "(in Python v3.7)") 时，它就会被移动到共享内存中。如果它已经被共享，那将不会有操作，否则将会触发一次额外的内存拷贝，而这将会拖慢整个进程。即使你有一个进程池把数据发送到一个进程，并把缓存送回来——这近乎于无操作，在发送下一个批次的数据时避免拷贝。

### 异步多进程训练(如Hogwild）

使用 [`torch.multiprocessing`](../multiprocessing.html#module-torch.multiprocessing "torch.multiprocessing")，可以异步训练一个模型，参数要么一直共享，要么周期性同步。在第一个情况下，我们建议传递整个模型的对象，而对于后一种情况，我们将以仅传递 [`state_dict()`](../nn.html#torch.nn.Module.state_dict "torch.nn.Module.state_dict")。

我们建议使用 [`multiprocessing.Queue`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue "(in Python v3.7)")在进程间传递 PyTorch 对象。当使用`fork`命令时，可以进行诸如继承共享内存中的张量和存储的操作，然而这个操作容易产生问题，应该小心使用，仅建议高级用户使用。Queue，尽管有时不是一个那么优雅的解决方案，但在所有的情况下都可以合理使用。

警告

你应该注意那些不在`if __name__ == '__main__'`中的全局声明。如果使用了一个不是`fork`的系统调用，它们将会在所有子进程中执行。

#### Hogwild

在[示例仓库](https://github.com/pytorch/examples/tree/master/mnist_hogwild)中可以找到一个具体的Hogwild实现，但除了完整的代码结构之外，下面也有一个简化的例子：

```py
import torch.multiprocessing as mp
from model import MyModel

def train(model):
    # Construct data_loader, optimizer, etc.
    for data, labels in data_loader:
        optimizer.zero_grad()
        loss_fn(model(data), labels).backward()
        optimizer.step()  # This will update the shared parameters

if __name__ == '__main__':
    num_processes = 4
    model = MyModel()
    # NOTE: this is required for the ``fork`` method to work
    model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

```

