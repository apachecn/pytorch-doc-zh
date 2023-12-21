# 多重处理最佳实践 [¶](#multiprocessing-best-practices "永久链接到此标题")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/multiprocessing>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/multiprocessing.html>


[`torch.multiprocessing`](../multiprocessing.html#module-torch.multiprocessing "torch.multiprocessing") 是 Python [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "(在 Python v3.12 中)") 模块。它支持完全相同的操作，但对其进行了扩展，以便通过 [`multiprocessing.Queue`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue "(in Python v3.12)") 会将其数据移至共享内存中，并且仅将句柄发送到另一个进程。


!!! note "笔记"

    当 [`Tensor`](../tensors.html#torch.Tensor "torch.Tensor") 被发送到另一个进程时， [`Tensor`](../tensors.html#torch.Tensor "torch.Tensor")数据是共享的。如果 [`torch.Tensor.grad`](../generated/torch.Tensor.grad.html#torch.Tensor.grad "torch.Tensor.grad") 不是 `None` ，它也会被共享。在 [`Tensor`](../tensors.html#torch.Tensor "torch.Tensor") 没有 [`torch.Tensor.grad`](../generated/torch.Tensor.grad.html#torch.Tensor.grad "torch.Tensor.grad") 字段被发送到其他进程，它创建一个标准进程特定的 `.grad`[`Tensor`](../tensors.html#torch.Tensor "torch.Tensor" ) 不会在所有进程之间自动共享，这与 [`Tensor`](../tensors.html#torch.Tensor "torch.Tensor") 的数据共享方式不同。


 这允许实现各种训练方法，例如 Hogwild、A3C 或任何其他需要异步操作的方法。


## 多处理中的 CUDA [¶](#cuda-in-multiprocessing "此标题的永久链接")


 CUDA运行时不支持`fork`启动方法；要在子进程中使用 CUDA，需要使用“spawn”或“forkserver”启动方法。


!!! note "笔记"

    可以通过使用`multiprocessing.get_context(...)`创建上下文或直接使用`multiprocessing.set_start_method(...)`来设置启动方法。


 与 CPU 张量不同，发送进程需要保留原始张量，而接收进程则保留张量的副本。它是在后台实现的，但要求用户遵循最佳实践才能使程序正确运行。例如，只要消费者进程引用张量，发送进程就必须保持活动状态，并且如果消费者进程通过致命信号异常退出，则重新计数无法拯救您。请参阅[本节](../multiprocessing.html#multiprocessing-cuda-sharing-details)。


 另请参阅：[使用 nn.parallel.DistributedDataParallel 而不是多处理或 nn.DataParallel](cuda.html#cuda-nn-ddp-instead)


## 最佳实践和技巧 [¶](#best-practices-and-tips "此标题的永久链接")


### 避免和解决死锁 [¶](#avoiding-and-fighting-deadlocks "永久链接到此标题")


 当新进程产生时，有很多事情可能会出错，其中最常见的死锁原因是后台线程。如果有任何线程持有锁或导入模块，并且调用了“fork”，则子进程很可能会处于损坏状态，并且会死锁或以不同的方式失败。请注意，即使您不这样做，Python 内置库也会这样做 
- 无需比 [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing "(in Python v3.12)") 。 [`multiprocessing.Queue`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue "(in Python v3.12)") 实际上是一个非常复杂的类，它产生多个使用的线程序列化、发送和接收对象，它们也可能导致上述问题。如果您发现自己处于这种情况，请尝试使用“SimpleQueue”，它不使用任何额外的线程。


 我们正在尽力让您轻松并确保不会发生这些僵局，但有些事情是我们无法控制的。如果您遇到暂时无法解决的任何问题，请尝试在论坛上联系，我们将看看是否可以解决该问题。


### 重用通过队列传递的缓冲区 [¶](#reuse-buffers-passed-through-a-queue "Permalink to this header")


 请记住，每次将 [`Tensor`](../tensors.html#torch.Tensor "torch.Tensor") 放入 [`multiprocessing.Queue`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue "(in Python v3.12)") ，它必须被移动到共享内存中。如果它已经共享，那么它是一个无操作，否则它将产生一个额外的内存副本可以减慢整个过程。即使您有一组进程向单个进程发送数据，也可以让它将缓冲区发回 - 这几乎是免费的，并且可以让您在发送下一批时避免复制。


### 异步多进程训练(例如 Hogwild)[¶](#asynchronous-multiprocess-training-e-g-hogwild "永久链接到此标题")


 使用 [`torch.multiprocessing`](../multiprocessing.html#module-torch.multiprocessing "torch.multiprocessing") ，可以异步训练模型，参数可以一直共享，也可以定期同步。在第一种情况下，我们建议发送整个模型对象，而在后者中，我们建议仅发送 [`state_dict()`](../generated/torch.nn.Module.html#torch.nn.Module.state_dict "torch.nn.Module.state_dict").


 我们建议使用 [`multiprocessing.Queue`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue "(in Python v3.12)") 在进程之间传递各种 PyTorch 对象。这是可能的，例如当使用“fork”启动方法时，继承共享内存中已有的张量和存储，但是它很容易出现错误，应该小心使用，并且只能由高级用户使用。队列虽然有时是一个不太优雅的解决方案，但在所有情况下都能正常工作。


!!! warning "警告"

    您应该小心使用不受 `if __name__ == '__main__'` 保护的全局语句。如果使用与`fork`不同的启动方法，它们将在所有子进程中执行。


#### Hogwild [¶](#hogwild "此标题的永久链接")


 具体的Hogwild实现可以在[示例存储库](https://github.com/pytorch/examples/tree/master/mnist_hogwild)中找到，但为了展示代码的整体结构，下面还有一个最小的示例：


```
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


## 多处理中的 CPU [¶](#cpu-in-multiprocessing "此标题的永久链接")


 不恰当的多处理会导致CPU超额认购，导致不同进程竞争CPU资源，导致效率低下。


 本教程将解释什么是 CPU 超额订阅以及如何避免它。


### CPU 超额订阅 [¶](#cpu-oversubscription "此标题的永久链接")


 CPU 超额认购是一个技术术语，指的是分配给系统的 vCPU 总数超过硬件上可用 vCPU 总数的情况。


 这导致CPU资源的严重争用。在这种情况下，进程之间会频繁切换，这会增加进程切换开销并降低系统整体效率。


 请参阅 CPU 超额订阅以及 [examplerepository](https://github.com/pytorch/examples/tree/main/mnist_hogwild) 中 Hogwildimplementation 中的代码示例。


 当使用 4 个进程在 CPU 上使用以下命令运行训练示例时：


```
python main.py --num-processes 4

```


 假设机器上有N个vCPU，执行上述命令将生成4个子进程。每个子进程都会为自己分配 NvCPU，因此需要 4*N 个 vCPU。但是，该机器只有 N 个可用 vCPU。因此，不同的进程会争夺资源，导致频繁的进程切换。


 以下观察结果表明存在 CPU 超额订阅：


1. CPU利用率高：通过使用`htop`命令，您可以观察到CPU利用率一直很高，经常达到或超过其最大容量。这表明对CPU资源的需求超过了可用的物理核心，导致进程之间对CPU时间的争夺和竞争。2．上下文切换频繁，系统效率低：在CPU超额使用的场景下，进程之间会竞争CPU时间，操作系统需要在不同进程之间快速切换，以公平地分配资源。这种频繁的上下文切换增加了开销并降低了整体系统效率。


### 避免 CPU 过度订阅 [¶](#avoid-cpu-oversubscription "永久链接到此标题")


 避免CPU超额使用的一个好方法是适当的资源分配。确保同时运行的进程或线程的数量不超过可用的CPU资源。


 在这种情况下，解决方案是在子进程中指定适当的线程数。这可以通过使用子进程中的“torch.set_num_threads(int)”函数设置每个进程的线程数来实现。


 假设机器上有 N 个 vCPU，并且将生成 M 个进程，则每个进程使用的最大 num_threads 值为 “floor(N/M)” 。为了避免 mnist_hogwildexample 中的 CPU 超额订阅，需要对 [examplerepository](https://github.com/pytorch/examples/tree/main/mnist_hogwild) 中的文件 `train.py` 进行以下更改。


```
def train(rank, args, model, device, dataset, dataloader_kwargs):
    torch.manual_seed(args.seed + rank)

    #### define the num threads used in current sub-processes
    torch.set_num_threads(floor(N/M))

    train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)

```


 使用 `torch.set_num_threads(floor(N/M))` 为每个进程设置“num_thread”。其中，您将 N 替换为可用 vCPU 的数量，将 M 替换为所选进程的数量。适当的“num_thread”值将根据手头的具体任务而变化。但是，作为一般准则，“num_thread”的最大值应为“floor(N/M)”，以避免 CPU 过度订阅。在 [mnist_hogwild](https://github.com/pytorch/example/tree/main/mnist_hogwild) 训练示例，避免 CPU 超额订阅后，可以实现 30 倍的性能提升。