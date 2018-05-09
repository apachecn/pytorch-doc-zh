.. _cuda-semantics:

CUDA 语义
==============

:mod:`torch.cuda` 被用于设置和运行 CUDA 操作. 它会记录当前选择的 GPU, 并且分配的所有 CUDA 张量将默认在上面创建. 可以使用 :any:`torch.cuda.device` 上下文管理器更改所选设备.

但是, 一旦张量被分配, 您可以直接对其进行操作, 而不需要考虑已选择的设备, 结果将始终放在与张量相关的设备上.


默认情况下, 不支持跨 GPU 操作, 有一些例外情况例如
:meth:`~torch.Tensor.copy_` 或其他方法有类似复制的功能如
 :meth:`~torch.Tensor.to` 和 :meth:`~torch.Tensor.cuda`.
除非启用对等的内存访问, 否则尝试任何夸设备的启动的操作将会导致错误.

下面我们用一个小例子来展示::

    cuda = torch.device('cuda')     # Default CUDA device
    cuda0 = torch.device('cuda:0')
    cuda2 = torch.device('cuda:2')  # GPU 2 (these are 0-indexed)

    x = torch.tensor([1., 2.], device=cuda0)
    # x.device is device(type='cuda', index=0)
    y = torch.tensor([1., 2.]).cuda()
    # y.device is device(type='cuda', index=0)

    with torch.cuda.device(1):
        # allocates a tensor on GPU 1
        a = torch.tensor([1., 2.], device=cuda)

        # transfers a tensor from CPU to GPU 1
        b = torch.tensor([1., 2.]).cuda()
        # a.device and b.device are device(type='cuda', index=1)

        # You can also use ``Tensor.to`` to transfer a tensor:
        b2 = torch.tensor([1., 2.]).to(device=cuda)
        # b.device and b2.device are device(type='cuda', index=1)

        c = a + b
        # c.device is device(type='cuda', index=1)

        z = x + y
        # z.device is device(type='cuda', index=0)

        # even within a context, you can specify the device
        # (or give a GPU index to the .cuda call)
        d = torch.randn(2, device=cuda2)
        e = torch.randn(2).to(cuda2)
        f = torch.randn(2).cuda(cuda2)
        # d.device, e.device, and f.device are all device(type='cuda', index=2)

异步执行
----------------------

默认情况下，GPU操作是异步的. 当你调用一个使用GPU的函数时，这些操作会 *排队* 到特定的设备,
但不一定会在以后执行.  这允许我们并行执行更多的计算，包括CPU或其他GPU上的操作。

一般情况下，异步计算的效果对调用者是不可见的，因为（1）每个设备按照它们排队的顺序执行操作，
（2）在CPU和GPU之间或两个GPU之间复制数据时，PyTorch自动执行必要的同步。 
因此，计算将按每个操作同步执行的方式进行。

您可以通过设置环境变量来强制同步计算
`CUDA_LAUNCH_BLOCKING=1`.  当GPU发生错误时，这可能非常方便
（使用异步执行时，只有在实际执行操作之后才会报告此类错误，因此堆栈跟踪不会显示请求的位置。）

一些例外的功能如 :meth:`~torch.Tensor.copy_` 允许
一个特定的 :attr:`async` 参数,当不必要时它让caller绕过同步。 
下面将解释CUDA流的另一个例外。

CUDA 流
^^^^^^^^^^^^

 `CUDA stream`_ 是属于特定设备的线性执行序列。 
 您通常不需要创建: 默认情况下，每个设备都使用自己的“默认”流.

每个流内部的操作按照它们创建的顺序进行序列化，但是来自不同流的操作可以以任何相对顺序并发执行，
除非显式同步功能 (如:meth:`~torch.cuda.synchronize` 或 :meth:`~torch.cuda.Stream.wait_stream`) 
被使用.  例如下列代码是不正确的::

    cuda = torch.device('cuda')
    s = torch.cuda.stream()  # Create a new stream.
    A = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)
    with torch.cuda.stream(s):
        # sum() may start execution before normal_() finishes!
        B = torch.sum(A)

当“当前流”是默认流时，PyTorch会在数据移动时自动执行必要的同步, 如上面解释的.
但是，使用非默认流时，用户有责任确保正确的同步.

.. _CUDA stream: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams

.. _cuda-memory-management:

Memory management
-----------------

内存管理
-----------------

PyTorch 使用缓存内存分配器来加速内存分配. 这允许在没有设备同步的情况下快速释放内存. 
但是, 由分配器管理的未使用的内存仍将显示为在 `nvidia-smi` 中使用.你可以
调用 :meth:`~torch.cuda.memory_allocated` 和 :meth:`~torch.cuda.max_memory_allocated` 
监视内存占用， 以及使用 :meth:`~torch.cuda.memory_cached` 和
 :meth:`~torch.cuda.max_memory_cached` 监视由缓存分配器管理的内存. 调用 :meth:`~torch.cuda.empty_cache` 
 可以从PyTorch释放所有 **未使用的** 缓存内存，以便其他GPU应用程序可以使用这些内存。

最佳实践
--------------

设备无关代码
^^^^^^^^^^^^^^^^^^^^

由于 PyTorch 的架构, 你可能需要明确写入设备无关 (CPU 或 GPU) 代码; 举个例子, 创建一个新的张量作为循环神经网络的初始隐藏状态.

第一步先确定是否使用 GPU. 一个常见的方式是使用 Python 的 ``argparse`` 模块来读入用户参数, 
并且有一个可以用来禁用 CUDA、能与 :meth:`~torch.cuda.is_available` 结合使用的标志. 
在下面的例子中, ``args.device`` 会产生一个 :class:`torch.device` 对象可以将Tensor移植CPU or CUDA

::

    import argparse
    import torch

    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    args = parser.parse_args()
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

现在我们有 ``args.device``, 我们可以使用它在所需的设备上创建一个
Tensor.

::

    x = torch.empty((8, 42), device=args.device)
    net = Network().to(device=args.device)

这可以在许多情况下用于生成设备不可知代码。 以下是使用dataloader时的示例:

::

    cuda0 = torch.device('cuda:0')  # CUDA GPU 0
    for i, x in enumerate(train_loader):
        x = x.to(cuda0)

在系统上使用多个 GPU 时, 您可以使用 ``CUDA_VISIBLE_DEVICES`` 环境标志来管理哪些 GPU 可用于 PyTorch.
如上所述, 要手动控制在哪个 GPU 上创建张量, 最好的方法是使用 :any:`torch.cuda.device` 上下文管理器. 

::

    print("Outside device is 0")  # On device 0 (default in most scenarios)
    with torch.cuda.device(1):
        print("Inside device is 1")  # On device 1
    print("Outside device is still 0")  # On device 0

如果您有一个Tensor, 并且想在同一个设备上创建一个相同类型的Tensor, 那么您可以使用 ``torch.Tensor.new_*`` 方法
(见 :class:`torch.Tensor`).
虽然前面提到 ``torch.*`` 工厂功能
(:ref:`tensor-creation-ops`) 取决于你传入的当前GPU上下文和属性参数, ``torch.Tensor.new_*`` 方法保留了Tensor的设备和其他属性.

当创建在向前传递期间需要在内部创建新的张量/变量的模块时, 建议使用这种做法
这是建立模块时推荐的做法，在正向传播期间需要在内部创建新的Tensor.

::

    cuda = torch.device('cuda')
    x_cpu = torch.empty(2)
    x_gpu = torch.empty(2, device=cuda)
    x_cpu_long = torch.empty(2, dtype=torch.int64)

    y_cpu = x_cpu.new_full([3, 2], fill_value=0.3)
    print(y_cpu)

        tensor([[ 0.3000,  0.3000],
                [ 0.3000,  0.3000],
                [ 0.3000,  0.3000]])

    y_gpu = x_gpu.new_full([3, 2], fill_value=-5)
    print(y_gpu)

        tensor([[-5.0000, -5.0000],
                [-5.0000, -5.0000],
                [-5.0000, -5.0000]], device='cuda:0')

    y_cpu_long = x_cpu_long.new_tensor([[1, 2, 3]])
    print(y_cpu_long)

        tensor([[ 1,  2,  3]])

如果你想创建一个与另一个张量有着相同类型和大小、并用 1 或 0 填充的张量, :meth:`~torch.ones_like` 或 :meth:`~torch.zeros_like` 可提供方便的辅助功能 (这也保存一个Tensor的
 :class:`torch.device` 和 :class:`torch.dtype` ).

::

    x_cpu = torch.empty(2, 3)
    x_gpu = torch.empty(2, 3)

    y_cpu = torch.ones_like(x_cpu)
    y_gpu = torch.zeros_like(x_gpu)


使用固定的内存缓冲区
^^^^^^^^^^^^^^^^^^^^

.. warning:
    这是一个高级提示. 如果您将要在低 RAM 上运行, 过度使用固定内存可能会导致严重的问题, 并且您应该意识到固定是一个代价很高的操作.

当副本来自固定 (页锁) 内存时, 主机到 GPU 的复制速度要快很多. CPU 张量和存储开放了一个 :meth:`~torch.Tensor.pin_memory` 方法, 它返回该对象的副本, 而它的数据放在固定区域中.

另外, 一旦固定了张量或存储, 就可以使用异步的 GPU 副本. 只需传递一个额外的 ``non_blocking=True`` 参数给 :meth:`~torch.Tensor.cuda` 调用. 这可以用于重叠数据传输与计算.

通过将 ``pin_memory=True`` 传递给其构造函数, 可以使 :class:`~torch.utils.data.DataLoader` 将 batch 返回到固定内存中. 

.. _cuda-nn-dataparallel-instead:

使用 nn.DataParallel 替代 multiprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

大多数涉及批量输入和多个 GPU 的情况应默认使用 :class:`~torch.nn.DataParallel` 来使用多个 GPU. 尽管有 GIL 的存在, 单个 Python 进程也可能使多个 GPU 饱和.

从 0.1.9 版本开始, 大量的 GPU (8+) 可能未被充分利用. 然而, 这是一个已知的问题, 也正在积极开发中. 和往常一样, 测试您的用例吧.

调用 :mod:`~torch.multiprocessing` 使用 CUDA 模型存在显著的注意事项; 除非您足够谨慎以满足数据处理需求, 否则您的程序很可能会出现错误或未定义的行为.
