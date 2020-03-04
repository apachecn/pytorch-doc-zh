# CUDA 语义

> 译者：[片刻](https://github.com/jiangzhonglian)
> 校验：[AlexJakin](https://github.com/AlexJakin)

[`torch.cuda`](../cuda.html#module-torch.cuda "torch.cuda") 用于设置和运行 CUDA 操作。它会跟踪当前选定的GPU，并且默认情况下会在该设备上创建您分配的所有 CUDA tensors。可以使用 [`torch.cuda.device`](../cuda.html#torch.cuda.device "torch.cuda.device") 上下文管理器更改所选设备。

但是，一旦分配了 tensor，就可以对其进行操作而不管所选择的设备如何，结果将始终与 tensor 放在同一设备上。

默认情况下不允许跨 GPU 操作，除了 copy_() 具有类似复制功能的其他方法，例如 to() 和 cuda()。除非您启用点对点内存访问，否则任何尝试在不同设备上传播的 tensor 上启动操作都会引发错误。

下面我们用一个小例子来展示:

```py
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

```

## 异步执行

默认情况下，GPU 操作是异步的。当您调用使用 GPU 的函数时，操作将排入特定设备，但不一定要在以后执行。这允许我们并行执行更多计算，包括在 CPU 或其他 GPU 上的操作。

通常，异步计算的效果对于调用者是不可见的，因为 (1) 每个设备按照它们排队的顺序执行操作，以及 (2) PyTorch 在 CPU 和 GPU 之间或两个 GPU 之间复制数据时自动执行必要的同步。因此，计算将如同每个操作同步执行一样进行。

您可以通过设置环境变量强制进行同步计算 `CUDA_LAUNCH_BLOCKING=1`。这在 GPU 上发生错误时非常方便。(使用异步执行时，直到实际执行操作后才会报告此类错误，因此堆栈跟踪不会显示请求的位置。）

异步计算的结果是没有同步的时间测量是不精确的。要获得精确的测量结果，应该在测量之前调用[`torch.cuda.synchronize()`](../cuda.html#torch.cuda.synchronize "torch.cuda.synchronize()")，或者使用[`torch.cuda.Event`](../cuda.html#torch.cuda.Event "torch.cuda.Event")记录时间如下：

```py
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

# 在这里执行一些操作

end_event.record()
torch.cuda.synchronize()  # Wait for the events to be recorded!
elapsed_time_ms = start_event.elapsed_time(end_event)
```

作为一个例外，有几个函数，例如 [`to()`](../tensors.html#torch.Tensor.to "torch.Tensor.to") 和 [`copy_()`](../tensors.html#torch.Tensor.copy_ "torch.Tensor.copy_") 允许一个显式 `non_blocking` 参数，它允许调用者在不需要时绕过同步。另一个例外是 CUDA streams，如下所述。

### CUDA streams

[CUDA stream](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams) 是执行的线性序列属于特定的设备。您通常不需要显式创建一个：默认情况下，每个设备使用自己的 “default” stream。

每个流内的操作按创建顺序进行序列化，但不同流的操作可以按任何相对顺序同时执行，除非使用显式同步功能(如  [`synchronize()`](../cuda.html#torch.cuda.synchronize "torch.cuda.synchronize") 或 [`wait_stream()`](../cuda.html#torch.cuda.Stream.wait_stream "torch.cuda.Stream.wait_stream"))。例如，以下代码不正确:

```py
cuda = torch.device('cuda')
s = torch.cuda.Stream()  # Create a new stream.
A = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)
with torch.cuda.stream(s):
    # sum() may start execution before normal_() finishes!
    B = torch.sum(A)

```

当 “current stream” 是 default stream 时，PyTorch 在数据移动时自动执行必要的同步，如上所述。但是，使用 non-default streams 时，用户有责任确保正确同步。

## 内存管理

PyTorch 使用缓存内存分配器来加速内存分配。这允许在没有设备同步的情况下快速释放内存。但是，分配器管理的未使用内存仍将显示为使用 `nvidia-smi`。您可以使用 [`memory_allocated()`](../cuda.html#torch.cuda.memory_allocated "torch.cuda.memory_allocated") 和 [`max_memory_allocated()`](../cuda.html#torch.cuda.max_memory_allocated "torch.cuda.max_memory_allocated") 监视张量占用的内存，并使用 [`memory_cached()`](../cuda.html#torch.cuda.memory_cached "torch.cuda.memory_cached") 和 [`max_memory_cached()`](../cuda.html#torch.cuda.max_memory_cached "torch.cuda.max_memory_cached") 监视缓存分配器管理的内存。调用 [`empty_cache()`](../cuda.html#torch.cuda.empty_cache "torch.cuda.empty_cache") 可以从 PyTorch 释放所有 **unused** 的缓存内存，以便其他 GPU 应用程序可以使用它们。但是，tensor 占用的 GPU 内存不会被释放，因此无法增加 PyTorch 可用的 GPU 内存量。

## 最佳做法

### 设备无关的代码

由于 PyTorch 的结构，您可能需要显式编写设备无关(CPU或GPU）代码; 一个例子可能是创建一个新的张量作为递归神经网络的初始隐藏状态。

第一步是确定是否应该使用GPU。常见的模式是使用Python的 `argparse` 模块读入用户参数，并有一个可用于禁用 CUDA 的标志，并结合使用 [`is_available()`](../cuda.html#torch.cuda.is_available "torch.cuda.is_available")。在下文中，`args.device` 结果 `torch.device` 可以用于将 tensor 移动到 CPU 或 CUDA 的对象。

```py
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

```

现在 `args.device` 我们可以使用它在所需的设备上创建 Tensor。

```py
x = torch.empty((8, 42), device=args.device)
net = Network().to(device=args.device)

```

这可以在许多情况下用于产生与设备无关的代码。以下是使用 dataloader 时的示例:

```py
cuda0 = torch.device('cuda:0')  # CUDA GPU 0
for i, x in enumerate(train_loader):
    x = x.to(cuda0)

```

在系统上使用多个 GPU 时，可以使用 `CUDA_VISIBLE_DEVICES` 环境标志来管理 PyTorch 可用的 GPU。如上所述，要手动控制创建张量的GPU，最佳做法是使用 [`torch.cuda.device`](../cuda.html#torch.cuda.device "torch.cuda.device") 上下文管理器。

```py
print("Outside device is 0")  # On device 0 (default in most scenarios)
with torch.cuda.device(1):
    print("Inside device is 1")  # On device 1
print("Outside device is still 0")  # On device 0

```

如果你有一个 tensor 并且想在同一个设备上创建一个相同类型的新 tensor，那么你可以使用一个 `torch.Tensor.new_*` 方法(参见参考资料 [`torch.Tensor`](../tensors.html#torch.Tensor "torch.Tensor")）。虽然前面提到的 `torch.*` factory 函数 ([Creation Ops](../torch.html#tensor-creation-ops))依赖于当前 GPU 上下文和您传入的属性参数，但 `torch.Tensor.new_*` 方法会保留设备和 tensor 的其他属性。

在创建在前向传递期间需要在内部创建新 tensor 的模块时，这是建议的做法。

```py
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

```

如果你想创建一个与另一个 tensor 相同类型和大小的 tensor，并用一个或零填充它，[`ones_like()`](../torch.html#torch.ones_like "torch.ones_like") 或 [`zeros_like()`](../torch.html#torch.zeros_like "torch.zeros_like") 作为方便的辅助函数(也保留 Tensor 的 `torch.device` 和 `torch.dtype`  )提供。

```py
x_cpu = torch.empty(2, 3)
x_gpu = torch.empty(2, 3)

y_cpu = torch.ones_like(x_cpu)
y_gpu = torch.zeros_like(x_gpu)

```

### 使用固定内存缓冲区

当源自固定(页面锁定）内存时，主机到 GPU 副本的速度要快得多。CPU tensor 和存储器公开一种 [`pin_memory()`](../tensors.html#torch.Tensor.pin_memory "torch.Tensor.pin_memory") 方法，该方法返回对象的副本，数据放在固定区域中。

此外，一旦您固定张量或存储，您就可以使用异步GPU副本。只需将一个额外的 `non_blocking=True` 参数传递给一个 [`cuda()`](../tensors.html#torch.Tensor.cuda "torch.Tensor.cuda") 调用。这可以用于通过计算重叠数据传输。

您可以  [`DataLoader`](../data.html#torch.utils.data.DataLoader "torch.utils.data.DataLoader") 通过传递 `pin_memory=True` 给构造函数使返回批处理放置在固定内存中。

## 使用 nn.DataParallel 而不是并行处理

涉及批量输入和多个 GPU 的大多数用例应默认 [`DataParallel`](../nn.html#torch.nn.DataParallel "torch.nn.DataParallel") 使用多个GPU。即使使用GIL，单个 Python 进程也可以使多个 GPU 饱和。

从版本 0.1.9 开始，可能无法充分利用大量 GPUs (8+)。但是，这是一个正在积极开发的已知问题。一如既往，测试您的用例。

使用带有 [`multiprocessing`](../multiprocessing.html#module-torch.multiprocessing "torch.multiprocessing") 的 CUDA 模型有一些重要的注意事项 ; 除非注意完全满足数据处理要求，否则您的程序可能会有不正确或未定义的行为。
