# CUDA 语义 [¶](#cuda-semantics "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/cuda>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/cuda.html>


[`torch.cuda`](../cuda.html#module-torch.cuda "torch.cuda") 用于设置和运行 CUDA 操作。它会跟踪当前选择的 GPU，并且默认情况下您分配的所有 CUDA 张量都将在该设备上创建。可以使用 [`torch.cuda.device`](../generated/torch.cuda.device.html#torch.cuda.device "torch.cuda.device") 上下文管理器更改所选设备。


 但是，一旦分配了张量，无论选择什么设备，都可以对其进行操作，并且结果将始终放置在与张量相同的设备上。


 默认情况下不允许跨 GPU 操作，但 [`copy_()`](../generated/torch.Tensor.copy_.html#torch.Tensor.copy_ "torch.Tensor.copy_") 除外以及其他具有类似复制功能的方法，例如 [`to()`](../generated/torch.Tensor.to.html#torch.Tensor.to "torch.Tensor.to") 和 [`cuda()` ](../generated/torch.Tensor.cuda.html#torch.Tensor.cuda "torch.Tensor.cuda") 。除非您启用点对点内存访问，否则任何启动操作张量的尝试都会分布在不同的设备上会引发错误。


 下面你可以找到一个展示这一点的小例子：


```
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


## Ampere 设备上的 TensorFloat-32(TF32) [¶](#tensorfloat-32-tf32-on-ampere-devices“此标题的永久链接”)


 从 PyTorch 1.7 开始，有一个名为 allowed_tf32 的新标志。此标志在 PyTorch 1.7 到 PyTorch 1.11 中默认为 True，在 PyTorch 1.12 及更高版本中默认为 False。此标志控制是否允许 PyTorch 使用 TensorFloat32 (TF32) 张量核心，自 Ampere 以来在新的 NVIDIA GPU 上可用，在内部计算 matmul(矩阵)乘法和批量矩阵乘法)和卷积。


 TF32 张量核心旨在通过将输入数据舍入为 10 位尾数，并以 FP32 精度累加结果，从而保持 FP32 动态范围，从而在 torch.float32 张量上实现 matmul 和卷积方面的更好性能。


 matmuls 和卷积是分开控制的，它们相应的标志可以在以下位置访问：


```
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

```


 请注意，除了 matmul 和卷积本身之外，内部使用 matmul 或卷积的函数和 nn 模块也会受到影响。其中包括 nn.Linear 、 nn.Conv* 、cdist、tensordot、仿射网格和网格样本、自适应日志 softmax、GRU 和 LSTM。


 要了解精度和速度，请参阅下面的示例代码：


```
a_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
b_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
ab_full = a_full @ b_full
mean = ab_full.abs().mean()  # 80.7277

a = a_full.float()
b = b_full.float()

# Do matmul at TF32 mode.
torch.backends.cuda.matmul.allow_tf32 = True
ab_tf32 = a @ b  # takes 0.016s on GA100
error = (ab_tf32 - ab_full).abs().max()  # 0.1747
relative_error = error / mean  # 0.0022

# Do matmul with TF32 disabled.
torch.backends.cuda.matmul.allow_tf32 = False
ab_fp32 = a @ b  # takes 0.11s on GA100
error = (ab_fp32 - ab_full).abs().max()  # 0.0031
relative_error = error / mean  # 0.000039

```


 从上面的例子中，我们可以看到，启用 TF32 后，速度快了约 7 倍，相对误差与双精度相比大约大 2 个数量级。如果需要完整的 FP32 精度，用户可以通过以下方式禁用 TF32：


```
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

```


 要在 C++ 中关闭 TF32 标志，您可以执行以下操作


```
at::globalContext().setAllowTF32CuBLAS(false);
at::globalContext().setAllowTF32CuDNN(false);

```


 有关 TF32 的更多信息，请参阅：



* [TensorFloat-32](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32
- precision-format/)
* [CUDA 11](https://devblogs.nvidia.com/cuda-11-features-revealed/)
* [Ampere 架构](https://devblogs.nvidia.com/nvidia-ampere-architecture-in-depth/)


## FP16 GEMM 中的精度降低 [¶](#reduced
- precision-reduction-in-fp16-gemms “此标题的永久链接”)


 fp16 GEMM 可能会通过一些中间降低的精度降低来完成(例如，在 fp16 而不是 fp32)。这些选择性的精度降低可以在某些工作负载(特别是具有大 k 维的工作负载)和 GPU 架构上实现更高的性能，但代价是数值精度和潜在的溢出。


 V100 的一些基准测试数据示例：


```
[--------------------------- bench_gemm_transformer --------------------------]
      [  m ,  k  ,  n  ]    |  allow_fp16_reduc=True  |  allow_fp16_reduc=False
1 threads: --------------------------------------------------------------------
      [4096, 4048, 4096]    |           1634.6        |           1639.8
      [4096, 4056, 4096]    |           1670.8        |           1661.9
      [4096, 4080, 4096]    |           1664.2        |           1658.3
      [4096, 4096, 4096]    |           1639.4        |           1651.0
      [4096, 4104, 4096]    |           1677.4        |           1674.9
      [4096, 4128, 4096]    |           1655.7        |           1646.0
      [4096, 4144, 4096]    |           1796.8        |           2519.6
      [4096, 5096, 4096]    |           2094.6        |           3190.0
      [4096, 5104, 4096]    |           2144.0        |           2663.5
      [4096, 5112, 4096]    |           2149.1        |           2766.9
      [4096, 5120, 4096]    |           2142.8        |           2631.0
      [4096, 9728, 4096]    |           3875.1        |           5779.8
      [4096, 16384, 4096]   |           6182.9        |           9656.5
(times in microseconds).

```


 如果需要完全降低精度，用户可以通过以下方式禁用 fp16 GEMM 中降低的精度：


```
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

```


 要切换 C++ 中降低的精度降低标志，可以这样做


```
at::globalContext().setAllowFP16ReductionCuBLAS(false);

```


## BF16 GEMM 中的精度降低 [¶](#reduced-precision-reduction-in-bf16-gemms "此标题的永久链接")


 BFloat16 GEMM 存在类似的标志(如上所述)。请注意，对于 BF16，此开关默认设置为 True ，如果您观察到工作负载中的数值不稳定，您可能希望将其设置为 False 。


 如果不希望降低精度，用户可以通过以下方式禁用 bf16 GEMM 中的降低精度：


```
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

```


 要切换 C++ 中降低的精度降低标志，可以这样做


```
at::globalContext().setAllowBF16ReductionCuBLAS(true);

```


## 异步执行 [¶](#asynchronous-execution "永久链接到此标题")


 默认情况下，GPU 操作是异步的。当您调用使用 GPU 的函数时，操作会“排队”到特定设备，但不一定要稍后执行。这使我们能够并行执行更多计算，包括 CPU 或其他 GPU 上的操作。


 一般来说，异步计算的效果对调用者来说是不可见的，因为(1)每个设备按照排队的顺序执行操作，(2)PyTorch 在 CPU 和 GPU 之间或两个 GPU 之间复制数据时自动执行必要的同步。因此，如果每个操作都是同步执行的，计算就会继续进行。


 您可以通过设置环境变量“CUDA_LAUNCH_BLOCKING=1”来强制同步计算。当 GPU 上发生错误时，这会很方便。(对于异步执行，只有在操作实际执行之后才会报告此类错误，因此堆栈跟踪不会显示请求的位置。)


 异步计算的结果是没有同步的时间测量不准确。为了获得精确的测量，应该在测量之前调用 [`torch.cuda.synchronize()`](../generated/torch.cuda.synchronize.html#torch.cuda.synchronize "torch.cuda.synchronize")，或者使用 [`torch.cuda.Event`](../generated/torch.cuda.Event.html#torch.cuda.Event "torch.cuda.Event") 记录时间，如下所示：


```
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

# Run some things here

end_event.record()
torch.cuda.synchronize()  # Wait for the events to be recorded!
elapsed_time_ms = start_event.elapsed_time(end_event)

```


 作为例外，有几个函数，例如 [`to()`](../generated/torch.Tensor.to.html#torch.Tensor.to "torch.Tensor.to") 和 [`copy_() `](../generated/torch.Tensor.copy_.html#torch.Tensor.copy_ "torch.Tensor.copy_") 承认一个显式的 `non_blocking` 参数，它允许调用者在不必要时绕过同步。另一个例外是 CUDA 流，如下所述。


### CUDA 流 [¶](#cuda-streams "此标题的永久链接")


 [CUDA 流](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams) 是属于特定设备的线性执行序列。您通常不需要显式创建一个：默认情况下，每个设备都使用自己的“默认”流。


 每个流内的操作按照它们创建的顺序进行序列化，但是来自不同流的操作可以以任何相对顺序同时执行，除非显式同步函数(例如 [`synchronize()`](../generated/torch.cuda.synchronize.html#torch.cuda.synchronize "torch.cuda.synchronize") 或 [`wait_stream()`](../generated/torch.cuda.Stream.html#torch.cuda.Stream.wait_stream "torch.使用 cuda.Stream.wait_stream") )。例如，下面的代码是不正确的：


```
cuda = torch.device('cuda')
s = torch.cuda.Stream()  # Create a new stream.
A = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)
with torch.cuda.stream(s):
    # sum() may start execution before normal_() finishes!
    B = torch.sum(A)

```


 当“当前流”是默认流时，PyTorch 在数据移动时自动执行必要的同步，如上所述。但是，当使用非默认流时，用户有责任确保正确的同步。


### 向后传递的流语义 [¶](#stream-semantics-of-backward-passes "Permalink to this header")


 每个向后 CUDA 操作都在用于其相应前向操作的同一流上运行。如果您的前向传递在不同流上并行运行独立操作，这有助于向后传递利用相同的并行性。


 相对于周围操作的向后调用的流语义与任何其他调用相同。向后传递会插入内部同步，以确保即使向后操作在多个流上运行(如上一段所述)。更具体地说，当调用 [`autograd.backward`](../generated/torch.autograd.backward.html#torch.autograd.backward "torch.autograd.backward") 、 [`autograd.grad`](../generated/torch.autograd.grad.html#torch.autograd.grad "torch.autograd.grad") 或 [ `tensor.backward`](../generated/torch.Tensor.backward.html#torch.Tensor.backward "torch.Tensor.backward") ，并可选择提供 CUDA 张量作为初始梯度 (例如， [`autograd.backward(..., grad_tensors=initial_grads)`](../generated/torch.autograd.backward.html#torch.autograd.backward "torch.autograd.backward") ， [`autograd.grad(..., grad_outputs=initial_grads)`](../generated/torch.autograd.grad.html#torch.autograd.grad "torch.autograd.grad") ，或 [ `tensor.backward(...,gradient=initial_grad)`](../generated/torch.Tensor.backward.html#torch.Tensor.backward "torch.Tensor.backward") ),的行为


1. 可选择填充初始梯度，2.调用向后传递，以及 3．使用渐变


 与任何一组操作具有相同的流语义关系：


```
s = torch.cuda.Stream()

# Safe, grads are used in the same stream context as backward()
with torch.cuda.stream(s):
    loss.backward()
    use grads

# Unsafe
with torch.cuda.stream(s):
    loss.backward()
use grads

# Safe, with synchronization
with torch.cuda.stream(s):
    loss.backward()
torch.cuda.current_stream().wait_stream(s)
use grads

# Safe, populating initial grad and invoking backward are in the same stream context
with torch.cuda.stream(s):
    loss.backward(gradient=torch.ones_like(loss))

# Unsafe, populating initial_grad and invoking backward are in different stream contexts,
# without synchronization
initial_grad = torch.ones_like(loss)
with torch.cuda.stream(s):
    loss.backward(gradient=initial_grad)

# Safe, with synchronization
initial_grad = torch.ones_like(loss)
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    initial_grad.record_stream(s)
    loss.backward(gradient=initial_grad)

```


#### BC 注意：在默认流上使用 grads [¶](#bc-note-using-grads-on-the-default-stream "Permalink to this header")


 在 PyTorch 的早期版本(1.9 及更早版本)中，autograd 引擎始终将默认流与所有向后操作同步，因此以下模式：


```
with torch.cuda.stream(s):
    loss.backward()
use grads

```


 只要“use grads”发生在默认流上，它就是安全的。在目前的 PyTorch 中，该模式不再安全。如果“backward()”和“use grads”位于不同的流上下文中，则必须同步流：


```
with torch.cuda.stream(s):
    loss.backward()
torch.cuda.current_stream().wait_stream(s)
use grads

```


 即使“use grads”位于默认流上。


## 内存管理 [¶](#memory-management "此标题的永久链接")


 PyTorch 使用缓存内存分配器来加速内存分配。这允许快速内存释放而无需设备同步。但是，分配器管理的未使用内存仍将显示为在“nvidia-smi”中使用。您可以使用 [`memory_allocated()`](../generated/torch.cuda.memory_allocated.html#torch.cuda.memory_alulated "torch.cuda.memory_alulated") 和 [`max_memory_allocated()` ](../generated/torch.cuda.max_memory_alulated.html#torch.cuda.max_memory_alulated "torch.cuda.max_memory_alulated") 监视张量占用的内存，并使用 [`memory_reserved()`](../generated /torch.cuda.memory_reserved.html#torch.cuda.memory_reserved "torch.cuda.memory_reserved") 和 [`max_memory_reserved()`](../generated/torch.cuda.max_memory_reserved.html#torch. cuda.max_memory_reserved "torch.cuda.max_memory_reserved") 来监视缓存分配器管理的内存总量。调用 [`empty_cache()`](../generated/torch.cuda.empty_cache.html#torch.cuda.empty_cache "torch.cuda.empty_cache") 释放 PyTorch 中所有**未使用的**缓存内存，以便这些可以被其他 GPU 应用程序使用。但是，张量占用的 GPU 内存不会被释放，因此无法增加 PyTorch 可用的 GPU 内存量。


 为了更好地了解 CUDA 内存随时间的使用情况，[了解 CUDA 内存使用情况](../torch_cuda_memory.html#torch-cuda-memory) 描述了用于捕获和可视化内存使用痕迹的工具。


 对于更高级的用户，我们通过 [`memory_stats()`](../generated/torch.cuda.memory_stats.html#torch.cuda.memory_stats "torch.cuda.memory_stats") 提供更全面的内存基准测试。我们还提供通过 [`memory_snapshot()`](../generated/torch.cuda.memory_snapshot.html#torch.cuda.memory_snapshot "torch.cuda.memory_snapshot" 捕获内存分配器状态的完整快照的功能) ，这可以帮助您了解代码生成的底层分配模式。


### 环境变量 [¶](#environment-variables "永久链接到此标题")


 使用缓存分配器可能会干扰内存检查工具，例如“cuda-memcheck”。要使用“cuda-memcheck”调试内存错误，请在环境中设置“PYTORCH_NO_CUDA_MEMORY_CACHING=1”以禁用缓存。


 缓存分配器的行为可以通过环境变量 `PYTORCH_CUDA_ALLOC_CONF` 进行控制。格式为 `PYTORCH_CUDA_ALLOC_CONF=<option>:<value>,<option2>:<value2>...` 可用选项：


* `backend` 允许选择底层分配器实现。 目前，有效的选项是 `native` ，它使用 PyTorch 的本机实现，以及 `cudaMallocAsync` ，它使用 [CUDA 的内置异步分配器](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/) 。 `cudaMallocAsync` 需要 CUDA 11.4 或更高版本。 默认是 `native` 。 “backend”适用于进程使用的所有设备，并且不能针对每个设备进行指定。
* `max_split_size_mb` 防止本机分配器分割大于此大小（以 MB 为单位）的块。 这可以减少碎片，并且可以允许完成一些边界工作负载而不会耗尽内存。 根据分配模式，性能成本可以从“零”到“大量”不等。 默认值是无限的，即所有块都可以拆分。 [`memory_stats()`](../generated/torch.cuda.memory_stats.html#torch.cuda.memory_stats "torch.cuda.memory_stats") 和 [`memory_summary()`](../generated/torch.cuda.memory_summary.html#torch.cuda.memory_summary "torch.cuda.memory_summary") 方法对于调整很有用。 对于由于“内存不足”而中止并显示大量非活动拆分块的工作负载，应将此选项用作最后的手段。 `max_split_size_mb` 仅对 `backend:native` 有意义。使用 `backend:cudaMallocAsync` 时，`max_split_size_mb` 将被忽略。
* `roundup_power2_divisions` 有助于将请求的分配大小舍入到最接近的 2 次幂除法并更好地利用块。 在本机 CUDACachingAllocator 中，大小以 512 块大小的倍数向上舍入，因此这对于较小的大小非常有效。 然而，这对于大型附近分配来说可能效率低下，因为每个分配都会分配到不同大小的块，并且这些块的重用被最小化。 这可能会创建大量未使用的块并浪费 GPU 内存容量。 此选项允许将分配大小舍入到最接近的 2 次方除法。 例如，如果我们需要对 1200 的大小进行向上舍入，并且除数为 4，则大小 1200 位于 1024 和 2048 之间，如果我们在它们之间进行 4 次除法，则值为 1024、1280、1536 和 1792。 因此，分配大小 1200 将四舍五入为 1280，作为最接近的 2 次方除法上限。 指定一个值以应用于所有分配大小，或指定一个键值对数组来为每个 2 的幂间隔单独设置 2 次幂除法。 例如，要为 256MB 以下的所有分配设置 1 个分区，为 256MB 和 512MB 之间的分配设置 2 个分区，为 512MB 到 1GB 之间的分配设置 4 个分区，为任何更大的分配设置 8 个分区，请将旋钮值设置为：[256:1,512:2,1024:4,>:8]。 `roundup_power2_divisions` 仅对 `backend:native` 有意义。 对于`backend:cudaMallocAsync`，`roundup_power2_divisions` 将被忽略。
* `garbage_collection_threshold` 有助于主动回收未使用的 GPU 内存，以避免触发昂贵的同步和回收所有操作 (release_cached_blocks)，这可能不利于延迟关键的 GPU 应用程序（例如服务器）。 设置此阈值（例如 0.8）后，如果 GPU 内存容量使用量超过阈值（即分配给 GPU 应用程序的总内存的 80%），分配器将开始回收 GPU 内存块。 该算法更喜欢首先释放旧的和未使用的块，以避免释放正在积极重用的块。 阈值应介于大于 0.0 和小于 1.0 之间。 `garbage_collection_threshold` 仅对 `backend:native` 有意义。 对于`backend:cudaMallocAsync`，`garbage_collection_threshold` 将被忽略。

* `expandable_segments` （实验性，默认值： False ）如果设置为 True ，此设置指示分配器创建 CUDA 分配，这些分配稍后可以扩展以更好地处理作业频繁更改分配大小的情况，例如更改批处理大小。 通常，对于大型 (>2MB) 分配，分配器会调用 cudaMalloc 来获取与用户请求大小相同的分配。 将来，如果这些分配的一部分是空闲的，则可以将其重新用于其他请求。 当程序发出许多大小完全相同或大小甚至是该大小的倍数的请求时，这种方法很有效。 许多深度学习模型都遵循这种行为。 然而，一个常见的例外是批量大小从一次迭代到下一次迭代略有变化，例如 在批量推理中。 当程序最初以批量大小 `N` 运行时，它将进行适合该大小的分配。如果将来它以大小 `N - 1` 运行，则现有分配仍然足够大。 但是，如果它以 `N + 1` 的大小运行，那么它将必须进行稍大的新分配。 并非所有张量的大小都相同。 有些可能是 `(N + 1)*A`，其他可能是 `(N + 1)*A*B`，其中 A 和 B 是模型中的一些非批量维度。 由于分配器会在现有分配足够大时重用现有分配，因此某些数量的 `(N + 1)*A` 分配实际上会适合已经存在的 `N*B*A` 段，尽管并不完美。 当模型运行时，它将部分填充所有这些段，在这些段的末尾留下不可用的空闲内存片。 分配器在某些时候需要 cudaMalloc 一个新的 `(N + 1)*A*B` 段。 如果没有足够的内存，那么现在无法恢复现有段末尾的空闲内存片。 对于 50 层以上深度的模型，此模式可能会重复 50 次以上，从而产生许多条子。
 


 可扩展的段允许分配器最初创建一个段，然后在需要更多内存时扩展其大小。它不是为每个段分配一个段，而是尝试使一个段(每个流)根据需要增长。现在，当 `N + 1` 情况运行时，分配将很好地平铺到一大段中，直到填满。然后请求更多内存并将其附加到段的末尾。此过程不会创建尽可能多的不可用内存条，因此更有可能成功找到该内存。


!!! note "笔记"

    [CUDA 内存管理 API](../cuda.html#cuda-memory-management-api) 报告的一些统计信息特定于 `backend:native` ，对于 `backend:cudaMallocAsync` 没有意义。请参阅每个函数的文档字符串了解详细信息。


## 使用 CUDA 的自定义内存分配器 [¶](#using-custom-memory-allocators-for-cuda "此标题的永久链接")


 可以将分配器定义为 C/C++ 中的简单函数并将它们编译为共享库，下面的代码显示了一个仅跟踪所有内存操作的基本分配器。


```
#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <iostream>
// Compile with g++ alloc.cc -o alloc.so -I/usr/local/cuda/include -shared -fPIC
extern "C" {
void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
 void *ptr;
 cudaMalloc(&ptr, size);
 std::cout<<"alloc "<<ptr<<size<<std::endl;
 return ptr;
}

void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
 std::cout<<"free "<<ptr<< " "<<stream<<std::endl;
 cudaFree(ptr);
}
}

```


 这可以通过 [`torch.cuda.memory.CUDAPluggableAllocator`](../generated/torch.cuda.CUDAPluggableAllocator.html#torch.cuda.CUDAPluggableAllocator "torch.cuda.memory.CUDAPluggableAllocator") 在 python 中使用。用户负责提供.so 文件的路径以及与上面指定的签名匹配的分配/释放函数的名称。


```
import torch

# Load the allocator
new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
    'alloc.so', 'my_malloc', 'my_free')
# Swap the current allocator
torch.cuda.memory.change_current_allocator(new_alloc)
# This will allocate memory in the device using the new allocator
b = torch.zeros(10, device='cuda')

```



```
import torch

# Do an initial memory allocator
b = torch.zeros(10, device='cuda')
# Load the allocator
new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
    'alloc.so', 'my_malloc', 'my_free')
# This will error since the current allocator was already instantiated
torch.cuda.memory.change_current_allocator(new_alloc)

```


## cuBLAS 工作空间 [¶](#cublas-workspaces "此标题的永久链接")


 对于 cuBLAS 句柄和 CUDA 流的每个组合，如果该句柄和流组合执行需要工作空间的 cuBLAS 内核，则将分配一个 cuBLAS 工作空间。为了避免重复分配工作空间，除非“torch._C”，否则不会释放这些工作空间。调用 `_cuda_clearCublasWorkspaces()`。每个分配的工作空间大小可以通过环境变量“CUBLAS_WORKSPACE_CONFIG”指定，格式为`:[SIZE]:[COUNT]`。例如，每个分配的默认工作空间大小为`CUBLAS_WORKSPACE_CONFIG=:4096:2:16:8` 指定总大小为 `2 * 4096 + 8 * 16 KiB` 。要强制 cuBLAS 避免使用工作区，请设置 `CUBLAS_WORKSPACE_CONFIG=:0:0` 。


## cuFFT 计划缓存 [¶](#cufft-plan-cache "此标题的永久链接")


 对于每个 CUDA 设备，cuFFT 计划的 LRU 缓存用于加速重复运行的 FFT 方法(例如，[`torch.fft.fft()`](../generated/torch.fft.fft.html#torch.fft.fft "torch.fft.fft") ) 在具有相同配置的相同几何形状的 CUDA 张量上。由于某些 cuFFT 计划可能会分配 GPU 内存，因此这些缓存具有最大容量。


 您可以通过以下API控制和查询当前设备缓存的属性：



* `torch.backends.cuda.cufft_plan_cache.max_size` 给出缓存的容量(在 CUDA 10 及更新版本上默认为 4096，在旧 CUDA 版本上默认为 1023)。设置此值会直接修改容量。
* `torch.backends.cuda.cufft_plan_cache.size` 给出当前驻留在缓存中的计划数量。
* `torch.backends.cuda.cufft_plan_cache.clear()` 清除缓存。


 要控制和查询非默认设备的计划缓存，您可以使用 [`torch.device`](../tensor_attributes.html#torch) 索引 `torch.backends.cuda.cufft_plan_cache` 对象或设备索引，并访问上述属性之一。例如，要设置设备“1”的缓存容量，可以编写`torch.backends.cuda.cufft_plan_cache[1].max_size = 10`。


## 即时编译 [¶](#just-in-time-compilation "永久链接到此标题")


 当在 CUDA 张量上执行时，PyTorch 即时编译一些操作，例如 torch.special.zeta。此编译可能非常耗时(最多几秒钟，具体取决于您的硬件和软件)，并且对于单个运算符来说可能会发生多次，因为许多 PyTorch 运算符实际上从各种内核中进行选择，每个内核都必须编译一次，具体取决于它们的内核输入。此编译每个进程发生一次，或者如果使用内核缓存则仅发生一次。


 默认情况下，如果定义了 XDG_CACHE_HOME，PyTorch 在 $XDG_CACHE_HOME/torch/kernels 中创建内核缓存，如果未定义，则在 $HOME/.cache/torch/kernels 中创建内核缓存(Windows 除外，其中没有内核缓存尚未支持)。缓存行为可以通过两个环境变量直接控制。如果 USE_PYTORCH_KERNEL_CACHE 设置为 0，则不会使用缓存，如果设置了 PYTORCH_KERNEL_CACHE_PATH，则该路径将用作内核缓存，而不是默认位置。


## 最佳实践 [¶](#best-practices "此标题的永久链接")


### 与设备无关的代码 [¶](#device-agnostic-code "固定链接到此标题")


 由于 PyTorch 的结构，您可能需要显式编写与设备无关(CPU 或 GPU)的代码；一个例子可能是创建一个新的张量作为循环神经网络的初始隐藏状态。


 第一步是确定是否应该使用 GPU。一种常见的模式是使用 Python 的 argparse 模块读取用户参数，并有一个可用于禁用 CUDA 的标志，与 [`is_available()`](../generated/torch.cuda.is_available.html#torch.cuda.is_available "torch.cuda.is_available").在下文中，`args.device` 生成一个 [`torch.device`](../tensor_attributes.html#torch.device "torch.device") 对象，可用于将张量移动到 CPU 或 CUDA。


```
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


!!! note "笔记"

    当评估给定环境中 CUDA 的可用性时( [`is_available()`](../generated/torch.cuda.is_available.html#torch.cuda.is_available "torch.cuda.is_available") )，PyTorch 的默认行为是调用 CUDA Runtime API 方法 [cudaGetDeviceCount](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaaffeab31a73cc55f) 。因为此调用依次初始化 CUDA 驱动程序 API(通过 [cuInit](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE_1g0a2f1517e1bd8502c7194c3a8c134bc3) )，如果尚未初始化，则后续的 forks运行 [`is_available()`](../generated/torch.cuda.is_available.html#torch.cuda.is_available "torch.cuda.is_available") 的进程将失败并出现 CUDA 初始化错误。


 在导入执行 [`is_available()`](../generated/torch.cuda.is_available.html#torch.cuda.is_available "torch.cuda.is_available") (或在直接执行之前)以便直接 [`is_available()`](../generated/torch.cuda.is_available.html#torch.cuda.is_available "torch.cuda.is_available") 尝试基于 NVML 的评估( [nvmlDeviceGetCount_v2](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1ga93623b195bff04bbe3490ca33c8a42d) )。如果基于 NVML 的评估成功(即 NVML 发现/初始化没有失败)，则 [`is_available()`](../generated/torch.cuda.is_available.html#torch.cuda.is_available "torch.cuda.is_available") 调用不会毒害后续分叉。


 如果 NVML 发现/初始化失败， [`is_available()`](../generated/torch.cuda.is_available.html#torch.cuda.is_available "torch.cuda.is_available") 将回退到标准 CUDA RuntimeAPI评估和前述的分叉约束将适用。


 请注意，上述基于 NVML 的 CUDA 可用性评估提供的保证比默认的 CUDARuntime API 方法(需要 CUDA 初始化才能成功)更弱。在某些情况下，基于 NVML 的检查可能会成功，但随后的 CUDA 初始化会失败。


 现在我们有了 `args.device` ，我们可以使用它在所需的设备上创建一个张量。


```
x = torch.empty((8, 42), device=args.device)
net = Network().to(device=args.device)

```


 这可以在许多情况下用于生成与设备无关的代码。下面是使用数据加载器的示例：


```
cuda0 = torch.device('cuda:0')  # CUDA GPU 0
for i, x in enumerate(train_loader):
    x = x.to(cuda0)

```


 在系统上使用多个 GPU 时，您可以使用“CUDA_VISIBLE_DEVICES”环境标志来管理哪些 GPU 可用于 PyTorch。如上所述，要手动控制在哪个 GPU 上创建张量，最佳实践是使用 [`torch.cuda.device`](../generated/torch.cuda.device.html#torch.cuda.device "torch.cuda.device") 上下文管理器。


```
print("Outside device is 0")  # On device 0 (default in most scenarios)
with torch.cuda.device(1):
    print("Inside device is 1")  # On device 1
print("Outside device is still 0")  # On device 0

```


 如果您有一个张量并且想在同一设备上创建相同类型的新张量，那么您可以使用 `torch.Tensor.new_*` 方法(请参阅 [`torch.Tensor`](../tensors.html#torch.Tensor "torch.Tensor") )。而前面提到的 `torch.*` 工厂函数( [Creation Ops](../torch.html#tensor-creation-ops) ) 取决于当前 GPU 上下文和您传入的属性参数，“torch.Tensor.new_*”方法保留设备和张量的其他属性。


 这是在创建需要在前向传播过程中内部创建新张量的模块时推荐的做法。


```
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


 如果要创建与另一个张量相同类型和大小的张量，并用 1 或 0 填充它， [`ones_like()`](../generated/torch.ones_like.html#torch.ones_like " torch.ones_like") 或 [`zeros_like()`](../generated/torch.zeros_like.html#torch.zeros_like "torch.zeros_like") 作为方便的辅助函数提供(也保留 [`torch.device张量的 `](../tensor_attributes.html#torch.device "torch.device") 和 [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")。


```
x_cpu = torch.empty(2, 3)
x_gpu = torch.empty(2, 3)

y_cpu = torch.ones_like(x_cpu)
y_gpu = torch.zeros_like(x_gpu)

```


### 使用固定内存缓冲区 [¶](#use-pinned-memory-buffers "永久链接到此标题")


!!! warning "警告"

    这是一个高级提示。如果过度使用固定内存，则在 RAM 不足时可能会导致严重问题，并且您应该意识到固定通常是一项昂贵的操作。


 当主机到 GPU 的副本源自固定(页面锁定)内存时，速度要快得多。 CPU 张量和存储公开了一个 [`pin_memory()`](../generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory "torch.Tensor.pin_memory") 方法，该方法返回对象，数据放入固定区域。


 此外，一旦固定张量或存储，您就可以使用异步 GPU 副本。只需将额外的 `non_blocking=True` 参数传递给 [`to()`](../generated/torch.Tensor.to.html#torch.Tensor.to "torch.Tensor.to") 或 [`cuda()`](../generated/torch.Tensor.cuda.html#torch.Tensor.cuda "torch.Tensor.cuda" ) 称呼。这可用于将数据传输与计算重叠。


 您可以通过将 `pin_memory=True` 传递给 [`DataLoader`](../data.html#torch.utils.data.DataLoader "torch.utils.data.DataLoader") 返回放置在固定内存中的批次构造函数。


### 使用 nn.parallel.DistributedDataParallel 而不是多处理或 nn.DataParallel [¶](#use-nn-parallel-distributeddataparallel-instead-of-multiprocessing-or-nn-dataparallel "永久链接到此标题")


 大多数涉及批量输入和多个 GPU 的用例应默认使用 [`DistributedDataParallel`](../generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel")使用多个 GPU。


 将 CUDA 模型与 [`multiprocessing`](../multiprocessing.html#module-torch.multiprocessing "torch.multiprocessing") 一起使用有一些重要的注意事项；除非注意完全满足数据处理要求，否则您的程序可能会出现不正确或未定义的行为。


 建议使用 [`DistributedDataParallel`](../generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") ，而不是 [`DataParallel` ](../generated/torch.nn.DataParallel.html#torch.nn.DataParallel "torch.nn.DataParallel") 进行多 GPU 训练，即使只有一个节点。


 [`DistributedDataParallel`](../generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") 和 [`DataParallel`](../generated/torch.nn.DataParallel.html#torch.nn.DataParallel "torch.nn.DataParallel") 是: [`DistributedDataParallel`](../generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") 使用多处理，为每个 GPU 创建一个进程，而 [`DataParallel`](../generated/torch.nn.DataParallel.html#torch.nn.DataParallel “torch.nn.DataParallel”)使用多线程。通过使用多处理，每个GPU都有其专用的进程，这避免了Python解释器的GIL带来的性能开销。


 如果您使用 [`DistributedDataParallel`](../generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") ，您可以使用 torch.distributed.launch用于启动程序的实用程序，请参阅[第三方后端](../distributed.html#distributed-launch) 。


## CUDA 图 [¶](#cuda-graphs“此标题的永久链接”)


 CUDA 图是 aCUDA 流及其依赖流执行的工作(主要是内核及其参数)的记录。有关底层 CUDA API 的一般原理和详细信息，请参阅[CUDA 图入门](https://developer.nvidia.com/blog/cuda-graphs/)和 CUDA 的[图形部分](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) C 编程指南。


 PyTorch 支持使用流捕获构建 CUDA 图(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture) ，它将 aCUDA 流置于 *捕获模式
* 。发送到捕获流的 CUDA 工作实际上并不在 GPU 上运行。相反，工作被记录在图表中。


 捕获后，可以“启动”图表以根据需要多次运行 GPU 工作。每次重播都使用相同的参数运行相同的内核。对于指针参数，这意味着使用相同的内存地址。通过在每次重播之前用新数据(例如，来自新批次)填充输入内存，您可以对新数据重新运行相同的工作。


### 为什么使用 CUDA 图？ [¶](#why-cuda-graphs“此标题的永久链接”)


 重放图牺牲了典型急切执行的动态灵活性，以换取**大大减少的CPU开销**。图的参数和内核是固定的，因此图重播会跳过参数设置和内核分派的所有层，包括 Python、C++ 和 CUDA 驱动程序开销。在底层，重放只需调用一次 [cudaGraphLaunch](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1accfe1da0c605a577c22d9751a09597) 即可将整个图的工作提交给 GPU。重放中的内核在 GPU 上的执行速度也稍快，但消除 CPU 开销是主要好处。


 如果您的网络的全部或部分是图形安全的(通常这意味着静态形状和静态控制流，但请参阅其他[约束](#capture-constraints))并且您怀疑其运行时间至少在某种程度上是CPU，您应该尝试CUDA图形-有限的。


### PyTorch API [¶](#pytorch-api“此标题的永久链接”)


!!! warning "警告"

    此 API 处于测试阶段，可能会在未来版本中发生变化。


 PyTorch 通过原始的 [`torch.cuda.CUDAGraph`](../generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph "torch.cuda.CUDAGraph") 类和两个方便的包装器 [`torch.cuda.graph`](../generated/torch.cuda.graph.html#torch.cuda.graph "torch.cuda.graph") 和 [`torch.cuda.make_graphed_callables`](../generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables")。


[`torch.cuda.graph`](../generated/torch.cuda.graph.html#torch.cuda.graph "torch.cuda.graph") 是一个简单、多功能的上下文管理器，可以在其上下文中捕获 CUDA 工作。在捕获之前，通过运行一些急切的迭代来预热要捕获的工作负载。预热必须发生在侧流上。由于图形在每次重播中读取和写入相同的内存地址，因此您必须维护对在捕获期间保存输入和输出数据的张量的长期引用。要在新输入数据上运行图形，请复制将新数据添加到捕获的输入张量，重播图形，然后从捕获的输出张量读取新输出。示例：


```
g = torch.cuda.CUDAGraph()

# Placeholder input used for capture
static_input = torch.empty((5,), device="cuda")

# Warmup before capture
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        static_output = static_input * 2
torch.cuda.current_stream().wait_stream(s)

# Captures the graph
# To allow capture, automatically sets a side stream as the current stream in the context
with torch.cuda.graph(g):
    static_output = static_input * 2

# Fills the graph's input memory with new data to compute on
static_input.copy_(torch.full((5,), 3, device="cuda"))
g.replay()
# static_output holds the results
print(static_output)  # full of 3 * 2 = 6

# Fills the graph's input memory with more data to compute on
static_input.copy_(torch.full((5,), 4, device="cuda"))
g.replay()
print(static_output)  # full of 4 * 2 = 8

```


 请参阅[全网络捕获](#whole-network-capture)、[与 torch.cuda.amp 一起使用](#graphs-with-amp) 和[与多个流一起使用](#multistream-capture) 以了解实际情况和高级模式。


[`make_graphed_callables`](../generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 更复杂。 [`make_graphed_callables`](../generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 接受 Python 函数和 [`torch.nn.Module`](../生成/torch.nn.Module.html#torch.nn.Module“torch.nn.Module”)s。对于每个传递的函数或模块，它都会创建前向传递和反向传递工作的单独图表。请参阅[部分网络捕获](#partial-network-capture)。


#### 约束 [¶](#constraints "此标题的永久链接")


 如果一组操作不违反以下任何约束，则它是“可捕获的”。


 约束适用于 [`torch.cuda.graph`](../generated/torch.cuda.graph.html#torch.cuda.graph "torch.cuda.graph") 上下文中的所有工作以及转发中的所有工作以及传递给 [`torch.cuda.make_graphed_callables()`](../generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 的任何可调用对象的向后传递。


 违反任何这些都可能会导致运行时错误：



* 捕获必须发生在非默认流上。 (如果您使用原始的 [`CUDAGraph.capture_begin`](../generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_begin "torch.cuda.CUDAGraph.capture_begin"，这只是一个问题) 和 [`CUDAGraph.capture_end`](../generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_end "torch.cuda.CUDAGraph.capture_end") 调用。 [`graph`](../generated/torch.cuda.graph.html#torch.cuda.graph "torch.cuda.graph") 和 [`make_graphed_callables()`](../generated/torch.cuda.make_graphed_callables. html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 为您设置一个侧流。)
* 禁止将 CPU 与 GPU 同步的操作(例如 `.item()` 调用)。
* CUDA RNG 操作允许，但必须使用默认生成器。例如，禁止显式构造新的 [`torch.Generator`](../generated/torch.Generator.html#torch.Generator "torch.Generator") 实例并将其作为 `generator` 参数传递给 RNG 函数。


 违反任何这些都可能会导致无提示的数字错误或未定义的行为：



* 在一个进程内，一次只能进行一次捕获。
* 当捕获正在进行时，任何未捕获的 CUDA 工作都不能在此进程中运行(在任何线程上)。
* 不会捕获 CPU 工作。如果捕获的操作包括 CPU 工作，则该工作将在重播期间被忽略。
* 每个重播都会读取和写入相同的(虚拟)内存地址。
* 禁止动态控制流(基于 CPU 或 GPU 数据)。
* 动态形状被禁止。该图假设捕获的操作序列中的每个张量在每次重播中都具有相同的大小和布局。
* 允许在捕获中使用多个流，但有[限制](#multistream-capture)。


#### 非约束 [¶](#non-constraints "此标题的永久链接")



* 捕获后，图表可以在任何流上重播。


### 全网捕获 [¶](#whole-network-capture "永久链接到此标题")


 如果您的整个网络都是可捕获的，您可以捕获并重放整个迭代：


```
N, D_in, H, D_out = 640, 4096, 2048, 1024
model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                            torch.nn.Dropout(p=0.2),
                            torch.nn.Linear(H, D_out),
                            torch.nn.Dropout(p=0.1)).cuda()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Placeholders used for capture
static_input = torch.randn(N, D_in, device='cuda')
static_target = torch.randn(N, D_out, device='cuda')

# warmup
# Uses static_input and static_target here for convenience,
# but in a real setting, because the warmup includes optimizer.step()
# you must use a few batches of real data.
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(3):
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(static_input)
        loss = loss_fn(y_pred, static_target)
        loss.backward()
        optimizer.step()
torch.cuda.current_stream().wait_stream(s)

# capture
g = torch.cuda.CUDAGraph()
# Sets grads to None before capture, so backward() will create
# .grad attributes with allocations from the graph's private pool
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    static_y_pred = model(static_input)
    static_loss = loss_fn(static_y_pred, static_target)
    static_loss.backward()
    optimizer.step()

real_inputs = [torch.rand_like(static_input) for _ in range(10)]
real_targets = [torch.rand_like(static_target) for _ in range(10)]

for data, target in zip(real_inputs, real_targets):
    # Fills the graph's input memory with new data to compute on
    static_input.copy_(data)
    static_target.copy_(target)
    # replay() includes forward, backward, and step.
    # You don't even need to call optimizer.zero_grad() between iterations
    # because the captured backward refills static .grad tensors in place.
    g.replay()
    # Params have been updated. static_y_pred, static_loss, and .grad
    # attributes hold values from computing on this iteration's data.

```


### 部分网络捕获 [¶](#partial-network-capture "此标题的固定链接")


 如果您的某些网络无法安全捕获(例如，由于动态控制流、动态形状、CPU 同步或基本的 CPU 端逻辑)，您可以急切地运行不安全部分并使用 [`torch.cuda.make _graphed_callables()`](../generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 仅绘制捕获安全部分的图形。


 默认情况下， [`make_graphed_callables()`](../generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 返回的可调用对象是自动分级感知的，并且可以在训练循环中用作您传递的函数或 [`nn.Module`](../generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") 的直接替换。


[`make_graphed_callables()`](../generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 内部创建 [`CUDAGraph`](../generated /torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph "torch.cuda.CUDAGraph") 对象，运行预热迭代，并根据需要维护静态输入和输出。因此(与 [`torch.cuda.graph`](../generated/torch.cuda.graph.html#torch.cuda.graph "torch.cuda.graph") 不同)您不需要手动处理这些。


 在下面的示例中，依赖于数据的动态控制流意味着网络无法端到端捕获，但是 [`make_graphed_callables()`](../generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 让我们能够以图形形式捕获和运行图形安全部分，无论：


```
N, D_in, H, D_out = 640, 4096, 2048, 1024

module1 = torch.nn.Linear(D_in, H).cuda()
module2 = torch.nn.Linear(H, D_out).cuda()
module3 = torch.nn.Linear(H, D_out).cuda()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(chain(module1.parameters(),
                                  module2.parameters(),
                                  module3.parameters()),
                            lr=0.1)

# Sample inputs used for capture
# requires_grad state of sample inputs must match
# requires_grad state of real inputs each callable will see.
x = torch.randn(N, D_in, device='cuda')
h = torch.randn(N, H, device='cuda', requires_grad=True)

module1 = torch.cuda.make_graphed_callables(module1, (x,))
module2 = torch.cuda.make_graphed_callables(module2, (h,))
module3 = torch.cuda.make_graphed_callables(module3, (h,))

real_inputs = [torch.rand_like(x) for _ in range(10)]
real_targets = [torch.randn(N, D_out, device="cuda") for _ in range(10)]

for data, target in zip(real_inputs, real_targets):
    optimizer.zero_grad(set_to_none=True)

    tmp = module1(data)  # forward ops run as a graph

    if tmp.sum().item() > 0:
        tmp = module2(tmp)  # forward ops run as a graph
    else:
        tmp = module3(tmp)  # forward ops run as a graph

    loss = loss_fn(tmp, target)
    # module2's or module3's (whichever was chosen) backward ops,
    # as well as module1's backward ops, run as graphs
    loss.backward()
    optimizer.step()

```


### 使用 torch.cuda.amp [¶](#usage-with-torch-cuda-amp "永久链接到此标题")


 对于典型的优化器，[`GradScaler.step`](../amp.html#torch.cuda.amp.GradScaler.step "torch.cuda.amp.GradScaler.step") 将 CPU 与 GPU 同步，这是在捕获。为了避免错误，请使用 [partial-network capture](#partial-network-capture) ，或者(如果前向、损失和后向是捕获安全的)捕获前向、损失和后向，但不捕获优化器步骤：


```
# warmup
# In a real setting, use a few batches of real data.
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            y_pred = model(static_input)
            loss = loss_fn(y_pred, static_target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
torch.cuda.current_stream().wait_stream(s)

# capture
g = torch.cuda.CUDAGraph()
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    with torch.cuda.amp.autocast():
        static_y_pred = model(static_input)
        static_loss = loss_fn(static_y_pred, static_target)
    scaler.scale(static_loss).backward()
    # don't capture scaler.step(optimizer) or scaler.update()

real_inputs = [torch.rand_like(static_input) for _ in range(10)]
real_targets = [torch.rand_like(static_target) for _ in range(10)]

for data, target in zip(real_inputs, real_targets):
    static_input.copy_(data)
    static_target.copy_(target)
    g.replay()
    # Runs scaler.step and scaler.update eagerly
    scaler.step(optimizer)
    scaler.update()

```


### 与多个流一起使用 [¶](#usage-with-multiple-streams "此标题的永久链接")


 捕获模式自动传播到与捕获流同步的任何流。在捕获中，您可以通过向不同流发出调用来公开并行性，但总体流依赖性 DAG 必须在捕获开始后从初始捕获流中分支出来，并在捕获之前重新加入初始流结束：


```
with torch.cuda.graph(g):
    # at context manager entrance, torch.cuda.current_stream()
    # is the initial capturing stream

    # INCORRECT (does not branch out from or rejoin initial stream)
    with torch.cuda.stream(s):
        cuda_work()

    # CORRECT:
    # branches out from initial stream
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        cuda_work()
    # rejoins initial stream before capture ends
    torch.cuda.current_stream().wait_stream(s)

```


!!! note "笔记"

    为了避免高级用户在 nsight 系统或 nvprof 中查看重播时感到困惑：与急切执行不同，该图将 capture 中的重要流 DAG 解释为提示，而不是命令。在重放期间，图表可能会将独立的 opson 重新组织到不同的流或以不同的顺序将它们排入队列(同时尊重原始 DAG 的整体依赖性)。


### 与 DistributedDataParallel 的用法 [¶](#usage-with-distributeddataparallel "此标题的永久链接")


#### NCCL < 2.9.6 [¶](#nccl-2-9-6“此标题的永久链接”)


 早于 2.9.6 的 NCCL 版本不允许捕获集合。您必须使用 [partial-network capture](#partial-network-capture) ，这会推迟所有归约发生在向后的图形部分之外。


 *在*使用以下命令包装网络之前，在可图形网络部分上调用 [`make_graphed_callables()`](../generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables")顺铂。


#### NCCL >= 2.9.6 [¶](#id5“此标题的永久链接”)


 NCCL 版本 2.9.6 或更高版本允许在图中进行集合。捕获[整个向后传递](#whole-network-capture) 的方法是一个可行的选项，但需要三个设置步骤。


1. 禁用DDP的内部异步错误处理：


```
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
torch.distributed.init_process_group(...)

```
2. Before full-backward capture, DDP must be constructed in a side-stream context:
 


```
with torch.cuda.stream(s):
    model = DistributedDataParallel(model)

```
3. Your warmup must run at least 11 DDP-enabled eager iterations before capture.


### 图形内存管理 [¶](#graph-memory-management "永久链接到此标题")


 捕获的图每次重放时都会作用于相同的虚拟地址。如果 PyTorch 释放内存，则稍后的重放可能会遇到非法内存访问。如果 PyTorch 将内存重新分配给新的张量，则重放可能会破坏这些张量看到的值。因此，必须为跨重放的图保留图使用的虚拟地址。 PyTorch 缓存分配器通过检测何时进行捕获并满足来自图形专用内存池的捕获分配来实现此目的。私有池将保持活动状态，直到其 [`CUDAGraph`](../generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph "torch.cuda.CUDAGraph") 对象和在 capturego 期间创建的所有张量超出范围。


 私人池是自动维护的。默认情况下，分配器为每个捕获创建单独的专用池。如果您捕获多个图形，这种保守的方法可确保图形重播不会破坏彼此的值，但有时会不必要地浪费内存。


#### 跨捕获共享内存 [¶](#sharing-memory-across-captures "永久链接到此标题")


 为了节省私有池中存储的内存， [`torch.cuda.graph`](../generated/torch.cuda.graph.html#torch.cuda.graph "torch.cuda.graph") 和 [`torch. cuda.make_graphed_callables()`](../generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 可选择允许不同的捕获共享相同的私有池。这是安全的对于共享私有池的一组图表，如果您知道它们将始终按照捕获的顺序重播，并且永远不会同时重播。


[`torch.cuda.graph`](../generated/torch.cuda.graph.html#torch.cuda.graph "torch.cuda.graph") 的 `pool` 参数是使用特定私有的提示池，可用于跨图共享内存，如下所示：


```
g1 = torch.cuda.CUDAGraph()
g2 = torch.cuda.CUDAGraph()

# (create static inputs for g1 and g2, run warmups of their workloads...)

# Captures g1
with torch.cuda.graph(g1):
    static_out_1 = g1_workload(static_in_1)

# Captures g2, hinting that g2 may share a memory pool with g1
with torch.cuda.graph(g2, pool=g1.pool()):
    static_out_2 = g2_workload(static_in_2)

static_in_1.copy_(real_data_1)
static_in_2.copy_(real_data_2)
g1.replay()
g2.replay()

```


 使用 [`torch.cuda.make_graphed_callables()`](../generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") ，如果你想绘制多个可调用的图形并且您知道它们将始终以相同的顺序运行(并且从不同时)将它们作为元组按照它们在实时工作负载中运行的顺序传递，并且 [`make_graphed_callables()`](../generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 将使用共享私有池捕获它们的图表。


 如果在实时工作负载中，您的可调用对象将以偶尔更改的顺序运行，或者如果它们同时运行，则将它们作为元组传递给 [`make_graphed_callables()`](../generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables")是不允许的。相反，您必须为每个单独调用 [`make_graphed_callables()`](../generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 。