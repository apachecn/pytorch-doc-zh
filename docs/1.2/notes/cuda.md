# CUDA语义

[ `torch.cuda`](../cuda.html#module-torch.cuda
"torch.cuda")用于设置和运行CUDA操作。它跟踪当前选择的GPU，而你分配所有CUDA张量将默认在该设备上创建。所选择的设备可以与[ `
torch.cuda.device`](../cuda.html#torch.cuda.device
"torch.cuda.device")上下文管理器被改变。

然而，一旦张量分配，你可以在上面做的操作，不论所选择的设备的，其结果将始终放置在相同的装置上张。

横GPU操作默认不允许，以[ `copy_除外（） `](../tensors.html#torch.Tensor.copy_
"torch.Tensor.copy_")等方法与复制样的功能，如[ `至（） `](../tensors.html#torch.Tensor.to
"torch.Tensor.to")和[ `CUDA（） `](../tensors.html#torch.Tensor.cuda
"torch.Tensor.cuda")。除非你能对等网络存储器存取，任何企图发动对分布在不同的设备上会产生一个错误张量欢声笑语。

下面你可以找到一个小例子展示了这一点：

    
    
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
    
        # You can also use ``Tensor.to``to transfer a tensor:
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
    

## 异步执行

默认情况下，GPU的操作都是异步的。当你调用一个使用GPU功能，操作排队的 __
到特定的设备，但直到后来不一定执行。这使我们能够并行执行更多的计算，包括CPU或其它GPU的操作。

一般情况下，异步计算的效果是不可见的呼叫者，因为（1）的每个设备在它们被排队的顺序来执行操作，和（2）PyTorch复制CPU和GPU之间或两个GPU之间的数据时自动执行必要的同步。因此，计算将继续进行，如果每一个操作同步执行。

可以通过设置环境变量 CUDA_LAUNCH_BLOCKING强制同步计算= 1 。当在GPU上发生错误，这是很方便的。
（随着异步执行，不报告这样的错误，直到之后实际执行的操作，因此堆栈跟踪不显示它被请求）。

作为例外，有几个功能，如[ `至（） `](../tensors.html#torch.Tensor.to "torch.Tensor.to")和[ `
copy_（） `](../tensors.html#torch.Tensor.copy_ "torch.Tensor.copy_")承认显式`
non_blocking`参数，它允许呼叫方旁路同步时，它是不必要的。另一个例外是CUDA流，解释如下。

### CUDA流

A [ CUDA流](http://docs.nvidia.com/cuda/cuda-c-programming-
guide/index.html#streams)是执行的线性序列属于特定的设备。你通常不需要明确创建一个：在默认情况下，每个设备使用自己的“默认”流。

每个流内的操作被序列在它们被创建的顺序，但来自不同流的操作可以以任何相对顺序同时执行，除非明确的同步功能（如[ `同步（） `
](../cuda.html#torch.cuda.synchronize "torch.cuda.synchronize")或[ `
wait_stream（） `](../cuda.html#torch.cuda.Stream.wait_stream
"torch.cuda.Stream.wait_stream")）被使用。例如，下面的代码是不正确：

    
    
    cuda = torch.device('cuda')
    s = torch.cuda.Stream()  # Create a new stream.
    A = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)
    with torch.cuda.stream(s):
        # sum() may start execution before normal_() finishes!
        B = torch.sum(A)
    

当“当前流”是默认流，PyTorch自动执行必要的同步数据时，到处移动，如上所述。然而，使用非默认流时，它是用户的责任，以确保正确的同步。

## 存储器管理

PyTorch使用缓存内存分配器，以加快内存分配。这允许快速内存释放不同步设备。然而，由分配器所管理的未使用的存储器仍然会显示为如果在`使用NVIDIA-
SMI`。您可以使用[ `memory_allocated（） `
](../cuda.html#torch.cuda.memory_allocated "torch.cuda.memory_allocated")和[ `
max_memory_allocated（） `](../cuda.html#torch.cuda.max_memory_allocated
"torch.cuda.max_memory_allocated")监视内存占用的由张量，并使用[ `memory_cached（） `
](../cuda.html#torch.cuda.memory_cached "torch.cuda.memory_cached")和[ `
max_memory_cached（） `](../cuda.html#torch.cuda.max_memory_cached
"torch.cuda.max_memory_cached")监视内存缓存分配器管理。主叫[ `empty_cache（） `
](../cuda.html#torch.cuda.empty_cache "torch.cuda.empty_cache")释放所有
**从PyTorch未使用**
高速缓存的存储器，使得那些可由其他GPU应用中。然而，由张量占用GPU内存不会被释放，因此它不能增加GPU的内存可用于PyTorch量。

## CUFFT计划缓存

对于每个CUDA设备，CUFFT的LRU高速缓存预案来加快重复运行FFT方法（例如，[ `torch.fft（） `
](../torch.html#torch.fft
"torch.fft")）上的CUDA张量与相同的结构相同的几何形状。由于一些CUFFT计划可能分配GPU内存，这些缓存有一个最大容量。

您可以控制和查询与以下API当前设备的高速缓存的性能：

  * `torch.backends.cuda.cufft_plan_cache.max_size`给出缓存的容量（默认为4096上CUDA 10和更新，和1023对旧CUDA版本）。设置这个值直接修改的能力。

  * `torch.backends.cuda.cufft_plan_cache.size`给出的当前驻留在缓存中的计划数。

  * `torch.backends.cuda.cufft_plan_cache.clear（） `清除缓存。

为了控制和非默认装置的查询计划的高速缓存，则可以用索引任一个`torch.device的`
torch.backends.cuda.cufft_plan_cache`对象
`对象或设备索引，并获得上述的属性之一。例如，要设置高速缓冲存储器的容量为设备`1`，可以写`
torch.backends.cuda.cufft_plan_cache [1] .max_size  =  10`。

## 最佳实践

### 设备无关的代码

由于PyTorch的结构，则可能需要明确写入设备无关的（CPU或GPU）代码;实例，可创建新的张量作为回归神经网络的初始隐蔽状态。

第一步是确定GPU是否应使用与否。一个常见的模式是使用Python的`argparse`模块在用户参数读取，并且具有可被用于禁用CUDA的标志，结合[
`is_available（） `](../cuda.html#torch.cuda.is_available
"torch.cuda.is_available")。在下文中，`args.device`结果在可用于张量移动到CPU或CUDA一个`
torch.device`对象。

    
    
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
    

现在，我们有`args.device`，我们可以用它到所需的设备上创建一个张量。

    
    
    x = torch.empty((8, 42), device=args.device)
    net = Network().to(device=args.device)
    

这可以在许多情况下，以产生设备无关代码的使用。下面是使用的DataLoader时的例子：

    
    
    cuda0 = torch.device('cuda:0')  # CUDA GPU 0
    for i, x in enumerate(train_loader):
        x = x.to(cuda0)
    

当使用多GPU的系统上工作，你可以使用`CUDA_VISIBLE_DEVICES
`环境标志来管理其GPU的可供PyTorch。如上所述，手动控制其中GPU上创建一个张量，最好的做法是使用一个[ `torch.cuda.device
`](../cuda.html#torch.cuda.device "torch.cuda.device")上下文管理器。

    
    
    print("Outside device is 0")  # On device 0 (default in most scenarios)
    with torch.cuda.device(1):
        print("Inside device is 1")  # On device 1
    print("Outside device is still 0")  # On device 0
    

如果你有一个张量，并希望创造在同一设备上同一类型的新张，那么你可以使用`torch.Tensor.new_ *`方法（参见[ `
torch.Tensor`](../tensors.html#torch.Tensor "torch.Tensor")）。虽然前面提到的`Torch 。*
`工厂函数（[ 创建行动 ](../torch.html#tensor-creation-ops)）取决于当前GPU上下文和你传递的属性参数`
torch.Tensor.new_ *`方法保持装置和张量的其他属性。

这是推荐的做法，当创建模块在新的张量需要在直传过程中内部创建的。

    
    
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
    

如果你想创建另一个张量的相同类型和大小的张量，并与任一或零填充，[ `ones_like（） `
](../torch.html#torch.ones_like "torch.ones_like")或[ `zeros_like（） `
](../torch.html#torch.zeros_like "torch.zeros_like")被设置作为方便的辅助功能（其也保留`
torch.device`和`torch.dtype 张量的`）。

    
    
    x_cpu = torch.empty(2, 3)
    x_gpu = torch.empty(2, 3)
    
    y_cpu = torch.ones_like(x_cpu)
    y_gpu = torch.zeros_like(x_gpu)
    

### 使用固定的内存缓冲区

主机到GPU副本要快得多，当他们从固定（锁定页）内存起源。 CPU张量和存储器露出[ `pin_memory（） `
](../tensors.html#torch.Tensor.pin_memory
"torch.Tensor.pin_memory")的方法，即返回对象的一个​​副本，其中放入一个钉扎区域数据。

此外，一旦你钉住张量或存储，您可以使用异步GPU副本。只是通过一个额外的`non_blocking =真 `参数向[ `至（） `
](../tensors.html#torch.Tensor.to "torch.Tensor.to")或[ `CUDA（） `
](../tensors.html#torch.Tensor.cuda "torch.Tensor.cuda")呼叫。这可以使用具有计算重叠的数据传输。

您可以在[ `的DataLoader`](../data.html#torch.utils.data.DataLoader
"torch.utils.data.DataLoader")回批通过传递`pin_memory =真 `给它的构造放置在固定的内存。

### 使用nn.DataParallel而不是多处理

涉及成批输入和多个GPU大多数使用情况应默认使用[ `数据并行 `](../nn.html#torch.nn.DataParallel
"torch.nn.DataParallel")利用多于一个的GPU。即使在GIL，一个Python程序可以饱和多个GPU。

随着0.1.9版本的GPU（8个）的大量可能没有被充分利用。然而，这是一个已知的问题，目前正在积极发展。与往常一样，测试你的使用情况。

有显著的注意事项使用CUDA型号[ `多重处理 `](../multiprocessing.html#module-
torch.multiprocessing "torch.multiprocessing")
;，除非小心地恰好满足数据处理的要求，很可能你的程序将有不正确的或不确定的行为。

[Next ![](../_static/images/chevron-right-orange.svg)](extending.html
"Extending PyTorch") [![](../_static/images/chevron-right-orange.svg)
Previous](cpu_threading_torchscript_inference.html "CPU threading and
TorchScript inference")

* * *

©版权所有2019年，Torch 贡献者。