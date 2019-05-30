# CUDA 语义

> 译者：[@Chris](https://github.com/Chriskuei)
> 
> 校对者：[@Twinkle](https://github.com/kemingzeng)

[`torch.cuda`](../cuda.html#module-torch.cuda "torch.cuda") 被用于设置和运行 CUDA 操作. 它会记录当前选择的 GPU, 并且分配的所有 CUDA 张量将默认在上面创建. 可以使用 [`torch.cuda.device`](../cuda.html#torch.cuda.device "torch.cuda.device") 上下文管理器更改所选设备.

但是, 一旦张量被分配, 您可以直接对其进行操作, 而不需要考虑已选择的设备, 结果将始终放在与张量相关的设备上.

默认情况下, 不支持跨 GPU 操作, 唯一的例外是 [`copy_()`](../tensors.html#torch.Tensor.copy_ "torch.Tensor.copy_"). 除非启用对等存储器访问, 否则对分布在不同设备上的张量尝试进行任何启动操作都将引发错误.

下面我们用一个小例子来展示:

```py
x = torch.cuda.FloatTensor(1)
# x.get_device() == 0
y = torch.FloatTensor(1).cuda()
# y.get_device() == 0

with torch.cuda.device(1):
    # allocates a tensor on GPU 1
    a = torch.cuda.FloatTensor(1)

    # transfers a tensor from CPU to GPU 1
    b = torch.FloatTensor(1).cuda()
    # a.get_device() == b.get_device() == 1

    c = a + b
    # c.get_device() == 1

    z = x + y
    # z.get_device() == 0

    # 即使在上下文里面, 你也可以在 .cuda 的参数中传入设备id
    d = torch.randn(2).cuda(2)
    # d.get_device() == 2

```

## 内存管理

PyTorch 使用缓存内存分配器来加速内存分配. 这允许在没有设备同步的情况下快速释放内存. 但是, 由分配器管理的未使用的内存仍将显示为在 `nvidia-smi` 中使用. 调用 [`empty_cache()`](../cuda.html#torch.cuda.empty_cache "torch.cuda.empty_cache") 可以从 PyTorch 中释放所有未使用的缓存内存, 以便其他 GPU 应用程序使用这些内存.

## 最佳实践

### 设备无关代码

由于 PyTorch 的架构, 你可能需要明确写入设备无关 (CPU 或 GPU) 代码; 举个例子, 创建一个新的张量作为循环神经网络的初始隐藏状态.

第一步先确定是否使用 GPU. 一个常见的方式是使用 Python 的 `argparse` 模块来读入用户参数, 并且有一个可以用来禁用 CUDA、能与 [`is_available()`](../cuda.html#torch.cuda.is_available "torch.cuda.is_available") 结合使用的标志. 在下面的例子中, `args.cuda` 会产生一个当需要时能将张量和模块转换为 CUDA 的标志:

```py
import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

```

如果需要将模块和张量发送到 GPU, `args.cuda` 可以使用如下:

```py
x = torch.Tensor(8, 42)
net = Network()
if args.cuda:
  x = x.cuda()
  net.cuda()

```

创建张量时, 可以定义一个默认的数据类型来替代 if 语句, 并使用它来转换所有的张量. 使用 dataLoader 的例子如下:

```py
dtype = torch.cuda.FloatTensor
for i, x in enumerate(train_loader):
    x = Variable(x.type(dtype))

```

在系统上使用多个 GPU 时, 您可以使用 `CUDA_VISIBLE_DEVICES` 环境标志来管理哪些 GPU 可用于 PyTorch. 如上所述, 要手动控制在哪个 GPU 上创建张量, 最好的方法是使用 [`torch.cuda.device`](../cuda.html#torch.cuda.device "torch.cuda.device") 上下文管理器:

```py
print("Outside device is 0")  # On device 0 (default in most scenarios)
with torch.cuda.device(1):
    print("Inside device is 1")  # On device 1
print("Outside device is still 0")  # On device 0

```

如果您有一个张量, 并且想在同一个设备上创建一个相同类型的张量, 那么您可以使用 [`new()`](../tensors.html#torch.Tensor.new "torch.Tensor.new") 方法, 它的使用和普通的张量构造函数一样. 虽然前面提到的方法取决于当前的 GPU 环境, 但是 [`new()`](../tensors.html#torch.Tensor.new "torch.Tensor.new") 保留了原始张量的设备信息.

当创建在向前传递期间需要在内部创建新的张量/变量的模块时, 建议使用这种做法:

```py
x_cpu = torch.FloatTensor(1)
x_gpu = torch.cuda.FloatTensor(1)
x_cpu_long = torch.LongTensor(1)

y_cpu = x_cpu.new(8, 10, 10).fill_(0.3)
y_gpu = x_gpu.new(x_gpu.size()).fill_(-5)
y_cpu_long = x_cpu_long.new([[1, 2, 3]])

```

如果你想创建一个与另一个张量有着相同类型和大小、并用 1 或 0 填充的张量, [`ones_like()`](../torch.html#torch.ones_like "torch.ones_like") 或 [`zeros_like()`](../torch.html#torch.zeros_like "torch.zeros_like") 可提供方便的辅助功能 (同时保留设备信息)

```py
x_cpu = torch.FloatTensor(1)
x_gpu = torch.cuda.FloatTensor(1)

y_cpu = torch.ones_like(x_cpu)
y_gpu = torch.zeros_like(x_gpu)

```

### 使用固定的内存缓冲区

当副本来自固定 (页锁) 内存时, 主机到 GPU 的复制速度要快很多. CPU 张量和存储开放了一个 [`pin_memory()`](../tensors.html#torch.Tensor.pin_memory "torch.Tensor.pin_memory") 方法, 它返回该对象的副本, 而它的数据放在固定区域中.

另外, 一旦固定了张量或存储, 就可以使用异步的 GPU 副本. 只需传递一个额外的 `async=True` 参数给 [`cuda()`](../tensors.html#torch.Tensor.cuda "torch.Tensor.cuda") 调用. 这可以用于重叠数据传输与计算.

通过将 `pin_memory=True` 传递给其构造函数, 可以使 [`DataLoader`](../data.html#torch.utils.data.DataLoader "torch.utils.data.DataLoader") 将 batch 返回到固定内存中.

### 使用 nn.DataParallel 替代 multiprocessing

大多数涉及批量输入和多个 GPU 的情况应默认使用 [`DataParallel`](../nn.html#torch.nn.DataParallel "torch.nn.DataParallel") 来使用多个 GPU. 尽管有 GIL 的存在, 单个 Python 进程也可能使多个 GPU 饱和.

从 0.1.9 版本开始, 大量的 GPU (8+) 可能未被充分利用. 然而, 这是一个已知的问题, 也正在积极开发中. 和往常一样, 测试您的用例吧.

调用 [`multiprocessing`](../multiprocessing.html#module-torch.multiprocessing "torch.multiprocessing") 使用 CUDA 模型存在显著的注意事项; 除非您足够谨慎以满足数据处理需求, 否则您的程序很可能会出现错误或未定义的行为.