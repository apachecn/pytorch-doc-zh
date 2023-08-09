# 新版本: PyTorch 1.7发布，带有CUDA 11，FFT的新API，Windows支持分布式训练等

> 发布: 2020年10月27日
> 
> 译者：[@片刻](https://github.com/jiangzhonglian)
> 
> 原文: <https://pytorch.org/blog/pytorch-1.7-released/>
> 
> 翻译: <https://pytorch.apachecn.org/LatestChanges/PyTorch_V1.7>

**来自 PyTorch团队**

今天，我们宣布PyTorch 1.7以及更新的域库的可用性。PyTorch 1.7版本包括许多新的API，包括支持NumPy兼容FFT操作、分析工具以及对分布式数据并行（DDP）和基于远程过程调用（RPC）的分布式训练的重大更新。此外，一些功能移动到[stable](https://pytorch.org/docs/stable/index.html#pytorch-documentation)，包括自定义C++类、内存分析器、通过自定义类似张量对象的扩展、RPC中的用户异步函数以及torch.distributed中的一些其他功能，如Per-RPC超时、DDP动态桶和RRef helper。

一些亮点包括：

*   CUDA 11现在正式支持[PyTorch.org](http://pytorch.org/)上的二进制文件
*   在autograd分析器中更新和添加RPC、torchscript和Stack跟踪的剖析和性能
*   （Beta）通过torch.fft支持NumPy兼容的快速傅里叶变换（FFT）
*   （Prototype）支持Nvidia A100代GPU和原生TF32格式
*   （Prototype）现在支持Windows上的分布式训练
*   torchvision
    *   （Stable）转换现在支持张量输入、批处理计算、GPU和torchscript
    *   （Stable）JPEG和PNG格式的原生图像I/O
    *   （Beta）新视频阅读器API
*   torchaudio
    *   （Stable）增加了对语音rec（wav2letter）、文本到语音（WaveRNN）和源分离（ConvTasNet）的支持

重申一下，从PyTorch 1.6开始，功能现在被归类为稳定、测试版和原型。你可以[在这里](https://pytorch.org/blog/pytorch-feature-classification-changes/)看到详细的公告。请注意，本博客中列出的原型功能可作为此版本的一部分使用。

[在这里](https://github.com/pytorch/pytorch/releases)找到完整的发布说明。

## 前端API

### \[BETA\] NumPy 兼容 torch.fft模块

FFT相关功能通常用于各种科学领域，如信号处理。虽然PyTorch历来支持一些与FFT相关的功能，但1.7版本增加了一个新的torch.fft模块，该模块使用与NumPy相同的API实现FFT相关功能。

这个新模块必须导入才能在1.7版本中使用，因为它的名称与历史（现已弃用）torch.fft函数冲突。

**示例用法：**

```
>>> import torch.fft
>>> t = torch.arange(4)
>>> t
tensor([0, 1, 2, 3])

>>> torch.fft.fft(t)
tensor([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])

>>> t = tensor([0.+1.j, 2.+3.j, 4.+5.j, 6.+7.j])
>>> torch.fft.fft(t)
tensor([12.+16.j, -8.+0.j, -4.-4.j,  0.-8.j])
```

*   [文稿](https://pytorch.org/docs/stable/fft.html#torch-fft)

### \[BETA\] C++支持变压器NN模块

自[PyTorch 1.5以来](https://pytorch.org/blog/pytorch-1-dot-5-released-with-new-and-updated-apis/)，我们继续保持python和C++前端API之间的奇偶校验。此更新允许开发人员使用来自C++前端的nn.transformer模块抽象。此外，开发人员不再需要从python/JIT中保存模块并加载到C++中，因为它现在可以直接在C++中使用。

*   [文稿](https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_transformer_impl.html#_CPPv4N5torch2nn15TransformerImplE)

### \[BETA\] torch.set_deterministic

可重现性（位对位确定性）可能有助于在调试或测试程序时识别错误。为了促进可重现性，PyTorch 1.7添加了`torch.set_deterministic(bool)`函数，该函数可以指导PyTorch操作员在可用时选择确定性算法，并在操作可能导致非确定性行为时抛出运行时错误。默认情况下，此函数控制的标志为假，行为没有变化，这意味着默认情况下，PyTorch可以非确定性地实现其操作。

更确切地说，当这面旗帜是真的时：

*   已知没有确定性实现的操作会抛出运行时错误；
*   使用确定性变体的操作使用这些变体（通常与非确定性版本相比具有性能惩罚）；以及
*   `torch.backends.cudnn.deterministic = True`设置好了。

请注意，对于**PyTorch程序的单个运行中的**确定性来说，这是必要的，**但还不够**。其他随机性来源，如随机数生成器、未知操作或异步或分布式计算，仍可能导致非确定性行为。

有关受影响的操作列表，请参阅`torch.set_deterministic(bool)`的文档。

*   [RFC](https://github.com/pytorch/pytorch/issues/15359)
*   [文稿](https://pytorch.org/docs/stable/generated/torch.set_deterministic.html)

## 性能评测

### \[BETA\] 堆栈跟踪已添加到分析器

用户现在不仅可以在分析器输出表中看到运算符名称/输入，还可以看到运算符在代码中的位置。工作流程几乎不需要改变就能利用这种能力。用户像以前一样使用[autograd分析器](https://pytorch.org/docs/stable/autograd.html#profiler)，但具有可选的新参数：`with_stack`和`group_by_stack_n`。注意：定期分析运行不应使用此功能，因为它增加了大量开销。

*   [详情](https://github.com/pytorch/pytorch/pull/43898/)
*   [文稿](https://pytorch.org/docs/stable/autograd.html)

## 分布式训练和RPC

### \[STABLE\] torchelastic 现在捆绑到 pytorch docker 图像中

Torchelastic 提供了当前 `torch.distributed.launch` CLI 的严格超集，并添加了容错和弹性功能。 如果用户对容错不感兴趣，他们可以通过设置 `max_restarts=0` 来获得准确的功能/行为奇偶校验，并增加自动分配的 `RANK` 和 `MASTER_ADDR|PORT` 的便利（与在 `torch.distributed.launch` 中手动指定相比）。

通过将`torchelastic`捆绑在与PyTorch相同的docker图像中，用户可以立即开始尝试TorchElastic，而无需单独安装`torchelastic`。除了方便之外，在现有的Kubeflow分布式PyTorch运算符中添加对弹性参数的支持时，这项工作也不错。

*   [用法示例以及如何开始](https://pytorch.org/elastic/0.2.0/examples.html)

### \[BETA\] 支持DDP中不均匀的数据集输入

PyTorch 1.7引入了一个新的上下文管理器，与使用`torch.nn.parallel.DistributedDataParallel`训练的模型一起使用，以便在不同进程中进行数据集大小不均匀的训练。此功能在使用DDP时具有更大的灵活性，并防止用户必须手动确保数据集大小在不同过程中相同。使用此上下文管理器，DDP将自动处理不均匀的数据集大小，这可以防止在训练结束时出现错误或挂起。

*   [RFC](https://github.com/pytorch/pytorch/issues/38174)
*   [文稿](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join)

### \[BETA\] NCCL可靠性-异步错误/超时处理

过去，由于集体陷入困境，NCCL的训练运行会无限期地挂起，给用户带来非常不愉快的体验。如果检测到潜在的挂起，此功能将中止卡住的集体，并抛出异常/崩溃过程。当与torchelastic（可以从最后一个检查点恢复训练过程）一起使用时，用户可以在分布式训练中具有更大的可靠性。此功能完全选择加入，位于需要显式设置的环境变量后面，以启用此功能（否则用户将看到与以前相同的行为）。

*   [RFC](https://github.com/pytorch/pytorch/issues/46874)
*   [文稿](https://pytorch.org/docs/stable/distributed.html?highlight=init_process_group#torch.distributed.init_process_group)

### \[BETA\] torchscript `RPC_REMOTE`和`RPC_SYNC`

`torch.distributed.rpc.rpc_async`在之前的版本中已在torchscript中可用。对于PyTorch 1.7，此功能将扩展到其余两个核心RPC API，`torch.distributed.rpc.rpc_sync`和`torch.distributed.rpc.remote`。这将完成torchscript中支持的主要RPC API，它允许用户在torchscript中使用现有的python RPC API（在脚本函数或脚本方法中，它释放了python全局解释器锁），并可能提高多线程环境中的应用程序性能。

*   [文稿](https://pytorch.org/docs/stable/rpc.html#rpc)
*   [例句](https://github.com/pytorch/pytorch/blob/58ed60c259834e324e86f3e3118e4fcbbfea8dd1/torch/testing/_internal/distributed/rpc/jit/rpc_test.py#L505-L525)

### \[BETA\] 支持 torchscript 的分布式优化器

PyTorch为训练算法提供了广泛的优化器，这些优化器已作为python API的一部分被反复使用。然而，用户通常希望使用多线程训练而不是多进程训练，因为它在大规模分布式训练中提供了更好的资源利用率和效率（例如分布式模型并行）或任何基于RPC的训练应用程序）。用户以前无法使用分布式优化器执行此操作，因为我们需要摆脱python全局解释器锁（GIL）限制来实现这一目标。

在PyTorch 1.7中，我们正在分布式优化器中启用torchscript支持，以删除GIL，并使在多线程应用程序中运行优化器成为可能。新的分布式优化器具有与以前完全相同的界面，但它会自动将每个工人中的优化器转换为torchscript，使每个GIL都免费。这是通过利用功能优化器概念来完成的，并允许分布式优化器将优化器的计算部分转换为torchscript。这将有助于分布式模型并行训练等用例，并使用多线程提高性能。

目前，唯一支持使用torchscript自动转换的优化器是`Adagrad`，如果没有torchscript支持，所有其他优化器仍将像以前一样工作。我们正在努力将覆盖范围扩大到所有PyTorch优化器，并预计未来版本会有更多的内容。启用torchscript支持的用法是自动的，与现有的python API完全相同，以下是如何使用它的例子：

```
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch import optim
from torch.distributed.optim import DistributedOptimizer

with dist_autograd.context() as context_id:
  # Forward pass.
  rref1 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))
  rref2 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 1))
  loss = rref1.to_here() + rref2.to_here()

  # Backward pass.
  dist_autograd.backward(context_id, [loss.sum()])

  # Optimizer, pass in optim.Adagrad, DistributedOptimizer will
  # automatically convert/compile it to torchscript (GIL-free)
  dist_optim = DistributedOptimizer(
     optim.Adagrad,
     [rref1, rref2],
     lr=0.05,
  )
  dist_optim.step(context_id)
```

*   [RFC](https://github.com/pytorch/pytorch/issues/46883)
*   [文稿](https://pytorch.org/docs/stable/rpc.html#module-torch.distributed.optim)

### \[BETA\] 基于RPC的剖析增强

PyTorch 1.6中首次引入了将PyTorch分析器与RPC框架结合使用的支持。在PyTorch 1.7中，进行了以下增强：

*   在RPC上实现了对分析torchscript函数的更好支持
*   在与RPC配合使用的情况分析器功能方面实现了奇偶校验
*   添加了对服务器端异步RPC函数的支持（withrpc`rpc.functions.async_execution)`装饰的功能）。

用户现在可以使用熟悉的分析工具，如`torch.autograd.profiler.profile()`和`with torch.autograd.profiler.record_function`，这与具有完整功能支持的RPC框架、配置文件异步函数和torchscript函数透明地工作。

*   [设计文档](https://github.com/pytorch/pytorch/issues/39675)
*   [例句](https://pytorch.org/tutorials/recipes/distributed_rpc_profiling.html)

### \[Prototype\] WINDOWS支持分布式训练

PyTorch 1.7为Windows平台上的`DistributedDataParallel`和集体通信提供了原型支持。在此版本中，支持仅涵盖基于Gloo的`ProcessGroup`和`FileStore`。

要跨多台机器使用此功能，请在`init_process_group`中提供来自共享文件系统的文件。

```
# initialize the process group
dist.init_process_group(
    "gloo",
    # multi-machine example:
    # init_method = "file://////{machine}/{share_folder}/file"
    init_method="file:///{your local file path}",
    rank=rank,
    world_size=world_size
)

model = DistributedDataParallel(local_model, device_ids=[rank])
```

*   [设计文档](https://github.com/pytorch/pytorch/issues/42095)
*   [文稿](https://pytorch.org/docs/master/distributed.html#backends-that-come-with-pytorch)
*   鸣谢（[gunandrose4u](https://github.com/gunandrose4u)）

## Mobile

PyTorch Mobile支持[iOS](https://pytorch.org/mobile/ios)和[Android](https://pytorch.org/mobile/android/)，[Cocoapods](https://cocoapods.org/)和[JCenter](https://mvnrepository.com/repos/jcenter)分别提供二进制包。您可以在[此处](https://pytorch.org/mobile/home/)了解有关PyTorch Mobile的更多信息。

### \[BETA\] PYTORCH移动缓存分配器用于性能改进

在一些移动平台上，如Pixel，我们观察到内存被更积极地返回到系统中。这导致频繁的页面故障，因为PyTorch是一个功能框架，无法为运营商保持状态。因此，对于大多数操作，每次执行时都会动态分配输出。为了改善由此导致的性能惩罚，PyTorch 1.7为CPU提供了一个简单的缓存分配器。分配器按张量大小缓存分配，目前只能通过PyTorch C++ API使用。缓存分配器本身由客户端拥有，因此分配器的生命周期也由客户端代码维护。然后，这种客户端拥有的缓存分配器可以与作用域保护`c10::WithCPUCachingAllocatorGuard`一起使用，以便在该范围内使用缓存分配。**示例用法：**

```
#include <c10/mobile/CPUCachingAllocator.h>
.....
c10::CPUCachingAllocator caching_allocator;
  // Owned by client code. Can be a member of some client class so as to tie the
  // the lifetime of caching allocator to that of the class.
.....
{
  c10::optional<c10::WithCPUCachingAllocatorGuard> caching_allocator_guard;
  if (FLAGS_use_caching_allocator) {
    caching_allocator_guard.emplace(&caching_allocator);
  }
  ....
  model.forward(..);
}
...
```

**注意**：缓存分配器仅在移动构建上可用，因此在移动构建之外使用缓存分配器将无效。

*   [文稿](https://github.com/pytorch/pytorch/blob/master/c10/mobile/CPUCachingAllocator.h#L13-L43)
*   [例句](https://github.com/pytorch/pytorch/blob/master/binaries/speed_benchmark_torch.cc#L207)

## torchvision

### \[STABLE\] 变换现在支持 Tensor 输入、批处理计算、GPU和torchscript

torchvision变换现在从`nn.Module`继承，可以进行火炬脚本并应用于火炬张量输入以及PIL图像。它们还支持具有批处理尺寸的Tensors，并在CPU/GPU设备上无缝工作：

```
import torch
import torchvision.transforms as T

# to fix random seed, use torch.manual_seed
# instead of random.seed
torch.manual_seed(12)

transforms = torch.nn.Sequential(
    T.RandomCrop(224),
    T.RandomHorizontalFlip(p=0.3),
    T.ConvertImageDtype(torch.float),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
)
scripted_transforms = torch.jit.script(transforms)
# Note: we can similarly use T.Compose to define transforms
# transforms = T.Compose([...]) and 
# scripted_transforms = torch.jit.script(torch.nn.Sequential(*transforms.transforms))

tensor_image = torch.randint(0, 256, size=(3, 256, 256), dtype=torch.uint8)
# works directly on Tensors
out_image1 = transforms(tensor_image)
# on the GPU
out_image1_cuda = transforms(tensor_image.cuda())
# with batches
batched_image = torch.randint(0, 256, size=(4, 3, 256, 256), dtype=torch.uint8)
out_image_batched = transforms(batched_image)
# and has torchscript support
out_image2 = scripted_transforms(tensor_image)
```

这些改进实现了以下新功能：

*   支持GPU加速
*   批量转换，例如根据视频的需要
*   转换多波段火炬张量图像（超过3-4个通道）
*   torchscript与您的模型一起转换，用于部署**注意：**torchscript支持的例外情况包括`Compose`、`RandomChoice`、`RandomOrder`、`Lambda`以及应用于PIL图像（如`ToPILImage`）。

### \[STABLE\] 用于JPEG和PNG格式的原生图像IO

torchvision 0.8.0引入了JPEG和PNG格式的原生图像读写操作。这些运算符支持torchscript，并以`uint8`格式返回`CxHxW`张量，因此现在可以成为您在C++环境中部署模型的一部分。

```
from torchvision.io import read_image

# tensor_image is a CxHxW uint8 Tensor
tensor_image = read_image('path_to_image.jpeg')

# or equivalently
from torchvision.io import read_file, decode_image
# raw_data is a 1d uint8 Tensor with the raw bytes
raw_data = read_file('path_to_image.jpeg')
tensor_image = decode_image(raw_data)

# all operators are torchscriptable and can be
# serialized together with your model torchscript code
scripted_read_image = torch.jit.script(read_image)
```

### \[STABLE\] 视网膜检测模型

此版本为视网膜网添加了预训练的模型，该模型具有来自[密集物体检测](https://arxiv.org/abs/1708.02002)的[焦距损失](https://arxiv.org/abs/1708.02002)的ResNet50主干。

### \[BETA\] 新视频阅读器API

此版本引入了新的视频阅读抽象，可以对视频的迭代进行更精细的控制。它支持图像和音频，并实现了迭代器接口，以便与迭代工具等其他python库互操作。

```
from torchvision.io import VideoReader

# stream indicates if reading from audio or video
reader = VideoReader('path_to_video.mp4', stream='video')
# can change the stream after construction
# via reader.set_current_stream

# to read all frames in a video starting at 2 seconds
for frame in reader.seek(2):
    # frame is a dict with "data" and "pts" metadata
    print(frame["data"], frame["pts"])

# because reader is an iterator you can combine it with
# itertools
from itertools import takewhile, islice
# read 10 frames starting from 2 seconds
for frame in islice(reader.seek(2), 10):
    pass
    
# or to return all frames between 2 and 5 seconds
for frame in takewhile(lambda x: x["pts"] < 5, reader):
    pass
```

**备注：**

*   为了使用Video Reader API测试版，您必须从源代码编译torchvision，并在系统中安装ffmpeg。
*   VideoReader API目前作为测试版发布，其API可能会根据用户反馈而更改。

## torchaudio

随着这个版本，torchaudio正在扩大对模型和[端到端应用程序](https://github.com/pytorch/audio/tree/master/examples)的支持，增加了wav2letter训练管道和端到端文本到语音和源分离管道。请在[github](https://github.com/pytorch/audio/issues/new?template=questions-help-support.md)上提交问题，以提供有关他们的反馈。

### \[STABLE\] 语音识别

在上一个版本中添加了用于语音识别的wav2letter模型的基础上，我们现在添加了一个带有LibriSpeech数据集的[wav2letter训练管道示例](https://github.com/pytorch/audio/tree/master/examples/pipeline_wav2letter)。

### \[STABLE\] 文本到语音

为了支持文本到语音应用程序，我们根据[该存储库](https://github.com/fatchord/WaveRNN)的实现，添加了一个基于WaveRNN模型的声码器。最初的实现是在“高效的神经音频合成”中引入的。我们还提供了一个[WaveRNN训练管道示例](https://github.com/pytorch/audio/tree/master/examples/pipeline_wavernn)，该[管道](https://github.com/pytorch/audio/tree/master/examples/pipeline_wavernn)使用本版本中添加到torchaudio的LibriTTS数据集。

### \[STABLE\] 源头分离

随着ConvTasNet模型的加入，基于论文“Conv-TasNet：超越语音分离的理想时间频率大小屏蔽”，torchiaudio现在也支持源分离。wsj-mix数据集提供了[ConvTasNet训练管道示例](https://github.com/pytorch/audio/tree/master/examples/source_separation)。

Cheers!

PyTorch 团队