# 新版本: PyTorch 1.6 发布，支持原生 AMP，微软作为 Windows 维护者加入

> 发布: 2020年07月28日
> 
> 译者：[@片刻](https://github.com/jiangzhonglian)
> 
> 原文: <https://pytorch.org/blog/pytorch-1.6-released/>
> 
> 翻译: <https://pytorch.apachecn.org/LatestChanges/PyTorch_V1.6>

**来自 PyTorch团队**

今天，我们宣布推出 PyTorch 1.6 以及更新的域库。我们还很高兴地宣布，Microsoft 团队[现在正在维护 Windows 版本和二进制文件](https://pytorch.org/blog/microsoft-becomes-maintainer-of-the-windows-version-of-pytorch)，并将支持 GitHub 上的社区以及 PyTorch Windows 讨论论坛。

PyTorch 1.6 版本包括许多新的 API、用于性能改进和分析的工具，以及基于分布式数据并行 (DDP) 和远程过程调用 (RPC) 的分布式训练的重大更新。其中一些亮点包括：

1.  现在原生支持自动混合精度 (AMP) 训练并且是一个稳定的功能（更多详细信息请参见[此处](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)）——感谢 NVIDIA 的贡献；
2.  现在添加了原生 TensorPipe 支持，用于专门为机器学习构建的张量感知、点对点通信原语；
3.  向前端 API 界面添加了对复杂张量的支持；
4.  新的分析工具提供张量级内存消耗信息；
5.  针对分布式数据并行 (DDP) 训练和远程过程调用 (RPC) 包的大量改进和新功能。

此外，从该版本开始，功能将分为稳定版、测试版和原型版。原型功能不包含在二进制发行版中，而是可以通过从源代码构建、使用 nightlies 或通过编译器标志来获得。[您可以在此处的](https://pytorch.org/blog/pytorch-feature-classification-changes/)帖子中详细了解此更改的含义。[您还可以在此处](https://github.com/pytorch/pytorch/releases)找到完整的发行说明。

## 性能与分析

### \[STABLE\] 自动混合精度 (AMP) 训练

AMP 允许用户轻松启用自动混合精度训练，从而在 Tensor Core GPU 上实现更高的性能和高达 50% 的内存节省。使用本机支持的`torch.cuda.amp`​​API，AMP 提供了混合精度的便捷方法，其中某些操作使用`torch.float32 (float)`数据类型，其他操作使用`torch.float16 (half)`. 一些操作，如线性层和卷积，在`float16`. 其他操作（例如缩减）通常需要 的动态范围`float32`。混合精度尝试将每个操作与其适当的数据类型相匹配。

*   设计文档（[链接](https://github.com/pytorch/pytorch/issues/25081)）
*   文档（[链接](https://pytorch.org/docs/stable/amp.html)）
*   使用示例（[链接](https://pytorch.org/docs/stable/notes/amp_examples.html)）

### \[BETA\] FORK/JOIN 并行性

此版本增加了对语言级构造的支持以及对 TorchScript 代码中粗粒度并行性的运行时支持。这种支持对于并行运行集成中的模型或并行运行循环网络的双向组件等情况非常有用，并且允许释放并行架构（例如多核CPU）的计算能力以实现任务级并行。

TorchScript 程序的并行执行是通过两个原语启用的：`torch.jit.fork`和`torch.jit.wait`。在下面的示例中，我们并行执行`foo`：

```
import torch
from typing import List

def foo(x):
    return torch.neg(x)

@torch.jit.script
def example(x):
    futures = [torch.jit.fork(foo, x) for _ in range(100)]
    results = [torch.jit.wait(future) for future in futures]
    return torch.sum(torch.stack(results))

print(example(torch.ones([])))
```

*   文档（[链接](https://pytorch.org/docs/stable/jit.html)）

### \[BETA\] 内存分析器

该`torch.autograd.profiler`API 现在包含一个内存分析器，可让您检查 CPU 和 GPU 模型内不同运算符的张量内存成本。

以下是 API 的使用示例：

```
import torch
import torchvision.models as models
import torch.autograd.profiler as profiler

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)
with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    model(inputs)

# NOTE: some columns were removed for brevity
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
# ---------------------------  ---------------  ---------------  ---------------
# Name                         CPU Mem          Self CPU Mem     Number of Calls
# ---------------------------  ---------------  ---------------  ---------------
# empty                        94.79 Mb         94.79 Mb         123
# resize_                      11.48 Mb         11.48 Mb         2
# addmm                        19.53 Kb         19.53 Kb         1
# empty_strided                4 b              4 b              1
# conv2d                       47.37 Mb         0 b              20
# ---------------------------  ---------------  ---------------  ---------------
```

*   公关（[链接](https://github.com/pytorch/pytorch/pull/37775)）
*   文档（[链接](https://pytorch.org/docs/stable/autograd.html#profiler)）

## 分布式训练和 RPC

### \[BETA\] 用于 RPC 的 TensorPipe 后端

PyTorch 1.6 引入了 RPC 模块的新后端，该模块利用 TensorPipe 库，这是一种针对机器学习的张量感知点对点通信原语，旨在补充 PyTorch 中分布式训练的当前原语（Gloo、MPI 等）这是集体性和阻塞性的。TensorPipe 的成对和异步性质使其适合超越数据并行的新网络范例：客户端-服务器方法（例如，用于嵌入的参数服务器、Impala 式 RL 中的参与者-学习者分离等）以及模型和管道并行训练（想想 GPipe）、八卦 SGD 等。

```
# One-line change needed to opt in
torch.distributed.rpc.init_rpc(
    ...
    backend=torch.distributed.rpc.BackendType.TENSORPIPE,
)

# No changes to the rest of the RPC API
torch.distributed.rpc.rpc_sync(...)
```

*   设计文档（[链接](https://github.com/pytorch/pytorch/issues/35251)）
*   文档（[链接](https://pytorch.org/docs/stable/rpc/index.html)）

### \[BETA\] DDP+RPC

PyTorch Distributed 支持两种强大的范例：用于模型的完全同步数据并行训练的 DDP 和允许分布式模型并行的 RPC 框架。以前，这两个功能独立工作，用户无法混合和匹配它们来尝试混合并行范例。

从 PyTorch 1.6 开始，我们使 DDP 和 RPC 能够无缝协作，以便用户可以结合这两种技术来实现数据并行和模型并行。一个例子是，用户希望将大型嵌入表放在参数服务器上并使用 RPC 框架进行嵌入查找，但将较小的密集参数存储在训练器上并使用 DDP 来同步密集参数。下面是一个简单的代码片段。

```
// On each trainer

remote_emb = create_emb(on="ps", ...)
ddp_model = DDP(dense_model)

for data in batch:
   with torch.distributed.autograd.context():
      res = remote_emb(data)
      loss = ddp_model(res)
      torch.distributed.autograd.backward([loss])
```

*   DDP+RPC教程（[链接](https://pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html)）
*   文档（[链接](https://pytorch.org/docs/stable/rpc/index.html)）
*   使用示例（[链接](https://github.com/pytorch/examples/pull/800)）

### \[BETA\] RPC - 异步用户函数

RPC 异步用户函数支持执行用户定义函数时在服务器端产生和恢复的能力。在此功能之前，当被调用者处理请求时，一个 RPC 线程会等待，直到用户函数返回。如果用户函数包含 IO（例如，嵌套 RPC）或信令（例如，等待另一个请求解除阻塞），则相应的 RPC 线程将闲置等待这些事件。因此，某些应用程序必须使用大量线程并发送额外的 RPC 请求，这可能会导致性能下降。要使用户函数在此类事件上屈服，应用程序需要： 1) 使用装饰器装饰函数`@rpc.functions.async_execution`；2) 让函数返回 a`torch.futures.Future`并将恢复逻辑安装为回调`Future`目的。请参阅下面的示例：

```
@rpc.functions.async_execution
def async_add_chained(to, x, y, z):
    return rpc.rpc_async(to, torch.add, args=(x, y)).then(
        lambda fut: fut.wait() + z
    )

ret = rpc.rpc_sync(
    "worker1", 
    async_add_chained, 
    args=("worker2", torch.ones(2), 1, 1)
)
        
print(ret)  # prints tensor([3., 3.])
```

*   使用异步用户函数的高性能批处理 RPC 教程（[链接](https://github.com/pytorch/tutorials/blob/release/1.6/intermediate_source/rpc_async_execution.rst)）
*   文档（[链接](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.functions.async_execution)）
*   使用示例（[链接](https://github.com/pytorch/examples/tree/master/distributed/rpc/batch)）

## 前端 API 更新

### \[BETA\] 复数

PyTorch 1.6 版本为复杂张量（包括 torch.complex64 和 torch.complex128 dtypes）提供了 beta 级支持。复数是可以用 a + bj 形式表示的数字，其中 a 和 b 是实数，j 是方程 x^2 = −1 的解。复数经常出现在数学和工程中，特别是在信号处理中，而复数神经网络领域是一个活跃的研究领域。复杂张量的测试版将支持常见的 PyTorch 和复杂张量功能，以及 Torchaudio、ESPnet 等所需的功能。虽然这是此功能的早期版本，我们预计它会随着时间的推移而改进，

## 移动更新

PyTorch 1.6 为移动设备上的推理带来了更高的性能和总体稳定性。我们消除了一些错误，继续维护并添加了一些新功能，同时提高了 CPU 后端上各种 ML 模型推理的 fp32 和 int8 性能。

### \[BETA\] 移动功能和性能

*   无状态和有状态 XNNPACK Conv 和 Linear 运算符
*   无状态 MaxPool2d + JIT 优化过程
*   JIT pass 优化：Conv + BatchNorm 融合、图形重写以用 xnnpack 操作替换 conv2d/线性、relu/hardtanh 融合、dropout 去除
*   QNNPACK 集成消除了重新量化规模限制
*   转换、线性和动态线性的每通道量化
*   禁用移动客户端跟踪可在 full-jit 构建上节省约 600 KB

## 更新的域库

### torchvision 0.7

torchvision 0.7 引入了两个新的预训练语义分割模型：[FCN ResNet50](https://arxiv.org/abs/1411.4038)和[DeepLabV3 ResNet50](https://arxiv.org/abs/1706.05587)，两者都在 COCO 上进行训练，并且使用比 ResNet101 主干网更小的内存占用。我们还为 torchvision 模型和算子引入了对 AMP（自动混合精度）自动转换的支持，它可以自动为不同的 GPU 操作选择浮点精度，以在保持准确性的同时提高性能。

*   发行说明（[链接](https://github.com/pytorch/vision/releases)）

### torchaudio 0.6

torchaudio 现在正式支持 Windows。此版本还引入了新的模型模块（包括 wav2letter）、新函数（contrast、cvm、dcshift、overdrive、vad、phaser、flanger、biquad）、数据集（GTZAN、CMU）以及新的可选 sox 后端，支持火炬脚本。

*   发行说明（[链接](https://github.com/pytorch/audio/releases)）

## 额外更新

### 黑客马拉松

全球 PyTorch 夏季黑客马拉松回来了！今年，团队可以虚拟地参加三个类别的比赛：

1.  **PyTorch 开发人员工具：**旨在为研究人员和开发人员提高 PyTorch 生产力和效率的工具或库
2.  **由 PyTorch 提供支持的 Web/移动应用程序：**具有由 PyTorch 支持的 Web/移动界面和/或嵌入式设备的应用程序
3.  **PyTorch Responsible AI 开发工具：**用于负责任的 AI 开发的工具、库或 Web/移动应用程序

这是与社区建立联系并练习机器学习技能的绝佳机会。

*   [参加黑客马拉松](http://pytorch2020.devpost.com/)
*   [观看教育视频](https://www.youtube.com/pytorch)

### LPCV挑战

2020 年 CVPR 低功耗视觉挑战赛[(LPCV) - 无人机视频在线赛道](https://lpcv.ai/2020CVPR/video-track)提交截止日期即将到来。您必须在 2020 年 7 月 31 日之前构建一个系统，该系统可以使用 PyTorch 和 Raspberry Pi 3B+ 准确地发现和识别无人机 (UAV) 捕获的视频中的字符。

### 原型特征

重申一下，PyTorch 中的原型功能是早期功能，我们希望在将其升级为测试版或稳定版之前收集反馈、评估其实用性并进行改进。以下功能不属于 PyTorch 1.6 版本的一部分，而是在带有单独文档/教程的 nightlies 中提供，以帮助促进早期使用和反馈。

#### 分布式 RPC/分析器

允许用户使用 autograd 分析器来分析使用的训练作业`torch.distributed.rpc`，并远程调用分析器以跨不同节点收集分析信息。[可以在此处](https://github.com/pytorch/pytorch/issues/39675)找到 RFC ，并且可以[在此处](https://github.com/pytorch/tutorials/tree/master/prototype_source)找到有关如何使用此功能的简短说明。

#### TorchScript 模块冻结

模块冻结是将模块参数和属性值内联到 TorchScript 内部表示的过程。参数和属性值被视为最终值，并且不能在冻结模块中修改。[可以在此处](https://github.com/pytorch/pytorch/pull/32178)找到此功能的 PR ，并且可以[在此处](https://github.com/pytorch/tutorials/tree/master/prototype_source)找到有关如何使用此功能的简短教程。

#### 图模式量化

Eager 模式量化要求用户对其模型进行更改，包括显式量化激活、模块融合、使用功能模块重写 torch 操作以及不支持功能量化。如果我们可以跟踪或编写模型脚本，则可以使用图形模式量化自动完成量化，而无需急切模式中的任何复杂性，并且可以通过`qconfig_dict`. 有关如何使用此功能的教程可以[在此处](https://github.com/pytorch/tutorials/tree/master/prototype_source)找到。

#### 量化数值套件

当量化有效时，它是好的，但当它不满足预期的精度时，很难知道出了什么问题。现在可以使用数值套件的原型来测量量化模块和浮点模块之间的比较统计数据。这可以使用 eager 模式进行测试，并且只能在 CPU 上进行测试，并提供更多支持。有关如何使用此功能的教程可以[在此处](https://github.com/pytorch/tutorials/tree/master/prototype_source)找到。

Cheers!

PyTorch 团队