# 新版本: PyTorch 2.1 版本，automatic dynamic shapes 编译，分布式 checkpoint

> 发布: 2023年10月04日
> 
> 译者：[@片刻](https://github.com/jiangzhonglian)
> 
> 原文: <https://pytorch.org/blog/pytorch-2-1>
> 
> 翻译: <https://pytorch.apachecn.org/LatestChanges/PyTorch_V2.1>

**来自 PyTorch团队**

我们很高兴地宣布PyTorch® 2.1（[发布说明](https://github.com/pytorch/pytorch/releases/tag/v2.1.0)）的发布！PyTorch 2.1提供automatic dynamic shapes支持intorch*.compile*，*torch.distributed.checkpoint*，用于并行保存/加载多个排名上的分布式训练作业，andtorch*.compile*支持NumPy API。

此外，此版本提供了许多性能改进（例如CPU inductor 改进，AVX512支持，scaled-dot-product-attention 支持）以及*torch.export*的原型发布，声音的全图捕获机制，以及基于*torch.export*的量化。

除了2.1，我们还发布了PyTorch域库的一系列更新。更多详细信息可以在图书馆更新博客中找到。

自2.0以来，此版本由6,682个提交和784个贡献者组成。我们要真诚地感谢我们敬业的社区的贡献。一如既往，我们鼓励您尝试这些，并在我们改进2.1时报告任何问题。有关如何开始使用PyTorch 2系列的更多信息，请访问我们的[入门](https://pytorch.org/get-started/pytorch-2.0/)页面。

总结：

*   *torch.compile*现在包括自动支持使用*automatic dynamic shapes*检测和最小化因张量形状变化而导致的重新编译*。*
*   *torch.distributed.checkpoint*允许从多个秩并行保存和加载模型，以及由于集群拓扑的变化而重新排序。
*   *torch.compile*现在可以通过将NumPy操作转换为PyTorch等效操作来编译NumPy操作。
*   *torch.compile*现在包括对Python 3.11的改进支持。
*   新的CPU性能功能包括 inductor 改进（例如bfloat16支持和动态形状）、AVX512内核支持和scaled-dot-product-attention 内核。
*   *torch.export*，一个声音的全图捕获机制被引入为原型功能，以及基于*torch.export*的量化。
*   *torch.sparse*现在包括对NVIDIA® GPU上semi-structed（2:4）稀疏性的原型支持。

| **Stable** | **Beta** | **Prototype** | **Performance Improvements** |
| --- | --- | --- | --- |
|   | Automatic Dynamic Shapes | *torch.export()* | AVX512内核支持 |
|   | *torch.distributed.checkpoint* | 基于Torch.export的量化 | scaled-dot-product-attention （SPDA）的CPU优化 |
|   | *torch.compile* + NumPy | semi-structed（2:4）稀疏 | bfloat16的CPU优化 |
|   | *torch.compile* + Python 3.11 | torchinductor 的 *cpp_wrapper* |   |
|   | *torch.compile + autograd.Function* |   |   |
|   | 第三方设备集成：*PrivateUse1* |   |   |

* 要查看公共2.1、2.0和1.13功能提交的完整列表，请单击[此处](https://docs.google.com/spreadsheets/d/1TzGkWuUMF1yTe88adz1dt2mzbIsZLd3PBasy588VWgk/edit?usp=sharing)。

## **测试版功能**

**[Beta版] Automatic Dynamic Shapes**

动态形状是*torch.compile*内置的功能，可以通过跟踪和生成基于张量符号形状而不是静态形状的代码来最大限度地减少重新编译（例如*[B，128，4]*而不是*[64，128，4]*）。这允许*torch.compile*生成一个单个内核，该内核可以适用于多种尺寸，但效率成本适中。动态形状在PyTorch 2.1中已大大稳定，如果由于输入形状不同而导致*torch.compile*通知重新编译，现在会自动启用。您可以通过将*dynamic=False*传递给torch.compile或settingtorch*._dynamo.config.automatic_dynamic_shapes = False*来禁用自动动态。

在PyTorch 2.1中，我们在CUDA和CPU上在各种模型类型（包括大型语言模型）上启用了动态形状，表现出了良好的性能。

有关动态形状的更多信息，请参阅[此文档](https://pytorch.org/docs/2.1/torch.compiler_dynamic_shapes.html)。

**[Beta] *torch.distributed.checkpoint***

*torch.distributed.checkpoint*能够并行从多个等级保存和加载模型。此外，检查点自动处理跨模型和优化器的完全限定名称（FQN）映射，从而实现跨不同集群拓扑的加载时间重新分级。

有关更多信息，请参阅*torch.distributed.checkpoint*[文档](https://pytorch.org/docs/2.1/distributed.checkpoint.html)和[教程](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)。

**[测试版] *torch.compile* + *NumPy***

*torch.compile*现在了解如何通过将NumPy操作转换为PyTorch等效操作来编译它们。由于这种集成以与设备无关的方式运行，您现在只需使用 *torch.compile，*就可以 GPU 加速 NumPy 程序，甚至混合 NumPy/PyTorch 程序。

有关 *torch.compile + NumPy 交互*的更多信息，请参阅 *torch.compile* FAQ 中的[此部分](https://pytorch.org/docs/2.1/torch.compiler_faq.html#does-numpy-work-with-torch-compile)，并关注 [PyTorch 博客](https://pytorch.org/blog/)以获取即将发布的有关此功能的博客。

**[Beta] *torch.compile* + Python 3.11**

*torch.compile*以前只支持Python版本3.8-3.10。用户现在可以在Python 3.11中使用*torch.compile*优化模型。

**[Beta] *torch.compile* + *autograd.Function***

*torch.compile*现在可以跟踪和优化用户定义的[autograd函数](https://pytorch.org/docs/stable/autograd.html#function)的后向功能，这为更重地使用扩展机制的模型解锁了训练优化。

**[Beta版]改进的第三方设备支持：*PrivateUse1***

第三方设备类型现在可以使用privateuse1调度密钥注册到PyTorch。这允许设备扩展将新内核注册到PyTorch，并将其与新密钥相关联，允许用户代码与内置设备类型等效。例如，要注册*“my_hardware_device*”，可以执行以下操作：

```
torch.rename_privateuse1_backend("my_hardware_device")
torch.utils.generate_methods_for_privateuse1_backend()
x = torch.randn((2, 3), device='my_hardware_device')
y = x + x # run add kernel on 'my_hardware_device'
```

为了验证此功能，*Ascend NPU*的OSS团队已通过*PrivateUse1*功能成功将[**torch_npu**](https://github.com/Ascend/pytorch)作为插件集成到pytorch中。

有关更多信息，请参阅[此处](https://pytorch.org/tutorials/advanced/privateuseone.html)的PrivateUse1教程。

## **原型功能**

**[Prototype] *torch.export()***

*torch.export()*提供了一个声音跟踪机制，用于从基于PT2.0提供的新技术的PyTorch程序中捕获完整的图形。

用户可以以数据流图的形式提取PyTorch程序的干净表示（导出IR），主要由对PyTorch运算符的直线调用组成。然后，导出IR可以转换、序列化、保存到文件、传输、加载回来，以便在有或没有Python的环境中执行。

有关更多信息，请参阅[此处](https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html)的教程。

**[Prototype] *torch.export-based* 的量化**

*torch.ao.quantization*现在支持基于PyTorch 2 *torch.export*的流的量化。这包括支持内置*XNNPACK*和*X64Inductor* *Quantizer*，以及指定自己的*Quantizer*的能力。

有关使用torch.export进行后训练静态量化的解释，请参阅[本教程](https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html)，有关使用torch.export进行静态量化的量化感知培训，请参阅[本教程](https://pytorch.org/tutorials/prototype/pt2e_quant_qat.html)。

有关如何编写自己的量化器的解释，请参阅[本教程](https://pytorch.org/tutorials/prototype/pt2e_quantizer.html)。

**[Prototype]NVIDIA® GPU的半结构化（2:4）稀疏性**

*torch.sparse*现在支持在半结构化稀疏（2:4）张量上创建和加速计算。有关格式的更多信息，请参阅NVIDIA[的](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/)博客。介绍半结构化稀疏性的最小示例如下：

```
from torch.sparse import to_sparse_semi_structured
 
x = torch.rand(64, 64).half().cuda()
mask = torch.tensor([0, 0, 1, 1]).tile((64, 16)).cuda().bool()
linear = nn.Linear(64, 64).half().cuda()

linear.weight = nn.Parameter(to_sparse_semi_structured(linear.weight.masked_fill(~mask, 0)))
linear(x)
```

要了解更多信息，请参阅[文档](https://pytorch.org/docs/2.1/sparse.html#sparse-semi-structured-tensors)和随附的[教程](https://pytorch.org/tutorials/prototype/semi_structured_sparse.html)。

**[Prototype] *torchinductor*的*cpp_wrapper***

*cpp_wrapper*可以通过在C++中生成内核包装器代码来减少在torhinductor中调用内核的Python开销。此功能仍处于原型阶段；它不支持今天在PT2中成功编译的所有程序。如果您发现用例的限制，请提交问题，以帮助我们确定优先级。

打开此功能的API是：

```
import torch
import torch._inductor.config as config
config.cpp_wrapper = True
```

有关更多信息，请参阅[教程](https://pytorch.org/tutorials/prototype/inductor_cpp_wrapper_tutorial.html)。

## **性能改进**

**AVX512内核支持**

在PyTorch 2.0中，即使CPU支持AVX512指令，也会使用AVX2内核。现在，如果CPU支持这些指令，PyTorch默认使用AVX512 CPU内核，这相当于在以前的版本中设置*ATEN_CPU_CAPABILITY=avx512*。可以通过设置*ATEN_CPU_CAPABILITY=avx2*来启用之前的行为*。*

**缩放点产品关注（SDPA）的CPU优化**

以前版本的PyTorch为变压器原语viatorch*.nn.functiona.scaled_dot_product_attention*提供了优化的CUDA实现。PyTorch 2.1包括优化的基于FlashAttention的CPU例程。

请参阅[此处](https://pytorch.org/docs/2.1/generated/torch.nn.functional.scaled_dot_product_attention.html)的文档。

**bfloat16的CPU优化**

PyTorch 2.1包括bfloat16的CPU优化，包括改进的矢量化支持和*火炬 inductor *代码器。
