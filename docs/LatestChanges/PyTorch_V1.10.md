# 新版本: PyTorch 1.10版本，包括CUDA Graphs API、前端和编译器改进

> 发布: 2021年10月21日
> 
> 译者：[@片刻](https://github.com/jiangzhonglian)
> 
> 原文: <https://pytorch.org/blog/pytorch-1.10-released/>
> 
> 翻译: <https://pytorch.apachecn.org/LatestChanges/PyTorch_V1.10>

**来自 PyTorch团队**

我们很高兴地宣布PyTorch 1.10的发布。自1.9以来，此版本由426名贡献者提交的3400多个提交组成。我们真诚地感谢我们的社区不断改进PyTorch。

PyTorch 1.10更新专注于提高PyTorch的训练和性能，以及开发人员的可用性。完整的发布说明可以[在这里](https://github.com/pytorch/pytorch/releases/tag/v1.10.0)找到。亮点包括：

1.  CUDA Graphs API集成在一起，以减少CUDA工作负载的CPU开销。
2.  几个前端API，如FX、torch.special和nn.Module参数化，已经从测试版转移到稳定。
3.  除了GPU外，JIT编译器中对自动融合的支持也扩展到CPU。
4.  Android NNAPI支持现已在测试版中提供。

除了1.10，我们还发布了PyTorch库的重大更新，您可以在[这篇博客文章中](https://pytorch.org/blog/pytorch-1.10-new-library-releases/)阅读。

## 前端API

### (Stable)使用FX进行Python代码转换

FX提供了一个Python平台，用于转换和降低PyTorch程序。这是一个用于传递编写者的工具套件，用于促进函数和nn.Module实例的Python到Python转换。该工具包旨在支持Python语言语义的子集，而不是整个Python语言，以促进转换的实现。有了1.10，外汇正在走向稳定。

您可以在[官方文档](https://pytorch.org/docs/master/fx.html)和使用`torch.fx`实现的程序转换的GitHub[示例](https://github.com/pytorch/examples/tree/master/fx)中了解有关FX的更多信息。

### (Stable)*torch.special*

一个类似于[SciPy的特殊模块的](https://docs.scipy.org/doc/scipy/reference/special.html)`torch.special module`，现在稳定可用。该模块有30个操作，包括伽马、贝塞尔和(高斯)错误函数。

有关更多详细信息，请参阅此[文档](https://pytorch.org/docs/master/special.html)。

### (Stable)nn.Module 参数化

`nn.Module`parametrizaton是一个功能，允许用户在不修改`nn.Module`本身的情况下对`nn.Module`的任何参数或缓冲区进行参数化，在稳定中可用。此版本增加了权重归一化(`weight_norm`)、正交参数化(矩阵约束和部分修剪)以及在创建自己的参数化时更大的灵活性。

有关更多详细信息，请参阅本[教程](https://pytorch.org/tutorials/intermediate/parametrizations.html)和一般[文档](https://pytorch.org/docs/master/generated/torch.nn.utils.parametrizations.spectral_norm.html?highlight=parametrize)。

### (Beta)CUDA图形API集成

PyTorch现在集成了CUDA Graphs API，以减少CUDA工作负载的CPU开销。

CUDA图形大大减少了CPU绑定的cuda工作负载的CPU开销，从而通过提高GPU利用率来提高性能。对于分布式工作负载，CUDA图形也减少了抖动，由于并行工作负载必须等待最慢的工人，因此减少抖动可以提高整体并行效率。

集成允许由cuda图捕获的网络部分与因图形限制而无法捕获的网络部分之间无缝互操作。

阅读[说明](https://pytorch.org/docs/master/notes/cuda.html#cuda-graphs)以了解更多详细信息和示例，并参考一般[文档](https://pytorch.org/docs/master/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph)以获取更多信息。

### \[Beta\]共轭视图

PyTorch对复tensor([torch.conj()](https://pytorch.org/docs/1.10.0/generated/torch.conj.html?highlight=conj#torch.conj))的共轭现在是一个常数时间操作，并返回具有共轭位集的输入tensor的视图，可以通过调用[torch.is\_conj()](https://pytorch.org/docs/1.10.0/generated/torch.is_conj.html?highlight=is_conj#torch.is_conj)看到。这已经在矩阵乘法、点积等其他各种PyTorch操作中得到了利用，将共轭与操作融合，从而在CPU和CUDA上实现显著的性能增益和内存节省。

## 分布式训练

### 分布式训练现已稳定发布

在1.10中，分布式软件包中有许多功能正在从测试版转向稳定：

*   **(Stable)远程模块**：此功能允许用户在远程工作者上操作模块，就像使用本地模块一样，其中RPC对用户是透明的。有关更多详细信息，请参阅此[文档](https://pytorch.org/docs/master/rpc.html#remotemodule)。
*   **(Stable)DDP通信钩子**：此功能允许用户覆盖DDP跨进程同步梯度的方式。有关更多详细信息，请参阅此[文档](https://pytorch.org/docs/master/rpc.html#remotemodule)。
*   **(Stable)ZeroRedundancyOptimizer**：此功能可以与DistributedDataParallel一起使用，以减少每个进程优化器状态的大小。有了这个稳定版本，它现在可以处理不同数据并行工人的不均匀输入。查看本[教程](https://pytorch.org/tutorials/advanced/generic_join.html)。我们还改进了参数分区算法，以更好地平衡跨进程的内存和计算开销。请参阅本[文档](https://pytorch.org/docs/master/distributed.optim.html)和本[教程](https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html)以了解更多信息。

## 性能优化和工具

### \[Beta\]在TorchScript中进行配置文件定向输出

TorchScript硬性要求源代码具有类型注释，以便编译成功。长期以来，只能通过反复试验(即通过逐一修复torch.jit.script生成的类型检查错误)来添加缺失或不正确的类型注释，这是低效和耗时的。

现在，我们通过利用MonkeyType等现有工具为 torch.jit.script 启用了配置文件定向键入，这使得该过程更简单、更快、更高效。有关更多详细信息，请参阅[文档](https://pytorch.org/docs/1.9.0/jit.html)。

### (Beta)CPU融合

在PyTorch 1.10中，我们为CPU添加了一个基于LLVM的JIT编译器，该编译器可以将`torch`库调用的序列融合在一起，以提高性能。虽然我们在GPU上拥有这种功能已经有一段时间了，但这个版本是我们第一次将编译带到CPU。  
您可以在这本[Colab笔记本](https://colab.research.google.com/drive/1xaH-L0XjsxUcS15GG220mtyrvIgDoZl6?usp=sharing)中为自己查看一些性能结果。

### (Beta)PyTorch分析器

PyTorch Profiler的目标是针对时间和/或内存成本最高的执行步骤，并可视化GPU和CPU之间的工作负载分布。PyTorch 1.10包括以下关键功能：

*   **增强的内存视图**：这有助于您更好地了解内存使用情况。此工具将通过在程序运行的各个点显示活动内存分配来帮助您避免内存不足错误。
*   **增强的自动化建议**：这有助于提供自动化性能建议，以帮助优化您的模型。这些工具建议更改批处理大小、TensorCore、内存减少技术等。
*   **增强的内核视图**：其他列显示网格和块大小，以及每个线程的共享内存使用和寄存器。
*   **分布式训练**：现在支持Gloo进行分布式训练工作。
*   **向前和向后通道中的相关运算符**：这有助于在跟踪视图中将向前通道中找到的运算符映射到向后通道，反之亦然。
*   **TensorCore**：此工具显示Tensor Core(TC)的使用情况，并为数据科学家和框架开发人员提供建议。
*   **NVTX**：对NVTX标记的支持是从传统的autograd分析器移植的。
*   **支持移动设备上的分析**：PyTorch分析器现在与TorchScript和移动后端具有更好的集成，可以为移动工作负载进行跟踪收集。

有关详细信息，请参阅此[文档](https://pytorch.org/docs/stable/profiler.html)。查看本[教程](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)，了解如何开始使用此功能。

## PyTorch手机

### (Beta)测试版中的Android NNAPI支持

去年，我们[发布了](https://medium.com/pytorch/pytorch-mobile-now-supports-android-nnapi-e2a2aeb74534)对Android神经网络API(NNAPI)的[原型支持](https://medium.com/pytorch/pytorch-mobile-now-supports-android-nnapi-e2a2aeb74534)。NNAPI允许Android应用程序在为手机供电的芯片中最强大、最高效的部件上运行计算密集型神经网络，包括GPU(图形处理单元)和NPU(专业神经处理单元)。

自原型以来，我们增加了更多的操作覆盖范围，增加了对加载时灵活形状的支持，以及在主机上运行模型进行测试的能力。使用[教程](https://pytorch.org/tutorials/prototype/nnapi_mobilenetv2.html)尝试此功能。

此外，转移学习步骤已添加到对象检测示例中。查看此GitHub[页面](https://github.com/pytorch/android-demo-app/tree/master/ObjectDetection#transfer-learning)以了解更多信息。请在[论坛](https://discuss.pytorch.org/c/mobile/18)上提供您的反馈或提问。您也可以查看[此演示文稿](https://www.youtube.com/watch?v=B-2spa3UCTU)以获取概述。

谢谢你的阅读。如果您对这些更新感兴趣，并想加入PyTorch社区，我们鼓励您加入[讨论论坛](https://discuss.pytorch.org/)并[打开GitHub问题](https://github.com/pytorch/pytorch/issues)。要从PyTorch获取最新消息，请在[Twitter](https://twitter.com/PyTorch)、[Medium](https://medium.com/pytorch)、[YouTube](https://www.youtube.com/pytorch)和[LinkedIn](https://www.linkedin.com/company/pytorch)上关注我们。

Cheers!

PyTorch 团队