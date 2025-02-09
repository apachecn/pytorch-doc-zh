# 新版本: PyTorch 2.5 发布博客

> 发布: 2024年10月17日
> 
> 译者：[@片刻](https://github.com/jiangzhonglian)
> 
> 原文: <https://pytorch.org/blog/pytorch2-5>
> 
> 翻译: <https://pytorch.apachecn.org/LatestChanges/PyTorch_V2.5>

**来自 PyTorch团队**

我们很高兴地宣布 PyTorch® 2.5 的发布（[发布说明](https://github.com/pytorch/pytorch/releases/tag/v2.5.0)）！此版本为 SDPA 提供了一个新的 cuDNN 后端，默认情况下，SDPA 用户在 H100 或更新的 GPU 上可以加速。此外，torch.compile 的区域编译提供了一种减少 torch.compile 冷启动时间的方法，它允许用户编译重复的 nn.Module（例如 LLM 中的转换器层）而无需重新编译。最后，TorchInductor CPP 后端通过 FP16 支持、CPP 包装器、AOT-Inductor 模式和最大自动调谐模式等众多增强功能提供了可靠的性能加速。

此版本由 504 位贡献者自 PyTorch 2.4 以来的 4095 次提交组成。我们衷心感谢我们敬业的社区所做的贡献。与往常一样，我们鼓励您尝试这些功能并在我们改进 2.5 时报告任何问题。有关如何开始使用 PyTorch 2 系列的更多信息，请参阅我们的[入门](https://pytorch.org/get-started/pytorch-2.0/)页面。

另外，请查看我们与[TorchRec](https://github.com/pytorch/torchrec)和[TorchFix](https://github.com/pytorch-labs/torchfix/releases/tag/v0.6.0)一起发布的新生态系统项目。

| **Beta** | **Prototype** | **Performance Improvements** |
| --- | --- | --- |
| SDPA 的 cuDNN 后端 | FlexAttention | |
| torch.compile 区域编译，无需重新编译 | 编译的 Autograd | |
| TorchDynamo 增加了对异常处理和 MutableMapping 类型的支持 | Flight Recorder | |
| TorchInductor CPU 后端优化 | 使用 GEMM 模板对 CPU 进行最大自动调谐支持 | |
| | Windows 上的 TorchInductor | |
| | CPU 路径上对 Eager 模式和 TorchInductor CPP 后端均支持 FP16 | |
| | 自动加载设备扩展 | |
| | 增强英特尔 GPU 支持 | |

* 要查看公开功能提交的完整列表，请单击[此处](https://docs.google.com/spreadsheets/d/1TzGkWuUMF1yTe88adz1dt2mzbIsZLd3PBasy588VWgk/edit?usp=sharing)。

## 测试版功能

### [Beta] SDPA 的 cuDNN 后端

*cuDNN“融合 Flash Attention”后端已为torch.nn. functional.scaled_dot_product_attention*推出。在 NVIDIA H100 GPU 上，这可以提供比 FlashAttentionV2 高达 75% 的速度提升。对于 H100 或更新 GPU 上的所有 SDPA 用户，此加速功能默认启用。

### [Beta] *torch.compile*区域编译无需重新编译

*通过torch._dynamo.config.inline_inbuilt_nn_modules*进行区域编译而无需重新编译，在 2.5+ 版本中默认为 True。此选项允许用户编译重复的*nn.Module*（例如 LLM 中的转换器层）而无需重新编译。与编译完整模型相比，此选项可以缩短编译延迟，性能下降 1%-5%。

请参阅[教程](https://pytorch.org/tutorials/recipes/regional_compilation.html)以了解更多信息。

### [Beta] TorchInductor CPU 后端优化

此功能增强了 Inductor 的 CPU 后端优化，包括 CPP 后端代码生成和与定制 CPU 内核的 FX 融合。Inductor CPU 后端支持常见数据类型和所有 Inductor IR 操作的矢量化，以及静态和符号形状。它兼容 Linux 和 Windows 操作系统，并支持默认 Python 包装器、CPP 包装器和 AOT-Inductor 模式。

此外，它还扩展了 GEMM 模板（在 2.5 中原型化）的最大自动调谐模式，从而进一步提升性能。后端支持各种 FX 融合，降低到定制内核，例如用于线性/卷积操作和 SDPA 的 oneDNN。Inductor CPU 后端在三个基准测试套件（TorchBench、Hugging Face 和 timms）中始终实现性能加速，在 193 个测试模型中，97.5% 的模型性能优于 Eager 模式。

## 原型特点

### [Prototype] FlexAttention

我们引入了一个灵活的 API，只需几行惯用的 PyTorch 代码即可实现各种注意力机制，例如滑动窗口、因果掩码和 PrefixLM。此 API 利用 torch.compile 生成融合的 FlashAttention 内核，从而消除了额外的内存分配并实现了与手写实现相当的性能。此外，我们使用 PyTorch 的自动求导机制自动生成后向传递。此外，我们的 API 可以利用注意力掩码中的稀疏性，从而比标准注意力实现有显著的改进。

更多信息和示例请参考官[方博文](https://pytorch.org/blog/flexattention/)和[Attention Gym](https://github.com/pytorch-labs/attention-gym)。

### [Prototype] 编译的 Autograd

编译型 Autograd 是 PT2 堆栈的扩展，允许捕获整个反向传递。与 AOT 调度程序跟踪的反向图不同，编译型 Autograd 跟踪被推迟到反向执行时间，这使得它不受正向传递图中断的影响，并允许它将反向挂钩记录到图中。

请参阅教程[以](https://pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html)了解更多信息。

### [Prototype] Flight Recorder

Flight recorder 是一款新的调试工具，可帮助调试卡住的作业。该工具的工作原理是在集体运行时不断捕获有关集体的信息。检测到卡住的作业后，可以使用该信息快速识别行为不当的等级/机器以及代码堆栈跟踪。

更多信息请参阅以下[教程](https://pytorch.org/tutorials/prototype/flight_recorder_tutorial.html)。

### [Prototype] 使用 GEMM 模板在 CPU 上实现最大自动调谐支持

torch.compile 中电感器 CPU 后端的最大自动调谐模式在编译时分析多个操作实现并选择性能最佳的实现。这对于 GEMM 相关操作特别有益，使用基于 C++ 模板的 GEMM 实现作为基于 ATen 的方法的替代方案，该方法使用 oneDNN 和 MKL 库。我们支持 FP32、BF16、FP16 和 INT8，并为 x86 CPU 提供结语融合。我们已经看到 Dynamo 基准测试套件的几何平均速度提高了 7%，LLM 推理的下一个标记延迟提高了 20%。

更多信息请参阅[教程](https://pytorch.org/tutorials/prototype/max_autotune_on_CPU_tutorial.html)。

### [Prototype] Windows 上的 TorchInductor CPU

torch.compile 中的电感器 CPU 后端现在可在 Windows 上运行。我们目前支持 Windows 电感器的 MSVC (cl)、clang (clang-cl) 和 Intel 编译器 (icx-cl)。

请参阅[教程](https://pytorch.org/tutorials/prototype/inductor_windows_cpu.html)了解更多详细信息。

### [Prototype] CPU 路径上的 FP16 支持 Eager 模式和 TorchInductor CPP 后端

Float16 是一种常用的简化浮点类型，用于提高神经网络推理/训练的性能。自此版本以来，CPU 路径上均支持 eager 和 TorchInductor 的 float16。

### [Prototype] 自动加载设备扩展

PyTorch 现在支持自动加载树外设备扩展，通过消除手动导入的需要来简化集成。此功能通过 torch.backends 入口点启用，通过确保无缝扩展加载简化了使用，同时允许用户在需要时通过环境变量禁用它。

请参阅[教程](https://pytorch.org/tutorials/prototype/python_extension_autoload.html)以了解更多信息。

### [Prototype] 增强英特尔 GPU 支持

英特尔 GPU 支持增强功能现已适用于英特尔® 数据中心 GPU Max 系列和英特尔® 客户端 GPU（内置英特尔® Arc™ 显卡的英特尔® 酷睿™ 超处理器和用于 dGPU 部件的英特尔® Arc™ 显卡），这是为了让您在 PyTorch 2.5 版本中更轻松地加速英特尔 GPU 上的机器学习工作流程。我们还在此版本中为英特尔® 客户端 GPU 启用了 Windows 上 PyTorch 的初始支持。

*   扩展 PyTorch 硬件后端支持矩阵，包括英特尔数据中心和客户端 GPU。  
*   SYCL\* 内核的实现增强了 Aten 运算符在英特尔 GPU 上的覆盖范围和执行，从而提升了 PyTorch 急切模式下的性能。
*   增强 torch.compile 的英特尔 GPU 后端，以提高广泛深度学习工作负载的推理和训练性能。

这些功能可通过 PyTorch 预览版和夜间二进制 PIP 轮盘获得。有关英特尔 GPU 支持的更多信息，请参阅[文档](https://pytorch.org/docs/main/notes/get_start_xpu.html)。
