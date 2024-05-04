# 新版本: PyTorch 2.2 版本，FlashAttention-v2 集成，AOTInductor

> 发布: 2024年01月30日
> 
> 译者：[@片刻](https://github.com/jiangzhonglian)
> 
> 原文: <https://pytorch.org/blog/pytorch2-2>
> 
> 翻译: <https://pytorch.apachecn.org/LatestChanges/PyTorch_V2.2>

**来自 PyTorch团队**

我们很高兴地宣布PyTorch® 2.2（[发布说明](https://github.com/pytorch/pytorch/releases/tag/v2.2.0)）的发布！PyTorch 2.2通过[FlashAttention-v2](https://arxiv.org/abs/2307.08691)集成以及*AOTInductor*（一种为非python服务器端部署构建的新的提前编译和部署工具）提供了约2倍的性能改进，以达到*[scaledot_product_attention](https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html)*。

此版本还包括对优化器的改进的*torch.compile*支持，一些新的电感优化，以及名为TORCH_LOGS的新日志记录机制。

请注意，我们正在[弃用macOS x86支持](https://github.com/pytorch/pytorch/issues/114602)，PyTorch 2.2.x将是支持macOS x64的最后一个版本。

除了2.2，我们还发布了对PyTorch域库的一系列更新。更多详细信息可以在图书馆更新博客中找到。

自PyTorch 2.1以来，此版本由3628个提交和521个贡献者组成。我们要真诚地感谢我们敬业的社区的贡献。一如既往，我们鼓励您尝试这些，并报告任何问题，因为我们改进了2.2。有关如何开始使用PyTorch 2系列的更多信息，请访问我们的[入门](https://pytorch.org/get-started/pytorch-2.0/)页面。

总结：

*   *[scaled_dot_product_attention](https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html)*（SDPA）现在支持*[FlashAttention-2](https://arxiv.org/abs/2307.08691)*，与之前的版本相比，速度提高了约2倍。
*   PyTorch 2.2引入了[TorchInductor](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)的一个新的提前扩展，称为*[AOTInductor](https://pytorch.org/docs/main/torch.compiler_aot_inductor.html)*，旨在为非python服务器端编译和部署PyTorch程序。
*   *torch.distributed*支持一种名为*[device_mesh](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html)*的新抽象，用于初始化和表示ProcessGroups。
*   PyTorch 2.2提供了一种标准化、可配置的日志记录机制，称为[TORCH_LOGS](https://pytorch.org/tutorials/recipes/torch_logs.html)。
*   PyTorch 2.2中包含一些*torch.compile*改进，包括改进了对编译优化器的支持，以及改进了TorchInductor融合和布局优化。
*   请注意，我们正在[弃用macOS x86支持](https://github.com/pytorch/pytorch/issues/114602)，PyTorch 2.2.x将是支持macOS x64的最后一个版本。



| **Stable** | **Beta** | **Performance Improvements** |
| --- | --- | --- |
| | FlashAttention-2集成 | Inductor 优化 |
| | AOTInductor | aarch64 优化 |
| | TORCH_LOGS | |
| | device_mesh | |
| | 优化器编译 | |

* 要查看公共功能提交的完整列表，请单击[此处](https://docs.google.com/spreadsheets/d/1TzGkWuUMF1yTe88adz1dt2mzbIsZLd3PBasy588VWgk/edit?usp=sharing)。

## 测试版功能

### [Beta] FlashAttention-2在*torch.nn.functional.scaled_dot_product_attention中*的支持

*[torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html)*（SDPA）现在支持FlashAttention-2，速度约为2倍（与之前版本相比），并在A100 GPU上达到理论最大FLOP/s的约50-73%。

[本文](https://arxiv.org/abs/2307.08691)提供了有关FlashAttention-2的更多信息。

有关如何使用SDPA的教程，请参阅[本教程](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)。

### [Beta] AOTInductor：torch.export-ed程序的提前编译和部署

AOTInductor是[TorchInductor](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)的扩展，旨在处理导出的PyTorch模型，优化它们，并生成共享库以及其他相关工件。这些编译的工件可以部署在非Python环境中，这些环境经常用于服务器端的推理。请注意，AOTInductor支持与Inductor相同的后端，包括CUDA、ROCm和CPU。

有关更多信息，请参阅[AOTInductor教程](https://pytorch.org/docs/main/torch.compiler_aot_inductor.html)。

### [Beta]通过TORCH_LOGS进行细粒度的可配置日志记录

PyTorch现在提供一个标准化的、可配置的日志记录机制，可用于分析各种子系统的状态，如编译和分布式操作。

日志可以通过TORCH_LOGS环境变量启用。例如，要将TorchDynamo的日志级别设置为logging.ERROR，将TorchInductor的日志级别设置为logging.DEBUG，请将*TORCH_LOGS=”-dynamo,+inductor”*传递给PyTorch。

有关更多信息，请参阅日志记录[文档](https://pytorch.org/docs/2.2/logging.html)和[教程](https://pytorch.org/tutorials/recipes/torch_logs.html)。

### [Beta] torch.distributed.device_mesh

PyTorch 2.2引入了一种新的抽象，用于表示分布式并行性所涉及的进程组，称为*torch.distributed.device_mesh*。这种抽象允许用户通过N维数组表示节点间和节点内进程组，例如，一个维度可以在FSDP中表示数据并行性，而另一个维度可以表示FSDP中的张量并行性。

有关更多信息，请参阅 [device_mesh 教程](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html)。

### [Beta]对*torch.compile*-ing优化器的改进

对torch.compile-ing优化器进行了一些改进，包括减少开销和对cuda图的支持。

有关改进的更多技术细节可在[dev-discuss](https://dev-discuss.pytorch.org/t/compiling-the-optimizer-with-pt2/1669)上找到，并且可以[在这里](https://pytorch.org/tutorials/recipes/compiling_optimizer.html)找到*torch.compile*-ing优化器的配方。

## 性能改进

### Inductor性能优化

TorchInductor中添加了许多性能优化，包括[对torch.concat的水平融合支持](https://github.com/pytorch/pytorch/pull/111437)，[改进了卷积布局优化](https://github.com/pytorch/pytorch/pull/114600)，以及改进了*scaled_dot_product_attention*[模式](https://github.com/pytorch/pytorch/pull/109156)[匹配](https://github.com/pytorch/pytorch/pull/110001)。

有关Inductor优化的完整列表，请参阅[发布说明](https://github.com/pytorch/pytorch/tree/v2.2.0)。

### aarch64性能优化

PyTorch 2.2包括aarch64的一些性能增强功能，包括支持[mkldnn权重预打包](https://github.com/pytorch/pytorch/pull/115037/files)，改进的[ideep](https://github.com/intel/ideep)[原始缓存](https://github.com/intel/ideep/pull/261)，以及通过对[OneDNN的](https://github.com/oneapi-src/oneDNN/)[固定格式内核改进](https://github.com/oneapi-src/oneDNN/pull/1590)提高推理速度。

有关aarch64优化的完整列表，请参阅[发布说明](https://github.com/pytorch/pytorch/tree/v2.2.0)。
