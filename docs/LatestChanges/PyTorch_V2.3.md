# 新版本: PyTorch 2.3 版本，公告

> 发布: 2024年04月24日
> 
> 译者：[@片刻](https://github.com/jiangzhonglian)
> 
> 原文: <https://pytorch.org/blog/pytorch2-3>
> 
> 翻译: <https://pytorch.apachecn.org/LatestChanges/PyTorch_V2.3>

**来自 PyTorch团队**

我们很高兴地宣布PyTorch® 2.3（[发布说明](https://github.com/pytorch/pytorch/releases/tag/v2.3.0)）的发布！PyTorch 2.3支持torch.compile中用户定义的Triton内核，允许用户从渴望中迁移自己的Triton内核，而无需经历性能回归或图形中断。张量并行性改进了使用原生PyTorch函数训练大型语言模型的体验，该函数已在100B参数模型的训练运行中进行了验证。此外，半结构化稀疏实现了半结构化稀疏作为张量子类，在密集矩阵乘法上观察到的速度高达1.6。

自PyTorch 2.2以来，此版本由3393个提交和426个贡献者组成。我们要真诚地感谢我们敬业的社区的贡献。一如既往，我们鼓励您尝试这些，并报告任何问题，因为我们改进了2.3。有关如何开始使用PyTorch 2系列的更多信息，请访问我们的[入门](https://pytorch.org/get-started/pytorch-2.0/)页面。

| **Beta** | **Prototype** | **Performance Improvements** |
| --- | --- | --- |
| torch.compile 中用户定义的 Triton内核 | torch.export 添加了新的API来指定 dynamic_shapes | 仅权重量化化引入 inductor CPU后端 |
| PyTorch分布式中的 Tensor 并行性 | 异步 checkpoint 生成 | |
| 支持半结构化稀疏性 | | |

* 要查看公共功能提交的完整列表，请单击[此处](https://docs.google.com/spreadsheets/d/1TzGkWuUMF1yTe88adz1dt2mzbIsZLd3PBasy588VWgk/edit?usp=sharing)。

## 测试版功能

### [Beta]支持 *torch.compile* 中用户定义的Triton内核

允许使用torch.compile原生执行包含三通内核的PyTorch代码。这使用户能够将包含三元内核的代码从渴望的PyTorch迁移到*torch.compile*，而不会遇到性能回归或图表中断。原生支持还为Torch Inductor预编译用户定义的Triton内核以及更好地围绕Triton内核组织代码创造了机会，从而进行进一步的优化。

您可以在[本教程](https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)中找到有关如何在torch.compile中使用用户定义的Triton内核的更多信息。

### [Beta]张量并行性引入了更有效的方法来训练LLM

张量并行API促进了GPU/主机之间的各种张量操作，并与FSDP集成进行二维并行（跨设备的张量并行+跨主机的数据并行）。它还提供了一个用于构建更高级Tensor并行API的低级API。该API已经过验证，以支持具有超过1000亿个参数的变压器模型的训练。

您可以在[本教程](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)中找到有关如何在工作流程中利用此的更多信息。

### [Beta]半结构化稀疏性为用户提供了一种利用加速稀疏推理和内存节省的方法

*torch.sparse.SparseSemiStructuredTensor*将半结构化稀疏作为张量子类，该子类在密集矩阵乘法上观察到高达1.6的速度。

特别是，它补充说：

*   量化可组合性的额外支持（混合d型、去定量融合）
*   更新了cuSPARSELt和CUTLASS内核
*   torch.compile支持

您可以[在这里](https://pytorch.org/tutorials/advanced/semi_structured_sparse.html)找到有关如何利用半结构化稀疏性的更多信息。

## 原型功能

### [Prototype] *torch.export*添加了新的API来指定*dynamic_shapes*

您现在可以使用*torch.export.Dim*更好地表示动态形状，方法是允许开发人员指定范围（最小和最大值），这些范围可以在不同的输入维度上重复使用，这些维度被限制为相等。

要了解有关*torch.export.Dim*的更多信息，以及如何使用它来表达更有趣的关系（如线性算术表达式），请查看[此处](https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html#constraints-dynamic-shapes)的教程。

### [Prototype]异步检查点生成

异步检查点生成允许用户在生成检查点时继续他们的训练循环，基本上卸载了大部分检查点成本。

通过这个[例子](https://github.com/pytorch/pytorch/blob/release/2.3/torch/distributed/checkpoint/examples/async_checkpointing_example.py)，您可以了解如何在自己的工作流程中利用这一点。

## 性能改进

### [Prototype]在 inductor CPU后端引入仅权重量化

PyTorch 2.3增强了火炬电感CPU后端的LLM推理性能。该项目[gpt-fast](https://github.com/pytorch-labs/gpt-fast)为使用*torch.compile*的变压器文本生成提供了简单高效的PyTorch原生加速。在2.3之前，只支持CUDA设备，此功能通过为int4和int8权重仅量化线性提供高度优化的内核来支持CPU对应。

有关更多信息/如何使用此功能，请参阅[gpt-fast README](https://github.com/pytorch-labs/gpt-fast#quantization)。
