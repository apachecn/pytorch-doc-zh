# 新版本: PyTorch 2.5 发布博客

> 发布: 2024年10月17日
> 
> 译者：[@片刻](https://github.com/jiangzhonglian)
> 
> 原文: <https://pytorch.org/blog/pytorch2-6>
> 
> 翻译: <https://pytorch.apachecn.org/LatestChanges/PyTorch_V2.6>

**来自 PyTorch团队**

我们很高兴地宣布 PyTorch® 2.6 的发布（[发布说明](https://github.com/pytorch/pytorch/releases/tag/v2.6.0)）！此版本对 PT2 进行了多项改进：`torch.compile`现在可以与 Python 3.13 一起使用；新的性能相关旋钮`torch.compiler.set_stance`；多项 AOTInductor 增强功能。除了 PT2 改进之外，另一个亮点是 X86 CPU 上的 FP16 支持。

注意：从此版本开始，我们将不会在 Conda 上发布，请参阅[[公告] 弃用 PyTorch 的官方 Anaconda 频道](https://github.com/pytorch/pytorch/issues/138506)了解详情。

对于此版本，随 CUDA 12.6.3（以及 Linux Aarch64、Linux ROCm 6.2.4 和 Linux XPU 二进制文件）一起提供的实验性 Linux 二进制文件是使用 CXX11_ABI=1 构建的，并[使用 Manylinux 2.28 构建平台](https://dev-discuss.pytorch.org/t/pytorch-linux-wheels-switching-to-new-wheel-build-platform-manylinux-2-28-on-november-12-2024/2581)。如果您使用自定义 C++ 或 CUDA 扩展构建 PyTorch 扩展，请更新这些版本以使用 CXX_ABI=1 并报告您发现的任何问题。对于下一个 PyTorch 2.7 版本，我们计划将所有 Linux 版本切换到 Manylinux 2.28 和 CXX11_ABI=1，请参阅[[RFC] PyTorch 下一个轮子构建平台：manylinux-2.28](https://github.com/pytorch/pytorch/issues/123649)了解详细信息和讨论。

此外，作为一项重要的安全改进措施，我们在此版本中更改了`weights_only`参数的默认值`torch.load`。这是一个破坏向后兼容性的更改，请参阅[此论坛帖子](https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573)了解更多详细信息。

自 PyTorch 2.5 以来，此版本由 520 位贡献者提交的 3892 份提交组成。我们衷心感谢我们敬业的社区所做的贡献。与往常一样，我们鼓励您尝试这些功能并在我们改进 PyTorch 时报告任何问题。有关如何开始使用 PyTorch 2 系列的更多信息，请参阅我们的[入门](https://pytorch.org/get-started/pytorch-2.0/)页面。

| **Beta** | **Prototype** | **Performance Improvements** |
| --- | --- | --- |
| torch.compiler.set_stance | 改进了 Intel GPU 上的 PyTorch 用户体验 | |
| torch.library.triton_op | X86 CPU 上的 LLM FlexAttention 支持 | |
| torch.compile 支持 Python 3.13 | Dim.AUTO | |
| AOTInductor 的新封装 API | 用于 AOTInductor 的 CUTLASS 和 CK GEMM/CONV 后端 | |
| AOTInductor：最小化器 | | |
| AOTInductor：ABI 兼容模式代码生成 | | |
| FP16 对 X86 CPU 的支持 | | |

* 要查看公开功能提交的完整列表，请单击[此处](https://docs.google.com/spreadsheets/d/1TzGkWuUMF1yTe88adz1dt2mzbIsZLd3PBasy588VWgk/edit?usp=sharing)。

## 测试版功能

### [Beta] torch.compiler.set_stance

此功能使用户能够指定`torch.compile`在编译函数的不同调用之间可以采取的不同行为（“立场”）。例如，其中一个立场是

“eager_on_recompile” 指示 PyTorch 在需要重新编译时急切地进行编码，并在可能的情况下重用缓存的编译代码。

有关更多信息，请参阅[set_stance 文档](https://pytorch.org/docs/2.6/generated/torch.compiler.set_stance.html#torch.compiler.set_stance)和[使用 torch.compiler.set_stance 的动态编译控制](https://pytorch.org/tutorials/recipes/torch_compiler_set_stance_tutorial.html)教程。

### [Beta] torch.library.triton_op

`torch.library.triton_op`提供了一种创建由用户定义的 triton 内核支持的自定义运算符的标准方法。

当用户将用户定义的 triton 内核转变为自定义运算符时，`torch.library.triton_op`可以`torch.compile`窥视实现，从而`torch.compile`优化其中的 triton 内核。

有关更多信息，请参阅[triton_op 文档](https://pytorch.org/docs/2.6/library.html#torch.library.triton_op)和[使用用户定义的 Triton 内核和 torch.compile](https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)教程。

### [Beta] torch.compile 支持 Python 3.13

`torch.compile`之前仅支持 Python 3.12 及以下版本。用户现在可以使用`torch.compile`Python 3.13 来优化模型。

### [Beta] AOTInductor 的新封装 API

引入了一种新的包格式“ [PT2 存档](https://docs.google.com/document/d/1RQ4cmywilnFUT1VE-4oTGxwXdc8vowCSZsrRgo3wFA8/edit?usp=sharing)”。这基本上包含 AOTInductor 需要使用的所有文件的 zip 文件，并允许用户将所需的所有内容发送到其他环境。还有将多个模型打包成一个工件的功能，以及将其他元数据存储在包内的功能。

有关更多详细信息，请参阅更新的[Python 运行时 torch.export AOTInductor 教程](https://pytorch.org/tutorials/recipes/torch_export_aoti_python.html)。

### [Beta] AOTInductor：最小化器

如果用户在使用 AOTInductor API 时遇到错误，AOTInductor Minifier 允许创建一个重现该错误的最小 nn.Module。

有关更多信息，请参阅[AOTInductor Minifier 文档](https://pytorch.org/docs/2.6/torch.compiler_aot_inductor_minifier.html)。

### [Beta] AOTInductor：ABI 兼容模式代码生成

AOTInductor 生成的模型代码依赖于 Pytorch cpp 库。由于 Pytorch 发展迅速，因此确保之前由 AOTInductor 编译的模型能够继续在较新的 Pytorch 版本上运行非常重要，即 AOTInductor 向后兼容。

为了保证应用程序二进制接口 (ABI) 向后兼容，我们在 libtorch 中精心定义了一组稳定的 C 接口，并确保 AOTInductor 生成的代码仅引用特定的 API 集，而不引用 libtorch 中的其他内容。我们将保持 C API 集在 Pytorch 版本之间稳定，从而为 AOTInductor 编译的模型提供向后兼容性保证。

### [Beta] FP16 支持 X86 CPU（eager 模式和 Inductor 模式）

Float16 数据类型通常用于在 AI 推理和训练中减少内存使用量并加快计算速度。最近推出的[带有 P-Cores 的 Intel® Xeon® 6](https://www.intel.com/content/www/us/en/products/details/processors/xeon/xeon6-p-cores.html?__hstc=132719121.160a0095c0ae27f8c11a42f32744cf07.1739101052423.1739101052423.1739104196345.2&__hssc=132719121.1.1739104196345&__hsfp=2543667465)等 CPU通过原生加速器[AMX](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html?__hstc=132719121.160a0095c0ae27f8c11a42f32744cf07.1739101052423.1739101052423.1739104196345.2&__hssc=132719121.1.1739104196345&__hsfp=2543667465)支持 Float16 数据类型。X86 CPU 上的 Float16 支持在 PyTorch 2.5 中作为原型功能引入，现在它已针对 eager 模式和 Torch.compile + Inductor 模式进行了进一步改进，使其成为 Beta 级功能，其功能和性能均已通过广泛的工作负载验证。

## 原型特点

### [Prototype] 改进了 Intel GPU 上的 PyTorch 用户体验

通过简化安装步骤、Windows 版本二进制分发和扩展支持的 GPU 模型（包括最新的英特尔® Arc™ B 系列独立显卡），英特尔 GPU 上的 PyTorch 用户体验得到进一步改善。希望在英特尔®[酷睿™ 超人工智能 PC](https://www.intel.com/content/www/us/en/products/docs/processors/core-ultra/ai-pc.html?__hstc=132719121.160a0095c0ae27f8c11a42f32744cf07.1739101052423.1739101052423.1739104196345.2&__hssc=132719121.1.1739104196345&__hsfp=2543667465)和[英特尔® Arc™ 独立显卡](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc.html?__hstc=132719121.160a0095c0ae27f8c11a42f32744cf07.1739101052423.1739101052423.1739104196345.2&__hssc=132719121.1.1739104196345&__hsfp=2543667465)上使用 PyTorch 模型进行微调、推理和开发的应用程序开发人员和研究人员现在可以直接使用适用于 Windows、Linux 和适用于 Linux 2 的 Windows 子系统的二进制版本安装 PyTorch。

*   简化的英特尔 GPU 软件堆栈设置，支持一键安装 torch-xpu PIP 轮子，以开箱即用的方式运行深度学习工作负载，消除了安装和激活英特尔 GPU 开发软件包的复杂性。
*   适用于英特尔 GPU 的 torch core、torchvision 和 torchaudio 的 Windows 二进制版本已推出，支持的 GPU 型号已从搭载英特尔® Arc™ 显卡的英特尔® 酷睿™ 超级处理器、[搭载英特尔® Arc™ 显卡的英特尔® 酷睿™ 超级系列 2](https://www.intel.com/content/www/us/en/products/details/processors/core-ultra.html?__hstc=132719121.160a0095c0ae27f8c11a42f32744cf07.1739101052423.1739101052423.1739104196345.2&__hssc=132719121.1.1739104196345&__hsfp=2543667465)和[英特尔® Arc™ A 系列显卡](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/desktop/a-series/overview.html?__hstc=132719121.160a0095c0ae27f8c11a42f32744cf07.1739101052423.1739101052423.1739104196345.2&__hssc=132719121.1.1739104196345&__hsfp=2543667465)扩展到最新的 GPU 硬件[英特尔® Arc™ B 系列显卡](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/desktop/b-series/overview.html?__hstc=132719121.160a0095c0ae27f8c11a42f32744cf07.1739101052423.1739101052423.1739104196345.2&__hssc=132719121.1.1739104196345&__hsfp=2543667465)。
*   进一步增强了使用 SYCL\* 内核的英特尔 GPU 上 Aten 运算符的覆盖范围，以实现流畅的急切模式执行，以及对英特尔 GPU 上的 torch.compile 进行错误修复和性能优化。

有关英特尔 GPU 支持的更多信息，请参阅[入门指南](https://pytorch.org/docs/main/notes/get_start_xpu.html)。

### [Prototype] X86 CPU 上的 LLM FlexAttention 支持

FlexAttention 最初是在 PyTorch 2.5 中引入的，旨在通过灵活的 API 为 Attention 变体提供优化的实现。在 PyTorch 2.6 中，通过 TorchInductor CPP 后端添加了对 FlexAttention 的 X86 CPU 支持。此新功能利用并扩展了当前的 CPP 模板功能，以基于现有的 FlexAttention API 支持广泛的注意变体（例如：PageAttention，这对于 LLM 推理至关重要），并在 x86 CPU 上带来优化的性能。借助此功能，可以轻松使用 FlexAttention API 在 CPU 平台上编写 Attention 解决方案并获得良好的性能。

### [Prototype] Dim.AUTO

`Dim.AUTO`允许使用自动动态形状`torch.export`。用户可以导出`Dim.AUTO` 并“发现”其模型的动态行为，其中自动推断最小/最大范围、维度之间的关系以及静态/动态行为。

与现有的用于指定动态形状的命名 Dims 方法相比，这是一种更加用户友好的体验，它要求用户在导出时充分了解其模型的动态行为。`Dim.AUTO`允许用户编写不依赖于模型的通用代码，从而增加了使用动态形状导出的易用性。

请参阅[torch.export 教程](https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html#constraints-dynamic-shapes)以获取更多信息。

### [Prototype] 用于 AOTInductor 的 CUTLASS 和 CK GEMM/CONV 后端

CUTLASS 和 CK 后端为 Inductor 中的 GEMM 自动调节添加了内核选项。此功能现在也可在 AOTInductor 中使用，可在 C++ 运行时环境中运行。这两个后端的一项重大改进是通过消除冗余内核二进制编译和动态形状支持来提高编译时速度。