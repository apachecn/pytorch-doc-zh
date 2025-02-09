# 新版本: PyTorch 2.4 发布博客

> 发布: 2024年07月24日
> 
> 译者：[@片刻](https://github.com/jiangzhonglian)
> 
> 原文: <https://pytorch.org/blog/pytorch2-4>
> 
> 翻译: <https://pytorch.apachecn.org/LatestChanges/PyTorch_V2.4>

**来自 PyTorch团队**

我们很高兴地宣布 PyTorch® 2.4 的发布（[发布说明](https://github.com/pytorch/pytorch/releases/tag/v2.4.0)）！PyTorch 2.4 增加了对最新版本 Python (3.12) 的支持`torch.compile`。AOTInductor 冻结通过允许序列化 MKLDNN 权重，为运行 AOTInductor 的开发人员提供了更多基于性能的优化。此外，`libuv`还引入了新的默认 TCPStore 服务器后端利用，这应该可以显著减少运行大规模作业的用户的初始化时间。最后，新的 Python 自定义运算符 API 使将自定义内核集成到 PyTorch 中比以前更容易，尤其是对于`torch.compile`。

自 PyTorch 2.3 以来，此版本由 3661 次提交和 475 位贡献者组成。我们要真诚地感谢我们敬业的社区所做的贡献。与往常一样，我们鼓励您尝试这些功能并在我们改进 2.4 时报告任何问题。有关如何开始使用 PyTorch 2 系列的更多信息，请参阅我们的[入门](https://pytorch.org/get-started/pytorch-2.0/)页面。


| **Beta** | **Prototype** | **Performance Improvements** |
| --- | --- | --- |
| Python 3.12 支持 torch.compile | FSDP2：基于 DTensor 的每参数分片 FSDP | 针对 AWS Graviton (aarch64-linux) 处理器的 torch.compile 优化 |
| AOTInductor 冻结 CPU | torch.distributed.pipelining，简化的管道并行性 | TorchInductor 中的 BF16 符号形状优化 |
| 新的高级 Python 自定义运算符 API | 可通过源代码构建获得英特尔 GPU | 利用 CPU 设备对 GenAI 项目进行性能优化 |
| 将 TCPStore 的默认服务器后端切换为 libuv | | |

* 要查看公开功能提交的完整列表，请单击[此处](https://docs.google.com/spreadsheets/d/1TzGkWuUMF1yTe88adz1dt2mzbIsZLd3PBasy588VWgk/edit?usp=sharing)。

## Beta 版功能

### [Beta] Python 3.12 支持*torch.compile*

`torch.compile()`之前仅支持 Python **3.8-3.11**，现在用户可以使用`torch.compile()`Python **3.12**来优化模型。

### [Beta] AOTInductor CPU 冻结

此功能允许用户在 CPU 上使用 AOTInductor 时打开冻结标志。借助此功能，AOTInductor 可以覆盖相同的操作场景，并达到与 Inductor CPP 后端相当的性能。在此支持之前，当模型包含 MKLDNN 运算符（涉及计算密集型运算符，例如卷积、线性、ConvTranspose 等）并且冻结处于开启状态时，这些模型将无法运行，因为 AOTInductor 不支持序列化具有不透明格式的 MKLDNN 权重。

工作流程如 AOTInductor[教程](https://pytorch.org/docs/main/torch.compiler_aot_inductor.html)中所述，此外，用户现在可以添加冻结标志以获得更好的性能：

```
export TORCHINDUCTOR_FREEZING=1
```

### [Beta] 新的高级 Python 自定义运算符 API

我们添加了一个新的更高级别的 Python 自定义运算符 API，这使得使用自定义运算符扩展 PyTorch 比以前更容易，这些运算符的行为类似于 PyTorch 的内置运算符。使用[新的高级 torch.library API](https://pytorch.org/docs/2.4/library.html#module-torch.library)注册的运算符保证与其他 PyTorch 子系统兼容；使用以前的[低级 torch.library API](https://pytorch.org/docs/2.4/library.html#low-level-apis)`torch.compile`在 Python 中编写自定义运算符需要深入了解 PyTorch 内部结构，并且有很多障碍。

请参阅[教程](https://pytorch.org/tutorials/advanced/python_custom_ops.html)以了解更多信息。

### [Beta] 将 TCPStore 的默认服务器后端切换为*libuv*

引入了 TCPStore 的新默认服务器后端，`libuv`该后端应能显著缩短初始化时间并提高可扩展性。在处理大规模作业时，这应该能让用户受益，启动时间大大缩短。

有关动机 + 后备指令的更多信息，请参阅本[教程](https://pytorch.org/tutorials/intermediate/TCPStore_libuv_backend.html)。

## 原型特点

### [Prototype] FSDP2：基于 DTensor 的每参数分片 FSDP

FSDP2 是一种新的完全分片数据并行实现，它使用 dim-0 每个参数分片来解决 FSDP1 的平面参数分片的基本可组合性挑战。

有关 FSDP2 动机/设计的更多信息，请参阅[Github 上的 RFC](https://github.com/pytorch/pytorch/issues/114299)。

### [Prototype] *torch.distributed.pipelining*，简化的管道并行性

流水线并行是深度学习的基本并行技术之一。它允许对模型的执行进行分区，以便多个微批次可以同时执行模型代码的不同部分。

`torch.distributed.pipelining`提供了一个工具包，允许在通用模型上轻松实现管道并行，同时还提供与其他常见 PyTorch 分布式功能（如 DDP、FSDP 或张量并行）的可组合性。

有关更多信息，请参阅我们的[文档](https://pytorch.org/docs/main/distributed.pipelining.html)和[教程](https://pytorch.org/tutorials/intermediate/pipelining_tutorial.html)。

### [Prototype] 可通过源代码构建获得英特尔 GPU

Linux 系统上的 PyTorch 中的英特尔 GPU 在英特尔® 数据中心 GPU Max 系列上提供了基本功能：eager 模式和 torch.compile。

对于 eager 模式，常用的 Aten 运算符使用 SYCL 编程语言实现。最关键性能的图形和运算符通过使用 oneAPI 深度神经网络 (oneDNN) 进行了高度优化。对于 torch.compile 模式，英特尔 GPU 后端集成到 Triton 上的 Inductor 中。

有关英特尔 GPU 源构建的更多信息，请参阅我们的[博客文章](https://www.intel.com/content/www/us/en/developer/articles/technical/pytorch-2-4-supports-gpus-accelerate-ai-workloads.html?__hstc=132719121.160a0095c0ae27f8c11a42f32744cf07.1739101052423.1739101052423.1739101052423.1&__hssc=132719121.4.1739101052423&__hsfp=2543667465)和[文档](https://pytorch.org/docs/main/notes/get_start_xpu.html)。

## 性能改进

### 针对 AWS Graviton (aarch64-linux) 处理器的*torch.compile优化*

AWS 针对 AWS Graviton3 处理器优化了 PyTorch torch.compile 功能。与基于 AWS Graviton3 的 Amazon EC2 实例上的多个自然语言处理 (NLP)、计算机视觉 (CV) 和推荐模型的默认 Eager 模式推理相比，此优化使 Hugging Face 模型推理的性能提高了 2 倍（基于 33 个模型的性能改进的几何平均值），TorchBench 模型推理的性能提高了 1.35 倍（45 个模型的性能改进的几何平均值）。

有关具体技术细节的更多信息，请参阅[博客文章](https://pytorch.org/blog/accelerated-pytorch-inference/)。

### TorchInductor 中的 BF16 符号形状优化

Pytorch 用户现在可以借助 beta BF16 符号形状支持体验到质量和性能的提升。虽然静态形状与符号形状相比可以提供额外的优化机会，但对于诸如具有不同批量大小和序列长度的推理服务或具有数据相关输出形状的检测模型等场景而言，它还不够。

使用 TorchBench、Huggingface 和 timms_model 进行验证，结果显示通过率与 BF16 静态形状场景相似，加速效果也相当。结合符号形状的优势、英特尔 CPU 提供的 BF16 AMX 指令硬件加速以及 PyTorch 2.4 中适用于静态和符号形状的通用 Inductor CPU 后端优化，BF16 符号形状的性能与 PyTorch 2.3 相比有显著提升。

使用该功能的 API：

```
model = ….
model.eval()
with torch.autocast(device_type=”cpu”, dtype=torch.bfloat16), torch.no_grad():
   compiled_model = torch.compile(model, dynamic=True)
```

### 利用 CPU 设备对 GenAI 项目进行性能优化

重点介绍 PyTorch 在 CPU 上的增强性能，如通过对[“Segment Anything Fast”](https://github.com/pytorch-labs/segment-anything-fast)和[“Diffusion Fast”](https://github.com/huggingface/diffusion-fast)项目的优化所展示的。但是，模型仅支持 CUDA 设备。我们在项目中加入了 CPU 支持，使用户能够利用 CPU 的增强功能来运行项目的实验。同时，我们还[为 SDPA 采用了块式注意掩码](https://github.com/pytorch/pytorch/pull/126961)，这可以显著减少峰值内存使用量并提高性能。我们还优化了 Inductor CPU 中的一系列[布局传播规则，](https://github.com/pytorch/pytorch/pull/126961)以提高性能。

为了方便使用，我们更新了 README 文件。使用此功能的 API 如下所示，只需`--device cpu`在命令行中提供即可：

*   对于快速细分任何内容：
    
    ```
    export SEGMENT_ANYTHING_FAST_USE_FLASH_4=0
    python run_experiments.py 16 vit_b <pytorch_github> <segment-anything_github>
    <path_to_experiments_data> --run-experiments --num-workers 32 --device cpu
    ```
    
*   对于扩散快：
    
    ```
    python run_benchmark.py --compile_unet --compile_vae --enable_fused_projections --device=cpu
    ```
    

用户可以按照指南运行实验并亲自观察性能改进，以及探索 FP32 和 BF16 数据类型的性能改进趋势。

此外，用户可以使用和 SDPA 获得良好的性能`torch.compile`。通过观察这些不同因素下的性能趋势，用户可以更深入地了解各种优化如何增强 PyTorch 在 CPU 上的性能。