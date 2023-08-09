# 新版本: PyTorch 1.8版本，包括编译器和分布式训练更新，以及新的移动端教程

> 发布: 2021年03月04日
> 
> 译者：[@片刻](https://github.com/jiangzhonglian)
> 
> 原文: <https://pytorch.org/blog/pytorch-1.8-released/>
> 
> 翻译: <https://pytorch.apachecn.org/LatestChanges/PyTorch_V1.8>

**来自 PyTorch团队**

我们很高兴地宣布PyTorch 1.8的可用性。自1.7以来，此版本由3000多个提交组成。它包括编译、代码优化、科学计算前端API的主要更新和新功能，以及通过pytorch.org提供的二进制文件支持AMD ROCm。它还为管道和模型并行性以及梯度压缩的大规模训练提供了改进的功能。一些亮点包括：

1.  支持通过`torch.fx`进行python到python函数转换；
2.  添加或稳定API以支持FFT（`torch.fft`）、线性代数函数（`torch.linalg`），增加了对复杂张量的autograd和支持，并进行了更新，以提高计算hessian和jacobians的性能；以及
3.  分布式训练的重大更新和改进包括：改进NCCL可靠性；管道并行支持；RPC分析；以及支持添加梯度压缩的通信钩子。在[此处](https://github.com/pytorch/pytorch/releases)查看完整的发布说明。

除了1.8，我们还发布了PyTorch库的重大更新，包括[TorchCSPRNG](https://github.com/pytorch/csprng)、[TorchVision](https://github.com/pytorch/vision)、[TorchText](https://github.com/pytorch/text)和[TorchAudio](https://github.com/pytorch/audio)。有关图书馆版本的更多信息，请参阅[此处](http://pytorch.org/blog/pytorch-1.8-new-library-releases)的帖子。如前所述，PyTorch版本中的功能分为稳定、测试版和原型。您可以在[此处](https://pytorch.org/blog/pytorch-feature-classification-changes/)的帖子中了解有关定义的更多信息。

## 新的和更新的API

PyTorch 1.8版本带来了大量新的和更新的API表面，包括NumPy兼容性的其他API，还支持在推理和训练时间改进和扩展代码性能的方法。以下是此版本中主要功能的简要摘要：

### \[Stable\] `Torch.fft`支持高性能NumPy风格的FFT

作为PyTorch支持科学计算的目标的一部分，我们投资改善了FFT支持，通过PyTorch 1.8，我们发布了`torch.fft`模块。该模块实现了与NumPy的`np.fft`模块相同的功能，但支持硬件加速和自动升级。

*   有关更多详细信息，请参阅此[博客文章](https://pytorch.org/blog/the-torch.fft-module-accelerated-fast-fourier-transforms-with-autograd-in-pyTorch/)
*   [文稿](https://pytorch.org/docs/1.8.0/fft.html)

### \[Beta\] 支持NumPy风格的线性代数函数`torch.linalg`

`torch.linalg`模块以NumPy的[np.linalg](https://numpy.org/doc/stable/reference/routines.linalg.html?highlight=linalg#module-numpy.linalg)模块为模型，为常见的线性代数运算带来了NumPy风格的支持，包括Cholesky分解、行列式、特征值和许多其他运算。

*   [文稿](https://pytorch.org/docs/1.8.0/linalg.html)

### \[BETA\] 使用FX进行PYTHON代码转换

FX允许您编写表单`transform(input_module : nn.Module)`\-> `nn.Module`的转换，您可以在其中输入`Module`实例并从中获取转换后的`Module`实例。

这种功能适用于许多场景。例如，基于FX的图形模式量化产品正在与FX同时作为原型发布。图形模式量化通过利用FX的程序捕获、分析和转换设施来自动化神经网络量化过程。我们还在与FX一起开发许多其他转型产品，我们很高兴与社区分享这个强大的工具包。

由于FX转换消耗和生成nn.Module实例，因此它们可以在许多现有的PyTorch工作流中使用。这包括工作流，例如，在Python中训练，然后通过TorchScript部署。

您可以在官方[文档](https://pytorch.org/docs/master/fx.html)中阅读有关外汇的更多信息。您还可以[在这里](https://github.com/pytorch/examples/tree/master/fx)找到几个使用`torch.fx`实现的程序转换的示例。我们不断改进FX，并邀请您在[论坛](https://discuss.pytorch.org/)或[问题跟踪器](https://github.com/pytorch/pytorch/issues)上分享您对工具包的任何反馈。

我们感谢[TorchScript](https://pytorch.org/docs/stable/jit.html)跟踪、[Apache MXNet](https://mxnet.apache.org/versions/1.7.0/)杂交以及最近的[JAX](https://github.com/google/jax)对通过跟踪进行程序获取的影响。我们还想感谢[Caffe2](https://caffe2.ai/)、[JAX](https://github.com/google/jax)和[TensorFlow](https://www.tensorflow.org/)作为简单、有向的数据流图形程序表示和转换对这些表示的值的启发。

## 分布式训练

PyTorch 1.8版本增加了许多新功能，并改进了可靠性和可用性。具体来说，支持：添加了[稳定级别的异步错误/超时处理](https://pytorch.org/docs/stable/distributed.html?highlight=init_process_group#torch.distributed.init_process_group)，以提高NCCL的可靠性；以及对[基于RPC的剖析](https://pytorch.org/docs/stable/rpc.html)的稳定支持。此外，我们通过使用DDP中的通信钩子，增加了对管道并行性以及梯度压缩的支持。详情如下：

### \[Beta\] 管道并行性

随着机器学习模型的规模不断扩大，传统的分布式数据并行（DDP）训练不再扩展，因为这些模型不适合单个GPU设备。新的管道并行功能提供了一个易于使用的PyTorch API，以利用管道并行作为训练循环的一部分。

*   [RFC](https://github.com/pytorch/pytorch/issues/44827)
*   [文稿](https://pytorch.org/docs/1.8.0/pipeline.html?highlight=pipeline#)

### \[Beta\] DDP通信挂钩

DDP通信钩子是一个通用接口，通过覆盖DistributedDataParallel中的香草allreduce来控制如何在工人之间通信梯度。提供了一些内置通信钩子，包括PowerSGD，用户可以轻松应用这些钩子中的任何一个来优化通信。此外，通信钩子接口还可以支持更高级用例的用户定义通信策略。

*   [RFC](https://github.com/pytorch/pytorch/issues/39272)
*   [文稿](https://pytorch.org/docs/1.8.0/ddp_comm_hooks.html?highlight=powersgd)

### 分布式训练的附加原型功能

除了此版本中主要的稳定和测试版分布式训练功能外，我们还在夜间版本中提供了许多原型功能，可以试用并提供反馈。我们在下面的文档草案中链接以供参考：

*   **(Prototype) ZeroRedundancyOptimizer**\-基于并与Microsoft DeepSpeed团队合作，此功能通过在`ProcessGroup`集团的所有参与进程中分片优化器状态来帮助减少每个进程的内存占用。有关更多详细信息，请参阅此[文档](https://pytorch.org/docs/master/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer)。
*   **(Prototype) 进程组NCCL Send/Recv**\-NCCL send/recv API在v2.7中引入，此功能在NCCL进程组中增加了对它的支持。此功能将为用户提供在Python层而不是C++层实现集体操作的选项。请参阅此[文档](https://pytorch.org/docs/master/distributed.html#distributed-communication-package-torch-distributed)和[代码示例](https://github.com/pytorch/pytorch/blob/master/torch/distributed/distributed_c10d.py#L899)以了解更多信息。
*   **(Prototype) 使用TensorPipe在RPC中支持CUDA**\-此功能应为具有多GPU机器的PyTorch RPC用户带来随之而来的速度改进，因为TensorPipe将在可用时自动利用NVLink，并在进程之间交换GPU张量时避免往返于主机内存的昂贵副本。当不在同一台机器上时，TensorPipe将退回到将张量复制到主机内存，并将其作为常规CPU张量发送。这也将改善用户体验，因为用户将能够在代码中将GPU张量视为常规CPU张量。有关更多详细信息，请参阅此[文档](https://pytorch.org/docs/1.8.0/rpc.html)。
*   **(Prototype) 远程模块**\-此功能允许用户在远程工作者上操作模块，就像使用本地模块一样，其中RPC对用户是透明的。过去，此功能是以临时方式实现的，总体而言，此功能将提高PyTorch上模型并行性的可用性。有关更多详细信息，请参阅此[文档](https://pytorch.org/docs/master/rpc.html#remotemodule)。

## PyTorch Mobile

对PyTorch Mobile的支持正在扩大，增加了一套新的教程，以帮助新用户更快地在设备上启动模型，并为现有用户提供一个工具，以从我们的框架中获得更多。这些包括：

*   [iOS上的图像分割DeepLabV3](https://pytorch.org/tutorials/beginner/deeplabv3_on_ios.html)
*   [Android上的图像分割DeepLabV3](https://pytorch.org/tutorials/beginner/deeplabv3_on_android.html)

我们的新演示应用程序还包括图像分割、对象检测、神经机器翻译、问题回答和视觉变压器的例子。它们在iOS和Android上都可用：

*   [iOS演示应用程序](https://github.com/pytorch/ios-demo-app)
*   [安卓演示应用程序](https://github.com/pytorch/android-demo-app)

除了改进MobileNetV3和其他型号的CPU性能外，我们还改进了Android GPU后端原型，以覆盖更广泛的型号和更快的推理：

*   [安卓教程](https://pytorch.org/tutorials/prototype/vulkan_workflow.html)

最后，我们将推出PyTorch Mobile Lite Interpreter作为此版本的原型功能。Lite Interpreter允许用户减少运行时二进制大小。请尝试这些，并向我们发送您对[PyTorch论坛的](https://discuss.pytorch.org/c/mobile/)反馈。我们的所有最新更新都可以在[PyTorch Mobile页面上](https://pytorch.org/mobile/home/)找到

### \[Prototype\] PyTorch Mobile Lite interpreter

PyTorch Lite Interpreter是PyTorch运行时的简化版本，可以在资源受限的设备上执行PyTorch程序，并减少二进制大小占用。与当前版本中的当前设备运行时相比，此原型功能将二进制大小减少了高达70%。

*   [iOS/Android教程](https://pytorch.org/tutorials/prototype/lite_interpreter.html)

## 性能优化

在1.8中，我们将发布对基准实用程序的支持，使用户能够更好地监控性能。我们还在开放一个新的自动量化API。详情请参阅以下内容：

### （Beta）基准实用程序

基准实用性允许用户进行准确的性能测量，并提供可组合工具来帮助基准制定和后期处理。这有望帮助PyTorch的贡献者快速了解他们的贡献如何影响PyTorch的性能。

示例：

```
from torch.utils.benchmark import Timer

results = []
for num_threads in [1, 2, 4]:
    timer = Timer(
        stmt="torch.add(x, y, out=out)",
        setup="""
            n = 1024
            x = torch.ones((n, n))
            y = torch.ones((n, 1))
            out = torch.empty((n, n))
        """,
        num_threads=num_threads,
    )
    results.append(timer.blocked_autorange(min_run_time=5))
    print(
        f"{num_threads} thread{'s' if num_threads > 1 else ' ':<4}"
        f"{results[-1].median * 1e6:>4.0f} us   " +
        (f"({results[0].median / results[-1].median:.1f}x)" if num_threads > 1 else '')
    )

1 thread     376 us   
2 threads    189 us   (2.0x)
4 threads     99 us   (3.8x)
```

*   [文稿](https://pytorch.org/docs/1.8.0/benchmark_utils.html?highlight=benchmark#)
*   [辅导课](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)

### (Prototype) FX图形模式量化

FX图形模式量化是PyTorch中新的自动量化API。它通过添加对功能的支持和自动化量化过程来改进Eager模式量化，尽管人们可能需要重构模型以使模型与FX图形模式量化兼容（使用`torch.fx`进行符号跟踪）。

*   [文稿](https://pytorch.org/docs/master/quantization.html#prototype-fx-graph-mode-quantization)
*   教程：
    *   [(Prototype) FX图形模式训练后动态量化](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_dynamic.html)
    *   [(Prototype) FX图形模式训练后静态Qunatization](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html)
    *   [(Prototype) FX图形模式量化用户指南](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html)

## 硬件支持

### \[Beta\] 在C++中为新后端扩展PyTorch Dispatcher的能力

In PyTorch 1.8, you can now create new out-of-tree devices that live outside the `pytorch/pytorch` repo. The tutorial linked below shows how to register your device and keep it in sync with native PyTorch devices.

*   [辅导课](https://pytorch.org/tutorials/advanced/extend_dispatcher.html)

### \[Beta\] AMD GPU二进制文件现已推出

从PyTorch 1.8开始，我们增加了对ROCm车轮的支持，可以轻松使用AMD GPU。您只需转到标准的[PyTorch安装选择器](https://pytorch.org/get-started/locally/)，选择ROCm作为安装选项，然后执行提供的命令。

感谢您的阅读，如果您对这些更新感到兴奋，并希望参与PyTorch的未来，我们鼓励您加入[讨论论坛](https://discuss.pytorch.org/)并[打开GitHub问题](https://github.com/pytorch/pytorch/issues)。

Cheers!

PyTorch 团队