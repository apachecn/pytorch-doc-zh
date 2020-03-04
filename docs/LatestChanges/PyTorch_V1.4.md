# 新版本: PyTorch 1.4发布，域库已更新

> 发布: 2020年01月15日
> 
> 原文: [PyTorch](https://pytorch.org/blog/pytorch-1-dot-4-released-and-domain-libraries-updated/)
> 
> 翻译: [ApacheCN](https://pytorch.apachecn.org/docs/LatestChanges/PyTorch_V1.4.html)

通过PyTorch团队

今天，我们宣布PyTorch 1.4的可用性以及PyTorch域库的更新。这些版本以[NeurIPS 2019](https://pytorch.org/blog/pytorch-adds-new-tools-and-libraries-welcomes-preferred-networks-to-its-community/)的公告为[基础](https://pytorch.org/blog/pytorch-adds-new-tools-and-libraries-welcomes-preferred-networks-to-its-community/)，在此我们共享了PyTorch Elastic的可用性，新的图像和视频分类框架以及PyTorch社区中添加了Preferred Networks。对于参加NeurIPS研讨会的人员，请在[此处](https://research.fb.com/neurips-2019-expo-workshops/)找到内容。

## [PYTORCH 1.4](https://pytorch.org/blog/pytorch-1-dot-4-released-and-domain-libraries-updated/#pytorch-14)

PyTorch 1.4版本增加了新功能，包括为PyTorch Mobile进行细粒度构建级别自定义的功能，以及新的实验性功能，包括对模型并行训练和Java语言绑定的支持。

### [PyTorch Mobile-构建级别自定义](https://pytorch.org/blog/pytorch-1-dot-4-released-and-domain-libraries-updated/#pytorch-mobile---build-level-customization)

在[1.3版本](https://pytorch.org/blog/pytorch-1-dot-3-adds-mobile-privacy-quantization-and-named-tensors/)的[PyTorch Mobile](https://pytorch.org/blog/pytorch-1-dot-3-adds-mobile-privacy-quantization-and-named-tensors/)开源之后，PyTorch 1.4添加了更多的移动支持，包括以细粒度级别自定义构建脚本的功能。这使移动开发人员可以通过仅包括其模型所使用的运算符来优化库的大小，并在此过程中显着减少其设备占用的空间。初步结果显示，例如，定制的MobileNetV2比预构建的PyTorch移动库小40％至50％。您可以[在此处](https://pytorch.org/mobile/home/)了解有关如何创建自己的自定义版本的更多信息，并且与往常一样，请在[PyTorch论坛](https://discuss.pytorch.org/c/mobile)上与社区互动，以提供您的任何反馈。

用于选择性地仅编译MobileNetV2所需的运算符的示例代码段：

```
# Dump list of operators used by MobileNetV2:
import torch, yaml
model = torch.jit.load('MobileNetV2.pt')
ops = torch.jit.export_opnames(model)
with open('MobileNetV2.yaml', 'w') as output:
    yaml.dump(ops, output)

```


```
# Build PyTorch Android library customized for MobileNetV2:
SELECTED_OP_LIST=MobileNetV2.yaml scripts/build_pytorch_android.sh arm64-v8a

# Build PyTorch iOS library customized for MobileNetV2:
SELECTED_OP_LIST=MobileNetV2.yaml BUILD_PYTORCH_MOBILE=1 IOS_ARCH=arm64 scripts/build_ios.sh

```


### [分布式模型并行训练（实验性）](https://pytorch.org/blog/pytorch-1-dot-4-released-and-domain-libraries-updated/#distributed-model-parallel-training-experimental)

随着模型的规模（例如RoBERTa）不断增加到数十亿个参数，模型并行训练对于帮助研究人员突破极限变得越来越重要。此版本提供了分布式RPC框架，以支持分布式模型并行训练。它允许远程运行功能和引用远程对象，而无需复制实际数据，并提供autograd和Optimizer API以透明地向后运行并跨RPC边界更新参数。

要了解有关API和此功能设计的更多信息，请参见以下链接：

* [API文档](https://pytorch.org/docs/stable/rpc.html)
* [分布式Autograd设计文档](https://pytorch.org/docs/stable/notes/distributed_autograd.html)
* [远程参考设计文档](https://pytorch.org/docs/stable/notes/rref.html)

有关完整的教程，请参见下面的链接：

* [完整的RPC教程](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html)
* [使用模型并行训练进行强化学习并使用LSTM的示例](https://github.com/pytorch/examples/tree/master/distributed/rpc)

与往常一样，您可以与社区成员联系并在[论坛](https://discuss.pytorch.org/c/distributed/distributed-rpc)上进行更多[讨论](https://discuss.pytorch.org/c/distributed/distributed-rpc)。

### [Java绑定（实验性）](https://pytorch.org/blog/pytorch-1-dot-4-released-and-domain-libraries-updated/#java-bindings-experimental)

除了支持Python和C ++，此版本还增加了对Java绑定的实验性支持。基于PyTorch Mobile中为Android开发的界面，新的绑定使您可以从任何Java程序调用TorchScript模型。请注意，Java绑定仅可用于此版本的Linux，并且仅用于推断。我们希望在以后的版本中扩展支持。有关如何在Java中使用PyTorch的信息，请参见下面的代码片段：


```
Module mod = Module.load("demo-model.pt1");
Tensor data =
    Tensor.fromBlob(
        new int[] {1, 2, 3, 4, 5, 6}, // data
        new long[] {2, 3} // shape
        );
IValue result = mod.forward(IValue.from(data), IValue.from(3.0));
Tensor output = result.toTensor();
System.out.println("shape: " + Arrays.toString(output.shape()));
System.out.println("data: " + Arrays.toString(output.getDataAsFloatArray()));

```

了解更多关于如何使用PyTorch从Java [这里](https://github.com/pytorch/java-demo)，看到完整的Javadoc API文档[在这里](https://pytorch.org/javadoc/1.4.0/)。

有关完整的1.4版本说明，请参见[此处](https://github.com/pytorch/pytorch/releases)。

## [域库](https://pytorch.org/blog/pytorch-1-dot-4-released-and-domain-libraries-updated/#domain-libraries)

PyTorch域库（例如torchvision，torchtext和torchaudio）使用常见的数据集，模型和转换对PyTorch进行了补充。我们很高兴与PyTorch 1.4核心版本一起共享所有三个域库的新版本。

### torchvision 0.5[](https://pytorch.org/blog/pytorch-1-dot-4-released-and-domain-libraries-updated/#torchvision-05)

torchvision 0.5的改进主要集中在增加对生产部署的支持，包括量化，TorchScript和ONNX。一些亮点包括：

* 现在，torchvision中的所有模型都可以使用torchscript编写，从而使其更易于移植到非Python生产环境中
* ResNets，MobileNet，ShuffleNet，GoogleNet和InceptionV3现在已经具有经过预先训练的模型的量化对象，并且还包括用于进行量化意识训练的脚本。
* 与Microsoft团队合作，我们增加了对所有型号的ONNX支持，包括Mask R-CNN。

[在此处](https://github.com/pytorch/vision/releases)了解有关Torchvision 0.5的更多信息。

### [torchaudio 0.4](https://pytorch.org/blog/pytorch-1-dot-4-released-and-domain-libraries-updated/#torchaudio-04)

torchaudio 0.4的改进集中在增强当前可用的转换，数据集和后端支持上。重点包括：

* SoX现在是可选的，并且新的可扩展后端分派机制公开了SoundFile作为SoX的替代方法。
* 数据集的界面已统一。这样可以添加两个大型数据集：LibriSpeech和Common Voice。
* 现在提供了新滤波器，例如双二阶滤波器，数据增强（例如时间和频率屏蔽），变换（例如MFCC），增益和抖动以及新特征计算（例如增量）。
* 转换现在支持批处理并且是jitable。
* 具有语音活动检测功能的交互式语音识别演示可用于实验。

[在此处](https://github.com/pytorch/audio/releases)了解有关Torchaudio 0.4的更多信息。

### [torchtext 0.5](https://pytorch.org/blog/pytorch-1-dot-4-released-and-domain-libraries-updated/#torchtext-05)

torchtext 0.5主要集中于对数据集加载器API的改进，包括与核心PyTorch API的兼容性，而且还增加了对无监督文本标记化的支持。重点包括：

* 为SentencePiece添加了绑定，以实现无监督的文本标记化。
* 添加了一个新的无监督学习数据集-enwik9。
* 对PennTreebank，WikiText103，WikiText2，IMDb进行了修订，以使其与torch.utils.data兼容。这些数据集位于实验文件夹中，我们欢迎您提供反馈。

[在此处](https://github.com/pytorch/text/releases)了解有关torchtext 0.5的更多信息。

*我们要感谢整个PyTorch团队和社区为这项工作做出的所有贡献。*

干杯!

PyTorch团队