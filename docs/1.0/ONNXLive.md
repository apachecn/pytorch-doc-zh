# ONNX 现场演示教程  

> 译者：[冯宝宝](https://github.com/PEGASUS1993)  

本教程将向您展示如何使用ONNX将已从PyTorch导出的神经模型传输模型转换为Apple CoreML格式。这将允许您在Apple设备上轻松运行深度学习模型，在这种情况下，可以从摄像机直播演示。  

## 什么是ONNX  
ONNX(开放式神经网络交换）是一种表示深度学习模型的开放格式。借助ONNX，AI开发人员可以更轻松地在最先进的工具之间移动模型，并选择最适合它们的组合。ONNX由合作伙伴社区开发和支持。 您可以访问 [onnx.ai](https://onnx.ai/)，了解有关ONNX的更多信息以及支持的工具。  

## 教程预览  

本教程将带你走过如下主要4步：  

1.  [下载(或训练）Pytorch风格装换模型](#download-or-train-pytorch-style-transfer-models)
2.  [将PyTorch模型转换至ONNX模型](#convert-the-pytorch-models-to-onnx-models)
3.  [将ONNX模型转换至CoreML模型](#convert-the-onnx-models-to-coreml-models)
4.  [在支持风格转换iOS App中运行CoreML模型](#run-the-coreml-models-in-a-style-transfer-ios-app)  

##  环境准备 

我们将在虚拟环境工作，以避免与您的本地环境冲突。在本教程中使用Python 3.6，但其他版本也应该可以正常工作。  

```py
python3.6 -m venv venv
source ./venv/bin/activate

```

我们需要安装Pytorch和 onnx->coreml 转换器：  

```py
pip install torchvision onnx-coreml

```

如果要在iPhone上运行iOS样式传输应用程序，还需要安装XCode。您也可以在Linux中转换模型，但要运行iOS应用程序本身，您将需要一台Mac。
  
## 下载(或训练）Pytorch风格装换模型  

在本教程中，我们将使用与pytorch一起发布的样式传输模型，地址为https://github.com/pytorch/examples/tree/master/fast_neural_style。如果您想使用其他PyTorch或ONNX模型，请随意跳过此步骤。  

这些模型用于在静态图像上应用样式传输，并且实际上没有针对视频进行优化以获得足够快的速度。但是，如果我们将分辨率降低到足够低，它们也可以很好地处理视频。  

我们先下载模型：
  
```py
git clone https://github.com/pytorch/examples
cd examples/fast_neural_style
``` 
如果您想自己训练模型，您刚刚克隆下载的的pytorch/examples存储库有更多关于如何执行此操作的信息。目前，我们只需使用存储库提供的脚本下载预先训练的模型：  

```py
./download_saved_models.sh

```

此脚本下载预先训练的PyTorch模型并将它们放入saved_models文件夹中。 你的目录中现在应该有4个文件，candy.pth，mosaic.pth，rain_princess.pth和udnie.pth。  

## 将PyTorch模型转换至ONNX模型  

现在我们已将预先训练好的PyTorch模型作为saved_models文件夹中的.pth文件，我们需要将它们转换为ONNX格式。模型定义在我们之前克隆的pytorch/examples存储库中，通过几行python我们可以将它导出到ONNX。在这种情况下，我们将调用torch.onnx._export而不是实际运行神经网络，它将PyTorch作为api提供，以直接从PyTorch导出ONNX格式的模型。但是，在这种情况下，我们甚至不需要这样做，因为脚本已经存在Neural_style / neural_style.py，它将为我们执行此操作。如果要将其应用于其他模型，也可以查看该脚本。  

从PyTorch导出ONNX格式本质上是追踪您的神经网络，因此这个api调用将在内部运行网络“虚拟数据”以生成图形。为此，它需要输入图像来应用样式转移，其可以简单地是空白图像。但是，此图像的像素大小很重要，因为这将是导出的样式传输模型的大小。为了获得良好的性能，我们将使用250x540的分辨率。如果您不太关心FPS，可以随意采取更大的分辨率，更多关于风格转移质量。    

让我们使用[ImageMagick](https://www.imagemagick.org/)创建我们想要的分辨率的空白图像：   

```py
convert -size 250x540 xc:white png24:dummy.jpg

```

然后用它来导出PyTorch模型用它来导出PyTorch模型：  

```py
python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/candy.pth --cuda 0 --export_onnx ./saved_models/candy.onnx
python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/udnie.pth --cuda 0 --export_onnx ./saved_models/udnie.onnx
python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/rain_princess.pth --cuda 0 --export_onnx ./saved_models/rain_princess.onnx
python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/mosaic.pth --cuda 0 --export_onnx ./saved_models/mosaic.onnx

```

你应该得到4个文件，`candy.onnx`，`mosaic.onnx`，`rain_princess.onnx`和`udnie.onnx`，由相应的`.pth`文件创建。  

## 将ONNX模型转换至CoreML模型  

现在我们有了ONNX模型，我们可以将它们转换为CoreML模型，以便在Apple设备上运行它们。为此，我们使用之前安装的onnx-coreml转换器。转换器附带一个convert-onnx-to-coreml脚本，上面的安装步骤添加到我们的路径中。遗憾的是，这对我们不起作用，因为我们需要将网络的输入和输出标记为图像，并且虽然这是转换器支持的，但只有在从python调用转换器时才支持它。  

通过查看样式传输模型(例如在像Netron这样的应用程序中打开.onnx文件），我们看到输入名为'0'，输出名为'186'。这些只是PyTorch分配的数字ID。我们需要将它们标记为图像。  

所以让我们创建一个python小文件并将其命名为onnx_to_coreml.py。这可以通过使用touch命令创建，并使用您喜欢的编辑器进行编辑，以添加以下代码行。  

```py
import sys
from onnx import onnx_pb
from onnx_coreml import convert

model_in = sys.argv[1]
model_out = sys.argv[2]

model_file = open(model_in, 'rb')
model_proto = onnx_pb.ModelProto()
model_proto.ParseFromString(model_file.read())
coreml_model = convert(model_proto, image_input_names=['0'], image_output_names=['186'])
coreml_model.save(model_out)

```

现在来运行:

```py
python onnx_to_coreml.py ./saved_models/candy.onnx ./saved_models/candy.mlmodel
python onnx_to_coreml.py ./saved_models/udnie.onnx ./saved_models/udnie.mlmodel
python onnx_to_coreml.py ./saved_models/rain_princess.onnx ./saved_models/rain_princess.mlmodel
python onnx_to_coreml.py ./saved_models/mosaic.onnx ./saved_models/mosaic.mlmodel

```

现在，您的saved_models目录中应该有4个CoreML模型：candy.mlmodel，mosaic.mlmodel，rain_princess.mlmodel和udnie.mlmodel。  

## 在支持风格转换iOS App中运行CoreML模型    

此存储库(即您当前正在阅读README.md的存储库）包含一个iOS应用程序，可以在手机摄像头的实时摄像头流上运行CoreML样式传输模型。  

```py
git clone https://github.com/onnx/tutorials

```

并在XCode中打开tutorials/examples/CoreML/NNXLive/ONNXLive.xcodeproj项目。我们建议使用XCode 9.3和iPhone X。在旧设备或XCode版本上可能会出现问题。   

在Models/文件夹中，项目包含一些.mlmodel文件。我们将用我们刚刚创建的模型替换它们。  

然后你在iPhone上运行应用程序就可以了。点击屏幕可切换模型。  

## 结论  

我们希望本教程能够概述ONNX的内容以及如何使用它来在框架之间转换神经网络，在这种情况下，神经风格的传输模型从PyTorch转移到CoreML。  

您可以随意尝试这些步骤并在自己的模型上进行测试。如果您遇到任何问题或想要提供反馈，请告诉我们。我们倾听你的想法。
