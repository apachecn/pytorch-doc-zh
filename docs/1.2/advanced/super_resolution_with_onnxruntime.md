# 4.(可选）从导出到PyTorch一个ONNX模型并使用运行它ONNX运行时

在本教程中，我们将介绍如何在PyTorch定义的模型转换成ONNX格式，然后用ONNX运行时运行它。

ONNX运行时是ONNX模型，跨多个平台和硬件(在Windows，Linux和Mac和两个CPU和GPU）有效地推论一个注重性能的发动机。
ONNX运行时已被证明大大增加了多种型号的性能，解释[此处](https://cloudblogs.microsoft.com/opensource/2019/05/22/onnx-
runtime-machine-learning-inferencing-0-4-release)

在本教程中，你需要安装[ ONNX ](https://github.com/onnx/onnx)和[
ONNX运行[HTG3。你可以得到的二进制建立ONNX和ONNX运行与`点子 安装 onnx  onnxruntime
[HTG13。需要注意的是ONNX运行与Python版本3.5到3.7兼容。`](https://github.com/microsoft/onnxruntime)

`注 `：本教程需要PyTorch主分支可通过以下的说明[这里被安装](https://github.com/pytorch/pytorch#from-
source)

    
    
    # Some standard imports
    import io
    import numpy as np
    
    from torch import nn
    import torch.utils.model_zoo as model_zoo
    import torch.onnx
    

超分辨率越来越多的图像，视频分辨率的方式，被广泛应用于图像处理和视频编辑。在本教程中，我们将使用一个小的超分辨率模型。

首先，让我们创建一个PyTorch超分辨模型。该模型采用在[中描述的“实时单幅图像和视频超分辨率采用高效的子像素卷积神经网络”的高效子像素卷积层 -
石等](https://arxiv.org/abs/1609.05158)用于提高图像的分辨率由高档的因素。该模型预计的图像作为输入的所述YCbCr的Y成分，并且输出在超分辨率放大的Y分量。

[该模型](https://github.com/pytorch/examples/blob/master/super_resolution/model.py)直接从PyTorch的例子来不加修改：

    
    
    # Super Resolution model definition in PyTorch
    import torch.nn as nn
    import torch.nn.init as init
    
    
    class SuperResolutionNet(nn.Module):
        def __init__(self, upscale_factor, inplace=False):
            super(SuperResolutionNet, self).__init__()
    
            self.relu = nn.ReLU(inplace=inplace)
            self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
            self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
            self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
            self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
            self._initialize_weights()
    
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.pixel_shuffle(self.conv4(x))
            return x
    
        def _initialize_weights(self):
            init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv4.weight)
    
    # Create the super-resolution model by using the above model definition.
    torch_model = SuperResolutionNet(upscale_factor=3)
    

通常情况下，你现在会训练这个模型;然而，在本教程中，我们反而会下载一些预训练的权重。请注意，这种模式并没有良好的精度全面训练，在这里仅用于演示目的。

它调用`torch_model.eval(） `或`torch_model.train(假）
`导出模型前，把该模型是非常重要的推论模式。既然喜欢在不同的推断和训练模式辍学或batchnorm运营商的行为，这是必需的。

    
    
    # Load pretrained model weights
    model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
    batch_size = 1    # just a random number
    
    # Initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))
    
    # set the model to inference mode
    torch_model.eval()
    

导出在PyTorch模型通过跟踪或脚本作品。这个教程将作为一个例子使用由跟踪导出的模型。要导出模型，我们称之为`torch.onnx.export(）
`功能。这将执行模式，记录的是什么运营商来计算输出跟踪。因为`出口 `运行模型，我们需要提供一个输入张量`×
[HTG11。只要它是正确的类型和尺寸在此的值可以是随机的。注意，输入尺寸将被固定在导出ONNX图形用于将输入的所有维的，除非指定为动态轴。在这个例子中，我们用的batch_size
1的输入导出模型，但然后指定所述第一尺寸为动态在`dynamic_axes`参数`torch.onnx.export (）
`。由此导出的模型将接受尺寸的输入[batch_size时，1，224，224]，其中的batch_size可以是可变的。`

要了解PyTorch的出口接口的详细信息，请查看[
torch.onnx文献[HTG1。](https://pytorch.org/docs/master/onnx.html)

    
    
    # Input to the model
    x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
    torch_out = torch_model(x)
    
    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # wether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                    'output' : {0 : 'batch_size'}})
    

我们还计算`torch_out`，该模型，我们将用它来验证ONNX运行中运行时，我们出口的模型计算相同的值后输出。

但在验证模型与ONNX运行时输出之前，我们将检查与ONNX的API的ONNX模型。首先，`
onnx.load(“super_resolution.onnx”）
`将加载保存的模型和将输出一个onnx.ModelProto结构(用于捆绑一个ML一个顶层文件/容器格式模型。详细信息[
onnx.proto文档](https://github.com/onnx/onnx/blob/master/onnx/onnx.proto)）。然后，`
onnx.checker.check_model(onnx_model）HTG8]
`将验证模型的结构，并确认该模型有一个有效的模式。所述ONNX图表的有效性是通过检查模型的版本，图的结构，以及作为节点，其输入和输出验证。

    
    
    import onnx
    
    onnx_model = onnx.load("super_resolution.onnx")
    onnx.checker.check_model(onnx_model)
    

现在，让我们计算使用ONNX运行的Python的API的输出。这一部分通常可以在一个单独的进程或另一台机器上完成，但我们会继续以同样的过程，使我们可以验证ONNX运行和PyTorch被计算为网络相同的值。

为了运行与ONNX运行模式，我们需要与所选择的配置参数(在这里我们使用默认配置）创建模型推断会话。一旦会话创建，我们评估使用的run(）API模型。这个调用的输出是含有ONNX运行时计算出的模型的输出列表。

    
    
    import onnxruntime
    
    ort_session = onnxruntime.InferenceSession("super_resolution.onnx")
    
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    

我们应该看到，PyTorch和ONNX运行时的输出数值上运行，与之相匹配的给定精度(RTOL = 1E-03和蒂=
1E-05）。作为一个侧面说明，如果他们不匹配，则有在ONNX出口的问题，请与我们联系在这种情况下。

## 运行使用图像上的模型ONNX运行时

到目前为止，我们已经从PyTorch导出的模型，并展示了如何加载和运行ONNX与伪张量作为输入运行它。

在本教程中，我们将使用广泛使用的一个著名的猫形象，它看起来像下面

![cat](img/cat_224x224.jpg)

首先，让我们使用标准的PIL Python库加载图像，预先对其进行处理。请注意，这是预处理的数据处理训练/测试神经网络的标准做法。

我们首先调整图像的大小，以适应模型的输入(224x224）的大小。然后我们图象分成了Y，Cb和Cr分量。这些组件代表灰度图像(Y）和蓝色差(Cb）和红色差(Cr）的色度分量。
Y分量是对人眼更敏感，我们感兴趣的是这部分，我们将改造。提取Y分量后，我们把它转换成这将是我们模型的输入张量。

    
    
    from PIL import Image
    import torchvision.transforms as transforms
    
    img = Image.open("./_static/img/cat.jpg")
    
    resize = transforms.Resize([224, 224])
    img = resize(img)
    
    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()
    
    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)
    img_y.unsqueeze_(0)
    

现在，作为下一步，让我们代表灰度调整猫形象的张量和运行ONNX运行超高分辨率模型如前所述。

    
    
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]
    

在这一点上，该模型的输出是一个张量。现在，我们将处理模型的输出从输出张建设回来的最终输出图像，并保存图像。后处理步骤已经从PyTorch实现超高分辨率模型[此处](https://github.com/pytorch/examples/blob/master/super_resolution/super_resolve.py)采用。

    
    
    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')
    
    # get the output image follow post-processing step from PyTorch implementation
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")
    
    # Save the image, we will compare this with the output image from mobile device
    final_img.save("./_static/img/cat_superres_with_ort.jpg")
    

![output\\_cat](img/cat_superres_with_ort.jpg)

ONNX运行时是一个跨平台的引擎，可以跨多个平台和两个CPU和GPU运行它。

ONNX运行时也可以部署到云中使用Azure的机器学习Services模型推理。更多信息[此处[HTG1。](https://docs.microsoft.com/en-
us/azure/machine-learning/service/concept-onnx)

关于ONNX运行时的性能[此处](https://github.com/microsoft/onnxruntime#high-
performance)更多信息。

有关ONNX运行[此处](https://github.com/microsoft/onnxruntime)更多信息。

**脚本的总运行时间：** (0分钟0.000秒）

[`Download Python source code:
super_resolution_with_onnxruntime.py`](../_downloads/58ce6e85b9b9e9647d302d6b48feccb0/super_resolution_with_onnxruntime.py)

[`Download Jupyter notebook:
super_resolution_with_onnxruntime.ipynb`](../_downloads/8c7f0be1e1c3803fcb4c41bcd9f4226b/super_resolution_with_onnxruntime.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](../intermediate/model_parallel_tutorial.html "1. Model Parallel
Best Practices") [![](../_static/images/chevron-right-orange.svg)
Previous](cpp_export.html "3. Loading a TorchScript Model in C++")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。



  * [HTG0 (可选）将模型从PyTorch导出到ONNX并使用ONNX Runtime运行	
    * 运行使用ONNX运行时的图像上的模型

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)







 
[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)



