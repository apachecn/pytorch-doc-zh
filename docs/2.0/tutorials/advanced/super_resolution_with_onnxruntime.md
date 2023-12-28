# (可选)将 PYTORCH 模型导出到 ONNX 并使用 ONNX 运行时运行它

> 译者：[masteryi-0018](https://github.com/masteryi-0018)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/advanced/super_resolution_with_onnxruntime>
>
> 原始地址：<https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html>

在本教程中，我们将介绍如何将 PyTorch 定义的模型转换为 ONNX 格式，然后使用 ONNX 运行时运行它。

ONNX 运行时是用于 ONNX 模型的以性能为中心的引擎，跨多个平台和硬件进行高效推理(Windows，Linux 和 Mac 以及 CPU 和 GPU 上)。事实证明，ONNX 运行时可以显著提高性能，查看[此处](https://cloudblogs.microsoft.com/opensource/2019/05/22/onnx-runtime-machine-learning-inferencing-0-4-release)所述的多种模型

对于本教程，您将需要安装 [ONNX](https://github.com/onnx/onnx) 和 [ONNX 运行时](https://github.com/microsoft/onnxruntime)。 您可以使用 `pip install onnx onnxruntime` 获得 ONNX 和 ONNX 运行时的二进制版本。ONNX 运行时建议使用 PyTorch 的最新稳定运行时。

```python
# Some standard imports
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
```

超分辨率是提高图像、视频分辨率的一种方式，并广泛用于图像处理或视频编辑。为此教程，我们将使用一个小的超分辨率模型。

首先，让我们在 PyTorch 中创建一个 `SuperResolution` 模型。该模型使用了 [“Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network” - Shi et al](https://arxiv.org/abs/1609.05158) 中所述的高效子像素卷积层来提高图像的分辨率受向上缩放因子的影响。该模型期望图像的 YCbCr 的 Y 分量作为输入，并以超分辨率输出放大的 Y 分量。

[模型](https://github.com/pytorch/examples/blob/master/super_resolution/model.py) 直接来自 PyTorch 的示例，无需修改：

```python
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
```

通常，您现在需要训练此模型；但是，对于本教程，我们将下载一些预先训练的权重。请注意，此模型没有经过完全训练以获得良好的准确性，在这里用于仅用于演示目的。

在导出模型之前，请先调用 `torch_model.eval()` 或 `torch_model.train(False)`，以将模型转换为推理模式，这一点很重要。也是必需的，因为像 `dropout` 或 `batchnorm` 这样的运算符在推理和训练模式下的行为会有所不同。

```python
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
```

在 PyTorch 中导出模型是通过 trace 或 script 编写的。本教程将通过 trace 导出模型为例进行演示。要导出模型，我们调用 `torch.onnx.export()` 函数。这将执行模型，并记录使用了什么运算符进行计算输出的轨迹。 因为 `export` 会运行模型，所以我们需要提供输入tensor `x`。tensor `x` 的形状与数据类型需要跟模型输入保持一致，值可以是随机的。请注意，除非指定模型输入输出为动态的，否则输入输出tensor形状将在导出 ONNX 计算图时固定为输入tensor的大小。在此示例中，我们导出输入为 `batch_size=1` 的模型，但随后在 `torch.onnx.export()` 的 `dynamic_axes` 参数中将第一维指定为动态。 因此，导出的模型将接受大小为 `[batch_size, 1, 224, 224]` 的输入，其中 `batch_size` 可以是可变的。

要了解有关 PyTorch 导出接口的更多详细信息，请查看[`torch.onnx`文档](https://pytorch.org/docs/master/onnx.html)。

```python
# Input to the model
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
```

我们还计算了模型运行之后的输出 `torch_out`，它将用来验证导出的模型在 ONNX 运行时中运行时是否计算出相同的值。

但在通过 ONNX 运行时验证模型的输出之前，我们将使用 ONNX 的 API 检查 ONNX 模型。首先，`onnx.load("super_resolution.onnx")` 将加载保存的模型并输出 `onnx.ModelProto` 结构(用于捆绑 ML 模型的顶级文件/容器格式。有关更多信息，请参见[`onnx.proto`文档](https://github.com/onnx/onnx/blob/master/onnx/onnx.proto)。然后，`onnx.checker.check_model(onnx_model)` 将验证模型的结构并确认模型具有有效的架构。通过检查模型的版本，图的结构以及节点及其输入和输出，可以验证 ONNX 图的有效性。

```python
import onnx

onnx_model = onnx.load("super_resolution.onnx")
onnx.checker.check_model(onnx_model)
```

现在，让我们使用 ONNX 运行时的 Python API 计算输出。 这部分通常可以在单独的过程中或在另一个过程中完成，但我们将继续相同的过程，以便我们可以验证 ONNX 运行时和 PyTorch 的输出是否相同。

为了使用 ONNX 运行时运行模型，我们需要创建一个具有所选配置的模型的推理会话参数(这里我们使用默认配置)。 创建会话后，我们使用 run() API 评估模型。 此调用的输出是包含模型输出的列表 由 ONNX 运行时计算。

```python
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
```

我们应该看到 PyTorch 和 ONNX 运行时的输出在数值上与给定的精度匹配(`rtol = 1e-03`和`atol = 1e-05`)。附带说明一下，如果它们不匹配，则说明 ONNX 导出器中存在问题，因此请与我们联系。

# 使用 ONNX 运行时进行图像推理

到目前为止，我们已经从 PyTorch 导出了一个模型，并展示了如何加载它并使用虚拟tensor作为输入在 ONNX 运行时中运行它。

在本教程中，我们将使用一个广泛使用的著名猫图像，如下图所示：

![猫](../../img/cat_224x224.jpg)

首先，让我们加载图片，使用标准的 PIL python 库对其进行预处理。请注意，此预处理是处理数据以训练/测试神经网络的标准做法。

我们首先调整图像大小以适合模型输入的大小(`224x224`)。然后，我们将图像分为 Y，Cb 和 Cr 分量。这些分量代表灰度图像(Y)，以及蓝差(Cb)和红差(Cr)色度分量。Y 分量对人眼更敏感，我们对将要转换的这个分量很感兴趣。提取 Y 分量后，我们将其转换为tensor，这将是模型的输入。

```python
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
```

现在，作为下一步，让我们以表示灰度调整大小的猫图像并在如前所述的 ONNX 运行时来运行超分辨率模型：

```python
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]
```

此时，模型的输出是一个tensor。现在，我们将处理模型的输出以构造回从输出tensor最终输出图像，并保存图像。后处理步骤采用了来自[此处](https://github.com/pytorch/examples/blob/master/super_resolution/super_resolve.py)的后处理步骤。

```python
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
```

![结果](../../img/cat_superres_with_ort.jpg)

ONNX 运行时是一个跨平台引擎，您可以跨多个平台以及 CPU 和 GPU 上。

ONNX 运行时也可以部署到云端进行模型推理，使用 Azure 机器学习服务。更多信息[在这里](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-onnx)。

有关 ONNX 运行时性能的详细信息，请单击[此处](https://github.com/microsoft/onnxruntime#high-performance)。

有关 ONNX 运行时的详细信息，请单击[此处](https://github.com/microsoft/onnxruntime)。