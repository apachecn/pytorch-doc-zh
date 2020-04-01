# 使用ONNX将模型从PyTorch传输到Caffe2和Mobile(移动端)  

> 译者：[冯宝宝](https://github.com/PEGASUS1993)  

在本教程中，我们将介绍如何使用ONNX将PyTorch中定义的模型转换为ONNX格式，然后将其加载到Caffe2中。一旦进入Caffe2，我们就可以运行模型来仔细检查它是否正确导出，然后我们展示了如何使用Caffe2功能(如移动导出器）在移动设备上执行模型。  

在本教程中，你需要安装onnx和Caffe2。您可以使用pip install onnx获取onnx的二进制版本。  

`注意`: 本教程需要PyTorch master分支，可以按照 [这里](https://github.com/pytorch/pytorch#from-source)说明进行安装。  

```py
# 一些包的导入
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

```
超分辨率是一种提高图像，视频分辨率的方法，广泛用于图像处理或视频剪辑。在本教程中，我们将首先使用带有虚拟输入的小型超分辨率模型。  

首先，让我们在PyTorch中创建一个SuperResolution模型。[这个模型](https://github.com/pytorch/examples/blob/master/super_resolution/model.py) 直接来自PyTorch的例子，没有修改：  

```py
# PyTorch中定义的Super Resolution模型
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

# 使用上面模型定义，创建super-resolution模型 
torch_model = SuperResolutionNet(upscale_factor=3)

```

通常，你现在会训练这个模型; 但是，对于本教程我们将下载一些预先训练的权重。请注意，此模型未经过充分训练来获得良好的准确性，此处仅用于演示目的。

```py
# 加载预先训练好的模型权重
del_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1    # just a random number

# 使用预训练的权重初始化模型
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

# 将训练模式设置为falsesince we will only run the forward pass.
torch_model.train(False)

```

在PyTorch中导出模型通过跟踪工作。要导出模型，请调用torch.onnx._export(）函数。这将执行模型，记录运算符用于计算输出的轨迹。因为_export运行模型，我们需要提供输入张量x。这个张量的值并不重要; 它可以是图像或随机张量，只要它是正确的大小。  

要了解有关PyTorch导出界面的更多详细信息，请查看[torch.onnx documentation](https://pytorch.org/docs/master/onnx.html)文档。

```py
# 输入模型
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)

# 导出模型
torch_out = torch.onnx._export(torch_model,             # model being run
                               x,                       # model input (or a tuple for multiple inputs)
                               "super_resolution.onnx", # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file

```

`torch_out` 是执行模型后的输出。通常您可以忽略此输出，但在这里我们将使用它来验证我们导出的模型在Caffe2中运行时计算相同的值。

现在让我们采用ONNX表示并在Caffe2中使用它。这部分通常可以在一个单独的进程中或在另一台机器上完成，但我们将继续在同一个进程中，以便我们可以验证Caffe2和PyTorch是否为网络计算相同的值：  

```py
import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend

# Load the ONNX ModelProto object. model is a standard Python protobuf object
model = onnx.load("super_resolution.onnx")

# prepare the caffe2 backend for executing the model this converts the ONNX model into a
# Caffe2 NetDef that can execute it. Other ONNX backends, like one for CNTK will be
# availiable soon.
prepared_backend = onnx_caffe2_backend.prepare(model)

# run the model in Caffe2

# Construct a map from input names to Tensor data.
# The graph of the model itself contains inputs for all weight parameters, after the input image.
# Since the weights are already embedded, we just need to pass the input image.
# Set the first input.
W = {model.graph.input[0].name: x.data.numpy()}

# Run the Caffe2 net:
c2_out = prepared_backend.run(W)[0]

# Verify the numerical correctness upto 3 decimal places
np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), c2_out, decimal=3)

print("Exported model has been executed on Caffe2 backend, and the result looks good!")

```

我们应该看到PyTorch和Caffe2的输出在数字上匹配最多3位小数。作为旁注，如果它们不匹配则存在Caffe2和PyTorch中的运算符以不同方式实现的问题，请在这种情况下与我们联系。  

## 使用ONNX转换SRResNET  

使用与上述相同的过程，我们参考[本文](https://arxiv.org/pdf/1609.04802.pdf)中提出的超分辨率转移了一个有趣的新模型“SRResNet”(感谢Twitter上的作者为本教程的目的提供了代码和预训练参数）。可在[此处](https://gist.github.com/prigoyal/b245776903efbac00ee89699e001c9bd)找到模型定义和预训练模型。下面是SRResNet模型输入，输出的样子。![SRResNet](img/b532c99eb24925fb30c040b3af3f71d0.jpg)  

## 在移动设备上运行模型  

到目前为止，我们已经从PyTorch导出了一个模型，并展示了如何加载它并在Caffe2中运行它。现在模型已加载到Caffe2中，我们可以将其转换为适合在移动设备上运行的格式[移动设备上运行](https://caffe2.ai/docs/mobile-integration.html)的格式。  

我们将使用Caffe2的[mobile_exporter](https://github.com/caffe2/caffe2/blob/master/caffe2/python/predictor/mobile_exporter.py)生成可在移动设备上运行的两个模型protobufs。 第一个用于使用正确的权重初始化网络，第二个实际运行执行模型。在本教程的其余部分，我们将继续使用小型超分辨率模型。  

```py
# extract the workspace and the model proto from the internal representation
c2_workspace = prepared_backend.workspace
c2_model = prepared_backend.predict_net

# 现在导入caffe2mobile_exporter
from caffe2.python.predictor import mobile_exporter

# call the Export to get the predict_net, init_net. These nets are needed for running things on mobile
init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_model, c2_model.external_input)

# 我们还将init_net和predict_net保存到我们稍后将用于在移动设备上运行它们的文件中
with open('init_net.pb', "wb") as fopen:
    fopen.write(init_net.SerializeToString())
with open('predict_net.pb', "wb") as fopen:
    fopen.write(predict_net.SerializeToString())

```  

`init_net`具有模型参数和嵌入在其中的模型输入，`predict_ne`t将用于指导运行时的`init_net`执行。 在本教程中，我们将使用上面生成的`init_net`和`predict_net`，并在正常的Caffe2后端和移动设备中运行它们，并验证两次运行中生成的输出高分辨率猫咪图像是否相同。  

在本教程中，我们将使用广泛使用的著名猫咪图像，如下所示：   

![cat](img/35d54d0c48ca1c52d56850a202a2c160.jpg)  

```py
# Some standard imports
from caffe2.proto import caffe2_pb2
from caffe2.python import core, net_drawer, net_printer, visualize, workspace, utils

import numpy as np
import os
import subprocess
from PIL import Image
from matplotlib import pyplot
from skimage import io, transform

```  

首先，让我们加载图像，使用标准的skimage python库对其进行预处理。 请注意，此预处理是处理用于训练/测试神经网络的数据的标准做法。  


```py
# 加载图像
img_in = io.imread("./_static/img/cat.jpg")

# 设置图片分辨率为 224x224
img = transform.resize(img_in, [224, 224])

# 保存好设置的图片作为模型的输入
io.imsave("./_static/img/cat_224x224.jpg", img)

```  

现在，作为下一步，让我们拍摄调整大小的猫图像并在Caffe2后端运行超分辨率模型并保存输出图像。 [这里](https://github.com/pytorch/examples/blob/master/super_resolution/super_resolve.py)的图像处理步骤已经从PyTorch实现的超分辨率模型中采用。  

```py
# 加载设置好的图片并更改为YCbCr的格式
img = Image.open("./_static/img/cat_224x224.jpg")
img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

# Let's run the mobile nets that we generated above so that caffe2 workspace is properly initialized
workspace.RunNetOnce(init_net)
workspace.RunNetOnce(predict_net)

# Caffe2 has a nice net_printer to be able to inspect what the net looks like and identify
# what our input and output blob names are.
print(net_printer.to_string(predict_net))

```   
`备注`： [YCbCr](https://baike.baidu.com/item/YCbCr/10012133?fr=aladdin) 

从上面的输出中，我们可以看到输入名为“9”，输出名为“27”(我们将数字作为blob名称有点奇怪，但这是因为跟踪JIT为模型生成了编号条目）。有点问题 后续校正。  


```py
# Now, let's also pass in the resized cat image for processing by the model.
workspace.FeedBlob("9", np.array(img_y)[np.newaxis, np.newaxis, :, :].astype(np.float32))

# run the predict_net to get the model output
workspace.RunNetOnce(predict_net)

# Now let's get the model output blob
img_out = workspace.FetchBlob("27")

```  

我们已经完成了在纯Caffe2后端运行我们的移动网络，现在，让我们在Android设备上执行该模型并获取模型输出。  

`注意`：对于Android开发，需要`adb shell`，否则教程的以下部分将无法运行。  

在我们在移动设备上运行模型的第一步中，我们把基于移动设备的本机速度测试基准二进制文件推送到adb。这个二进制文件可以在移动设备上执行模型，也可以导出我们稍后可以检索的模型输出。二进制文件可在[此处](https://github.com/caffe2/caffe2/blob/master/caffe2/binaries/speed_benchmark.cc)获得。要构建二进制文件，请按照[此处](https://github.com/caffe2/caffe2/blob/master/scripts/build_android.sh)的说明执行`build_android.sh`脚本。  

`注意`： 你需要已经安装了`ANDROID_NDK`,并且设置环境变量`ANDROID_NDK=path to ndk root`。  

```py
# let's first push a bunch of stuff to adb, specify the path for the binary
CAFFE2_MOBILE_BINARY = ('caffe2/binaries/speed_benchmark')

# we had saved our init_net and proto_net in steps above, we use them now.
# Push the binary and the model protos
os.system('adb push ' + CAFFE2_MOBILE_BINARY + ' /data/local/tmp/')
os.system('adb push init_net.pb /data/local/tmp')
os.system('adb push predict_net.pb /data/local/tmp')

# Let's serialize the input image blob to a blob proto and then send it to mobile for execution.
with open("input.blobproto", "wb") as fid:
    fid.write(workspace.SerializeBlob("9"))

# push the input image blob to adb
os.system('adb push input.blobproto /data/local/tmp/')

# Now we run the net on mobile, look at the speed_benchmark --help for what various options mean
os.system(
    'adb shell /data/local/tmp/speed_benchmark '                     # binary to execute
    '--init_net=/data/local/tmp/super_resolution_mobile_init.pb '    # mobile init_net
    '--net=/data/local/tmp/super_resolution_mobile_predict.pb '      # mobile predict_net
    '--input=9 '                                                     # name of our input image blob
    '--input_file=/data/local/tmp/input.blobproto '                  # serialized input image
    '--output_folder=/data/local/tmp '                               # destination folder for saving mobile output
    '--output=27,9 '                                                 # output blobs we are interested in
    '--iter=1 '                                                      # number of net iterations to execute
    '--caffe2_log_level=0 '
)

# get the model output from adb and save to a file
os.system('adb pull /data/local/tmp/27 ./output.blobproto')

# We can recover the output content and post-process the model using same steps as we followed earlier
blob_proto = caffe2_pb2.BlobProto()
blob_proto.ParseFromString(open('./output.blobproto').read())
img_out = utils.Caffe2TensorToNumpyArray(blob_proto.tensor)
img_out_y = Image.fromarray(np.uint8((img_out[0,0]).clip(0, 255)), mode='L')
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")
final_img.save("./_static/img/cat_superres_mobile.jpg")

```

现在，您可以比较图像cat_superres.jpg(来自纯caffe2后端执行的模型输出）和cat_superres_mobile.jpg(来自移动执行的模型输出），并看到两个图像看起来相同。 如果它们看起来不一样，那么在移动设备上执行会出现问题，在这种情况下，请联系Caffe2社区。你应该期望看到输出图像如下所示：  

![output\_cat](img/7afef7064c414bbb62bcbe068eca19a0.jpg)  

使用上述步骤，您可以轻松地在移动设备上部署模型。 另外，有关caffe2移动后端的更多信息，请查看[caffe2-android-demo](https://caffe2.ai/docs/AI-Camera-demo-android.html)。   
