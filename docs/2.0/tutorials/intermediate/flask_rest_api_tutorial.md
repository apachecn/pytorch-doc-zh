# 通过带有 FLASK 的 REST API 在 PYTHON 中部署 PYTORCH

> 译者：[masteryi-0018](https://github.com/masteryi-0018)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/flask_rest_api_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html>

作者： [Avinash Sajjanshetty](https://avi.im/)

在本教程中，我们将使用 Flask 部署一个 PyTorch 模型，并公开一个用于模型推理的 REST API。特别的，我们将部署一个预训练检测图像的DenseNet 121模型。

> 提示
>
> 这里使用的所有代码都是在MIT许可下发布的，并且可以在[Github](https://github.com/avinassh/pytorch-flask-api)上找到。

这是关于部署 PyTorch 模型的系列教程中的第一个。在生产中，以这种方式使用 Flask 是迄今为止部署您的 PyTorch 模型最简单的入门方法，但它不适用于具有高性能要求的场合。为此：

- 如果您已经熟悉 TorchScript，可以直接进入教程中的[使用C++加载 TorchScript 模型](https://pytorch.org/tutorials/advanced/cpp_export.html)。
- 如果您首先需要复习 TorchScript，请查看我们的 [TorchScript介绍](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) 教程。

## 接口定义

我们将首先定义我们的 API 端点、请求和响应类型。我们 API 端点将采用包含图像的参数的 HTTP POST 请求。响应将是 JSON 包含预测的响应：`/predictfile`

```
{"class_id": "n02124075", "class_name": "Egyptian_cat"}
```

## 依赖

通过运行以下命令安装所需的依赖项：

```
$ pip install Flask==2.0.1 torchvision==0.10.0
```

## 简单的网络服务器

以下是一个简单的 Web 服务器，摘自 Flask 的文档

```
from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World!'
```

将上面的代码片段保存在一个名为`app.py`的文件中，您现在可以运行Flask开发服务器，键入：

```
$ FLASK_ENV=development FLASK_APP=app.py flask run
```

当您在网络浏览器中访问时，您将被`http://localhost:5000/Hello World!`的文字问候

我们将对上面的代码片段进行细微的更改，以便它适合我们的 API 定义。首先，我们将该方法重命名为`predict`，我们将更新的终结点路径`/predict`。由于图像文件将通过 HTTP POST 请求，我们将对其进行更新，以便它也只接受 POST 请求：

```
@app.route('/predict', methods=['POST'])
def predict():
    return 'Hello World!'
```

我们还将更改响应类型，以便它返回的 JSON 响应包含 ImageNet类ID 和名称。更新后的`app.py`文件变为：

```
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})
```

## 推理

在接下来的部分中，我们将重点介绍如何编写推理代码。这将涉及两部分，一部分是我们准备图像以便可以将图像送到 DenseNet，接下来，我们将编写代码从模型获得实际预测结果。

### 准备图像

DenseNet 模型要求的输入为 3 通道 RGB 分辨率为224 x 224 图像。我们还将使用所需的平均值和标准差对图像张量进行归一化。您可以[在此处](https://pytorch.org/vision/stable/models.html)阅读有关它的更多信息。

我们将使用`torchvision`中的`transforms`库中的函数并构建一个转换流程，根据需要转换我们的图像。你可以[在此处](https://pytorch.org/vision/stable/transforms.html)阅读有关转换的更多信息。

```
import io

import torchvision.transforms as transforms
from PIL import Image

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)
```

上面的方法以字节为单位获取图像数据，应用一系列转换并返回一个张量。要测试上述方法，请在字节模式（首先替换 *https://pytorch.org/tutorials/_static/img/sample_file.jpeg* 为实际计算机上文件的路径），然后查看是否返回张量：

```
with open("https://pytorch.org/tutorials/_static/img/sample_file.jpeg", 'rb') as f:
    image_bytes = f.read()
    tensor = transform_image(image_bytes=image_bytes)
    print(tensor)
```

### 预测

现在将使用预训练的 DenseNet 121 模型来预测图像种类。我们将使用`torchvision`库中，加载模型并获得一个推理结果。虽然我们将在本例中使用预训练模型，但您可以对您自己的模型使用相同的方法。有关加载您的模型的更多信息请查看[本教程](https://pytorch.org/tutorials/beginner/saving_loading_models.html)。

```
from torchvision import models

# Make sure to set `weights` as `'IMAGENET1K_V1'` to use the pretrained weights:
model = models.densenet121(weights='IMAGENET1K_V1')
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat
```

张量将包含预测类 id 的索引。但是，我们需要一个人类可读的类名。为此，我们需要一个类 ID 到名称映射。下载[此文件](https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json)并记住您保存它的位置（或者，如果您正在按照本教程中的确切步骤进行操作，将其保存在 *tutorials/_static* ）。此文件包含 ImageNet 类 ID 到`y_hat`的映射 ImageNet 类名。我们将加载这个`imagenet_class_index.json` JSON 文件并获取预测的索引。

```
import json

imagenet_class_index = json.load(open('https://pytorch.org/tutorials/_static/imagenet_class_index.json'))

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]
```

在使用`imagenet_class_index`字典之前，首先我们将转换张量值为字符串值，因为`imagenet_class_index`字典中的键是字符串。我们将测试我们上面的方法：

```
with open("https://pytorch.org/tutorials/_static/img/sample_file.jpeg", 'rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes=image_bytes))
```

您应该得到如下响应：

```
['n02124075', 'Egyptian_cat']
```

数组中的第一项是 ImageNet类ID，第二项是人类可读名称。

> 注意
>
> 您是否注意到变量不是方法的一部分？或者为什么模型是全局变量？加载`model`模型在内存和计算方面操作成本高昂。如果我们在方法中加载模型，那么它将不必要地加载调用`get_prediction`方法的时间。由于我们正在构建一个 Web 服务器，因此每秒可能有数千个请求，我们不应该浪费时间为每个`get_prediction`推理冗余加载模型。所以，我们保留模型在内存中仅加载一次。在生产系统，有必要有效地使用计算能够大规模地处理请求，因此通常应该在处理请求之前加载模型。

## 将模型集成到我们的 API 服务器中

在最后一部分中，我们将模型添加到 Flask API 服务器。因为我们的 API 服务器应该获取一个图像文件`predict`，我们将更新我们的方法以从请求中读取文件：

```
from flask import request

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})
```

该`app.py`文件现已完成。以下是完整版本，取代包含保存文件的路径的路径，运行：

```
import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request


app = Flask(__name__)
imagenet_class_index = json.load(open('<PATH/TO/.json/FILE>/imagenet_class_index.json'))
model = models.densenet121(weights='IMAGENET1K_V1')
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()
```

让我们测试一下我们的网络服务器！运行：

```
$ FLASK_ENV=development FLASK_APP=app.py flask run
```

我们可以使用[请求库](https://pypi.org/project/requests/)向我们的应用程序发送 POST 请求：

```
import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('<PATH/TO/.jpg/FILE>/cat.jpg','rb')})
```

打印 *resp.json()* 现在将显示以下内容：

```
{"class_id": "n02124075", "class_name": "Egyptian_cat"}
```

## 后续步骤

我们编写的服务器非常简单，可能无法完成您的生产应用程序需要所有操作。所以，这里有一些事情你可以做得更好：

- 在请求中，终结点假定始终存在图像文件。这可能不适用于所有/predict请求。我们的用户可能会使用不同的参数发送图像或根本不发送任何图像。

- 用户也可以发送非图像类型的文件。由于我们不处理错误，这将破坏我们的服务器。添加显式错误处理引发异常的路径，将使我们能够更好地处理错误的输入

- 即使模型可以识别大量类别的图像，它可能无法识别所有图像。加强执行处理模型无法识别图像中的任何内容的情况。

- 我们在开发模式下运行 Flask 服务器，这不适合在生产环境中部署。您可以查看[本教程](https://flask.palletsprojects.com/en/1.1.x/tutorial/deploy/)，了解如何在生产环境中部署 Flask 服务器。

- 您还可以通过创建一个带有表单的页面来添加 UI，该页面采用图像和显示预测。查看类似项目的[演示](https://pytorch-imagenet.herokuapp.com/)及其[源代码](https://github.com/avinassh/pytorch-flask-api-heroku)。

- 在本教程中，我们只展示了如何构建一个可以返回一次一个图像预测的服务。我们可以修改我们的服务，以便能够返回一次多张图片。此外，服务[流处理器](https://github.com/ShannonAI/service-streamer)库会自动将请求排队到服务，并将它们采样到小批处理中可以将其输入到模型中。您可以查看[本教程](https://github.com/ShannonAI/service-streamer/wiki/Vision-Recognition-Service-with-Flask-and-service-streamer)。

- 最后，我们鼓励您查看有关部署 PyTorch 模型的其他教程链接到页面顶部。
