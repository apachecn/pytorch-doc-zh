# 1.通过REST API与部署PyTorch在Python烧瓶

**作者** ：[阿维纳什Sajjanshetty ](https://avi.im)

在本教程中，我们将使用瓶部署PyTorch模型和暴露的模型推断一个REST API。特别是，我们将部署一个预训练DenseNet 121模型检测的图像。

小费

这里使用的所有的代码是在MIT许可下发布，并可在[ Github上[HTG1。](https://github.com/avinassh/pytorch-
flask-api)

这代表了一个教程系列的第一个在生产中部署PyTorch车型。以这种方式使用瓶是目前为止最简单的方法来启动服务您PyTorch车型，但它不是一个用例性能要求较高的工作。为了那个原因：

> [HTG2如果你已经很熟悉TorchScript，您可以直接跳到我们的[加载++
](https://pytorch.org/tutorials/advanced/cpp_export.html)一个TorchScript模型用C教程。

>   * 如果你首先需要在TorchScript复习，看看我们的【HTG0]介绍一个TorchScript 教程。

>

## API定义

首先，我们将定义我们的API端点，请求和响应类型。我们的API端点将位于`/预测 `这需要HTTP POST请求与包含该图像的`文件
`参数。该响应将是包含预测JSON响应的：

    
    
    {"class_id": "n02124075", "class_name": "Egyptian_cat"}
    

## 依赖性

安装运行下面的命令所需的依赖关系：

    
    
    $ pip install Flask==1.0.3 torchvision-0.3.0
    

## 简单的Web服务器

下面是一个简单的Web服务器，从瓶资料为准

    
    
    from flask import Flask
    app = Flask(__name__)
    
    
    @app.route('/')
    def hello():
        return 'Hello World!'
    

保存上面的代码在一个名为`app.py`您现在可以通过键入运行瓶开发服务器文件：

    
    
    $ FLASK_ENV=development FLASK_APP=app.py flask run
    

当您访问`HTTP：//本地主机：5000 /`在你的网页浏览器，你将与`你好 世界打招呼！`文本

我们将上述片断的轻微变化，所以它适合我们的API定义。首先，我们将重命名为`预测 `的方法。我们将更新端点路径`/预测
[HTG7。由于图像文件将通过HTTP POST请求被发送，我们会随时更新，以便它也只接受POST请求：`

    
    
    @app.route('/predict', methods=['POST'])
    def predict():
        return 'Hello World!'
    

我们也将改变响应类型，因此它返回一个包含ImageNet类ID和名称的JSON响应。更新`app.py`文件将是现在：

    
    
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})
    

## 推理

在接下来的章节中，我们将集中精力编写推理代码。这将涉及到两个部分，一是我们准备的图像，以便它可以被输送到DenseNet和明年，我们将编写代码即可获得从模型的实际预测。

### 准备图像

DenseNet模型需要图像是尺寸224 X
224的3通道RGB图像的我们也将正常化与所需的平均和标准偏差值的图像张量。你可以阅读更多关于它的[此处[HTG1。](https://pytorch.org/docs/stable/torchvision/models.html)

我们将使用`变换 `从`torchvision
`库，并建立一个管道改造的要求，它改变我们的图像。你可以阅读更多关于变换[此处[HTG9。](https://pytorch.org/docs/stable/torchvision/transforms.html)

    
    
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
    

上述方法需要图像数据以字节为单位，应用一系列的变换，并返回一个张量。为了检验上述方法，读取字节模式下的图像文件(第一替换
../_static/img/sample_file.jpeg 的实际路径到计算机上的文件），看看如果你得到一个张量背部：

    
    
    with open("../_static/img/sample_file.jpeg", 'rb') as f:
        image_bytes = f.read()
        tensor = transform_image(image_bytes=image_bytes)
        print(tensor)
    

日期：

    
    
    tensor([[[[ 0.4508,  0.4166,  0.3994,  ..., -1.3473, -1.3302, -1.3473],
              [ 0.5364,  0.4851,  0.4508,  ..., -1.2959, -1.3130, -1.3302],
              [ 0.7077,  0.6392,  0.6049,  ..., -1.2959, -1.3302, -1.3644],
              ...,
              [ 1.3755,  1.3927,  1.4098,  ...,  1.1700,  1.3584,  1.6667],
              [ 1.8893,  1.7694,  1.4440,  ...,  1.2899,  1.4783,  1.5468],
              [ 1.6324,  1.8379,  1.8379,  ...,  1.4783,  1.7352,  1.4612]],
    
             [[ 0.5728,  0.5378,  0.5203,  ..., -1.3704, -1.3529, -1.3529],
              [ 0.6604,  0.6078,  0.5728,  ..., -1.3004, -1.3179, -1.3354],
              [ 0.8529,  0.7654,  0.7304,  ..., -1.3004, -1.3354, -1.3704],
              ...,
              [ 1.4657,  1.4657,  1.4832,  ...,  1.3256,  1.5357,  1.8508],
              [ 2.0084,  1.8683,  1.5182,  ...,  1.4657,  1.6583,  1.7283],
              [ 1.7458,  1.9384,  1.9209,  ...,  1.6583,  1.9209,  1.6408]],
    
             [[ 0.7228,  0.6879,  0.6531,  ..., -1.6476, -1.6302, -1.6476],
              [ 0.8099,  0.7576,  0.7228,  ..., -1.6476, -1.6476, -1.6650],
              [ 1.0017,  0.9145,  0.8797,  ..., -1.6476, -1.6650, -1.6999],
              ...,
              [ 1.6291,  1.6291,  1.6465,  ...,  1.6291,  1.8208,  2.1346],
              [ 2.1868,  2.0300,  1.6814,  ...,  1.7685,  1.9428,  2.0125],
              [ 1.9254,  2.0997,  2.0823,  ...,  1.9428,  2.2043,  1.9080]]]])
    

### 预测

现在将使用预训练DenseNet 121模型预测图像类。我们将使用一个从`torchvision
`库，加载模型，并得到一个推论。虽然我们将在这个例子中使用预训练的模型，你可以使用自己的模型同样的方法。查看更多有关此[ 教程
](../beginner/saving_loading_models.html)加载你的模型。

    
    
    from torchvision import models
    
    # Make sure to pass `pretrained`as `True`to use the pretrained weights:
    model = models.densenet121(pretrained=True)
    # Since we are using our model only for inference, switch to `eval`mode:
    model.eval()
    
    
    def get_prediction(image_bytes):
        tensor = transform_image(image_bytes=image_bytes)
        outputs = model.forward(tensor)
        _, y_hat = outputs.max(1)
        return y_hat
    

预测的类ID的张量`y_hat
`将包含索引。然而，我们需要人类可读的类名。为此，我们需要一个等级ID名称映射。下载[这个文件](https://s3.amazonaws.com/deep-
learning-models/image-models/imagenet_class_index.json)为`
imagenet_class_index.json`，并记住您保存它(或者，如果你是以下在本教程中的具体步骤，它保存在教程/ _static
）。此文件包含ImageNet类ID来ImageNet类名的映射。我们将加载这个JSON文件，并得到预测指数的类名。

    
    
    import json
    
    imagenet_class_index = json.load(open('../_static/imagenet_class_index.json'))
    
    def get_prediction(image_bytes):
        tensor = transform_image(image_bytes=image_bytes)
        outputs = model.forward(tensor)
        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())
        return imagenet_class_index[predicted_idx]
    

使用`imagenet_class_index`词典之前，我们首先将张量的值转换为字符串值，因为在`imagenet_class_index
`字典中的键是字符串。我们将测试我们上面的方法：

    
    
    with open("../_static/img/sample_file.jpeg", 'rb') as f:
        image_bytes = f.read()
        print(get_prediction(image_bytes=image_bytes))
    

Out:

    
    
    ['n02124075', 'Egyptian_cat']
    

你应该得到这样的回应：

    
    
    ['n02124075', 'Egyptian_cat']
    

在阵列中的第一项是ImageNet类ID和第二项是人类可读的名称。

Note

你有没有注意到`模型 `变量不是`get_prediction
`方法的一部分？为什么是模型中的全局变量？加载模型可以是在存储器和计算方面是昂贵的操作。如果我们在`get_prediction
`方法加载模型，那么它会得到不必要加载的每一个方法被调用的时间。因为，我们正在建立一个Web服务器，有可能是每秒数千次的请求，我们不应该浪费时间冗余负载对每个推理模型。所以，我们一直在内存中加载只有一次的模型。在生产系统中，有必要提高效率你计算的使用能够在大规模服务请求，所以你一般应为请求提供服务之前加载模型。

## 在我们的API服务器集成模型

在这最后一部分，我们将我们的模型添加到我们的瓶API服务器。由于我们的API服务器应该采取一个图像文件，我们会随时更新我们的`预测
`方法来读取请求的文件：

    
    
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
    

在`app.py`文件现已完成。以下是完整版;与你保存的文件，它应该运行的路径替换路径：

    
    
    import io
    import json
    
    from torchvision import models
    import torchvision.transforms as transforms
    from PIL import Image
    from flask import Flask, jsonify, request
    
    
    app = Flask(__name__)
    imagenet_class_index = json.load(open('<PATH/TO/.json/FILE>/imagenet_class_index.json'))
    model = models.densenet121(pretrained=True)
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
    

让我们来测试我们的网络服务器！跑：

    
    
    $ FLASK_ENV=development FLASK_APP=app.py flask run
    

我们可以使用[请求](https://pypi.org/project/requests/)库发送POST请求到我们的应用程序：

    
    
    import requests
    
    resp = requests.post("http://localhost:5000/predict",
                         files={"file": open('<PATH/TO/.jpg/FILE>/cat.jpg','rb')})
    

印刷 resp.json(）现在会显示以下内容：

    
    
    {"class_id": "n02124075", "class_name": "Egyptian_cat"}
    

## 接下来的步骤

我们写的服务器是很琐碎，可能不是你所需要的生产应用程序的一切。所以，这里有一些事情可以做，以更好地使其：

  * 端点`/预测 `假定总是会有在该请求的图像文件。这可能不是适用于所有要求如此。我们的用户可以发送图像具有不同的参数或者根本不发送图像。
  * 用户可以发送过多非图像类型的文件。由于我们没有处理错误，这将打破我们的服务器。并称将抛出一个异常明确的处理错误的道路，使我们能够更好地处理无效输入
  * 尽管该模型可识别大量的图像类，也未必能够识别的所有图像。加强对办案时模型无法识别图像中的任何实施。
  * 我们运行的发展模式，这是不适合在生产部署瓶服务器。您可以检查出[本教程](https://flask.palletsprojects.com/en/1.1.x/tutorial/deploy/)在生产部署瓶服务器。
  * 您也可以通过创建与需要的图像，并显示预测的形式添加页面的UI。检查出一个类似的项目和它的[源代码](https://github.com/avinassh/pytorch-flask-api-heroku)的[演示[HTG1。](https://pytorch-imagenet.herokuapp.com/)
  * 在本教程中，我们只展示了如何构建，可以在同一时间返回预测单个图像服务。我们可以修改我们的服务能马上回家多个图像的预测。此外，[服务流光](https://github.com/ShannonAI/service-streamer)库自动排队请求您的服务和样品它们变成可被送入模型迷你批次。您可以检查出[本教程[HTG3。](https://github.com/ShannonAI/service-streamer/wiki/Vision-Recognition-Service-with-Flask-and-service-streamer)
  * 最后，我们建议您检查出部署PyTorch模型链接到页面的顶部我们的其他教程。

**脚本的总运行时间：** (0分钟0.925秒）
