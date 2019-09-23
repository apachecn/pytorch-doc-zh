# 神经网络传输使用PyTorch

**作者** ：[亚历克西黄灯笼](https://alexis-jacq.github.io)

**由** 编辑：[温斯顿鲱鱼](https://github.com/winston6)

## 简介

本教程介绍了如何实现[神经类型算法](https://arxiv.org/abs/1508.06576)开发由Leon A.
Gatys，亚历山大S.埃克和Matthias贝特格。神经风格，或神经传输，让您拍摄图像，并用新的艺术风格重现。该算法需要三个图像，输入图像，内容的图像，和一个样式图象，并且改变输入到类似于内容的图像的内容和样式图像的艺术风格。

![content1](../_images/neuralstyle.png)

## 基本原理

原理很简单：我们定义两个距离，一个是内容（ \（D_C \）HTG1]），一个用于样式（ \（D_S \）HTG3]）。  \（D_C
\）的含量如何不同是两个图像之间的措施而 \（D_S
\）措施的样式如何不同是两个图像之间。然后，我们把第三图像，输入，并将其转换以最大限度地减小其内容的距离与内容的图像，并与风格像它的风格距离。现在，我们可以导入必要的软件包，并开始神经传递。

## 导入包和选择设备

下面是实现神经传送所需的包的列表。

  * `炬 `，`torch.nn`，`numpy的 `（包赛前必读用于与神经网络PyTorch）
  * `torch.optim`（有效梯度下坡）
  * `PIL`，`PIL.Image`，`matplotlib.pyplot`（负载和显示图像）
  * `torchvision.transforms`（变换PIL图像转换成张量）
  * `torchvision.models`（火车或载预训练的模型）
  * `复制 `（深拷贝的模型;系统封装）

    
    
    from __future__ import print_function
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    
    from PIL import Image
    import matplotlib.pyplot as plt
    
    import torchvision.transforms as transforms
    import torchvision.models as models
    
    import copy
    

接下来，我们需要选择在运行网络，设备和导入的内容和风格的图像。大图像运行的神经传递算法需要更长的时间，并在GPU上运行时，会快很多。我们可以使用`
torch.cuda.is_available（） `，以检测是否有可用的GPU。接下来，我们设置了`torch.device
`在整个教程中使用。另外，`。要（装置） `方法用于张量或模块移动到期望的设备。

    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

## 加载图像

现在，我们将导入的风格和内容的图像。原始PIL图像具有值0和255之间，但是，当转化入火炬张量，它们的值被转换为0和1之间的图像也需要被调整到具有相同的尺寸。要注意的重要细节是，从火炬库神经网络与张量值范围从0到1的培训。如果你尝试用0到255张图像喂网络，然后激活功能的地图将无法意义上的预期内容和风格。但是，从来自Caffe库预训练的网络被训练，用0至255张量的图像。

Note

通过以下链接下载到运行教程所需的图像：[ picasso.jpg
](https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg)和[
dancing.jpg [HTG3。下载这两个图像，并将它们与名称`图像
`在当前工作目录添加到目录中。](https://pytorch.org/tutorials/_static/img/neural-
style/dancing.jpg)

    
    
    # desired size of the output image
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
    
    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    
    
    def image_loader(image_name):
        image = Image.open(image_name)
        # fake batch dimension required to fit network's input dimensions
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)
    
    
    style_img = image_loader("./data/images/neural-style/picasso.jpg")
    content_img = image_loader("./data/images/neural-style/dancing.jpg")
    
    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"
    

现在，让我们创建一个由再转它的一个副本，以PIL格式，并使用`plt.imshow
`显示复制显示图像的功能。我们会尽量显示内容和风格的图像，以确保他们正确导入。

    
    
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    
    plt.ion()
    
    def imshow(tensor, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001) # pause a bit so that plots are updated
    
    
    plt.figure()
    imshow(style_img, title='Style Image')
    
    plt.figure()
    imshow(content_img, title='Content Image')
    

  * ![../_images/sphx_glr_neural_style_tutorial_001.png](../_images/sphx_glr_neural_style_tutorial_001.png)
  * ![../_images/sphx_glr_neural_style_tutorial_002.png](../_images/sphx_glr_neural_style_tutorial_002.png)

## 损失函数

### 丢失内容

内容损失是代表内容的距离为一个单独层的加权的版本的功能。该函数采用特征地图 \（F_ {XL} \）的层 \（L \）在一个网络处理输入 \（X
\）并返回该内容加权距离 \（W_ {CL} .D_C ^ L（X，C）\）的图像之间 \（X \）和内容图像 \（C \）。所述内容图像的特征映射（
\（F_ {CL} \））必须由功能，以计算其含量距离是已知的。我们用一个构造函数 \（F_ {CL} \）作为输入实现该功能作为炬模块。的距离 \（\ |
F_ {XL} - F_ {CL} \ | ^ 2 \）是两组特征地图之间的均方误差，并且可以使用`[计算HTG19 ] nn.MSELoss `。

我们将直接正在用于计算内容距离的卷积层（一个或多个）之后添加该内容损耗模块。此每个网络被供给的输入图像内容的损失将在所希望的层被计算并因为汽车研究所的，所有的梯度将被计算时间的方法。现在，为了使内容损耗层透明我们必须定义计算含量损失，然后返回层的输入`
向前 `方法。所计算的损失被保存为模块的参数。

    
    
    class ContentLoss(nn.Module):
    
        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            # we 'detach' the target content from the tree used
            # to dynamically compute the gradient: this is a stated value,
            # not a variable. Otherwise the forward method of the criterion
            # will throw an error.
            self.target = target.detach()
    
        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input
    

Note

**重要细节** ：虽然这个模块被命名为`ContentLoss
`，它是不是一个真正的PyTorch损失函数。如果你要定义你的内容的损失为PyTorch损失函数，你必须创建一个PyTorch
autograd功能重新计算/在`后退 `方法手动实现梯度。

### 风格损失

风格损失模块类似地实现对内容的损失模块。它将作为其计算该层的风格损失的网络中的透明层。为了计算的样式的损失，我们需要计算克矩阵 \（G_ {XL}
\）。甲克矩阵是通过它的转置矩阵的给定矩阵相乘的结果。在本申请中给出的矩阵是特征的重整的版本映射 \（F_ {XL} \）层 \（L \） 。  \（F_
{XL} \）被整形以形成 \（\帽子{F} _ {XL} \），A  \（K \） X  \（N \）矩阵，其中 \（K \）是特征图中的层 \（L
\）和数\ （N \）是任何量化特征地图 \（F_ {XL} ^ķ\）的长度。例如，的第一行\（\帽子{F} _ {XL} \）对应于第一量化特征地图
\（F_ {XL} ^ 1 \）。

最后，克矩阵必须由在矩阵元素的总数量除以每个元素进行归一化。这种归一化是为了抵消这一事实 \（\帽子{F} _ {XL} \）具有大 \（N
\）维产量较大的革兰氏矩阵值的矩阵。这些较大的值将导致第一层（池层之前），以具有梯度下降期间产生更大的影响。风格特征往往是在网络的更深层所以这归一化步骤是至关重​​要的。

    
    
    def gram_matrix(input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
    
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    
        G = torch.mm(features, features.t())  # compute the gram product
    
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)
    

现在的风格损耗模块看起来几乎完全一样的内容损失模块。样式距离使用之间的均方误差还计算\（G_ {XL} \）和 \（G_ {SL} \）。

    
    
    class StyleLoss(nn.Module):
    
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()
    
        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input
    

## 导入模型

现在，我们需要进口预训练的神经网络。我们将使用19层VGG网络就像在纸中使用的一个。

PyTorch的实现VGG的是分成两个子`序贯 `模块的模块：`特征 `（含有卷积和集中层），和`分类 `（含有完全连接层）。我们将使用`功能
`模块，因为我们需要的个体卷积层的输出来衡量内容和风格的损失。一些层具有比训练评估过程中不同的行为，所以我们必须用`.eval（） `设置网络为评估模式。

    
    
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    

此外，VGG网络上的图像训练与由平均归一化每个信道= [0.485，0.456，0.406]和std =
[0.229，0.224，0.225]。我们将使用它们发送到其网络之前正常化的形象。

    
    
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    # create a module to normalize input image so we can easily put it in a
    # nn.Sequential
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)
    
        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std
    

A `顺序 `模块包含的子模块的有序列表。例如，`vgg19.features
`包含在深度的正确的顺序排列的序列（Conv2d，RELU，MaxPool2d，Conv2d，RELU
...）。我们需要他们检测的卷积层后立即加入我们的内容损失和风格损失层。要做到这一点，我们必须创建一个具有内容损失和风格损失模块正确地插入一个新的`顺序
`模块。

    
    
    # desired depth layers to compute style/content losses :
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    
    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        cnn = copy.deepcopy(cnn)
    
        # normalization module
        normalization = Normalization(normalization_mean, normalization_std).to(device)
    
        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []
    
        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)
    
        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
    
            model.add_module(name, layer)
    
            if name in content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)
    
            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
    
        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
    
        model = model[:(i + 1)]
    
        return model, style_losses, content_losses
    

接下来，我们选择输入图像。您可以使用内容的图像或白噪声的副本。

    
    
    input_img = content_img.clone()
    # if you want to use white noise instead uncomment the below line:
    # input_img = torch.randn(content_img.data.size(), device=device)
    
    # add the original input image to the figure:
    plt.figure()
    imshow(input_img, title='Input Image')
    

![../_images/sphx_glr_neural_style_tutorial_003.png](../_images/sphx_glr_neural_style_tutorial_003.png)

## 梯度下降

正如莱昂Gatys，算法的作者，建议[此处](https://discuss.pytorch.org/t/pytorch-tutorial-for-
neural-transfert-of-artistic-style/336/20?u=alexis-jacq)，我们将使用L-
BFGS算法来运行我们的梯度下降。训练不同的网络，我们希望培养的输入图像，以尽量减少对内容/格式的损失。我们将创建一个PyTorch L-BFGS优化`
optim.LBFGS`和我们的形象传递给它的张量来优化。

    
    
    def get_input_optimizer(input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer
    

最后，我们必须定义执行的神经传递的功能。对于网络中的每个迭代中，它被馈送的更新的输入，并计算新的损失。我们将运行`后退
`每个损耗模块的方法来dynamicaly计算其梯度。优化需要一个“关闭”功能，重新评估的模件，并返回损失。

我们还有最后一个约束来解决。该网络可以尝试与超过该图像的0到1张量范围内的值，以优化的输入。我们可以通过校正所述输入值是网络运行每次之间0至1解决这个问题。

    
    
    def run_style_transfer(cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=300,
                           style_weight=1000000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)
        optimizer = get_input_optimizer(input_img)
    
        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:
    
            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)
    
                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0
    
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
    
                style_score *= style_weight
                content_score *= content_weight
    
                loss = style_score + content_score
                loss.backward()
    
                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
    
                return style_score + content_score
    
            optimizer.step(closure)
    
        # a last correction...
        input_img.data.clamp_(0, 1)
    
        return input_img
    

最后，我们可以运行的算法。

    
    
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)
    
    plt.figure()
    imshow(output, title='Output Image')
    
    # sphinx_gallery_thumbnail_number = 4
    plt.ioff()
    plt.show()
    

![../_images/sphx_glr_neural_style_tutorial_004.png](../_images/sphx_glr_neural_style_tutorial_004.png)

日期：

    
    
    Building the style transfer model..
    Optimizing..
    run [50]:
    Style Loss : 4.169304 Content Loss: 4.235329
    
    run [100]:
    Style Loss : 1.145476 Content Loss: 3.039176
    
    run [150]:
    Style Loss : 0.716769 Content Loss: 2.663749
    
    run [200]:
    Style Loss : 0.476047 Content Loss: 2.500893
    
    run [250]:
    Style Loss : 0.347092 Content Loss: 2.410895
    
    run [300]:
    Style Loss : 0.263698 Content Loss: 2.358449
    

**脚本的总运行时间：** （1分钟9.573秒）

[`Download Python source code:
neural_style_tutorial.py`](../_downloads/7d103bc16c40d35006cd24e65cf978d0/neural_style_tutorial.py)

[`Download Jupyter notebook:
neural_style_tutorial.ipynb`](../_downloads/f16c4cab7b50f6dea0beb900dee4bf0e/neural_style_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](../beginner/fgsm_tutorial.html "Adversarial Example Generation")
[![](../_static/images/chevron-right-orange.svg)
Previous](../intermediate/spatial_transformer_tutorial.html "Spatial
Transformer Networks Tutorial")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * 神经网络传输使用PyTorch 
    * 介绍
    * 基本原理
    * 导入包和选择设备
    * 加载图像
    * 损失函数
      * 内容丢失
      * 风格损失
    * 导入模型
    * 梯度下降

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)

## 文件

对于PyTorch访问完整的开发文档

[View Docs](https://pytorch.org/docs/stable/index.html)

## 教程

获取详细的教程，对于初学者和高级开发者

[View Tutorials](https://pytorch.org/tutorials)

## 资源

查找开发资源，并得到回答您的问题

[View Resources](https://pytorch.org/resources)

[](https://pytorch.org/)

  * [ PyTorch ](https://pytorch.org/)
  * [入门](https://pytorch.org/get-started)
  * [特点](https://pytorch.org/features)
  * [生态系统](https://pytorch.org/ecosystem)
  * [博客](https://pytorch.org/blog/)
  * [资源](https://pytorch.org/resources)

  * [支持](https://pytorch.org/support)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [讨论](https://discuss.pytorch.org)
  * [ Github的问题](https://github.com/pytorch/pytorch/issues)
  * [松弛](https://pytorch.slack.com)
  * [贡献](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md)

  * 跟着我们
  * 邮箱地址

[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)

  * 入门
  * 特点
  * 生态系统
  * [博客](https://pytorch.org/blog/)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [资源](https://pytorch.org/resources)
  * [ Github的](https://github.com/pytorch/pytorch)

