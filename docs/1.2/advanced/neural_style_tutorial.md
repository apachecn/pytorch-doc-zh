# 使用PyTorch进行神经网络传递

> **作者**：[Alexis Jacq](https://alexis-jacq.github.io)
>
> **编辑**：[Winston Herring](https://github.com/winston6)
> 
> 译者：[片刻](https://github.com/jiangzhonglian)
> 
> 校验：[片刻](https://github.com/jiangzhonglian)

## 简介

本教程介绍了如何实现 由Leon A. Gatys，Alexander S. Ecker和Matthias Bethge开发的[神经风格算法](https://arxiv.org/abs/1508.06576)。神经风格或神经传递，使您可以拍摄图像并以新的艺术风格对其进行再现。该算法获取三个图像，即输入图像，内容图像和样式图像，然后更改输入以使其类似于内容图像的内容和样式图像的艺术风格。

![https://pytorch.org/tutorials/_images/neuralstyle.png](https://pytorch.org/tutorials/_images/neuralstyle.png)

## 基本原理

原理很简单：我们定义了两个距离，一个用于内容 $$(D_S)$$ 和一种样式 $$(D_S)$$。 $$(D_C)$$ 测量两个图像之间的内容有多不同，而 $$(D_S)$$ 测量两个图像之间样式的差异。然后，我们获取第三个图像输入，并将其转换为最小化与内容图像的内容距离和与样式图像的样式距离。现在我们可以导入必要的包并开始神经传递。

## 导入软件包并选择设备

以下是实现神经传递所需的软件包列表。

* `torch`，`torch.nn`，`numpy`(与PyTorch神经网络包赛前必读）
* `torch.optim` (有效的梯度下降）
* `PIL`，`PIL.Image`，`matplotlib.pyplot`(加载和显示图像）
* `torchvision.transforms` (将PIL图像转换为张量）
* `torchvision.models` (训练或加载预训练模型）
* `copy` (以深层复制模型；系统包）

    
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
    

接下来，我们需要选择要在哪个设备上运行网络并导入内容和样式图像。 在大图像上运行神经传递算法需要更长的时间，并且在GPU上运行时会更快。 我们可以使用`torch.cuda.is_available()`来检测是否有GPU。 接下来，我们设置`torch.device`以在整个教程中使用。 `.to(device)`方法也用于将张量或模块移动到所需的设备。

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

## 加载图像

现在，我们将导入样式和内容图像。原始的PIL图像的值在0到255之间，但是当转换为Torch张量时，其值将转换为0到1之间。图像也需要调整大小以具有相同的尺寸。需要注意的一个重要细节是，使用从0到1的张量值对Torch库中的神经网络进行训练。如果尝试为网络提供0到255张量图像，则激活的特征图将无法感知预期的内容和风格。但是，来自Caffe库的预训练网络使用0到255张量图像进行训练。

> Note
> 通过以下链接下载到运行教程所需的图像：[picasso.jpg](https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg)和[dancing.jpg](https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg)。下载这两个图像并将它们添加到images当前工作目录中具有名称的目录中。 
    
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
    
现在，让我们创建一个通过将图像的副本转换为PIL格式并使用来显示图像的功能`plt.imshow`。我们将尝试显示内容和样式图像，以确保正确导入它们。
    
    
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
    

   ![https://pytorch.org/tutorials/_images/sphx_glr_neural_style_tutorial_001.png](https://pytorch.org/tutorials/_images/sphx_glr_neural_style_tutorial_001.png)
   ![https://pytorch.org/tutorials/_images/sphx_glr_neural_style_tutorial_002.png](https://pytorch.org/tutorials/_images/sphx_glr_neural_style_tutorial_002.png)

## 损失函数

### 内容损失

内容损失是代表单个图层内容距离的加权版本的函数。该功能获取特征图$$F_{XL}$$ 一层$$L$$在网络中处理输入$$X$$并返回加权内容距离$$w_{CL}.D_C^L(X,C)$$图像之间$$X$$和内容图片$$C$$。内容图像的特征图($$F_{CL}$$函数必须知道）才能计算内容距离。我们将此函数作为带有构造函数的torch模块来实现$$F_{CL}$$作为输入。距离$$\|F_{XL} - F_{CL}\|^2$$是两组要素图之间的均方误差，可以使用进行计算`nn.MSELoss`。

我们将直接在用于计算内容距离的卷积层之后添加此内容丢失模块。这样，每次向网络提供输入图像时，都会在所需层上计算内容损失，并且由于自动渐变，将计算所有梯度。现在，为了使内容丢失层透明，我们必须定义一种`forward`方法来计算内容丢失，然后返回该层的输入。计算出的损耗将保存为模块的参数。
    
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
    

> Note
> **重要细节**：尽管此模块名为`ContentLoss`，但它不是真正的PyTorch损失函数。如果要将内容损失定义为PyTorch损失函数，则必须创建一个PyTorch autograd函数以在`backward`方法中手动重新计算/实现梯度。

### 风格损失

风格损失模块类似地实现对内容的损失模块。它将作为其计算该层的风格损失的网络中的透明层。为了计算的样式的损失，我们需要计算克矩阵$$G_{XL}$$。甲克矩阵是通过它的转置矩阵的给定矩阵相乘的结果。在本申请中给出的矩阵是特征的重整的版本映射$$F_{XL}$$层$$L$$。$$F_{XL}$$重塑形成$$\hat{F}_{XL}$$，$$K$$ X $$N$$矩阵，其中 $$K$$ 是特征图中的层$$L$$和数$$N$$是任何量化特征地图$$F_{XL}^ķ$$的长度。例如，的第一行 $$\hat{F}_{XL}$$ 对应于第一量化特征地图 $$F_{XL}^1$$。

最后，克矩阵必须由在矩阵元素的总数量除以每个元素进行归一化。这种归一化是为了抵消这一事实$$\hat{F}_{XL}$$具有大$$N$$维产量较大的革兰氏矩阵值的矩阵。这些较大的值将导致第一层(池层之前），以具有梯度下降期间产生更大的影响。风格特征往往是在网络的更深层所以这归一化步骤是至关重​​要的。


    def gram_matrix(input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
    
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    
        G = torch.mm(features, features.t())  # compute the gram product
    
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

现在，样式丢失模块看起来几乎与内容丢失模块完全一样。样式距离也可以使用$$G_{XL}$$ 和 $$G_{SL}$$。
    
    class StyleLoss(nn.Module):
    
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()
    
        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input
    
## 导入模型

现在我们需要导入一个预训练的神经网络。我们将使用19层VGG网络，就像本文中使用的那样。

PyTorch的VGG实现是一个模块，分为两个子 `Sequential`模块：(`features`包含卷积和池化层）和`classifier`(包含完全连接的层）。我们将使用该`features`模块，因为我们需要各个卷积层的输出来测量内容和样式损失。某些层在训练期间的行为与评估不同，因此我们必须使用将网络设置为评估模式`.eval()`。

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    
另外，在图像上训练VGG网络，每个通道的均值通过 mean=[0.485，0.456，0.406]和 std=[0.229，0.224，0.225]归一化。在将其发送到网络之前，我们将使用它们对图像进行规范化。

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

`Sequential`模块包含子模块的有序列表。 例如，`vgg19.features`包含以正确的深度顺序对齐的序列(Conv2d，ReLU，MaxPool2d，Conv2d，ReLU…）。 我们需要在检测到的卷积层之后立即添加内容丢失层和样式丢失层。 为此，我们必须创建一个新`Sequential`模块，该模块具有正确插入的内容丢失和样式丢失模块。   

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
    
接下来，我们选择输入图像。您可以使用内容图像或白噪声的副本。
    
    input_img = content_img.clone()
    # if you want to use white noise instead uncomment the below line:
    # input_img = torch.randn(content_img.data.size(), device=device)
    
    # add the original input image to the figure:
    plt.figure()
    imshow(input_img, title='Input Image')
    

![https://pytorch.org/tutorials/_images/sphx_glr_neural_style_tutorial_003.png](https://pytorch.org/tutorials/_images/sphx_glr_neural_style_tutorial_003.png)

## 梯度下降

正如该算法的作者Leon Gatys在[此处](https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq)建议的那样，我们将使用L-BFGS算法来运行我们的梯度下降。与训练网络不同，我们希望训练输入图像以最大程度地减少内容/样式损失。我们将创建一个PyTorch L-BFGS优化器`optim.LBFGS`，并将图像作为张量传递给它进行优化。

    def get_input_optimizer(input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer
    
最后，我们必须定义一个执行神经传递的函数。对于网络的每次迭代，它都会被提供更新的输入并计算新的损耗。我们将运行`backward`每个损失模块的方法来动态计算其梯度。优化器需要“关闭”功能，该功能可以重新评估模数并返回损耗。

我们还有最后一个约束要解决。网络可能会尝试使用超出图像的0到1张量范围的值来优化输入。我们可以通过在每次网络运行时将输入值校正为0到1之间来解决此问题。
    
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
    
最后，我们可以运行算法。
    
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)
    
    plt.figure()
    imshow(output, title='Output Image')
    
    # sphinx_gallery_thumbnail_number = 4
    plt.ioff()
    plt.show()
    

![https://pytorch.org/tutorials/_images/sphx_glr_neural_style_tutorial_004.png](https://pytorch.org/tutorials/_images/sphx_glr_neural_style_tutorial_004.png)

Out:

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
    

**脚本的总运行时间：** (1分钟9.573秒）

[`Download Python source code:
neural_style_tutorial.py`](../_downloads/7d103bc16c40d35006cd24e65cf978d0/neural_style_tutorial.py)

[`Download Jupyter notebook:
neural_style_tutorial.ipynb`](../_downloads/f16c4cab7b50f6dea0beb900dee4bf0e/neural_style_tutorial.ipynb)
