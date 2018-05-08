# -*- coding: utf-8 -*-
"""
用 PyTorch 做 神经转换 (Neural Transfer) 
==========================================
**原作者**: `Alexis Jacq <https://alexis-jacq.github.io>`_

介绍
------------

欢迎观看! 这篇教程解释了如何实现 Leon A. Gatys, Alexander S. Ecker 和 
Matthias Bethge 几位学者发明的
`Neural-Style <https://arxiv.org/abs/1508.06576>`__ 算法 .

题中的神经描述的是什么?
~~~~~~~~~~~~~~~~~~

神经风格, 或者说神经转换是一种算法, 它输入一张内容图像 (例如海龟), 一张风格图像 
(例如艺术波浪), 然后返回内容图像的内容, 此时返回的内容像是被艺术风格图像的风格渲染过: 

.. figure:: /_static/img/neural-style/neuralstyle.png
   :alt: content1

它是如何工作的?
~~~~~~~~~~~~~~~~~~

原理很简单: 我们定义两个距离, 一个是关于内容的 (:math:`D_C`) , 另一个是关于风格的 
(:math:`D_S`) . :math:`D_C` 衡量两张图像的内容有多么不同, 而 :math:`D_S` 
衡量两张图像的风格有多么不同. 接着我们拿出我们的输入, 也就是第三张图像 (例如全噪声), 
然后我们转换它, 同时最小化它与内容图像的内容距离和它与风格图像的风格距离.

好吧, 它具体是怎么工作的?
^^^^^^^^^^^^^^^^^^^^^^

继续深入需要一些数学知识. 令 :math:`C_{nn}` 代表一个预训练好的深度卷积神经网络, 
:math:`X` 代表任何图像. :math:`C_{nn}(X)` 是神经网络输入 :math:`X` 后的结果
(包括在所有层的特征映射). 令 :math:`F_{XL} \in C_{nn}(X)` 代表在深度为 :math:`L` 
层处的特征映射, 都矢量化和级联为一个单一矢量. 我们简单地用 :math:`F_{XL}` 定义 
:math:`X` 在 :math:`L` 层的内容. 如果 :math:`Y` 是另一张和 :math:`X` 相同大小的图像, 
我们定义这两张图像在 :math:`L` 层的内容距离如下:

.. math:: D_C^L(X,Y) = \|F_{XL} - F_{YL}\|^2 = \sum_i (F_{XL}(i) - F_{YL}(i))^2

式中 :math:`F_{XL}(i)` 是 :math:`F_{XL}` 的第 :math:`i^{th}` 个元素.
定义风格要更繁琐一些. 令满足 :math:`k \leq K` 的 :math:`F_{XL}^k` 代表  
:math:`L` 层矢量化的 :math:`K` 个特征映射中的第 :math:`k^{th}` 个. 
图像 :math:`X` 在 :math:`L` 层的风格 :math:`G_{XL}` 定义为满足 
:math:`k \leq K` 的所有矢量化特征映射 :math:`F_{XL}^k` 的克产物 (Gram produce). 
换句话说, :math:`G_{XL}` 是一个 :math:`K`\ x\ :math:`K` 的矩阵, 其在 
:math:`k^{th}` 行和 :math:`l^{th}` 列的每个元素 :math:`G_{XL}(k,l)` 
是 :math:`F_{XL}^k` 和 :math:`F_{XL}^l` 之间的矢量产物 :

.. math::

    G_{XL}(k,l) = \langle F_{XL}^k, F_{XL}^l\\rangle = \sum_i F_{XL}^k(i) . F_{XL}^l(i)

式中 :math:`F_{XL}^k(i)` 是 :math:`F_{XL}^k` 的第 :math:`i^{th}` 个元素. 
我们可以把 :math:`G_{XL}(k,l)` 当做特征映射 :math:`k` 和 :math:`l` 相关性的衡量. 
那样的话, :math:`G_{XL}` 代表了 :math:`X` 在 :math:`L` 层特征向量的相关性矩阵. 
注意 :math:`G_{XL}` 的尺寸只决定于特征映射的数量, 不被 :math:`X` 的尺寸所影响. 
然后如果 :math:`Y` 是 *任意尺寸* 的另一张图像, 我们定义在 :math:`L` 层的风格距离如下: 

.. math::

    D_S^L(X,Y) = \|G_{XL} - G_{YL}\|^2 = \sum_{k,l} (G_{XL}(k,l) - G_{YL}(k,l))^2

要想一次性地在一些层最小化一个可变图像 :math:`X` 与目标内容图像 :math:`C` 
间的 :math:`D_C(X,C)`, 和 :math:`X` 与目标风格图像 :math:`S` 间的 
:math:`D_S(X,S)` , 我们计算并加和每个目标层每个距离的梯度 (对 :math:`X` 求导).

.. math::

    \\nabla_{\textit{total}}(X,S,C) = \sum_{L_C} w_{CL_C}.\\nabla_{\textit{content}}^{L_C}(X,C) + \sum_{L_S} w_{SL_S}.\\nabla_{\textit{style}}^{L_S}(X,S)

式中 :math:`L_C` 和 :math:`L_S` 分别是内容和风格的目标层(任意陈述), 
:math:`w_{CL_C}` 和 :math:`w_{SL_S}` 是风格和内容关于每个目标层的权重(任意陈述). 
然后我们对 :math:`X` 进行梯度下降:

.. math:: X \leftarrow X - \\alpha \\nabla_{\textit{total}}(X,S,C)

好吧, 数学的部分就到此为止. 如果你想要更加深入(比如怎么计算梯度),  **我们推荐你阅读原始论文** (作者是 Leon
A. Gatys 和 AL), 论文中这部分解释地更好更清晰. 

对于在 PyTorch 中的实现, 我们已经有了我们需要的一切: 
事实上就是 PyTorch, 所有的梯度都被为你自动且动态地计算(当你从库中使用函数时). 
这就是为什么算法的实现在 PyTorch 中变得非常轻松.

PyTorch 实现
----------------------

如果你不确定是否理解了以上数学公式, 你也可以通过实现它, 在过程中有所领悟. 
如果你正在探索 PyTorch , 我们推荐你先阅读这篇教程 

包
~~~~~~~~

我们将会依赖下列这些包:

-  ``torch``, ``torch.nn``, ``numpy`` (indispensables packages for
   neural networks with PyTorch)
-  ``torch.optim`` (efficient gradient descents)
-  ``PIL``, ``PIL.Image``, ``matplotlib.pyplot`` (load and display
   images)
-  ``torchvision.transforms`` (treat PIL images and transform into torch
   tensors)
-  ``torchvision.models`` (train or load pre-trained models)
-  ``copy`` (to deep copy the models; system package)
"""

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


######################################################################
# Cuda
# ~~~~
#
# 如果你的计算机里有 GPU, 推荐在上面运行算法, 尤其是当你要尝试
# 大型网络时 (就像 VGG). 有鉴于此, 我们有 ``torch.cuda.is_available()``, 
# 如果你的计算机有可用 GPU 则会返回 True. 然后我们用 ``.to(device)`` 方法
# 将张量或模块送到想要送的设备上. 当我们想将这些模块重新移回 CPU 
# 的时候(比如要用 numpy), 我们用 ``.cpu()`` 方法. 
#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# 读取图像
# ~~~~~~~~~~~
#
# 为了简化其实现, 让我们从导入一个相同维度的风格和内容图像开始. 
# 然后我们将它们缩放到想要的输入图像尺寸 (在例子中是 128 和 512, 
# 取决你的 GPU 是否可用) 然后把它们转化为 torch 张量, 以待喂入一个神经网络.
#
# .. Note::
#     这里是教程需要的图像的下载链接: 
#     `picasso.jpg <http://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg>`__ 和
#     `dancing.jpg <http://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg>`__.
#     下载这两张图像然后把它们加入到名为 ``images`` 的目录中.
# 


# 想要的输出图像尺寸
imsize = 512 if torch.cuda.is_available() else 128  # 如果没有 GPU 则使用小尺寸

loader = transforms.Compose([
    transforms.Resize(imsize),  # 缩放图像
    transforms.ToTensor()])  # 缩放图像


def image_loader(image_name):
    image = Image.open(image_name)
    # 由于神经网络输入的需要, 添加 batch 的维度
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("images/picasso.jpg")
content_img = image_loader("images/dancing.jpg")

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"


######################################################################
# 导入的 PIL 图像像素值的范围为 0 到 255. 转化为 torch 张量后, 
# 它们的值范围变为了 0 到and 1. 这是个重要的细节:
# torch 库中的神经网络被使用 0-1 的张量图像训练. 如果你尝试用 
# 0-255 的张量图像喂入神经网络, 激活的特征映射就没用了. 这不是
# 使用 Caffe 库中预训练的神经网络, Caffe 中是用 0-255 的张量图像训练的. 
#
# 显示图像
# ~~~~~~~~~~~~~~
#
# 我们将使用 ``plt.imshow`` 来显示图像. 
# 所以我们需要先把它们转回 PIL 图像. 
#

unloader = transforms.ToPILImage()  # 转回 PIL 图像

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # 克隆是为了不改变它
    image = image.squeeze(0)      # 移除 batch 维度
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # 暂停一会, 让绘图更新


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')


######################################################################
# 内容损失
# ~~~~~~~~~~~~
#
# 内容损失是一个在网络输入为 :math:`X` 的层 :math:`L` 输入特征映射 
# :math:`F_{XL}` 的函数, 返回此图像与内容图像间的加权内容距离 
# :math:`w_{CL}.D_C^L(X,C)`. 之后, 权重 :math:`w_{CL}` 和目标内容 
# :math:`F_{CL}` 就成为了函数的参数. 我们把这个函数作为 torch 模块来实现, 把这些参
# 数作为构造器的输入. 这个距离 :math:`\|F_{XL} - F_{YL}\|^2` 是两个特征映射集的
# 均方误差, 可以用作为第三个参数的标准的 ``nn.MSELoss`` 来计算. 
#
# 我们会在每个目标层加入我们的内容损失作为额外的神经网络模块. 这样, 每次我们都会给神经
# 网络投喂一张输入图像 :math:`X`, 所有的损失都会在目标层被计算, 多亏了自动梯度计算, 
# 所有梯度都会被搞定. 要实现, 我们只需写出转换模块的 ``forward`` 方法, 这个模块就变
# 成了网络的 ''transparent layer (透明层)'', 计算好的损失被存为模块的参数. 
#
# 最后, 我们定义一个假的 ``backward`` 方法, 它仅仅只调用后向方法 ``nn.MSELoss`` 
# 来重构梯度. 
# 这个方法返回计算好的损失: 运行梯度下降时要想显示风格和内容损失的变化, 这会非常有用. 
#

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # 我们会从所使用的树中“分离”目标内容
        # 动态地计算梯度: 它是个状态值, 不是变量.
        # 否则评价指标的前向方法会抛出错误. 
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


######################################################################
# .. Note::
#    **重要细节**: 这个模块虽然叫做 ``ContentLoss``, 却不是个真正的 Pytorch 
#    损失函数. 如果你想像 Pytorch 损失一样定义你的内容损失, 你得新建一个 Pytorch 
#    自动求导函数并手动得在 ``backward`` 方法中重算/实现梯度. 
#
# 风格损失
# ~~~~~~~~~~
#
# 对于风格损失, 我们首先需要定义一个给定输入 :math:`X` 在 :math:`L` 
# 层的特征映射 :math:`F_{XL}` 时计算克产物 :math:`G_{XL}` 的模块. 令 
# :math:`\hat{F}_{XL}` 表示 :math:`F_{XL}` 重变形为 
# :math:`K`\ x\ :math:`N` 的版本, 这里 :math:`K` 是 :math:`L` 层特征
# 映射的数量, :math:`N` 是任意矢量化特征映射 :math:`F_{XL}^k` 的长度. 
# :math:`\hat{F}_{XL}` 的第 :math:`k^{th}` 行是 :math:`F_{XL}^k`. 
# 可以验证 :math:`\hat{F}_{XL} \cdot \hat{F}_{XL}^T = G_{XL}`. 鉴于此, 
# 实现我们的模块就很容易了:
#

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b= 特征映射的数量
    # (c,d)= 一个特征映射的维度 (N=c*d)

    features = input.view(a * b, c * d)  # 将 F_XL 转换为 \hat F_XL

    G = torch.mm(features, features.t())  # 计算克产物 (gram product)

    # 我们用除以每个特征映射元素数量的方法
    # 标准化克矩阵 (gram matrix) 的值
    return G.div(a * b * c * d)


######################################################################
# 特征映射的维度 :math:`N` 越长, 则克矩阵 (gram matrix) 的值越大. 
# 因此如果我们不用 :math:`N` 来标准化, 在梯度下降过程中第一层 
# (在池化层之前) 的损失计算就会过于重要. 我们当然不希望这样, 
# 因为我们感兴趣的风格特征都在最深的那些层!
#
# 接着, 风格损失模块被以和内容损失模块相同的方式实现, 
# 但是它和克矩阵中的输入和目标不同
# 

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


######################################################################
# 读取神经网络
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# 现在, 我们要导入一个预训练好的神经网络. 和论文一样, 我们用预训练
# 的 19 层 VGG 网络 (VGG19).
#
# PyTorch对 VGG 的实现模块分为两个子 ``Sequential`` 模块: 
# ``features`` (包括卷积和池化层) 和 ``classifier`` (包括全连接层). 
# 我们只对 ``features`` 感兴趣:
#
# 一些层在训练和验证中有不同的行为. 因为我们要将之用作特征提取, 所有要用 ``.eval()`` 
# 将网络设置为验证模式.
#

cnn = models.vgg19(pretrained=True).features.to(device).eval()

######################################################################
# 另外, VGG 网络训练使用的图像是三个通道分别被均值为 [0.485, 0.456, 0.406] 和
# 标准差为 [0.229, 0.224, 0.225] 标准化过. 我们在用网络处理图像之前, 要用它们
# 做标准化. 
#

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# 创建一个模块来标准化输入图像, 以便我们能在 nn.Sequential 中轻松使用. 
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view 处理均值和标准差使他们形如 [C x 1 x 1], 以直接适应形状为 
        # [B x C x H x W] 的图像张量.
        # B 是批大小. C 是通道数. H 是高度而 W 是宽度.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # 标准化
        return (img - self.mean) / self.std


######################################################################
# ``Sequential`` (顺序) 模块包含一个子模块的列表. 比如, 
# ``vgg19.features`` 包含一个以正确深度排列的序列 (Conv2d, ReLU, 
# Maxpool2d, Conv2d, ReLU...), 就如我们在 *Content loss* 部分讲到的, 
# 我们想要把我们的风格和内容损失模块以想要的深度作为 '透明层' 加入到
# 我们的网络中. 为了这样, 我们建立了一个新的 ``Sequential`` (顺序) 
# 模块, 在其中我们把 ``vgg19`` 和我们的损失模块以正确的顺序加入:
#

# 希望计算风格/内容损失的层 :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # 标准化模块
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # 仅为了有一个可迭代的 内容/风格 损失的列表 
    content_losses = []
    style_losses = []

    # 假设 cnn 模型是 nn.Sequential, 我们建一个新的 nn.Sequential 来使
    # 模块能够被线性连续地激活. 
    model = nn.Sequential(normalization)

    i = 0  # 每次看见卷积就递增
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # 空间内操作的版本并不很好地适合我们下面插入的内容和风格损失. 
            # 所以我们在这里用空间外操作.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # 加入内容损失:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # 加入风格损失:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # 现在我们在最后的内容和样式损失之后修剪图层
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


######################################################################
# .. Note::
#    在这篇论文中他们推荐将最大池化层更改为平均池化层. 
#    AlexNet是一个比 VGG19 更小的网络, 用它实现的话我们也不会看到
#    任何结果质量的不同. 而如果你想做这个替代的话, 可以用这些代码:
#
#    ::
#
#        # avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size,
#        #                         stride=layer.stride, padding = layer.padding)
#        # model.add_module(name,avgpool)


######################################################################
# 输入图像
# ~~~~~~~~~~~
#
# 为了简化代码, 我们用与内容和风格图像同样尺寸的图像做输入. 
# 这个图像可以是白噪声的, 也可以是一份内容图像的拷贝.
#

input_img = content_img.clone()
# 如果你想用白噪声做输入, 请取消下面的注释行:
# input_img = Variable(torch.randn(content_img.data.size())).type(dtype)

# 在绘图中加入原始的输入图像:
plt.figure()
imshow(input_img, title='Input Image')


######################################################################
# Gradient descent
# ~~~~~~~~~~~~~~~~
#
# 由于本算法的作者 Leon Gatys 的建议 
# `here <https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq>`__, 
# 我们将使用 L-BFGS 算法来跑我们的梯度下降. 和训练一个网络不同的是, 
# 我们希望训练输入图像来最小化 内容/风格 损失. 我们想简单地建一个 
# PyTorch L-BFGS 优化器 ``optim.LBFGS`` , 传入我们的图像作为变量进行优化. 我们
# 使用 ``.requires_grad_()`` 来确保这个图像需要梯度. 
#

def get_input_optimizer(input_img):
    # 这行显示了输入是一个需要梯度计算的参数
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


######################################################################
# **最后一步**: 循环进行梯度下降. 每一步中我们必须喂给神经网络更新后
# 的输入以计算新的损失, 我们要运行每个损失的 ``backward`` 方法来动态
# 计算他们的梯度并呈现梯度下降的每一步. 这个优化器需要一个 "closure"
# : 一个重新评估模型并返回损失的函数. 
#
# 然而, 这里有一个小问题. 被优化的图像的像素值会在 :math:`-\infty` 
# 和 :math:`+\infty` 之间波动, 而不是继续保持在 0 到 1. 换句话说, 
# 图像可能会被完美地优化成荒谬的值. 事实上, 我们必须在限制下使用
# 优化器来使我们的输入图像一直保持正确的值. 有一个简单的解决方案: 
# 在每一步, 都校正图像使其保持 0-1 范围的值. 
#

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
            # 校正更新后的输入图像值
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            (style_score * style_weight + content_score * content_weight).backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # 最后一次的校正...
    input_img.data.clamp_(0, 1)

    return input_img

######################################################################
# 最后, 运行算法

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()
