# -*- coding: utf-8 -*-
"""
用 PyTorch 做 神经转换（Neural Transfer）
==========================================
**原作者**: `Alexis Jacq <https://alexis-jacq.github.io>`_

介绍
------------

欢迎观看! 这篇教程解释了如何实现 Leon A. Gatys, Alexander S. Ecker 和 
Matthias Bethge 几位学者发明的
`Neural-Style <https://arxiv.org/abs/1508.06576>`__ 算法 .

题中神经描述的是什么?
~~~~~~~~~~~~~~~~~~

神经风格, 或者说神经转换是一种算法，它输入一张内容图像 (例如海龟), 一张风格图像 
(例如艺术波浪), 然后返回内容图像的内容，这个内容像是被艺术风格图像的风格渲染过: 

.. figure:: /_static/img/neural-style/neuralstyle.png
   :alt: content1

它是如何工作的?
~~~~~~~~~~~~~~~~~

原理很简单: 我们定义两个距离, 一个是关于内容的(:math:`D_C`) , 另一个是关于风格的
 (:math:`D_S`). :math:`D_C` 衡量两张图像的内容有多么不同, 而 :math:`D_S`
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
:math:`k \leq K` 的所有矢量化特征映射 :math:`F_{XL}^k` 的豆产物 (Gram produce). 
换句话说, :math:`G_{XL}` 是一个 :math:`K`\ x\ :math:`K` 的矩阵, 其在 
:math:`k^{th}` 行和 :math:`l^{th}` 列的每个元素 :math:`G_{XL}(k,l)` 
是 :math:`F_{XL}^k` 和 :math:`F_{XL}^l` 之间的矢量产物 :

.. math::

    G_{XL}(k,l) = \langle F_{XL}^k, F_{XL}^l\\rangle = \sum_i F_{XL}^k(i) . F_{XL}^l(i)

式中 :math:`F_{XL}^k(i)` 是 :math:`F_{XL}^k` 的第 :math:`i^{th}` 个元素. 
我们可以把 :math:`G_{XL}(k,l)` 当做特征映射 :math:`k` 和 :math:`l` 相关性的衡量. 
那样的话, :math:`G_{XL}` 代表了 :math:`X` 在 :math:`L` 层特征向量的相关性矩阵. 
注意 :math:`G_{XL}` 的尺寸只决定于特征映射的数量，不被 :math:`X` 的尺寸所影响. 
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

如果你不确定是否理解了以上数学公式, 你也可以在实现它的过程中有所领悟. 
如果你正在探索 PyTorch , 我们推荐你先阅读这篇教程 
:doc:`Introduction to PyTorch </beginner/deep_learning_60min_blitz>`.

包
~~~~~~~~

我们将会依赖下列这些包:

-  ``torch``, ``torch.nn``, ``numpy`` (indispensables packages for
   neural networks with PyTorch)
-  ``torch.autograd.Variable`` (dynamic computation of the gradient wrt
   a variable)
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
from torch.autograd import Variable
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
# 如果你的计算机有可用 GPU 则会返回 True. 然后我们用 ``.cuda()`` 方法
# 将可分配的进程和模块从 CPU 移动到 GPU. 当我们想将这些模块重新移回 CPU 
# 的时候(比如要用 numpy), 我们用 ``.cpu()`` 方法. 
# 最后, ``.type(dtype)`` 会用来将一个 ``torch.FloatTensor`` 
# 转化为 用于 GPU 进程输入的 ``torch.cuda.FloatTensor``.
#

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


######################################################################
# 读取图像
# ~~~~~~~~~~~
#
# 为了简化其实现, 让我们从导入一个相同维度的风格和内容图像开始. 
# 然后我们将它们缩放到想要的输入图像尺寸 (在例子中是 128 和 512, 
# 取决你的 GPU 是否可用) 然后把它们转化为 torch 张量, 以待喂入一个神经网络.
#
# .. 注释::
#     这里是教程需要的图像的下载链接: 
#     `picasso.jpg <http://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg>`__ 和
#     `dancing.jpg <http://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg>`__.
#     下载这两张图像然后把它们加入到名为 ``images`` 的目录中.
# 


# 想要的输出图像尺寸
imsize = 512 if use_cuda else 128  # 如果没有 GPU 则使用小尺寸

loader = transforms.Compose([
    transforms.Scale(imsize),  # 缩放图像
    transforms.ToTensor()])  # 将其转化为 torch 张量


def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    # 由于神经网络输入的需要, 添加 batch 的维度
    image = image.unsqueeze(0)
    return image


style_img = image_loader("images/picasso.jpg").type(dtype)
content_img = image_loader("images/dancing.jpg").type(dtype)

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
    image = tensor.clone().cpu()  # 克隆是为了不改变它
    image = image.view(3, imsize, imsize)  # 移除 batch 维度
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # 暂停一会, 让绘图更新


plt.figure()
imshow(style_img.data, title='Style Image')

plt.figure()
imshow(content_img.data, title='Content Image')


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

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # 我们会从所使用的树中“分离”目标内容
        self.target = target.detach() * weight
        # 动态地计算梯度: 它是个状态值, 不是变量.
        # 否则评价指标的前向方法会抛出错误。
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


######################################################################
# .. 注意::
#    **重要细节**: 这个模块虽然叫做 ``ContentLoss``, 却不是个真正的 Pytorch 
#    损失函数. 如果你想像 Pytorch 损失一样定义你的内容损失, 你得新建一个 Pytorch 
#    自动求导函数并手动得在 ``backward`` 方法中重算/实现梯度. 
#
# 风格损失
# ~~~~~~~~~~
#
# For the style loss, we need first to define a module that compute the
# gram produce :math:`G_{XL}` given the feature maps :math:`F_{XL}` of the
# neural network fed by :math:`X`, at layer :math:`L`. Let
# :math:`\hat{F}_{XL}` be the re-shaped version of :math:`F_{XL}` into a
# :math:`K`\ x\ :math:`N` matrix, where :math:`K` is the number of feature
# maps at layer :math:`L` and :math:`N` the lenght of any vectorized
# feature map :math:`F_{XL}^k`. The :math:`k^{th}` line of
# :math:`\hat{F}_{XL}` is :math:`F_{XL}^k`. We let you check that
# :math:`\hat{F}_{XL} \cdot \hat{F}_{XL}^T = G_{XL}`. Given that, it
# becomes easy to implement our module:
#

class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


######################################################################
# The longer is the feature maps dimension :math:`N`, the bigger are the
# values of the gram matrix. Therefore, if we don't normalize by :math:`N`,
# the loss computed at the first layers (before pooling layers) will have
# much more importance during the gradient descent. We dont want that,
# since the most interesting style features are in the deepest layers!
#
# Then, the style loss module is implemented exactly the same way than the
# content loss module, but we have to add the ``gramMatrix`` as a
# parameter:
#

class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


######################################################################
# Load the neural network
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# Now, we have to import a pre-trained neural network. As in the paper, we
# are going to use a pretrained VGG network with 19 layers (VGG19).
#
# PyTorch's implementation of VGG is a module divided in two child
# ``Sequential`` modules: ``features`` (containing convolution and pooling
# layers) and ``classifier`` (containing fully connected layers). We are
# just interested by ``features``:
#

cnn = models.vgg19(pretrained=True).features

# move it to the GPU if possible:
if use_cuda:
    cnn = cnn.cuda()


######################################################################
# A ``Sequential`` module contains an ordered list of child modules. For
# instance, ``vgg19.features`` contains a sequence (Conv2d, ReLU,
# Maxpool2d, Conv2d, ReLU...) aligned in the right order of depth. As we
# said in *Content loss* section, we wand to add our style and content
# loss modules as additive 'transparent' layers in our network, at desired
# depths. For that, we construct a new ``Sequential`` module, in wich we
# are going to add modules from ``vgg19`` and our loss modules in the
# right order:
#

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, style_img, content_img,
                               style_weight=1000, content_weight=1,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    model = nn.Sequential()  # the new Sequential module network
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    # move these modules to the GPU if possible:
    if use_cuda:
        model = model.cuda()
        gram = gram.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  # ***

    return model, style_losses, content_losses


######################################################################
# .. Note::
#    In the paper they recommend to change max pooling layers into
#    average pooling. With AlexNet, that is a small network compared to VGG19
#    used in the paper, we are not going to see any difference of quality in
#    the result. However, you can use these lines instead if you want to do
#    this substitution:
#
#    ::
#
#        # avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size,
#        #                         stride=layer.stride, padding = layer.padding)
#        # model.add_module(name,avgpool)


######################################################################
# Input image
# ~~~~~~~~~~~
#
# Again, in order to simplify the code, we take an image of the same
# dimensions than content and style images. This image can be a white
# noise, or it can also be a copy of the content-image.
#

input_img = content_img.clone()
# if you want to use a white noise instead uncomment the below line:
# input_img = Variable(torch.randn(content_img.data.size())).type(dtype)

# add the original input image to the figure:
plt.figure()
imshow(input_img.data, title='Input Image')


######################################################################
# Gradient descent
# ~~~~~~~~~~~~~~~~
#
# As Leon Gatys, the author of the algorithm, suggested
# `here <https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq>`__,
# we will use L-BFGS algorithm to run our gradient descent. Unlike
# training a network, we want to train the input image in order to
# minimise the content/style losses. We would like to simply create a
# PyTorch  L-BFGS optimizer, passing our image as the variable to optimize.
# But ``optim.LBFGS`` takes as first argument a list of PyTorch
# ``Variable`` that require gradient. Our input image is a ``Variable``
# but is not a leaf of the tree that requires computation of gradients. In
# order to show that this variable requires a gradient, a possibility is
# to construct a ``Parameter`` object from the input image. Then, we just
# give a list containing this ``Parameter`` to the optimizer's
# constructor:
#

def get_input_param_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer


######################################################################
# **Last step**: the loop of gradient descent. At each step, we must feed
# the network with the updated input in order to compute the new losses,
# we must run the ``backward`` methods of each loss to dynamically compute
# their gradients and perform the step of gradient descent. The optimizer
# requires as argument a "closure": a function that reevaluates the model
# and returns the loss.
#
# However, there's a small catch. The optimized image may take its values
# between :math:`-\infty` and :math:`+\infty` instead of staying between 0
# and 1. In other words, the image might be well optimized and have absurd
# values. In fact, we must perform an optimization under constraints in
# order to keep having right vaues into our input image. There is a simple
# solution: at each step, to correct the image to maintain its values into
# the 0-1 interval.
#

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300,
                       style_weight=1000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        style_img, content_img, style_weight, content_weight)
    input_param, optimizer = get_input_param_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.data[0], content_score.data[0]))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_param.data.clamp_(0, 1)

    return input_param.data

######################################################################
# Finally, run the algorithm

output = run_style_transfer(cnn, content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()
