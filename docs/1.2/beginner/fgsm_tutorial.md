# 对抗性实施例代

**作者：** [弥敦道Inkawhich ](https://github.com/inkawhich)

如果你正在读这篇文章，希望你能明白一些机器学习模型的有效性如何。研究正不断ML车型更快，更准确，更高效。然而，设计和培训模式的一个经常被忽视的方面是安全性和稳健性，尤其是在谁愿意来愚弄模型对手的脸。

本教程将提高你的意识，以ML车型的安全漏洞，并会深入了解对抗机器学习的热门话题。你可能会惊讶地发现，加入不易察觉的扰动到图像 _可以_
导致截然不同的模型性能。考虑到这是一个教程中，我们将探讨在图像分类通过例子的话题。具体来说，我们将使用的第一个也是最流行的攻击方式之一，快速倾斜的符号攻击（FGSM），愚弄的MNIST分类。

## 威胁模型

对于背景下，有许多种类的敌对攻击，每一个不同的目标和攻击者的知识假设。然而，一般的总体目标是扰动的至少量添加到所述输入数据，以使所期望的错误分类。有几种类型的攻击者的知识的假设，其中两个是：
**白盒** 和 **黑盒[HTG3。 A _白盒_ 攻击假定攻击者具有充分的知识，并获得了模型，包括体系结构，输入，输出，和权重。 A _黑箱_
攻击假定攻击者只能访问输入和模型的输出，并且一无所知底层架构或权重。也有几种类型的目标，包括 **误分类** 和 **源/目标误分类** 。的 _误判_
一个目标是指对手只希望输出的分类是错误的，但并不关心新的分类是什么。 A _源/目标误分类_
表示对手想要改变图像是特定源类的最初使得其被归类为特定的目标类。**

在这种情况下，FGSM攻击是一种 _白盒_ 攻击与 _误判_ 的目标。在这样的背景信息，现在我们可以详细讨论了攻击。

## 快速倾斜的符号攻击

之一的第一和最流行的对抗攻击日期被称为 _快速梯度注册攻击（FGSM）_
并且由Goodfellow等说明。人。在[解释和治理对抗性实施例](https://arxiv.org/abs/1412.6572)。这种攻击是非常强大的，可是直觉。它的目的是通过充分利用他们学习的方式来攻击神经网络，
_梯度[HTG5。这个想法是简单的，而不是工作，通过调整基于所述backpropagated梯度的权重，以尽量减少损失，攻击
_调整输入数据以最大化基于相同backpropagated梯度的丧失_
。换句话说，该攻击使用的损失w.r.t输入数据的梯度，然后调整输入数据以最大化损失。_

在我们跳进代码，让我们来看看著名的[ FGSM ](https://arxiv.org/abs/1412.6572)熊猫例子，提取一些符号。

![fgsm_panda_image](../_images/fgsm_panda_image.png)

从该图中， \（\ mathbf {X} \）是正确归类为“熊猫”， \（Y \）原始输入图像是用于地面实况标签 \（\ mathbf {X} \），
\（\ mathbf {\ THETA} \）表示的模型参数，并 \（j（\ mathbf {\ THETA} ，\ mathbf
{X}，y）的\）是用于训练网络的损失。攻击backpropagates梯度回输入的数据来计算 \（\ nabla_ {X}Ĵ（\ mathbf {\
THETA}，\ mathbf {X}，y）的\）。然后，它调整由小步骤中的输入数据（ \（\小量\）或 \（0.007 \）在画面）的方向（即，
\（符号（\ nabla_ {X}Ĵ（\ mathbf {\ THETA}，\ mathbf {X}，y）的）\）），其将最大限度地损失。将得到的扰动图像，
\（X'\），然后错误分类由目标网络为‘长臂猿’时，它仍然是明确了‘熊猫’ _。_

现在希望本教程的动机很明显，所以让我们跳进实施。

    
    
    from __future__ import print_function
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    import numpy as np
    import matplotlib.pyplot as plt
    

## 实现

在本节中，我们将讨论的输入参数的教程，确定受到攻击的模型，然后编码攻击和运行一些测试。

### 输入

只有三个输入本教程，并定义如下：

  * **epsilons** \- 小量值的列表以用于运行。它保持0在列表中，因为它代表了原始的测试集模型的性能是非常重要的。此外，直观我们希望越大ε，更明显的扰动，但是在分解模型精度方面更有效的攻击。由于数据范围这里是 \（[0,1] \），没有小量值不应超过1。
  * **pretrained_model** \- 路径，将其用[训练预训练的模型MNIST pytorch /示例/ MNIST ](https://github.com/pytorch/examples/tree/master/mnist)。为简单起见，下载预训练的模型[此处[HTG5。](https://drive.google.com/drive/folders/1fn83DF14tWmit0RTKWRhPq5uVXt73e0h?usp=sharing)
  * **use_cuda** \- 布尔标志到如果需要和可用使用CUDA。请注意，本教程为CPU不会花费太多的时间与CUDA GPU的并不重要。

    
    
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    pretrained_model = "data/lenet_mnist_model.pth"
    use_cuda=True
    

### 模式下的攻击

如所提到的，在攻击该模型是从[ pytorch /示例/ MNIST
](https://github.com/pytorch/examples/tree/master/mnist)相同MNIST模型。你可以训练并保存自己的MNIST模型，或者你可以下载和使用所提供的模型。的
_净_ 定义和测试的DataLoader这里已经从MNIST示例复制。本部分的目的是定义模型和的DataLoader，然后初始化模型并加载预训练的权重。

    
    
    # LeNet Model definition
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)
    
        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    # MNIST Test dataset and dataloader declaration
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=1, shuffle=True)
    
    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    
    # Initialize the network
    model = Net().to(device)
    
    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
    
    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()
    

日期：

    
    
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ../data/MNIST/raw/train-images-idx3-ubyte.gz
    Extracting ../data/MNIST/raw/train-images-idx3-ubyte.gz to ../data/MNIST/raw
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ../data/MNIST/raw/train-labels-idx1-ubyte.gz
    Extracting ../data/MNIST/raw/train-labels-idx1-ubyte.gz to ../data/MNIST/raw
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw/t10k-images-idx3-ubyte.gz
    Extracting ../data/MNIST/raw/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz
    Extracting ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw
    Processing...
    Done!
    CUDA Available:  True
    

### FGSM攻击

现在，我们可以定义通过扰乱原来的输入产生对抗的例子功能。的`fgsm_attack`函数有三个输入， _图像_ 是原始干净图像（ \（X \）），
_的ε-_ 为逐像素扰动量（ \（\小量\））和 _data_grad_ 是损失WRT的梯度来确定输入图像（ \（\ nabla_ { X}Ĵ（\
mathbf {\ THETA}，\ mathbf {X}，y）的\））。然后，该函数产生扰动的图像作为

\\[perturbed\\_image = image + epsilon*sign(data\\_grad) = x + \epsilon *
sign(\nabla_{x} J(\mathbf{\theta}, \mathbf{x}, y))\\]

最后，为了保持数据的原始范围，对于扰动的图像被夹到范围 \（[0,1] \）。

    
    
    # FGSM attack code
    def fgsm_attack(image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image
    

### 测试功能

最后，本教程的中央结果来源于`测试 `功能。该测试功能每次调用执行对MNIST测试集一个完整的测试步骤，并报告最终精度。然而，请注意，这个功能也需要一个
_的ε-_ 输入。这是因为`测试 `函数将报告一个模型，它是受到攻击从对手与强度
\（\小量\）的准确性。更具体地，在测试组中的每个样本，所述函数计算所述损失WRT输入数据（ \（数据\ _grad \））的梯度，产生具有`扰动的图像
fgsm_attack`（ \（扰动\ _data
\）），然后检查是否被扰动的例子是对抗性。除了测试模型的准确性，功能也节省并返回稍后显现一些成功的例子对抗性。

    
    
    def test( model, device, test_loader, epsilon ):
    
        # Accuracy counter
        correct = 0
        adv_examples = []
    
        # Loop over all examples in test set
        for data, target in test_loader:
    
            # Send the data and label to the device
            data, target = data.to(device), target.to(device)
    
            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True
    
            # Forward pass the data through the model
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    
            # If the initial prediction is wrong, dont bother attacking, just move on
            if init_pred.item() != target.item():
                continue
    
            # Calculate the loss
            loss = F.nll_loss(output, target)
    
            # Zero all existing gradients
            model.zero_grad()
    
            # Calculate gradients of model in backward pass
            loss.backward()
    
            # Collect datagrad
            data_grad = data.grad.data
    
            # Call FGSM Attack
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
    
            # Re-classify the perturbed image
            output = model(perturbed_data)
    
            # Check for success
            final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            if final_pred.item() == target.item():
                correct += 1
                # Special case for saving 0 epsilon examples
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
    
        # Calculate final accuracy for this epsilon
        final_acc = correct/float(len(test_loader))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    
        # Return the accuracy and an adversarial example
        return final_acc, adv_examples
    

### 运行攻击

实施的最后一部分是实际运行攻击。在这里，我们运行在 _epsilons_ 输入的每个的ε-
值全测试步骤。对于每一个小量，我们也节省了最终的准确度和未来的部分要绘制一些成功的例子对抗性。注意印刷精度如何降低作为的ε值增加。另外，请注意 \（\小量=
0 \）的情况下表示原始测试精度，没有攻击。

    
    
    accuracies = []
    examples = []
    
    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)
    

Out:

    
    
    Epsilon: 0      Test Accuracy = 9810 / 10000 = 0.981
    Epsilon: 0.05   Test Accuracy = 9426 / 10000 = 0.9426
    Epsilon: 0.1    Test Accuracy = 8510 / 10000 = 0.851
    Epsilon: 0.15   Test Accuracy = 6826 / 10000 = 0.6826
    Epsilon: 0.2    Test Accuracy = 4301 / 10000 = 0.4301
    Epsilon: 0.25   Test Accuracy = 2082 / 10000 = 0.2082
    Epsilon: 0.3    Test Accuracy = 869 / 10000 = 0.0869
    

## 结果

### 精确度和小量

第一个结果是精度与小量的情节。正如先前提到的，因为小量增加，我们预计测试精度降低。这是因为更大的epsilons意味着我们采取的是将最大限度地损失方向以更大的一步。注意在曲线的趋势，即使的ε值线性间隔不是线性的。例如，在
\（\小量= 0.05 \）的精度比下仅约4％\（\小量= 0 \），但精度在 \ （\小量= 0.2 \）大于低25％\（\小量= 0.15
\）。另外，请注意该模型的准确度命中随机精度\之间 10级分类器（\小量= 0.25 \）和 \（\小量= 0.3 \）。

    
    
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()
    

![../_images/sphx_glr_fgsm_tutorial_001.png](../_images/sphx_glr_fgsm_tutorial_001.png)

### 样品对抗性实施例

记住没有免费的午餐的想法？在这种情况下，作为小量增加了测试精度降低 **BUT**
扰动变得更容易察觉。在现实中，有精度降解和攻击者必须考虑感之间的权衡。在这里，我们显示出对每一个小量值成功对抗的例子一些例子。情节的每行显示一个不同的小量值。第一行是
\（\小量= 0 \），其表示不具有扰动原来的“干净”的图像实例。各图像的标题显示了“原始分类 - & GT ;对抗性分类。”通知，扰动开始成为在
\（\小量= 0.15 \）明显，是相当明显在 \（\小量= 0.3 \）。然而，在所有的情况下，人类仍然能够识别正确的类，尽管添加了噪音的。

    
    
    # Plot several examples of adversarial samples at each epsilon
    cnt = 0
    plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()
    

![../_images/sphx_glr_fgsm_tutorial_002.png](../_images/sphx_glr_fgsm_tutorial_002.png)

## 下一步去哪里？

希望这个教程提供一些见解对立的机器学习的话题。有许多潜在的方向从这里走。这次攻击是对抗攻击的研究一开始就和因为有一直为如何攻击和对手防守ML车型很多后续的想法。事实上，在2017年NIPS有一个对抗性的攻防竞争和许多在比赛中使用的方法在本文中描述：[对抗性攻击和防御比赛[HTG1。在防守上的工作还通向使机器学习模型的想法更多
_健壮_ 在一般情况下，双方自然扰动和adversarially制作的投入。](https://arxiv.org/pdf/1804.00097.pdf)

去另一个方向是在不同的领域对抗攻击和防御。对抗性的研究不限于图像域，检查出[上的语音至文本模式这个](https://arxiv.org/pdf/1801.01944.pdf)攻击。但也许更多地了解对抗机器学习的最佳方式是让你的手脏。尝试实施从2017年NIPS竞争不同的攻击，看看它与FGSM的不同之处。然后，尝试从自己的攻击防御模型。

**脚本的总运行时间：** （2分钟57.229秒）

[`Download Python source code:
fgsm_tutorial.py`](../_downloads/c9aee5c8955d797c051f02c07927b0c0/fgsm_tutorial.py)

[`Download Jupyter notebook:
fgsm_tutorial.ipynb`](../_downloads/fba7866856a418520404ba3a11142335/fgsm_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](dcgan_faces_tutorial.html "DCGAN Tutorial")
[![](../_static/images/chevron-right-orange.svg)
Previous](../advanced/neural_style_tutorial.html "Neural Transfer Using
PyTorch")

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

  * 对抗性实施例代
    * 威胁模型
    * 快速倾斜的符号攻击
    * 实现
      * 输入
      * 型号受到攻击
      * FGSM攻击
      * 测试函数
      * 运行攻击
    * 结果
      * 精度VS的Epsilon 
      * [HTG0样品对抗性实施例
    * 下一步去哪里？ 

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

