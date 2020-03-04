# 对抗性实例生成

> **作者**: [Nathan Inkawhich](https://github.com/inkawhich)
> 
> 译者：[片刻](https://github.com/jiangzhonglian)
> 
> 校验：[片刻](https://github.com/jiangzhonglian)

如果您正在阅读本文，希望您能体会到某些机器学习模型的有效性。研究不断推动ML模型更快，更准确和更高效。但是，设计和训练模型的一个经常被忽略的方面是安全性和鲁棒性，尤其是在面对想要欺骗模型的对手的情况下。

本教程将提高您对ML模型的安全漏洞的认识，并深入了解对抗性机器学习的热门话题。您可能会惊讶地发现，对图像添加无法察觉的扰动会导致模型性能大不相同。鉴于这是一个教程，我们将通过图像分类器上的示例来探讨该主题。具体来说，我们将使用第一种也是最流行的攻击方法之一，即快速梯度符号攻击(FGSM）来欺骗MNIST分类器。

## 威胁模型

就上下文而言，有许多类别的对抗性攻击，每种攻击者都有不同的目标和对攻击者知识的假设。但是，总的来说，总体目标是向输入数据添加最少的扰动，以引起所需的错误分类。攻击者的知识有几种假设，其中两种是：`white-box`和`black-box`。一个`white-box`攻击假设攻击者有充分的知识和访问模型，包括建筑，输入，输出，和权重。一个`black-box`攻击假设攻击者只能访问输入和模型的输出，并且一无所知底层架构或权重。目标也有几种类型，包括**错误分类**和**源/目标错误分类**。一个错误分类的目标意味着对手只希望输出分类错误，而不关心新分类是什么。一个源/目标误分类装置对手想要改变图像是特定源类的最初使得其被归类为特定的目标类。

在这种情况下，FGSM攻击是`white-box`攻击，目的是进行错误分类。有了这些背景信息，我们现在就可以详细讨论攻击了。

## 快速梯度符号攻击

迄今为止，最早的也是最流行的对抗性攻击之一被称为“ 快速梯度符号攻击”(FGSM），由Goodfellow et. al. 在[解释和利用对抗例子中的运用](https://arxiv.org/abs/1412.6572)描述。攻击非常强大，而且直观。它旨在利用神经网络的学习方式，梯度来攻击神经网络。这个想法很简单，不是根据反向传播的梯度通过调整权重来使损失最小化，而是根据相同的反向传播的梯度来调整输入数据以使损失最大化。换句话说，攻击使用输入数据的损失梯度，然后调整输入数据以使损失最大化。

在进入代码之前，让我们看一下著名的 [FGSM](https://arxiv.org/abs/1412.6572) panda示例并提取一些表示法。

![https://pytorch.org/tutorials/_images/fgsm_panda_image.png](https://pytorch.org/tutorials/_images/fgsm_panda_image.png)


从图中 $$x$$ 是正确分类为 “panda” 的原始输入图像， $$y$$ 是地面真相标签 $$x$$，$$\mathbf{\theta}$$ 代表模型参数，并且 $$J(\mathbf{\theta}, \mathbf{x}, y)$$ 是用于训练网络的损失。攻击会将梯度反向传播回输入数据以进行计算 $$\nabla_{x} J(\mathbf{\theta}, \mathbf{x}, y)$$。然后，通过一小步调整输入数据($$\epsilon$$要么 0.007 在图片中）的方向(即 $$sign(\nabla_{x} J(\mathbf{\theta}, \mathbf{x}, y))$$），这将使损失最大化。产生的扰动图像，$$x'$$然后 ，在目标网络仍明显是 “panda” 的情况下，它会被目标网络误分类为“gibbon”。

希望本教程的动机已经明确，所以让我们进入实现过程。
    
    from __future__ import print_function
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    import numpy as np
    import matplotlib.pyplot as plt


## 实现

在本节中，我们将讨论本教程的输入参数，定义受到攻击的模型，然后编写攻击代码并运行一些测试。

### 输入

本教程只有三个输入，定义如下：

* **epsilons** - 用于运行的epsilon值列表。在列表中保留0很重要，因为它代表原始测试集上的模型性能。同样，从直觉上讲，我们期望ε越大，扰动越明显，但是从降低模型准确性的角度来看，攻击越有效。由于这里的数据范围是[0,1]，则epsilon值不得超过1。
* **pretrained_model** - 使用 [pytorch/examples/mnist](https://github.com/pytorch/examples/tree/master/mnist) 训练的预训练MNIST模型的路径 。为简单起见，请在此处下载预训练的模型。
* **use_cuda** - 布尔标志，如果需要和可用，则使用CUDA。请注意，具有CUDA的GPU在本教程中并不重要，因为CPU不会花费很多时间。
    
    
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    pretrained_model = "data/lenet_mnist_model.pth"
    use_cuda=True
    

### 受到攻击的模型

如前所述，受到攻击的模型与 [pytorch/examples/mnist](https://github.com/pytorch/examples/tree/master/mnist) 中的MNIST模型相同 。您可以训练并保存自己的MNIST模型，也可以下载并使用提供的模型。该网的定义和测试的 DataLoader 这里已经从MNIST实例中复制。本部分的目的是定义模型和数据加载器，然后初始化模型并加载预训练的权重。

    
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
    
Out:

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

现在，我们可以通过干扰原始输入来定义创建对抗示例的函数。该`fgsm_attack`函数需要三个输入，图像是原始的干净图像$$(x)$$，epsilon是像素方向的扰动量$$(\epsilon)$$，而 data_grad 是输入图片$$\nabla_{x} J(\mathbf{\theta}, \mathbf{x}, y)$$。该函数然后创建扰动图像为

$$perturbed\_image = image + epsilon*sign(data\_grad) = x + \epsilon * sign(\nabla_{x} J(\mathbf{\theta}, \mathbf{x}, y))$$

最后，为了保持数据的原始范围，将受干扰的图像裁剪到一定范围 [0,1]。   
    
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

最后，本教程的主要结果来自该`test`函数。每次调用此测试功能都会在MNIST测试集中执行完整的测试步骤，并报告最终精度。但是，请注意，此功能还需要输入epsilon。这是因为该`test`功能报告了受到对手强大攻击的模型的准确性$$\epsilon$$。更具体地说，对于测试集中的每个样本，函数都会计算输入数据的损耗梯度$$(data_grad)$$，使用`fgsm_attack` $$(perturbed_data)$$，然后检查受干扰的示例是否具有对抗性。除了测试模型的准确性外，该函数还保存并返回一些成功的对抗示例，以供以后可视化。
    
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
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex))
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex))
    
        # Calculate final accuracy for this epsilon
        final_acc = correct/float(len(test_loader))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    
        # Return the accuracy and an adversarial example
        return final_acc, adv_examples
    

### 运行攻击

实现的最后一部分是实际运行攻击。在这里，我们为*epsilons*输入中的每个*epsilon*值运行一个完整的测试步骤。对于每个*epsilon*，我们还保存最终精度，并在接下来的部分中绘制一些成功的对抗示例。请注意，随着 $$\epsilon$$ 值的增加，打印的精度如何降低。另外，请注意 $$\epsilon=0$$ 外壳代表原始的测试准确性，没有任何攻击。

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

### Accuracy vs Epsilon

第一个结果是 accuracy 与ε曲线的关系。如前所述，随着ε的增加，我们期望测试精度会降低。这是因为较大的ε意味着我们朝着将损失最大化的方向迈出了更大的一步。请注意，即使epsilon值是线性间隔的，曲线中的趋势也不是线性的。例如，在ϵ=0.05 仅比 ϵ=0 约低4％，但accuracy为 ϵ=0.2 比 ϵ=0.15 低25％。另外，请注意，对于介于 ϵ=0.25 和 ϵ=0.3。

    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()
    

![https://pytorch.org/tutorials/_images/sphx_glr_fgsm_tutorial_001.png](https://pytorch.org/tutorials/_images/sphx_glr_fgsm_tutorial_001.png)

### 对抗示例

还记得没有免费午餐的想法吗？在这种情况下，随着ε的增加，测试精度降低，**BUT**扰动变得更容易察觉。实际上，攻击者必须考虑准确性降低和可感知性之间的权衡。在这里，我们展示了每个epsilon值的成功对抗示例。绘图的每一行显示不同的ε值。第一行是ϵ=0代表原始“干净”图像且无干扰的示例。每个图像的标题显示“原始分类->对抗分类”。请注意，扰动在以下位置开始变得明显ϵ=0.15 并且在 ϵ=0.3。然而，在所有情况下，尽管增加了噪音，人类仍然能够识别正确的类别。


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
    

![https://pytorch.org/tutorials/_images/sphx_glr_fgsm_tutorial_002.png](https://pytorch.org/tutorials/_images/sphx_glr_fgsm_tutorial_002.png)

## 下一步去哪里？

希望本教程对对抗性机器学习主题有所了解。从这里可以找到许多潜在的方向。这种攻击代表了对抗性攻击研究的最开始，并且由于随后有很多关于如何攻击和防御对手的ML模型的想法。实际上，在NIPS 2017上有一个对抗性的攻击和防御竞赛，并且本文描述了该竞赛中使用的许多方法：[对抗性的攻击和防御竞赛](https://arxiv.org/pdf/1804.00097.pdf)。国防方面的工作还引发了使机器学习模型总体上更加健壮的想法，以适应自然扰动和对抗性输入。

另一个方向是不同领域的对抗性攻击和防御。对抗性研究不仅限于图像领域，请查看[这种](https://arxiv.org/pdf/1801.01944.pdf)对语音到文本模型的攻击。但是，也许更多地了解对抗性机器学习的最好方法是弄脏您的手。尝试实施与NIPS 2017竞赛不同的攻击，并查看其与FGSM的不同之处。然后，尝试保护模型免受自己的攻击。

**脚本的总运行时间：** (2分钟57.229秒）

[`Download Python source code:
fgsm_tutorial.py`](../_downloads/c9aee5c8955d797c051f02c07927b0c0/fgsm_tutorial.py)

[`Download Jupyter notebook:
fgsm_tutorial.ipynb`](../_downloads/fba7866856a418520404ba3a11142335/fgsm_tutorial.ipynb)
