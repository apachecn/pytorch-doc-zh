


 没有10



 单击
 [此处](#sphx-glr-download-beginner-fgsm-tutorial-py)
 下载完整的示例代码








 对抗性示例生成
 [¶](#adversarial-example- Generation "永久链接到此标题")
===============================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/fgsm_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/fgsm_tutorial.html>




**作者：** 
[Nathan Inkawhich](https://github.com/inkawhich)




 如果您正在阅读本文，希望您能够体会到一些
机器学习模型的有效性。研究不断推动机器学习模型
变得更快、更准确、更高效。然而，设计和训练模型时
经常被忽视的一个方面是安全性和
鲁棒性，尤其是面对想要愚弄模型的对手时
。




 本教程将提高您对机器学习模型的安全漏洞的认识，并深入了解对抗性机器学习的热门话题。您可能会惊讶地发现，向图像添加难以察觉的
扰动
*可能*
会导致模型性能
截然不同。鉴于这是一个教程，我们将通过图像分类器的示例来探索该主题。具体来说，我们将使用
第一种也是最流行的攻击方法之一，即快速梯度符号攻击
(FGSM)，来欺骗 MNIST 分类器。





 威胁模型
 [¶](#threat-model "永久链接到此标题")
-----------------------------------------------------------------------------


就上下文而言，对抗性攻击有许多类别，每种攻击都有不同的目标和假设攻击者的知识。然而，一般来说，总体目标是向输入数据添加最少量的扰动，以导致所需的错误分类。攻击者的知识有多种假设，其中两种是：
 **白盒** 
 和
 **黑盒** 
 。 *白盒* 攻击假设攻击者完全了解并有权访问模型，包括架构、输入、输出和权重。 
 *黑盒* 
 攻击假设
攻击者只能访问模型的输入和输出，
对底层架构或权重一无所知。 
还有几种类型的目标，包括
 **错误分类** 
 和
 **源/目标错误分类** 
 。 
 *错误分类* 
 的目标意味着
对手只希望输出分类错误，但不关心新分类是什么。 
 *源/目标
错误分类* 
 表示攻击者想要更改
最初属于特定源类的图像，
以便将其分类为
特定目标类。




 在这种情况下，FGSM 攻击是一种
 *白盒* 
 攻击，其目标是
 *错误分类* 
 。有了这些背景信息，我们现在可以
详细讨论这次攻击。






 快速梯度符号攻击
 [¶](#fast-gradient-sign-attack "永久链接到此标题")
------------------------------------------------------------------------------------------



 迄今为止第一个也是最流行的对抗性攻击之一被称为
 *快速梯度符号攻击 (FGSM)* 
 并由 Goodfellow 等人描述。 al. in
 [解释和利用对抗性
示例](https://arxiv.org/abs/1412.6572) 
.这种攻击非常强大，而且直观。它旨在通过
利用神经网络的学习方式
 *梯度* 
 来攻击神经网络。这个想法很简单，攻击不是通过根据反向传播梯度调整权重来最小化损失，而是根据相同的反向传播梯度调整输入数据以最大化损失*。换句话说，
攻击使用损失相对于输入数据的梯度，然后
调整输入数据以使损失最大化。




 在我们进入代码之前，让’s 看一下著名的
 [FGSM](https://arxiv.org/abs/1412.6572) 
 panda 示例并提取
一些符号。




![fgsm_panda_image](https://pytorch.org/tutorials/_images/fgsm_panda_image.png)


 从图中，
 
 \(\mathbf{x}\)
 
 是原始输入图像
正确分类为“panda”，
 \ n \(y\)
 
 是
的真实标签

 
 \(\mathbf{x}\)
 
 ,
 
 \(\mathbf {\theta}\)
 
 表示模型
参数，
 
 \(J(\mathbf{\theta}, \mathbf{x}, y)\)\ n 
 是用于训练网络的损失。攻击将
梯度反向传播回输入数据以计算
 
 \(abla_{x} J(\mathbf{\theta}, \mathbf{x}, y)\ \)

 。然后，它将输入数据调整一小步（

 \(\epsilon\)
 
 或

 \(0.007\)

图片中的
 ) 方向（即
 
 \(sign(abla_{x} J(\mathbf{\theta}, \mathbf{x}, y))\)
 
 ) 这将使损失最大化。生成的扰动图像 
 
 \(x'\)
 
 被目标网络
 *错误分类* 
 为 “gibbon”它仍然
显然是 “panda”。




 希望现在本教程的动机已经明确，所以让我们跳转
 实现。






```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

```






 实现
 [¶](#implementation "永久链接到此标题")
--------------------------------------------------------------------------------



 在本节中，我们将讨论教程的输入参数，
定义受到攻击的模型，然后编写攻击代码并运行一些测试。




### 
 输入
 [¶](#inputs "此标题的永久链接")



 本教程只有三个输入，定义如下：



* `epsilons`
 - 用于运行的 epsilon 值列表。在列表中保留 0 很重要，因为它代表模型在原始测试集上的性能。此外，直观上我们预计 epsilon 越大，扰动就越明显，但在降低模型准确性方面攻击就越有效。由于此处的数据范围为
 
 \([0,1]\)
 
 ，因此 epsilon
值不应超过 1。
* `pretrained_model`
 - 预训练的路径MNIST 模型已使用
 [pytorch/examples/mnist](https://github.com/pytorch/examples/tree/master/mnist)
 进行训练。
为简单起见，请下载预训练模型
 [此处](https://drive.google.com/file/d/1HJV2nUHJqclXQ8flKvcWmjZ-OU5DGatl/view?usp=drive_link) 
.
* `use_cuda`
 - 如果需要且可用，使用 CUDA 的布尔标志
请注意，具有 CUDA 的 GPU 对于本教程来说并不重要，因为 CPU
不会花费太多时间。





```
epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "data/lenet_mnist_model.pth"
use_cuda=True
# Set random seed for reproducibility
torch.manual_seed(42)

```






```
<torch._C.Generator object at 0x7f12f701ee50>

```





### 
 模型受到攻击
 [¶](#model-under-attack "永久链接到此标题")



 如前所述，受到攻击的模型与 [pytorch/examples/mnist](https://github.com/pytorch/examples/tree/master/mnist) 中的 MNIST 模型相同。
您可以训练并保存您自己的 MNIST 模型，或者您可以下载并使用
提供的模型。这里的
 *Net* 
 定义和测试数据加载器是从 MNIST 示例中复制的。本节的目的是
定义模型和数据加载器，然后初始化模型并加载
预训练的权重。






```
# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])),
        batch_size=1, shuffle=True)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location=device))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

```






```
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ../data/MNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/9912422 [00:00<?, ?it/s]
100%|##########| 9912422/9912422 [00:00<00:00, 345164142.10it/s]
Extracting ../data/MNIST/raw/train-images-idx3-ubyte.gz to ../data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ../data/MNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/28881 [00:00<?, ?it/s]
100%|##########| 28881/28881 [00:00<00:00, 149181888.95it/s]
Extracting ../data/MNIST/raw/train-labels-idx1-ubyte.gz to ../data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/1648877 [00:00<?, ?it/s]
100%|##########| 1648877/1648877 [00:00<00:00, 316589214.77it/s]
Extracting ../data/MNIST/raw/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/4542 [00:00<?, ?it/s]
100%|##########| 4542/4542 [00:00<00:00, 29906638.57it/s]
Extracting ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw

CUDA Available:  True

Net(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (dropout1): Dropout(p=0.25, inplace=False)
  (dropout2): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=9216, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)

```





### 
 FGSM 攻击
 [¶](#fgsm-attack "此标题的固定链接")


现在，我们可以定义通过扰动原始输入来创建对抗性示例的函数。 
 `fgsm_attack`
 函数需要三个
输入，
 *image* 
 是原始的干净图像 (
 
 \(x\)
 
 ),
 * epsilon* 
 是逐像素扰动量 (
 
 \(\epsilon\)
 
 )，并且
 *data_grad* 
 是损失相对于输入图像
(
 
 \(abla_{x} J(\mathbf{\theta}, \mathbf{x}, y)\)
 
 )。该函数
然后将扰动图像创建为




 \[扰动\_image = 图像 + epsilon*sign(data\_grad) = x + \epsilon * 符号(abla_{x} J(\ \mathbf{\theta}, \mathbf{x}, y))

\]
 

 最后，为了保持数据的原始范围，
对扰动图像进行裁剪范围
 
 \([0,1]\)
 
.






```
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

# restores the tensors to their original scale
def denorm(batch, mean=[0.1307], std=[0.3081]):
 """
 Convert a batch of tensors to their original scale.

 Args:
 batch (torch.Tensor): Batch of normalized tensors.
 mean (torch.Tensor or list): Mean used for normalization.
 std (torch.Tensor or list): Standard deviation used for normalization.

 Returns:
 torch.Tensor: batch of tensors without normalization applied to them.
 """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

```





### 
 测试函数
 [¶](#testing-function "永久链接到此标题")



 最后，本教程的中心结果来自
 `test`
 函数。每次调用此测试函数都会在 MNIST 测试集上执行完整的测试步骤并报告最终准确性。但是，请注意，
此函数还接受
 *epsilon* 
 输入。这是因为
 `test`
 函数报告了受到强度
 
 \(\epsilon\)
 
 对手攻击的模型的准确性。更具体地说，对于测试集中的每个样本，该函数计算输入数据的损失梯度 (
 
 \(data\_grad\)
 
 )，创建一个使用
 `fgsm_attack`
 (
 
 \(perturbed\_data\)
 
 ) 扰动图像，然后检查
扰动的示例是否是对抗性的。除了测试模型的
准确性之外，该函数还保存并返回一些
成功的对抗性示例，以便稍后可视化。






```
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

        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = data.grad.data

        # Restore the data to its original scale
        data_denorm = denorm(data)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

```





### 
 运行攻击
 [¶](#run-attack "永久链接到此标题")



 实施的最后一部分是实际运行攻击。在这里，
我们对
 *epsilons* 
 输入中的每个 epsilon 值运行完整的测试步骤。
对于每个 epsilon，我们还保存最终精度和一些成功
的对抗示例，以在接下来的部分中绘制。请注意
打印的精度如何随着 epsilon 值的增加而降低。另外，
请注意
 
 \(\epsilon=0\)
 
 情况代表原始测试精度，
没有受到攻击。






```
accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

```






```
Epsilon: 0      Test Accuracy = 9912 / 10000 = 0.9912
Epsilon: 0.05   Test Accuracy = 9605 / 10000 = 0.9605
Epsilon: 0.1    Test Accuracy = 8743 / 10000 = 0.8743
Epsilon: 0.15   Test Accuracy = 7111 / 10000 = 0.7111
Epsilon: 0.2    Test Accuracy = 4877 / 10000 = 0.4877
Epsilon: 0.25   Test Accuracy = 2717 / 10000 = 0.2717
Epsilon: 0.3    Test Accuracy = 1418 / 10000 = 0.1418

```







 结果
 [¶](#results "此标题的永久链接")
-----------------------------------------------------



### 
 准确度与 Epsilon
 [¶](#accuracy-vs-epsilon "此标题的永久链接")



 第一个结果是精度与 epsilon 图。正如前面提到的，随着 epsilon 的增加，我们预计测试精度会降低。这是因为较大的 epsilon 意味着我们在最大化损失的方向上迈出了更大的一步。请注意，即使 epsilon 值是线性间隔的，曲线中的趋势也不是线性的。例如，
 
 \(\epsilon=0.05\)
 
 的准确度仅比
 
 
 \(\epsilon=0\)\ 低 4% 左右n 
 ，但
 
 \(\epsilon=0.2\)
 
 的精度比
 
 \(\epsilon=0.15\)
低 25%
 
 。另外，请注意模型的准确度
达到 10 类分类器的随机准确度
 
 \(\epsilon=0.25\)
 
 和
 
 \(\epsilon= 0.3\)

.






```
plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

```



![准确度与 Epsilon](https://pytorch.org/tutorials/_images/sphx_glr_fgsm_tutorial_001.png)


### 
 对抗性示例示例
 [¶](#sample-adversarial-examples "此标题的永久链接")



 还记得天下没有免费的午餐吗？在这种情况下，随着 epsilon 的增加
测试精度会降低
 **但是** 
扰动变得更容易
被察觉。实际上，攻击者必须考虑准确性下降和可感知性之间的权衡。在这里，我们展示了每个 epsilon 值的成功对抗示例的一些示例。该图的每一行显示不同的 epsilon 值。第一行是
 
 \\(\\epsilon=0\\)
 
 示例，表示没有扰动的原始
\xe2\x80\x9cclean\xe2\x80\x9d 图像。每个图像的标题显示
\xe2\x80\x9原始分类 -> 对抗性分类。\xe2\x80\x9d 注意，
扰动开始在
 
 \\(\\epsilon=0.15\ \)
 
 并且
在
 
 \\(\\epsilon=0.3\\)
 
 处非常明显。然而，在所有情况下，尽管噪声增加，
人类仍然能够识别正确的类别。






```
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
            plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title(f"{orig} -> {adv}")
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()

```



![7 -> 7, 9 -> 9, 0 -> 0, 3 -> 3, 5 -> 5, 2 -> 8, 1 -> 3, 3 -> 5, 4 -> 6, 4 -> 9, 9 -> 4, 5 -> 6, 9 -> 5, 9 -> 5, 3 -> 2, 3 -> 5, 5 -> 3, 1 -> 6, 4 -> 9, 7 -> 9, 7 -> 2, 8 -> 2, 4 -> 8, 3 -> 7, 5 -> 3, 8 -> 3, 0 -> 8, 6 -> 5, 2 -> 3, 1 -> 8, 1 -> 9, 1 -> 8, 5 -> 8, 7 -> 8, 0 -> 2](https://pytorch.org/tutorials/_images/sphx_glr_fgsm_tutorial_002.png)




 下一步去哪里？
 [¶](#where-to-go-next "此标题的永久链接")
--------------------------------------------------------------------------



 希望本教程能够让您对对抗性机器学习主题有一些深入了解。从这里有许多潜在的方向。
这种攻击代表了对抗性攻击研究的开始
，因为关于如何攻击和防御对手的 ML 模型已经有许多后续想法。事实上，在 NIPS 2017 上有一场
对抗性攻击和防御竞赛，竞赛中使用的许多方法
在本文中有描述：
 [对抗性攻击和防御竞赛](https://arxiv.org/pdf/1804.00097.pdf) 
 。这项工作
非防御还引出了一种想法，即让机器学习模型
更
*鲁棒*
一般来说，适应自然扰动和对抗性
精心设计的输入。



另一个方向是不同域中的对抗性攻击和防御。对抗性研究不仅限于图像领域，请查看
[此](https://arxiv.org/pdf/1801.01944.pdf)
对nspeech-to-text模型的攻击。但也许了解更多关于对抗性机器学习的最好方法就是亲自动手。尝试实施与 NIPS 2017 竞赛不同的攻击，看看与 FGSM 有何不同。然后，尝试保护模型免受
您自己的攻击。




 根据可用资源，进一步的方向是修改
代码以支持批处理、并行和/或分布式
vs 在上面针对每个 
 `epsilon 一次处理一个攻击
 

 test()`
 循环。




**脚本总运行时间:** 
 ( 3 分 56.768 秒)






[`下载
 

 Python
 

 源
 

 代码:
 

 fgsm_tutorial.py`](../_downloads/377bf4a7b1761e5f081e057385870d8e/fgsm_tutorial.py ）






[`下载
 

 Jupyter
 

 笔记本:
 

 fgsm_tutorial.ipynb`](../_downloads/56c122e1c18e5e07666673e900acaed5/fgsm_tutorial.ipynb)






[Sphinx-Gallery 生成的图库](https://sphinx-gallery.github.io)









