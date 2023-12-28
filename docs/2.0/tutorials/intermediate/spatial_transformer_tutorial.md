# 空间变压器网络教程 [¶](#spatial-transformer-networks-tutorial "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/spatial_transformer_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html>




**作者** 
 :
 [Ghassen HAMROUNI](https://github.com/GHamrouni)




![https://pytorch.org/tutorials/_images/FSeq.png](https://pytorch.org/tutorials/_images/FSeq.png)


 在本教程中，您将学习如何使用称为空间变换器网络的视觉注意机制来增强网络。您可以在 [DeepMind 论文](https://arxiv.org/abs/1506.02025) 中阅读有关空间变换器
网络的更多信息


空间变换网络是对任何空间变换的可微分注意力的概括。空间变换网络
(简称 STN)允许神经网络学习如何对输入图像执行空间
变换，以增强模型的几何
方差。
例如，它可以裁剪感兴趣的区域，缩放并纠正图像的方向。它可能是一种有用的机制，因为 CNN
 对于旋转和缩放以及更一般的仿射变换
不是不变的。




 STN 最好的事情之一是能够简单地将其插入
任何现有的 CNN，只需很少的修改。






```
# License: BSD
# Author: Ghassen Hamrouni

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

plt.ion()   # interactive mode

```






```
<contextlib.ExitStack object at 0x7f93c7feaf20>

```





## 正在加载数据 [¶](#loading-the-data "永久链接到此标题")




 在这篇文章中，我们使用经典的 MNIST 数据集进行实验。使用
通过空间变换器
网络增强的标准卷积网络。






```
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose(
                       [transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=4)
# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=4)

```






```
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./MNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/9912422 [00:00<?, ?it/s]
100%|##########| 9912422/9912422 [00:00<00:00, 322702592.79it/s]
Extracting ./MNIST/raw/train-images-idx3-ubyte.gz to ./MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./MNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/28881 [00:00<?, ?it/s]
100%|##########| 28881/28881 [00:00<00:00, 46770538.16it/s]
Extracting ./MNIST/raw/train-labels-idx1-ubyte.gz to ./MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./MNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/1648877 [00:00<?, ?it/s]
100%|##########| 1648877/1648877 [00:00<00:00, 162999160.87it/s]
Extracting ./MNIST/raw/t10k-images-idx3-ubyte.gz to ./MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./MNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/4542 [00:00<?, ?it/s]
100%|##########| 4542/4542 [00:00<00:00, 34202026.51it/s]
Extracting ./MNIST/raw/t10k-labels-idx1-ubyte.gz to ./MNIST/raw

```





## 描述空间变换器网络 [¶](#depicting-spatial-transformer-networks "永久链接到此标题")




 空间变换器网络可归结为三个主要组件：



* 定位网络是一个常规的 CNN，它对变换参数进行回归。从未从该数据集中
显式学习变换，而是网络自动学习
提高全局精度的空间变换。
* 网格生成器在输入图像中生成
对应于输出图像中每个像素的坐标网格.
* 采样器使用变换参数并将其应用于
输入图像。



![https://pytorch.org/tutorials/_images/stn-arch.png](https://pytorch.org/tutorials/_images/stn-arch.png)



 注意




 我们需要包含
affine_grid 和 grid_sample 模块的最新版本的 PyTorch。







```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net().to(device)

```





## 训练模型 [¶](#training-the-model "永久链接到此标题")




 现在，让’s使用SGD算法来训练模型。网络正在以监督方式学习分类任务。同时
模型以端到端的方式自动学习 STN。






```
optimizer = optim.SGD(model.parameters(), lr=0.01)


def train(epoch):
 [model.train](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train "torch.nn.Module. train")()
 for batch_idx, (data, target) in enumerate([train_loader](https://pytorch.org/docs/stable/data.html#torch.utils.data. DataLoader "torch.utils.data.DataLoader")):
 data, target = data.to([device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device "torch.device ")), target.to([device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device "torch.device"))

 [optimizer.zero_grad] (https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD.zero_grad“torch.optim.SGD.zero_grad”)()
输出=模型(数据) 
 损失 = [F.nll_loss](https://pytorch.org/docs/stable/generated/torch.nn.function.nll_loss.html#torch.nn.function.nll_loss "torch.nn.function.nll_loss")(输出，目标)
 loss.backward()
 [optimizer.step](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim。 SGD.step "torch.optim.SGD.step")()
 if batch_idx % 500 == 0:
 print('训练纪元: {} [{}/{} ({:.0f }%)]\tLoss: {:.6f}'.format(
 epoch, batch_idx * len(data), len([train_loader.dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST "torchvision.datasets.MNIST")),
 100. * batch_idx /len([train\ _loader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader "torch.utils.data.DataLoader")), loss.item()))
#
 # 一个简单的测试程序来测量 STN 在 MNIST 上的性能。
#


def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))

```





## 可视化 STN 结果 [¶](#visualizing-the-stn-results "固定链接到此标题")




 现在，我们将检查学习到的视觉注意力
机制的结果。




 我们定义了一个小辅助函数，以便在训练时
可视化
转换。






```
def convert_image_np(inp):
 """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.


def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

for epoch in range(1, 20 + 1):
    train(epoch)
    test()

# Visualize the STN transformation on some input batch
visualize_stn()

plt.ioff()
plt.show()

```



![数据集图像，转换后的图像](https://pytorch.org/tutorials/_images/sphx_glr_spatial_transformer_tutorial_001.png)



```
/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/nn/functional.py:4358: UserWarning:

Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.

/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/nn/functional.py:4296: UserWarning:

Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.

Train Epoch: 1 [0/60000 (0%)]   Loss: 2.315648
Train Epoch: 1 [32000/60000 (53%)]      Loss: 1.047744
/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/nn/_reduction.py:42: UserWarning:

size_average and reduce args will be deprecated, please use reduction='sum' instead.


Test set: Average loss: 0.2656, Accuracy: 9264/10000 (93%)

Train Epoch: 2 [0/60000 (0%)]   Loss: 0.533303
Train Epoch: 2 [32000/60000 (53%)]      Loss: 0.331733

Test set: Average loss: 0.1831, Accuracy: 9462/10000 (95%)

Train Epoch: 3 [0/60000 (0%)]   Loss: 0.387849
Train Epoch: 3 [32000/60000 (53%)]      Loss: 0.215252

Test set: Average loss: 0.1148, Accuracy: 9656/10000 (97%)

Train Epoch: 4 [0/60000 (0%)]   Loss: 0.338932
Train Epoch: 4 [32000/60000 (53%)]      Loss: 0.213857

Test set: Average loss: 0.1616, Accuracy: 9491/10000 (95%)

Train Epoch: 5 [0/60000 (0%)]   Loss: 0.305876
Train Epoch: 5 [32000/60000 (53%)]      Loss: 0.217289

Test set: Average loss: 0.1351, Accuracy: 9609/10000 (96%)

Train Epoch: 6 [0/60000 (0%)]   Loss: 0.221662
Train Epoch: 6 [32000/60000 (53%)]      Loss: 0.145264

Test set: Average loss: 0.0708, Accuracy: 9782/10000 (98%)

Train Epoch: 7 [0/60000 (0%)]   Loss: 0.114100
Train Epoch: 7 [32000/60000 (53%)]      Loss: 0.190583

Test set: Average loss: 0.0742, Accuracy: 9766/10000 (98%)

Train Epoch: 8 [0/60000 (0%)]   Loss: 0.293466
Train Epoch: 8 [32000/60000 (53%)]      Loss: 0.070622

Test set: Average loss: 0.0616, Accuracy: 9821/10000 (98%)

Train Epoch: 9 [0/60000 (0%)]   Loss: 0.092730
Train Epoch: 9 [32000/60000 (53%)]      Loss: 0.080178

Test set: Average loss: 0.0776, Accuracy: 9766/10000 (98%)

Train Epoch: 10 [0/60000 (0%)]  Loss: 0.095328
Train Epoch: 10 [32000/60000 (53%)]     Loss: 0.227478

Test set: Average loss: 0.0738, Accuracy: 9776/10000 (98%)

Train Epoch: 11 [0/60000 (0%)]  Loss: 0.162715
Train Epoch: 11 [32000/60000 (53%)]     Loss: 0.114082

Test set: Average loss: 0.0616, Accuracy: 9810/10000 (98%)

Train Epoch: 12 [0/60000 (0%)]  Loss: 0.105073
Train Epoch: 12 [32000/60000 (53%)]     Loss: 0.184882

Test set: Average loss: 0.0538, Accuracy: 9839/10000 (98%)

Train Epoch: 13 [0/60000 (0%)]  Loss: 0.129685
Train Epoch: 13 [32000/60000 (53%)]     Loss: 0.138069

Test set: Average loss: 0.0522, Accuracy: 9838/10000 (98%)

Train Epoch: 14 [0/60000 (0%)]  Loss: 0.046923
Train Epoch: 14 [32000/60000 (53%)]     Loss: 0.100477

Test set: Average loss: 0.0514, Accuracy: 9849/10000 (98%)

Train Epoch: 15 [0/60000 (0%)]  Loss: 0.063011
Train Epoch: 15 [32000/60000 (53%)]     Loss: 0.158940

Test set: Average loss: 0.0734, Accuracy: 9765/10000 (98%)

Train Epoch: 16 [0/60000 (0%)]  Loss: 0.076108
Train Epoch: 16 [32000/60000 (53%)]     Loss: 0.149375

Test set: Average loss: 0.0452, Accuracy: 9857/10000 (99%)

Train Epoch: 17 [0/60000 (0%)]  Loss: 0.266226
Train Epoch: 17 [32000/60000 (53%)]     Loss: 0.184768

Test set: Average loss: 0.0857, Accuracy: 9746/10000 (97%)

Train Epoch: 18 [0/60000 (0%)]  Loss: 0.112116
Train Epoch: 18 [32000/60000 (53%)]     Loss: 0.089787

Test set: Average loss: 0.0583, Accuracy: 9833/10000 (98%)

Train Epoch: 19 [0/60000 (0%)]  Loss: 0.065648
Train Epoch: 19 [32000/60000 (53%)]     Loss: 0.143108

Test set: Average loss: 0.0442, Accuracy: 9863/10000 (99%)

Train Epoch: 20 [0/60000 (0%)]  Loss: 0.071892
Train Epoch: 20 [32000/60000 (53%)]     Loss: 0.154807

Test set: Average loss: 0.0489, Accuracy: 9851/10000 (99%)

```




**脚本的总运行时间:** 
 ( 2 分 4.495 秒)
