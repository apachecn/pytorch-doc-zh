


 使用 TensorBoard 可视化模型、数据和训练
 [¶](#visualizing-models-data-and-training-with-tensorboard "永久链接到此标题")
===================================================================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/tensorboard_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html>




 在
 [60 分钟闪电战](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) 
 中，
我们向您展示如何加载数据，
通过我们定义为的模型提供数据
 `nn.Module` 的子类
，
在训练数据上训练此模型，并在测试数据上测试它。
为了查看’s 发生了什么，我们打印出一些统计数据作为模型
is 
但是，我们可以做得更好：PyTorch 与 TensorBoard 集成，
TensorBoard 是一种旨在可视化神经网络训练运行结果的工具。
本教程使用 [Fashion-MNIST 数据集](https://github.com/zalandoresearch/fashion-mnist) 说明其部分功能，可使用 torchvision.datasets 将其读入 PyTorch 
 
 。




 在本教程中，我们’ 将学习如何：




> 
> 
> 1. 读入数据并进行适当的转换（与之前的教程几乎相同）。
> 2. 设置 TensorBoard。
> 3. 写入 TensorBoard。
> 4.使用 TensorBoard 检查模型架构。
> 5. 使用 TensorBoard 创建我们在上一个教程中创建的可视化的交互式版本，代码较少
> 
> 
> 
>



 具体来说，在第 5 点，我们’ 将看到：




> 
> 
> * 检查训练数据的几种方法
> * 如何在训练时跟踪我们的模型\xe2\x80\x99s 性能
> * 如何评估我们的模型\xe2\训练后的性能为 x80\x99。
> 
> 
> 
>



 我们’ 将以与 [CIFAR-10 教程](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) 中类似的样板代码开始 
 :






```
# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)
testset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=2)


testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=2)

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

```




 我们’ 将根据该教程定义类似的模型架构，仅
进行少量修改，以说明图像现在
是一个通道而不是三个通道，并且是 28x28 而不是 32x32：






```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

```




 我们’ 将定义与之前相同的
 `优化器`
 和
 `标准`
:






```
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

```





 1. TensorBoard 设置
 [¶](#tensorboard-setup "永久链接到此标题")
--------------------------------------------------------------------------- -



 现在我们’ll 设置 TensorBoard，从
 `torch.utils`
 导入
 `tensorboard`
 并定义
 `SummaryWriter`
 ，这是我们用于将信息写入到的关键对象张量板。






```
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

```




 请注意，这一行单独创建了一个
 `runs/fashion_mnist_experiment_1`
 文件夹。






 2. 写入 TensorBoard
 [¶](#writing-to-tensorboard "永久链接到此标题")
------------------------------------------------------------------------------------------



 现在让’s 将图像写入我们的 TensorBoard - 具体来说，是一个网格 -
使用
 [make_grid](https://pytorch.org/vision/stable/utils.html#torchvision.utils.make_grid) 
.






```
# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)

```




 正在运行






```
tensorboard --logdir=runs

```




 从命令行然后导航到
 <http://localhost:6006>
 应显示以下内容。



![https://pytorch.org/tutorials/_static/img/tensorboard_first_view.png](https://pytorch.org/tutorials/_static/img/tensorboard_first_view.png)

 现在你知道如何使用 TensorBoard 了！然而，这个示例可以在 Jupyter Notebook 中完成 - TensorBoard 真正擅长的地方是创建交互式可视化。我们’ 接下来将介绍其中之一，
在本教程结束时还会介绍更多内容。






 3. 使用 TensorBoard 检查模型
 [¶](#inspect-the-model-using-tensorboard "固定链接到此标题")
------------------------------------------------------------------------------------------------------------------------------



 TensorBoard’s 的优势之一是其能够可视化复杂模型
结构。让’s可视化我们构建的模型。






```
writer.add_graph(net, images)
writer.close()

```




 现在刷新 TensorBoard 后，您应该会看到一个 “Graphs” 选项卡，
如下所示：



![https://pytorch.org/tutorials/_static/img/tensorboard_model_viz.png](https://pytorch.org/tutorials/_static/img/tensorboard_model_viz.png)

 双击 “Net” 查看它展开，查看
构成模型的各个操作的详细视图。



TensorBoard 有一个非常方便的功能，可以可视化高维数据，例如低维空间中的图像数据；我们’
接下来会介绍这个。






 4. 添加 “Projector” 到 TensorBoard
 [¶](#adding-a-projector-to-tensorboard "永久链接到此标题")
----------------------------------------------------------------------------------------------------------------


我们可以通过 [add_embedding](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter 可视化高维数据的低维表示.add_embedding) 
 方法






```
# helper function
def select_n_random(data, labels, n=100):
 '''
 Selects n random datapoints and their corresponding labels from a dataset
 '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# select random images and their target indices
images, labels = select_n_random(trainset.data, trainset.targets)

# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
writer.close()

```


现在，在 TensorBoard 的 \xe2\x80\x9cProjector\xe2\x80\x9d 选项卡中，您可以看到这 100 个
图像（每个图像都是 784 维）向下投影到三维空间。此外，这是交互式的：您可以单击
并拖动来旋转三维投影。最后，有一些提示可以使可视化更容易查看：选择左上角的 \xe2\x80\x9ccolor: label\xe2\x80\x9d\，以及启用 \xe2\x80\x9cnight 模式\xe2\ x80\x9d，这将使
图像更容易看到，因为它们的背景是白色的:



![https://pytorch.org/tutorials/_static/img/tensorboard_projector.png](https://pytorch.org/tutorials/_static/img/tensorboard_projector.png)

 现在我们’已经彻底检查了我们的数据，让’s显示TensorBoard
如何让跟踪模型训练和评估更加清晰，
从训练开始。






 5. 使用 TensorBoard 跟踪模型训练
 [¶](#tracking-model-training-with-tensorboard "固定链接到此标题")
-----------------------------------------------------------------------------------------------------------------------------



 在前面的示例中，我们只是
 *打印* 
 模型’s 运行损失
 每 2000 次迭代。现在，我们’ll 将运行损失记录到
TensorBoard，并通过
 `plot_classes_preds`
 函数查看模型
所做的预测。






```
# helper functions

def images_to_probs(net, images):
 '''
 Generates predictions and corresponding probabilities from a trained
 network and a list of images
 '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
 '''
 Generates matplotlib Figure using a trained network, along with images
 and labels from a batch, that shows the network's top prediction along
 with its probability, alongside the actual label, coloring this
 information based on whether the prediction was correct or not.
 Uses the "images_to_probs" function.
 '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

```




 最后，让 ’s 使用之前教程中的相同模型训练代码来训练模型，但每 1000 个
批次将结果写入 TensorBoard，而不是打印到控制台；这是使用 [add_scalar](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalar) 函数完成的。




 此外，在训练时，我们’ 将生成一个图像，显示模型’s
预测与该批次中包含的四个图像的实际结果。






```
running_loss = 0.0
for epoch in range(1):  # loop over the dataset multiple times

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(trainloader) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(net, inputs, labels),
                            global_step=epoch * len(trainloader) + i)
            running_loss = 0.0
print('Finished Training')

```




 现在，您可以查看标量选项卡来查看
在 15,000 次训练迭代中绘制的运行损失：



![https://pytorch.org/tutorials/_static/img/tensorboard_scalar_runs.png](https://pytorch.org/tutorials/_static/img/tensorboard_scalar_runs.png)

 此外，我们可以查看模型在整个学习过程中对任意批次做出的预测。请参阅 “Images” 选项卡并在 “predictions vs.actuals” 可视化下向下滚动
以查看此内容；
这向我们展示了，例如，仅经过 3000 次训练迭代，
该模型就已经能够区分视觉上不同的
类别，例如衬衫、运动鞋和外套，尽管’
不像后来在训练中
那样自信：



![https://pytorch.org/tutorials/_static/img/tensorboard_images.png](https://pytorch.org/tutorials/_static/img/tensorboard_images.png)

 在之前的教程中，我们在模型训练完成后查看每类的准确度；在这里，我们’将使用TensorBoard绘制精度召回
曲线（很好的解释
[此处](https://www.scikit-yb.org/en/latest/api/classifier/prcurve.html ) 
 )
对于每个类。






 6. 使用 TensorBoard 评估经过训练的模型
 [¶](#assessing-trained-models-with-tensorboard "永久链接到此标题")
-------------------------------------------------------------------------------------------------------------------------------





```
# 1. gets the probability predictions in a test_size x num_classes Tensor
# 2. gets the preds in a test_size Tensor
# takes ~10 seconds to run
class_probs = []
class_label = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]

        class_probs.append(class_probs_batch)
        class_label.append(labels)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_label = torch.cat(class_label)

# helper function
def add_pr_curve_tensorboard(class_index, test_probs, test_label, global_step=0):
 '''
 Takes in a "class_index" from 0 to 9 and plots the corresponding
 precision-recall curve
 '''
    tensorboard_truth = test_label == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_truth,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

# plot all the pr curves
for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, test_label)

```




 现在，您将看到 “PR Curves” 选项卡，其中包含每个类别的精确召回
曲线。继续探索吧；您’ 将看到，在某些类上，模型在曲线” 下几乎有 100% “area，
而在其他类上，该区域较低：



![https://pytorch.org/tutorials/_static/img/tensorboard_pr_curves.png](https://pytorch.org/tutorials/_static/img/tensorboard_pr_curves.png)

 ’s 是 TensorBoard 和 PyTorch’s 集成的介绍有了它。
当然，您可以在 JupyterNotebook 中执行 TensorBoard 所做的所有操作
，但是使用 TensorBoard，您可以在默认情况下获得交互式视觉效果。









