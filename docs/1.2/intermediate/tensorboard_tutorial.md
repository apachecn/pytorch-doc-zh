# 使用 TensorBoard 可视化模型，数据和训练

> 译者：[片刻](https://github.com/jiangzhonglian)
> 
> 校验：[片刻](https://github.com/jiangzhonglian)

在[60分钟闪电战](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)中，我们向您展示了如何加载数据，如何通过定义为的子类的`nn.Module`模型提供数据，如何在训练数据上训练该模型以及如何在测试数据上对其进行测试。为了了解发生了什么，我们在模型训练期间打印一些统计数据，以了解训练是否在进行。但是，我们可以做得更好：PyTorch与TensorBoard集成在一起，TensorBoard是一种工具，用于可视化神经网络训练运行的结果。本教程使用[Fashion-MNIST数据集](https://github.com/zalandoresearch/fashion-mnist)说明了其某些功能，该 数据集 可以使用torchvision.datasets读取到PyTorch中。

在本教程中，我们将学习如何：

* 读入数据并进行适当的转换(与先前的教程几乎相同）。
* 设置TensorBoard。
* 写入TensorBoard。
* 使用TensorBoard检查模型架构。
* 使用TensorBoard以更少的代码创建我们在上一个教程中创建的可视化的交互式版本

具体来说，在第5点，我们将看到：

* 检查我们训练数据的几种方法
* 在训练过程中如何跟踪模型的性能
* 训练后如何评估模型的性能。
* 我们将从与[CIFAR-10教程](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)类似的样板代码开始：

```py
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
    
我们将在该教程中定义一个类似的模型体系结构，仅需进行少量修改即可解决以下事实：图像现在是一个通道而不是三个通道，而图像是28x28而不是32x32：

```py
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
    

我们将`optimizer`与`criterion`之前定义相同：

```py
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```
    

## 1. TensorBoard设置

现在我们将设置TensorBoard，`tensorboard`从我们的关键对象导入`torch.utils`并定义它`SummaryWriter`，该关键对象用于将信息写入TensorBoard。

```py
from torch.utils.tensorboard import SummaryWriter

# default `log_dir`is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
```  

请注意，这条线单独创建一个`runs/fashion_mnist_experiment_1`文件夹中。

## 2.写入TensorBoard

现在，让我们写我们的TensorBoard形象-具体而言，一个网格-使用[make_grid](https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid)。

```py
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)
```


现在运行

```py
tensorboard --logdir=runs
```    

在命令行，然后导航到[https://localhost:6006/](https://localhost:6006)应该显示如下。

![https://pytorch.org/tutorials/_static/img/tensorboard_first_view.png](https://pytorch.org/tutorials/_static/img/tensorboard_first_view.png)

现在您知道如何使用TensorBoard了！但是，此示例可以在Jupyter Notebook中完成-TensorBoard真正擅长的地方是创建交互式可视化。我们将在接下来的内容中介绍其中之一，并在本教程结束时介绍更多内容。

## 3. 使用TensorBoard检查模型
TensorBoard的优势之一是其可视化复杂模型结构的能力。让我们可视化我们构建的模型。

```py    
writer.add_graph(net, images)
writer.close()
```

现在刷新TensorBoard后，您应该会看到一个“ Graphs”标签，如下所示：

![https://pytorch.org/tutorials/_static/img/tensorboard_model_viz.png](https://pytorch.org/tutorials/_static/img/tensorboard_model_viz.png)


继续并双击 “Net” 以展开它，查看组成模型的各个操作的详细视图。

TensorBoard具有非常方便的功能，用于可视化高维数据，例如在低维空间中的图像数据；接下来我们将介绍。

## 4. 在TensorBoard中添加一个“投影仪”

我们可以通过 [add_embedding](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_embedding) 方法可视化高维数据的低维表示

```py
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
 
现在，在TensorBoard的“投影仪”选项卡中，您可以看到这100张图像-每个图像784维-向下投影到三维空间中。此外，这是交互式的：您可以单击并拖动以旋转三维投影。最后，一些技巧可以使可视化效果更容易看到：在左上方选择“颜色：标签”，并启用“夜间模式”，这将使图像更容易看到，因为它们的背景是白色的：

![https://pytorch.org/tutorials/_static/img/tensorboard_projector.png](https://pytorch.org/tutorials/_static/img/tensorboard_projector.png)

现在我们已经彻底检查了我们的数据，让我们展示了TensorBoard如何从训练开始就可以使跟踪模型训练和评估更加清晰。

## 5. 使用TensorBoard跟踪模型训练

在前面的示例中，我们仅每2000次迭代打印一次模型的运行损失。现在，我们将运行损失记录到TensorBoard中，并通过模型查看模型所做的预测`plot_classes_preds`。


```py
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
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig
```

最后，让我们使用与之前教程中相同的模型训练代码来训练模型，但是每1000批将结果写入TensorBoard，而不是打印到控制台。这是使用 [add_scalar](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalar) 函数完成的 。

另外，在训练过程中，我们将生成一幅图像，显示该批次中包含的四幅图像的模型预测与实际结果。

```py
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

现在，您可以查看“标量”选项卡，以查看在15,000次训练迭代中绘制的运行损失：


![https://pytorch.org/tutorials/_static/img/tensorboard_scalar_runs.png](https://pytorch.org/tutorials/_static/img/tensorboard_scalar_runs.png)

此外，我们可以看看预测在整个学习任意批量制造的模型。查看“图像”选项卡，然后在“预测与实际”可视化条件下向下滚动以查看此内容；这向我们表明，例如，仅经过3000次训练迭代，该模型就已经能够区分出视觉上截然不同的类，例如衬衫，运动鞋和外套，尽管它并没有像后来的训练那样充满信心：

![https://pytorch.org/tutorials/_static/img/tensorboard_images.png](https://pytorch.org/tutorials/_static/img/tensorboard_images.png)


在之前的教程中，我们研究了模型训练后的每班准确性；在这里，我们将使用TensorBoard绘制每个类的精确调用曲线([此处](https://www.scikit-yb.org/en/latest/api/classifier/prcurve.html)有很好的解释 )。

## 6. 使用TensorBoard评估经过训练的模型

```py
# 1. gets the probability predictions in a test_size x num_classes Tensor
# 2. gets the preds in a test_size Tensor
# takes ~10 seconds to run
class_probs = []
class_preds = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]
        _, class_preds_batch = torch.max(output, 1)

        class_probs.append(class_probs_batch)
        class_preds.append(class_preds_batch)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_preds = torch.cat(class_preds)

# helper function
def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

# plot all the pr curves
for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, test_preds)
``` 

现在，您将看到一个“ PR Curves”选项卡，其中包含每个类别的精确调用曲线。继续戳一下；您会看到，在某些类别上，模型的“曲线下面积”接近100％，而在另一些类别上，该面积更低：

![https://pytorch.org/tutorials/_static/img/tensorboard_pr_curves.png](https://pytorch.org/tutorials/_static/img/tensorboard_pr_curves.png)

这是TensorBoard和PyTorch与之集成的介绍。当然，你可以做一切TensorBoard确实在Jupyter笔记本电脑，但TensorBoard，你得到了默认情况下交互的视觉效果。
