# 可视化模型，数据，并与TensorBoard培训

在[
60分钟闪电战](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)，我们向您展示如何在数据加载，通过我们定义为`
nn.Module
`一个子类的模型给它，这个训练模型训练数据，并测试它的测试数据。要看到发生了什么，我们打印出一些统计数据，该模型训练得到一个有意义的培训是进展。但是，我们可以做的比这更好：PyTorch与TensorBoard，设计可视化神经网络训练运行结果的工具集成。这个教程说明了一些它的功能，使用[时装-
MNIST数据集](https://github.com/zalandoresearch/fashion-mnist)可使用
torchvision.datasets 读入PyTorch。

在本教程中，我们将学习如何：

>   1. 读入数据，并用适当的变换（几乎等同于现有教程）。

>   2. 设置TensorBoard。

>   3. 写到TensorBoard。

>   4. 使用TensorBoard检查的模型体系结构。

>   5. 使用TensorBoard创建我们在上一个教程中创建可视化的互动形式，用更少的代码

>

具体来说，在点＃5，我们会看到：

>   * 一对夫妇的方式来检查我们的训练数据

>   * 因为训练

>   * 如何，一旦被训练评估我们的模型中的表现如何跟踪我们的模型的性能。

>

我们将与类似的样板代码在[
CIFAR-10教程](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)开始：

    
    
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
    # (used in the `plot_classes_preds`function below)
    def matplotlib_imshow(img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    

我们将从教程定义了一个类似的模型架构，使细微的修改考虑到一个事实，即图像是现在一个通道，而不是三个和28x28，而不是32×32：

    
    
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
    
    

我们将定义相同`优化 `和`标准 `从之前：

    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    

## 1\. TensorBoard设置

现在，我们将建立TensorBoard，导入`tensorboard`从`torch.utils`和定义`SummaryWriter
`，我们已经对TensorBoard写入信息的关键对象。

    
    
    from torch.utils.tensorboard import SummaryWriter
    
    # default `log_dir`is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    
    

请注意，这条线单独创建一个`运行/ fashion_mnist_experiment_1`文件夹中。

## 2.写入TensorBoard

现在，让我们写一个像我们TensorBoard - 具体而言，一个网格 - 使用[ make_grid
[HTG1。](https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid)

    
    
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    # create grid of images
    img_grid = torchvision.utils.make_grid(images)
    
    # show images
    matplotlib_imshow(img_grid, one_channel=True)
    
    # write to tensorboard
    writer.add_image('four_fashion_mnist_images', img_grid)
    
    

现在运行

    
    
    tensorboard --logdir=runs
    

在命令行，然后导航到[的https：//本地主机：6006 ](https://localhost:6006)应该显示如下。

![intermediate/../../_static/img/tensorboard_first_view.png](intermediate/../../_static/img/tensorboard_first_view.png)

现在你知道如何使用TensorBoard！其中TensorBoard的确有过人之处是创建交互式可视化 -
这个例子，但是，可以在Jupyter笔记本电脑来完成。我们接下来将介绍，还有几个由教程结束的一个。

## 3.使用检查模型TensorBoard

一个TensorBoard的优势之一是它的可视化复杂的模型结构的能力。让我们想象，我们构建的模型。

    
    
    writer.add_graph(net, images)
    writer.close()
    
    

现在，在刷新TensorBoard你应该会看到一个“图形”选项卡，看起来像这样：

![intermediate/../../_static/img/tensorboard_model_viz.png](intermediate/../../_static/img/tensorboard_model_viz.png)

来吧，在“网络”双击看到它扩大，看到单独的操作组成模型的详细视图。

TensorBoard具有可视化，如在低维空间中的图像数据的高维数据的非常方便的功能;我们将讨论这个未来。

## 4.添加一个“投影”到TensorBoard

我们可以通过[ add_embedding
](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_embedding)方法可视化高维数据的低维表示

    
    
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
    
    

现在TensorBoard的“投影机”选项卡，可以看到这些100张图像 - 每一个都是784维 -
投射分解成三维空间。此外，这是互动的：你可以点击并拖动旋转三维投影。最后，一对夫妇的提示，使可视化更容易看到：选择“颜色：标签”的左上角，以及使“夜间模式”，这将使得图像更容易看到，因为他们的背景是白色的：

![intermediate/../../_static/img/tensorboard_projector.png](intermediate/../../_static/img/tensorboard_projector.png)

现在我们已经彻底检查我们的数据，让我们向TensorBoard如何才能使跟踪模型的训练和评估更清晰，开始训练。

## 5.跟踪模型训练TensorBoard

在前面的例子中，我们简单地 _印刷_ 模型的运行每2000次迭代的损失。现在，我们将代替登录运行损失TensorBoard，以期到模型可通过`
plot_classes_preds`功能使得预测一起。

    
    
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
    
    

最后，让我们使用相同的模型训练码从之前的教程训练模型，但是写结果TensorBoard每1000个批次，而不是打印到控制台;这是使用[ add_scalar
](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalar)函数来完成。

此外，我们训练，我们将生成展示模型的预测与包含在该批次的四个图像的实际效果的图像。

    
    
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
    
    

现在，您可以看看标量选项卡查看正在运行的损失在绘制培训15000次迭代：

![intermediate/../../_static/img/tensorboard_scalar_runs.png](intermediate/../../_static/img/tensorboard_scalar_runs.png)

此外，我们可以看看预测在整个学习任意批量制造的模型。请参阅“图像”选项卡，然后向下滚动在“预测与实际数据”可视化看到这个;这告诉我们，例如，仅仅3000的训练迭代后，该模型已经能够视觉上不同的类，区分衬衫，运动鞋，和外套，虽然它，因为它在训练之后变成上是不是有信心：

![intermediate/../../_static/img/tensorboard_images.png](intermediate/../../_static/img/tensorboard_images.png)

在现有的教程中，我们看到每个类的准确性，一旦模型被训练;在这里，我们将使用TensorBoard绘制精确召回曲线（很好的解释[此处](https://www.scikit-
yb.org/en/latest/api/classifier/prcurve.html)）为每个类。

## 6.评估训练的模型与TensorBoard

    
    
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
    
    

现在，您将看到一个包含每个类的精确召回曲线的“公关曲线”选项卡。来吧，闲逛;你会看到，在一些类模型具有“曲线下面积”近100％，而对别人这方面是下：

![intermediate/../../_static/img/tensorboard_pr_curves.png](intermediate/../../_static/img/tensorboard_pr_curves.png)

这是一个介绍到TensorBoard和PyTorch与它集成。当然，你可以做一切TensorBoard确实在Jupyter笔记本电脑，但TensorBoard，你得到了默认情况下交互的视觉效果。

[Next ![](../_static/images/chevron-right-
orange.svg)](../beginner/saving_loading_models.html "Saving and Loading
Models") [![](../_static/images/chevron-right-orange.svg)
Previous](../beginner/deploy_seq2seq_hybrid_frontend_tutorial.html "Deploying
a Seq2Seq Model with TorchScript")

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

  * 可视化模型，数据，和与训练TensorBoard 
    * [HTG0 1. TensorBoard设置
    * [HTG0 2.写入TensorBoard 
    * 3.检查使用TensorBoard模型
    * 4.添加一个“投影”到TensorBoard 
    * [HTG0 5.跟踪模型训练TensorBoard 
    * 6.评估训练的模型与TensorBoard 

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

