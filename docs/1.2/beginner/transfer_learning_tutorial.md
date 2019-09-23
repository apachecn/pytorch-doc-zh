# 迁移学习教程

**作者** ：[ Sasank Chilamkurthy ](https://chsasank.github.io)

在本教程中，您将学习如何使用迁移学习训练网络。你可以阅读更多关于[
cs231n票据转让学习](https://cs231n.github.io/transfer-learning/)

引用这些笔记，

>
[HTG0在实践中，很少有人训练的整个卷积网络从头开始（与随机初始化），因为它是比较少见到有足够大的数据集。相反，它是常见的pretrain上的非常大的数据集（例如ImageNet，其中包含与1000个类别1200000个图像）一个ConvNet，然后使用ConvNet无论是作为初始化或对于感兴趣的任务的固定特征提取。

这两大转移学习情境如下所示：

  * **微调的convnet** ：除了随机initializaion，我们初始化一个预训练的网络的网络，就像是在imagenet 1000集训练之一。培训的其余神色如常。
  * **ConvNet为固定特征提取** ：在这里，我们将冻结的权重的所有不同的是最终的完全连接层的网络。这最后的完全连接层被替换为一个新的随机的权重也只有这层进行训练。

    
    
    # License: BSD
    # Author: Sasank Chilamkurthy
    
    from __future__ import print_function, division
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
    import numpy as np
    import torchvision
    from torchvision import datasets, models, transforms
    import matplotlib.pyplot as plt
    import time
    import os
    import copy
    
    plt.ion()   # interactive mode
    

## 负载数据

我们将使用torchvision和torch.utils.data包加载数据。

我们今天要解决的问题是训练的模型进行分类 **蚂蚁** 和
**蜜蜂HTG3。我们每次约120训练图像蚂蚁和蜜蜂。有75个每一类验证图像。通常情况下，这是一个非常小的数据集在一概而论，如果从头开始培训。由于我们使用的迁移学习，我们应该能够概括得相当好。**

该数据集是imagenet的一个很小的子集。

Note

从[此处](https://download.pytorch.org/tutorial/hymenoptera_data.zip)下载数据，并将其解压到当前目录。

    
    
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    data_dir = 'data/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

### 可视化的几个图像

让我们想象一些训练图像，以便了解数据扩充。

    
    
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
    
    
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    
    imshow(out, title=[class_names[x] for x in classes])
    

![../_images/sphx_glr_transfer_learning_tutorial_001.png](../_images/sphx_glr_transfer_learning_tutorial_001.png)

## 培养模式

现在，让我们写一个通用函数来训练模型。在这里，我们将说明：

  * 安排学习率
  * 保存最好的模式

在下文中，参数`调度 `是从`torch.optim.lr_scheduler`的LR调度对象。

    
    
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()
    
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
    
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
    
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
    
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()
    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
    
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
    
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
    

### 可视化模型预测

泛型函数来显示一些图像预测

    
    
    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()
    
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)
    
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
    
                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                    imshow(inputs.cpu().data[j])
    
                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)
    

## 微调修道院

加载一个预训练的模型和复位最终完全连接层。

    
    
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)
    
    model_ft = model_ft.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    

### 火车和评价

它应该承担CPU周围15-25分钟。在GPU的是，它需要不到一分钟。

    
    
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)
    

日期：

    
    
    Epoch 0/24
    ----------
    train Loss: 0.6751 Acc: 0.7049
    val Loss: 0.1834 Acc: 0.9346
    
    Epoch 1/24
    ----------
    train Loss: 0.5892 Acc: 0.7746
    val Loss: 1.0048 Acc: 0.6667
    
    Epoch 2/24
    ----------
    train Loss: 0.6568 Acc: 0.7459
    val Loss: 0.6047 Acc: 0.8366
    
    Epoch 3/24
    ----------
    train Loss: 0.4196 Acc: 0.8320
    val Loss: 0.4388 Acc: 0.8562
    
    Epoch 4/24
    ----------
    train Loss: 0.5883 Acc: 0.8033
    val Loss: 0.4013 Acc: 0.8889
    
    Epoch 5/24
    ----------
    train Loss: 0.6684 Acc: 0.7705
    val Loss: 0.2666 Acc: 0.9412
    
    Epoch 6/24
    ----------
    train Loss: 0.5308 Acc: 0.7787
    val Loss: 0.4803 Acc: 0.8693
    
    Epoch 7/24
    ----------
    train Loss: 0.3464 Acc: 0.8566
    val Loss: 0.2385 Acc: 0.8954
    
    Epoch 8/24
    ----------
    train Loss: 0.4586 Acc: 0.7910
    val Loss: 0.2064 Acc: 0.9020
    
    Epoch 9/24
    ----------
    train Loss: 0.3438 Acc: 0.8402
    val Loss: 0.2336 Acc: 0.9020
    
    Epoch 10/24
    ----------
    train Loss: 0.2405 Acc: 0.9016
    val Loss: 0.1866 Acc: 0.9346
    
    Epoch 11/24
    ----------
    train Loss: 0.2335 Acc: 0.8852
    val Loss: 0.2152 Acc: 0.9216
    
    Epoch 12/24
    ----------
    train Loss: 0.3441 Acc: 0.8402
    val Loss: 0.2298 Acc: 0.9020
    
    Epoch 13/24
    ----------
    train Loss: 0.2513 Acc: 0.9098
    val Loss: 0.2204 Acc: 0.9020
    
    Epoch 14/24
    ----------
    train Loss: 0.2745 Acc: 0.8934
    val Loss: 0.2439 Acc: 0.8889
    
    Epoch 15/24
    ----------
    train Loss: 0.2978 Acc: 0.8607
    val Loss: 0.2817 Acc: 0.8497
    
    Epoch 16/24
    ----------
    train Loss: 0.2560 Acc: 0.8975
    val Loss: 0.1933 Acc: 0.9281
    
    Epoch 17/24
    ----------
    train Loss: 0.2326 Acc: 0.9098
    val Loss: 0.2176 Acc: 0.9085
    
    Epoch 18/24
    ----------
    train Loss: 0.2274 Acc: 0.9016
    val Loss: 0.2084 Acc: 0.9346
    
    Epoch 19/24
    ----------
    train Loss: 0.3091 Acc: 0.8689
    val Loss: 0.2270 Acc: 0.9150
    
    Epoch 20/24
    ----------
    train Loss: 0.2540 Acc: 0.8975
    val Loss: 0.1957 Acc: 0.9216
    
    Epoch 21/24
    ----------
    train Loss: 0.3203 Acc: 0.8648
    val Loss: 0.1969 Acc: 0.9216
    
    Epoch 22/24
    ----------
    train Loss: 0.3048 Acc: 0.8443
    val Loss: 0.1981 Acc: 0.9346
    
    Epoch 23/24
    ----------
    train Loss: 0.2526 Acc: 0.9016
    val Loss: 0.2415 Acc: 0.8889
    
    Epoch 24/24
    ----------
    train Loss: 0.3041 Acc: 0.8689
    val Loss: 0.1894 Acc: 0.9346
    
    Training complete in 1m 7s
    Best val Acc: 0.941176
    
    
    
    visualize_model(model_ft)
    

![../_images/sphx_glr_transfer_learning_tutorial_002.png](../_images/sphx_glr_transfer_learning_tutorial_002.png)

## ConvNet为固定特征提取

在这里，我们需要冻结所有网络，除了最后一层。我们需要设置`requires_grad  ==  假 `冻结参数，使梯度不`计算向后（） `。

您可以将文档[此处](https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-
from-backward)在阅读更多关于这一点。

    
    
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
    
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    
    model_conv = model_conv.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    

### 火车和评价

在CPU这将需要大约一半的时间比以前的情况。这是预期的梯度不需要计算对于大多数网络。然而，前确实需要进行计算。

    
    
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=25)
    

Out:

    
    
    Epoch 0/24
    ----------
    train Loss: 0.6073 Acc: 0.6598
    val Loss: 0.2511 Acc: 0.8954
    
    Epoch 1/24
    ----------
    train Loss: 0.5457 Acc: 0.7459
    val Loss: 0.5169 Acc: 0.7647
    
    Epoch 2/24
    ----------
    train Loss: 0.4023 Acc: 0.8320
    val Loss: 0.2361 Acc: 0.9150
    
    Epoch 3/24
    ----------
    train Loss: 0.5150 Acc: 0.7869
    val Loss: 0.5423 Acc: 0.8039
    
    Epoch 4/24
    ----------
    train Loss: 0.4142 Acc: 0.8115
    val Loss: 0.2257 Acc: 0.9216
    
    Epoch 5/24
    ----------
    train Loss: 0.6364 Acc: 0.7418
    val Loss: 0.3133 Acc: 0.8889
    
    Epoch 6/24
    ----------
    train Loss: 0.5543 Acc: 0.7664
    val Loss: 0.1959 Acc: 0.9412
    
    Epoch 7/24
    ----------
    train Loss: 0.3552 Acc: 0.8443
    val Loss: 0.2013 Acc: 0.9477
    
    Epoch 8/24
    ----------
    train Loss: 0.3538 Acc: 0.8525
    val Loss: 0.1825 Acc: 0.9542
    
    Epoch 9/24
    ----------
    train Loss: 0.3954 Acc: 0.8402
    val Loss: 0.1959 Acc: 0.9477
    
    Epoch 10/24
    ----------
    train Loss: 0.3615 Acc: 0.8443
    val Loss: 0.1779 Acc: 0.9542
    
    Epoch 11/24
    ----------
    train Loss: 0.3951 Acc: 0.8320
    val Loss: 0.1730 Acc: 0.9542
    
    Epoch 12/24
    ----------
    train Loss: 0.4111 Acc: 0.8156
    val Loss: 0.2573 Acc: 0.9150
    
    Epoch 13/24
    ----------
    train Loss: 0.3073 Acc: 0.8525
    val Loss: 0.1901 Acc: 0.9477
    
    Epoch 14/24
    ----------
    train Loss: 0.3288 Acc: 0.8279
    val Loss: 0.2114 Acc: 0.9346
    
    Epoch 15/24
    ----------
    train Loss: 0.3472 Acc: 0.8525
    val Loss: 0.1989 Acc: 0.9412
    
    Epoch 16/24
    ----------
    train Loss: 0.3309 Acc: 0.8689
    val Loss: 0.1757 Acc: 0.9412
    
    Epoch 17/24
    ----------
    train Loss: 0.3963 Acc: 0.8197
    val Loss: 0.1881 Acc: 0.9608
    
    Epoch 18/24
    ----------
    train Loss: 0.3332 Acc: 0.8484
    val Loss: 0.2175 Acc: 0.9412
    
    Epoch 19/24
    ----------
    train Loss: 0.3419 Acc: 0.8320
    val Loss: 0.1932 Acc: 0.9412
    
    Epoch 20/24
    ----------
    train Loss: 0.3471 Acc: 0.8689
    val Loss: 0.1851 Acc: 0.9477
    
    Epoch 21/24
    ----------
    train Loss: 0.2843 Acc: 0.8811
    val Loss: 0.1772 Acc: 0.9477
    
    Epoch 22/24
    ----------
    train Loss: 0.4024 Acc: 0.8402
    val Loss: 0.1818 Acc: 0.9542
    
    Epoch 23/24
    ----------
    train Loss: 0.2409 Acc: 0.8975
    val Loss: 0.2211 Acc: 0.9346
    
    Epoch 24/24
    ----------
    train Loss: 0.3838 Acc: 0.8238
    val Loss: 0.1918 Acc: 0.9412
    
    Training complete in 0m 34s
    Best val Acc: 0.960784
    
    
    
    visualize_model(model_conv)
    
    plt.ioff()
    plt.show()
    

![../_images/sphx_glr_transfer_learning_tutorial_003.png](../_images/sphx_glr_transfer_learning_tutorial_003.png)

**脚本的总运行时间：** （1分钟53.655秒）

[`Download Python source code:
transfer_learning_tutorial.py`](../_downloads/07d5af1ef41e43c07f848afaf5a1c3cc/transfer_learning_tutorial.py)

[`Download Jupyter notebook:
transfer_learning_tutorial.ipynb`](../_downloads/62840b1eece760d5e42593187847261f/transfer_learning_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](deploy_seq2seq_hybrid_frontend_tutorial.html "Deploying a Seq2Seq
Model with TorchScript") [![](../_static/images/chevron-right-orange.svg)
Previous](examples_nn/dynamic_net.html "PyTorch: Control Flow + Weight
Sharing")

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

  * 迁移学习教程
    * 负载数据
      * 可视化几个图像
    * 训练模型
      * 可视化模型预测
    * 微调的convnet 
      * 火车和评价
    * ConvNet为固定特征提取
      * 火车和评价

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

