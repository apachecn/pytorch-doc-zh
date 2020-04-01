# 迁移学习教程

> **作者**：[Sasank Chilamkurthy](https://chsasank.github.io)
> 
> 译者：[DrDavidS](https://github.com/DrDavidS)
> 
> 校验：[DrDavidS](https://github.com/DrDavidS)

在本教程中，您将学习如何使用迁移学习训练网络。你可以在[
cs231n笔记](https://cs231n.github.io/transfer-learning/)中阅读更多关于迁移学习的内容。

引用笔记，

>
在实践中，很少有人从头开始训练整个卷积网络(随机初始化），因为足够大的数据集是相对少见的。相反，通常在非常大的数据集(例如 ImageNet，其包含具有1000个类别的120万张图片）上预先训练一个卷积神经网络，然后使用这个卷积神经网络对目标任务进行初始化或用作固定特征提取器。

如下是两个主要的迁移学习场景：

  * **微调卷积神经网络** 我们使用预训练网络来初始化网络，而不是随机初始化，比如一个已经在imagenet 1000数据集上训练好的网络一样。其余训练和往常一样。
  * **将卷积神经网络作为固定特征提取器** ：在这里，我们将冻结除最终全连接层之外的整个网络的权重。最后一个全连接层被替换为具有随机权重的新层，并且仅训练该层。

    
    
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
    

## 加载数据

我们将使用 torchvision 和 torch.utils.data 包来加载数据。

今天，我们要解决的问题是训练一个模型来对**蚂蚁**和**蜜蜂**进行分类。我们**蚂蚁**和**蜜蜂**分别准备了大约120个训练图像，并且每类还有75个验证图像。通常，如果从头开始训练，这是一个非常小的数据集。由于我们正在使用迁移学习，我们应该能够合理地进行泛化。

该数据集是imagenet的一个很小的子集。

注意

>从[此处](https://download.pytorch.org/tutorial/hymenoptera_data.zip)下载数据，并将其解压到当前目录。

    
    
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
    

### 可视化一些图像

让我们通过可视化一些训练图像，来理解什么是数据增强。

    
    
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
    

![img/sphx_glr_transfer_learning_tutorial_001.png](https://pytorch.org/tutorials/_images/sphx_glr_transfer_learning_tutorial_001.png)

## 训练模型

现在, 让我们编写一个通用函数来训练一个模型。这里, 我们将会举例说明:

  * 调整学习率
  * 保存最好的模型

下面函数中, `scheduler` 参数是 `torch.optim.lr_scheduler` 中的学习率调整(LR scheduler）对象.

    
    
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
    

### 模型预测的可视化

用于显示少量预测图像的通用函数

    
    
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
    

## 微调卷积神经网络

加载预训练模型并重置最后的全连接层。

    
    
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
    

### 训练与评价

在CPU上训练需要大约15-25分钟。但是在GPU上，它只需不到一分钟。
    
    
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)
    

输出：

    
    
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
    

![img/sphx_glr_transfer_learning_tutorial_002.png](https://pytorch.org/tutorials/_images/sphx_glr_transfer_learning_tutorial_002.png)

## 将卷积神经网络为固定特征提取器

在这里，我们需要冻结除最后一层之外的所有网络。我们需要设置`requires_grad  ==  False `来冻结参数，以便在`backward()`中不会计算梯度。

您可以在[此处](https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-
from-backward)的文档中阅读更多相关信息。

    
    
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
    

### 训练与评价

在CPU上，与前一个场景相比，大概只花费一半的时间。这在预料之中，因为不需要为绝大多数网络计算梯度。当然，我们还是需要计算前向传播。
    
    
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
    

![img/sphx_glr_transfer_learning_tutorial_003.png](https://pytorch.org/tutorials/_images/sphx_glr_transfer_learning_tutorial_003.png)

**脚本的总运行时间：** (1分钟53.655秒）

[由Sphinx-Gallery生成的图库](https://sphinx-gallery.readthedocs.io)



