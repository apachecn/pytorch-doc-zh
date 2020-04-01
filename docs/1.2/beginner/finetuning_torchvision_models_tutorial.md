# 微调 TorchVision 模型

> **作者**：[Nathan Inkawhich](https://github.com/inkawhich)
> 
> 译者：[片刻](https://github.com/jiangzhonglian)
> 
> 校验：[片刻](https://github.com/jiangzhonglian)

在本教程中，我们将更深入地研究如何微调和特征提取[Torchvision模型](https://pytorch.org/docs/stable/torchvision/models.html)，所有这些模型都已在1000类Imagenet数据集上进行了预训练。本教程将深入研究如何使用几种现代的CNN架构，并将建立一种直观的方法来微调任何PyTorch模型。由于每种模型的架构都不同，因此没有适用于所有场景的样板微调代码。相反，研究人员必须查看现有的体系结构，并对每个模型进行自定义调整。

在本文档中，我们将执行两种类型的迁移学习：微调和特征提取。在**微调**中，我们从预先训练的模型开始，并为新任务更新模型的所有参数，实质上是对整个模型进行重新训练。在特征提取中，我们从预先训练的模型开始，仅更新最终的层权重，从中得出预测值。之所以称为特征提取，是因为我们将预训练的CNN用作固定的特征提取器，并且仅更改输出层。有关转学的更多技术信息，请参见[此处](https://cs231n.github.io/transfer-learning/)和[此处](https://ruder.io/transfer-learning/)。

通常，两种转移学习方法都遵循相同的几个步骤：

* 初始化预训练模型
* 重塑最终图层，使其输出数量与新数据集中的类数相同
* 为优化算法定义我们要在训练期间更新哪些参数
* 运行训练步骤

```
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
```

Out:
    
    PyTorch Version:  1.2.0
    Torchvision Version:  0.4.0
    

## 输入

这是要更改运行的所有参数。我们将使用可以在[此处下载](https://download.pytorch.org/tutorial/hymenoptera_data.zip)的hymenoptera_data数据集 。该数据集包含**bees**和**ants**两类，其结构使得我们可以使用 [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder) 数据集，而不必编写自己的自定义数据集。下载数据并将`data_dir`输入设置为数据集的根目录。输入的`model_name`是您要使用的模型的名称，必须从以下列表中进行选择：
    
    [resnet, alexnet, vgg, squeezenet, densenet, inception]

其他输入如下：`num_classes`是数据集中的类数，`batch_size`是用于训练的批次大小，可以根据您计算机的能力进行调整，`num_epochs`是我们要运行的训练时期的数量，以及`feature_extract`是一个布尔值，它定义了我们是微调还是特征提取。如果`feature_extract = False`，则微调模型并更新所有模型参数。 如果`feature_extract = True`，则仅更新最后一层参数，其他参数保持固定。
    
    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "./data/hymenoptera_data"
    
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "squeezenet"
    
    # Number of classes in the dataset
    num_classes = 2
    
    # Batch size for training (change depending on how much memory you have)
    batch_size = 8
    
    # Number of epochs to train for
    num_epochs = 15
    
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True
    

## 辅助函数

在编写用于调整模型的代码之前，让我们定义一些辅助函数。

### 模型训练和验证码

`train_model`函数处理给定模型的训练和验证。作为输入，它采用PyTorch模型，数据加载器字典，损失函数，优化器，要训练和验证的指定时期数以及当模型是Inception模型时的布尔标志。 `is_inception`标志用于适应Inception v3模型，因为该体系结构使用辅助输出，并且总体模型损失同时考虑了辅助输出和最终输出，如[此处](https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958)所述。 该函数针对指定的时期数进行训练，并且在每个时期之后运行完整的验证步骤。 它还跟踪最佳模型(在验证准确性方面），并且在训练结束时返回最佳模型。 在每个时期之后，将打印训练和验证准确性。
    
    def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
        since = time.time()
    
        val_acc_history = []
    
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
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
    
                        _, preds = torch.max(outputs, 1)
    
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
    
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
    
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
    
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history
    

### 设置模型参数`.requires_grad`属性

当我们进行特征提取时，此辅助函数将模型中参数的`.requires_grad`属性设置为`False`。默认情况下，当我们加载预训练的模型时，所有参数都具有`.requires_grad = True`，如果我们从头开始或进行微调训练，这很好。但是，如果我们要进行特征提取，并且只想为新初始化的图层计算梯度，那么我们希望所有其他参数都不需要梯度。稍后将更有意义。
    
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
    

## 初始化和重塑网络

现在到最有趣的部分。我们在这里处理每个网络的重塑。注意，这不是自动过程，并且对于每个型号都是唯一的。回想一下，CNN模型的最后一层(通常是FC层的倍数）具有与数据集中的输出类数相同的节点数。由于所有模型都已在Imagenet上进行了预训练，因此它们都具有大小为1000的输出层，每个类一个节点。这里的目标是重塑最后一层，使其具有与以前相同的输入数量，并且具有与数据集中的类数相同的输出数量。在以下各节中，我们将讨论如何分别更改每个模型的体系结构。但是首先，有一个关于微调和特征提取之间差异的重要细节。

特征提取时，我们只想更新最后一层的参数，换句话说，我们只想更新我们要重塑的层的参数。因此，我们不需要计算不变的参数的梯度，因此为了提高效率，我们将`.requires_grad`属性设置为`False`。这很重要，因为默认情况下，此属性设置为`True`。然后，当我们初始化新图层时，默认情况下，新参数的值为`.requires_grad = True`，因此仅新图层的参数将被更新。当我们进行微调时，我们可以将所有`.required_grad`的设置保留为默认值`True`。

最后，请注意 inception_v3 要求输入大小为(299,299)，而所有其他模型都期望为(224,224)。

### Resnet

Resnet在[用于图像识别的深度残差学习](https://arxiv.org/abs/1512.03385)中进行了介绍。 有几种不同大小的变体，包括Resnet18，Resnet34，Resnet50，Resnet101和Resnet152，所有这些都可以从Torchvision模型中获得。 这里我们使用Resnet18，因为我们的数据集很小，只有两个类。 当我们打印模型时，我们看到最后一层是完全连接的层，如下所示：
    
    (fc): Linear(in_features=512, out_features=1000, bias=True)
    

因此，我们必须将`model.fc`重新初始化为具有512个输入要素和2个输出要素的线性层，其具有：

    model.fc = nn.Linear(512, num_classes)
    

### Alexnet

Alexnet在[《使用深度卷积神经网络的ImageNet分类》](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)一书中进行了介绍，并且是ImageNet数据集上第一个非常成功的CNN。 当我们打印模型架构时，我们看到模型输出来自分类器的第六层
    
    (classifier): Sequential(
        ...
        (6): Linear(in_features=4096, out_features=1000, bias=True)
     )

为了将模型与我们的数据集一起使用，我们将该层重新初始化为
    
    model.classifier[6] = nn.Linear(4096,num_classes)

### VGG

VGG在[用于大型图像识别的甚深度卷积网络](https://arxiv.org/pdf/1409.1556.pdf)中被介绍。TorchVision提供了八种不同长度的VGG版本，有些具有批归一化层。在这里，我们将VGG-11与批处理归一化一起使用。 输出层类似于Alexnet，即
    
    (classifier): Sequential(
        ...
        (6): Linear(in_features=4096, out_features=1000, bias=True)
     )

因此，我们使用相同的技术来修改输出层
    
    model.classifier[6] = nn.Linear(4096,num_classes)
    

### Squeezenet

论文SqueezeNet中描述了Squeeznet体系结构：[AlexNet级别的精度，参数减少了50倍，模型尺寸小于0.5MB](https://arxiv.org/abs/1602.07360)，并且使用的输出结构与此处显示的任何其他模型都不相同。 TorchVision有两个版本的Squeezenet，我们使用1.0版。输出来自1x1卷积层，这是分类器的第一层：
    
    (classifier): Sequential(
        (0): Dropout(p=0.5)
        (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
        (2): ReLU(inplace)
        (3): AvgPool2d(kernel_size=13, stride=1, padding=0)
     )
    
为了修改网络，我们将Conv2d层重新初始化为深度为2的输出特征图为 
    
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    

### Densenet

Densenet在[《密集连接卷积网络》](https://arxiv.org/abs/1608.06993)一文中进行了介绍。 TorchVision有Densenet的四个变体，但这里我们仅使用Densenet-121。输出层是具有1024个输入要素的线性层：
    
    (classifier): Linear(in_features=1024, out_features=1000, bias=True)
    
为了重塑网络，我们将分类器的线性层重新初始化为
    
    model.classifier = nn.Linear(1024, num_classes)

### Inception V3

最后，在[重新思考计算机视觉的初始架构](https://arxiv.org/pdf/1512.00567v1.pdf)中首次描述了Inception v3。该网络是唯一的，因为在训练时它具有两个输出层。第二个输出称为辅助输出，包含在网络的AuxLogits部分中。 主要输出是网络末端的线性层。注意，在测试时，我们仅考虑主要输出。 加载模型的辅助输出和主要输出打印为：
    
    (AuxLogits): InceptionAux(
        ...
        (fc): Linear(in_features=768, out_features=1000, bias=True)
     )
     ...
    (fc): Linear(in_features=2048, out_features=1000, bias=True)

要微调此模型，我们必须重塑这两层的形状。这可以通过以下步骤完成
    
    model.AuxLogits.fc = nn.Linear(768, num_classes)
    model.fc = nn.Linear(2048, num_classes)
    
注意，许多模型具有相似的输出结构，但是每个模型的处理方式都必须略有不同。另外，请检查重塑网络的打印模型架构，并确保输出要素的数量与数据集中的类的数量相同。
    
    def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0
    
        if model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
    
        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224
    
        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224
    
        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = num_classes
            input_size = 224
    
        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224
    
        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs,num_classes)
            input_size = 299
    
        else:
            print("Invalid model name, exiting...")
            exit()
    
        return model_ft, input_size
    
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    
    # Print the model we just instantiated
    print(model_ft)

Out:

    SqueezeNet(
      (features): Sequential(
        (0): Conv2d(3, 96, kernel_size=(7, 7), stride=(2, 2))
        (1): ReLU(inplace=True)
        (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        (3): Fire(
          (squeeze): Conv2d(96, 16, kernel_size=(1, 1), stride=(1, 1))
          (squeeze_activation): ReLU(inplace=True)
          (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
          (expand1x1_activation): ReLU(inplace=True)
          (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (expand3x3_activation): ReLU(inplace=True)
        )
        (4): Fire(
          (squeeze): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
          (squeeze_activation): ReLU(inplace=True)
          (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
          (expand1x1_activation): ReLU(inplace=True)
          (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (expand3x3_activation): ReLU(inplace=True)
        )
        (5): Fire(
          (squeeze): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
          (squeeze_activation): ReLU(inplace=True)
          (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
          (expand1x1_activation): ReLU(inplace=True)
          (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (expand3x3_activation): ReLU(inplace=True)
        )
        (6): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        (7): Fire(
          (squeeze): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
          (squeeze_activation): ReLU(inplace=True)
          (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
          (expand1x1_activation): ReLU(inplace=True)
          (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (expand3x3_activation): ReLU(inplace=True)
        )
        (8): Fire(
          (squeeze): Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1))
          (squeeze_activation): ReLU(inplace=True)
          (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
          (expand1x1_activation): ReLU(inplace=True)
          (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (expand3x3_activation): ReLU(inplace=True)
        )
        (9): Fire(
          (squeeze): Conv2d(384, 48, kernel_size=(1, 1), stride=(1, 1))
          (squeeze_activation): ReLU(inplace=True)
          (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
          (expand1x1_activation): ReLU(inplace=True)
          (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (expand3x3_activation): ReLU(inplace=True)
        )
        (10): Fire(
          (squeeze): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1))
          (squeeze_activation): ReLU(inplace=True)
          (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
          (expand1x1_activation): ReLU(inplace=True)
          (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (expand3x3_activation): ReLU(inplace=True)
        )
        (11): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        (12): Fire(
          (squeeze): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
          (squeeze_activation): ReLU(inplace=True)
          (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
          (expand1x1_activation): ReLU(inplace=True)
          (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (expand3x3_activation): ReLU(inplace=True)
        )
      )
      (classifier): Sequential(
        (0): Dropout(p=0.5, inplace=False)
        (1): Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
        (2): ReLU(inplace=True)
        (3): AdaptiveAvgPool2d(output_size=(1, 1))
      )
    )
    

## 加载数据

既然我们知道输入大小必须为多少，就可以初始化数据转换，图像数据集和数据加载器。请注意，模型已经过硬编码规范化值的预训练，[如下所述](https://pytorch.org/docs/master/torchvision/models.html)。
    
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    print("Initializing Datasets and Dataloaders...")
    
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Out:
    
    Initializing Datasets and Dataloaders...

## 创建优化器

既然模型结构正确，那么微调和特征提取的最后一步就是创建一个仅更新所需参数的优化器。回想一下，在加载了预训练的模型之后，但是在重塑之前，如果`feature_extract = True`，我们将所有参数的`.requires_grad`属性手动设置为`False`。然后，默认情况下，重新初始化的图层的参数为.requires_grad = True。 因此，现在我们知道应该优化所有具有`.requires_grad = True`的参数。接下来，我们列出此类参数，并将此列表输入SGD算法构造函数。

要验证这一点，请查看打印的参数以进行学习。 进行微调时，此列表应该很长，并且包括所有模型参数。 但是，在提取特征时，此列表应简短，并且仅包括重塑图层的权重和偏差。
    
    # Send the model to GPU
    model_ft = model_ft.to(device)
    
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

Out:

    Params to learn:
             classifier.1.weight
             classifier.1.bias

## 运行训练和验证步骤

最后，最后一步是为模型设置损失，然后针对设定的时期数运行训练和验证功能。注意，根据时期数，此步骤在CPU上可能需要一段时间。同样，默认学习率并非对所有模型都最佳，因此要获得最大的准确性，有必要分别针对每个模型进行调整。
    
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    
    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
    
Out:

    Epoch 0/14
    ----------
    train Loss: 0.5200 Acc: 0.7336
    val Loss: 0.3895 Acc: 0.8366
    
    Epoch 1/14
    ----------
    train Loss: 0.3361 Acc: 0.8566
    val Loss: 0.3015 Acc: 0.8954
    
    Epoch 2/14
    ----------
    train Loss: 0.2721 Acc: 0.8770
    val Loss: 0.2938 Acc: 0.8954
    
    Epoch 3/14
    ----------
    train Loss: 0.2776 Acc: 0.8770
    val Loss: 0.2774 Acc: 0.9150
    
    Epoch 4/14
    ----------
    train Loss: 0.1881 Acc: 0.9139
    val Loss: 0.2715 Acc: 0.9150
    
    Epoch 5/14
    ----------
    train Loss: 0.1561 Acc: 0.9467
    val Loss: 0.3201 Acc: 0.9150
    
    Epoch 6/14
    ----------
    train Loss: 0.2536 Acc: 0.9016
    val Loss: 0.3474 Acc: 0.9150
    
    Epoch 7/14
    ----------
    train Loss: 0.1781 Acc: 0.9303
    val Loss: 0.3262 Acc: 0.9150
    
    Epoch 8/14
    ----------
    train Loss: 0.2321 Acc: 0.8811
    val Loss: 0.3197 Acc: 0.8889
    
    Epoch 9/14
    ----------
    train Loss: 0.1616 Acc: 0.9344
    val Loss: 0.3161 Acc: 0.9346
    
    Epoch 10/14
    ----------
    train Loss: 0.1510 Acc: 0.9262
    val Loss: 0.3199 Acc: 0.9216
    
    Epoch 11/14
    ----------
    train Loss: 0.1485 Acc: 0.9385
    val Loss: 0.3198 Acc: 0.9216
    
    Epoch 12/14
    ----------
    train Loss: 0.1098 Acc: 0.9590
    val Loss: 0.3331 Acc: 0.9281
    
    Epoch 13/14
    ----------
    train Loss: 0.1449 Acc: 0.9385
    val Loss: 0.3556 Acc: 0.9281
    
    Epoch 14/14
    ----------
    train Loss: 0.1405 Acc: 0.9303
    val Loss: 0.4227 Acc: 0.8758
    
    Training complete in 0m 20s
    Best val Acc: 0.934641

## 与从头开始训练的模型比较

只是为了好玩，让我们看看如果我们不使用转移学习，该模型将如何学习。 微调与特征提取的性能在很大程度上取决于数据集，但与从头开始训练的模型相比，总体而言，两种转移学习方法在训练时间和总体准确性方面均产生良好的结果。

    # Initialize the non-pretrained version of the model used for this run
    scratch_model,_ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
    scratch_model = scratch_model.to(device)
    scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
    scratch_criterion = nn.CrossEntropyLoss()
    _,scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))
    
    # Plot the training curves of validation accuracy vs. number
    #  of training epochs for the transfer learning method and
    #  the model trained from scratch
    ohist = []
    shist = []
    
    ohist = [h.cpu().numpy() for h in hist]
    shist = [h.cpu().numpy() for h in scratch_hist]
    
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
    plt.plot(range(1,num_epochs+1),shist,label="Scratch")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.show()

![https://pytorch.org/tutorials/_images/sphx_glr_finetuning_torchvision_models_tutorial_001.png](https://pytorch.org/tutorials/_images/sphx_glr_finetuning_torchvision_models_tutorial_001.png)

Out:

    Epoch 0/14
    ----------
    train Loss: 0.7032 Acc: 0.5205
    val Loss: 0.6931 Acc: 0.4641
    
    Epoch 1/14
    ----------
    train Loss: 0.6931 Acc: 0.5000
    val Loss: 0.6931 Acc: 0.4641
    
    Epoch 2/14
    ----------
    train Loss: 0.6931 Acc: 0.4549
    val Loss: 0.6931 Acc: 0.4641
    
    Epoch 3/14
    ----------
    train Loss: 0.6931 Acc: 0.5041
    val Loss: 0.6931 Acc: 0.4641
    
    Epoch 4/14
    ----------
    train Loss: 0.6931 Acc: 0.5041
    val Loss: 0.6931 Acc: 0.4641
    
    Epoch 5/14
    ----------
    train Loss: 0.6931 Acc: 0.5656
    val Loss: 0.6931 Acc: 0.4641
    
    Epoch 6/14
    ----------
    train Loss: 0.6931 Acc: 0.4467
    val Loss: 0.6931 Acc: 0.4641
    
    Epoch 7/14
    ----------
    train Loss: 0.6932 Acc: 0.5123
    val Loss: 0.6931 Acc: 0.4641
    
    Epoch 8/14
    ----------
    train Loss: 0.6931 Acc: 0.4918
    val Loss: 0.6931 Acc: 0.4641
    
    Epoch 9/14
    ----------
    train Loss: 0.6931 Acc: 0.4754
    val Loss: 0.6931 Acc: 0.4641
    
    Epoch 10/14
    ----------
    train Loss: 0.6931 Acc: 0.4795
    val Loss: 0.6931 Acc: 0.4641
    
    Epoch 11/14
    ----------
    train Loss: 0.6931 Acc: 0.5205
    val Loss: 0.6931 Acc: 0.4641
    
    Epoch 12/14
    ----------
    train Loss: 0.6931 Acc: 0.4754
    val Loss: 0.6931 Acc: 0.4641
    
    Epoch 13/14
    ----------
    train Loss: 0.6932 Acc: 0.4590
    val Loss: 0.6931 Acc: 0.4641
    
    Epoch 14/14
    ----------
    train Loss: 0.6932 Acc: 0.5082
    val Loss: 0.6931 Acc: 0.4641
    
    Training complete in 0m 29s
    Best val Acc: 0.464052
    

## 最后的思考和下一步是什么


尝试运行其他一些模型，看看精度如何。另外，请注意，特征提取花费的时间更少，因为在向后传递中，我们不必计算大多数梯度。 这里有很多地方。 你可以：

* 使用更困难的数据集运行此代码，并查看迁移学习的更多好处
* 使用此处描述的方法，使用转移学习来更新不同的模型，也许是在新的领域(例如NLP，音频等）
* 对模型满意后，可以将其导出为ONNX模型，也可以使用混合前端对其进行跟踪以提高速度和优化机会。

**脚本的总运行时间**：(0分钟57.562秒）
