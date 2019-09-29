# 微调Torchvision模型

**作者：** [弥敦道Inkawhich ](https://github.com/inkawhich)

在本教程中，我们将采取在如何微调和特征提取[
torchvision模型](https://pytorch.org/docs/stable/torchvision/models.html)，所有这些都被预先训练的1000级Imagenet数据集更深入的了解。本教程将给予在如何与一些现代CNN架构工作的深入看，将建立一个直觉微调任何PyTorch模型。由于每个模型架构是不同的，没有样板代码细化和微调，将在所有情况下工作。相反，研究人员必须着眼于现有的架构，并为每个模型自定义调整。

在本文中，我们将执行两种类型的迁移学习的：和细化和微调特征提取。在 **微调** ，我们先从预训练模式和更新 _所有为我们的新任务模型的参数_
，在本质上再培训整个模型。在 **特征提取**
，我们先从一个预训练的模型和仅更新从中我们推导预测最终的层的权重。这就是所谓的特征提取，因为我们使用预训练的CNN作为一个固定的功能提取，只改变输出层。有关迁移学习更多的技术信息，请参阅[此处](https://cs231n.github.io/transfer-
learning/)和[此处[HTG9。](https://ruder.io/transfer-learning/)

一般这两种传输的学习方法遵循相同的几个步骤：

  * 初始化预训练模式
  * 重塑最后的层（一个或多个），以具有相同的数量的输出作为类在新的数据集的数目
  * 定义哪些参数，我们要在训练期间更新优化算法
  * 运行训练步骤

    
    
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
    

日期：

    
    
    PyTorch Version:  1.2.0
    Torchvision Version:  0.4.0
    

## 输入

这里是所有的参数，为运行而改变。我们将使用 _hymenoptera_data_ 数据集可以下载[此处[HTG3。此数据集包含两类， **蜜蜂HTG5]和
**蚂蚁** ，其结构是这样，我们可以使用[ ImageFolder
](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder)数据集，而不是写自己的自定义数据集。下载数据和`
DATA_DIR`输入设置为数据集的根目录下。在`MODEL_NAME
`输入你想使用，必须从这个列表中选择模型的名称：**](https://download.pytorch.org/tutorial/hymenoptera_data.zip)

    
    
    [resnet, alexnet, vgg, squeezenet, densenet, inception]
    

其它输入如下：`num_classes`是类数据集中的数目，`的batch_size`是用于训练的批量大小和可根据您的机器的性能进行调整，`
num_epochs`是我们要运行训练时期的编号，`feature_extract`是一个布尔值，定义，如果我们微调或特征提取。如果`
feature_extract  =  假 `，模型被微调，并且所有模型参数被更新。如果`feature_extract  =  真
`，只有最后层参数被更新，其它的保持固定。

    
    
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

之前我们写的代码，用于调整模型，让定义一些辅助功能。

### 模型训练和验证码

在`train_model
`函数处理给定模型的训练和验证。作为输入，它需要一个PyTorch模型，dataloaders的词典，损失函数，优化器，一个指定数目的历元的训练和验证，和一个布尔标志，用于当模型是一个启模型。的
_is_inception_ 标志用于以容纳 _启V3_
模型，因为该架构使用的辅助输出和整体模型损耗方面都辅助输出和最终的输出，如所描述的[此处[HTG9。该功能用于训练历元的指定数目和每个时期后运行一个完整的验证步骤。它也跟踪性能最佳的模型（在验证准确性方面），并在训练结束返回表现最好的模型。每个历元之后，训练和验证的精度进行打印。](https://discuss.pytorch.org/t/how-
to-optimize-inception-model-with-auxiliary-classifiers/7958)

    
    
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
    

### 设置模型参数.requires_grad属性

这个辅助功能设置`.requires_grad`的参数模型中的属性设置为false，当我们特征提取。默认情况下，当我们加载预训练模型的所有参数都`
.requires_grad =真
`，这是很好的，如果我们从头开始或培训细化和微调。然而，如果我们特征提取，只要计算新初始化层梯度那么我们希望所有的其他参数不要求梯度。这将在后面更有意义。

    
    
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
    

## 初始化和重塑网络

现在到了最有趣的部分。这里我们处理每一个网络的重塑。请注意，这不是一个自动的过程，是唯一的每个模型。回想一下，CNN的模式，这是经常倍FC层的最终层，具有相同的数作为输出类别的数据集中的节点数量。由于所有的模型都被预先训练上Imagenet，它们都具有尺寸1000，为每个类一个节点的输出层。这样做的目的是为了重塑最后一层像以前一样有相同数量的输入，并拥有相同数量的输出作为类中的数据集数。在下面的章节中，我们将讨论如何单独改变每个模型的架构。但首先，有关于和细化和微调特征提取的区别一个重要的细节。

当特征提取，我们只希望更新的最后一层的参数，或者换句话说，我们只需要更新我们正在重塑层（一个或多个）的参数。因此，我们并不需要计算的参数，我们不改变梯度，所以效率，我们的.requires_grad属性设置为False。因为默认情况下，该属性设置为True，这是非常重要的。然后，当我们初始化新的层，并默认新参数有`
.requires_grad =真 `所以才有了新层的参数将被更新。当我们微调我们可以把所有的.required_grad的一套以真默认值。

最后，注意inception_v3需要输入的内容是（299299），而所有其他型号的预期（224224）。

### RESNET

RESNET在图像识别纸张[深残余学习引入。有不同的尺寸，包括Resnet18，Resnet34，Resnet50，Resnet101和Resnet152，所有这些都可以从torchvision模型的几个变种。这里我们使用Resnet18，因为我们的数据集很小，只有两个班。当我们打印模式，我们看到，最后一层是完全连接层，如下图所示：](https://arxiv.org/abs/1512.03385)

    
    
    (fc): Linear(in_features=512, out_features=1000, bias=True)
    

因此，我们必须重新初始化`model.fc`是一个线性层512点输入的特征和2层输出的功能：

    
    
    model.fc = nn.Linear(512, num_classes)
    

### Alexnet

Alexnet是与深卷积神经网络纸[
ImageNet分类介绍，是第一个非常成功的CNN在ImageNet数据集。当我们打印模型架构，我们可以看到模型输出来自分类的第6层](https://papers.nips.cc/paper/4824-imagenet-
classification-with-deep-convolutional-neural-networks.pdf)

    
    
    (classifier): Sequential(
        ...
        (6): Linear(in_features=4096, out_features=1000, bias=True)
     )
    

要使用我们的数据使用的模型中，我们初始化该层

    
    
    model.classifier[6] = nn.Linear(4096,num_classes)
    

### VGG

VGG在文献[非常深卷积网络推出的大型图像识别[HTG1。
Torchvision提供各种长度和一些有一批归一化层的8个版本VGG的。这里我们使用VGG-11批标准化。输出层类似于Alexnet，即](https://arxiv.org/pdf/1409.1556.pdf)

    
    
    (classifier): Sequential(
        ...
        (6): Linear(in_features=4096, out_features=1000, bias=True)
     )
    

因此，我们使用相同的技术来修改输出层

    
    
    model.classifier[6] = nn.Linear(4096,num_classes)
    

### Squeezenet

所述Squeeznet体系结构在论文中描述[ SqueezeNet：用50个更少的参数和AlexNet级精度& LT ;
0.5MB模型大小](https://arxiv.org/abs/1602.07360)，并使用不同的输出结构比任何其他模型的这里显示。
Torchvision有Squeezenet的两个版本中，我们使用1.0版本。输出来自这是分类器的第一层是1x1卷积层：

    
    
    (classifier): Sequential(
        (0): Dropout(p=0.5)
        (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
        (2): ReLU(inplace)
        (3): AvgPool2d(kernel_size=13, stride=1, padding=0)
     )
    

修改网络，我们重新初始化Conv2d层具有深度为2的输出特性图作为

    
    
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    

### Densenet

Densenet是在论文[密集连接的卷积网络](https://arxiv.org/abs/1608.06993)引入。
Torchvision有Densenet的四个变种，但在这里我们只使用Densenet-121。输出层是用1024个输入特征的线性层：

    
    
    (classifier): Linear(in_features=1024, out_features=1000, bias=True)
    

重塑网络，我们重新初始化分类的线性层

    
    
    model.classifier = nn.Linear(1024, num_classes)
    

### 盗梦空间V3

最后，启V3在[反思盗梦空间架构计算机视觉HTG1]首次描述。该网络是独一无二的，因为它有训练的时候两个输出层。第二输出被称为辅助输出，并且包含在该网络的AuxLogits一部分。初级输出是在网络的端部的线性层。请注意，测试我们只考虑主输出时。辅助输出和所加载的模型的主要输出被打印为：](https://arxiv.org/pdf/1512.00567v1.pdf)

    
    
    (AuxLogits): InceptionAux(
        ...
        (fc): Linear(in_features=768, out_features=1000, bias=True)
     )
     ...
    (fc): Linear(in_features=2048, out_features=1000, bias=True)
    

微调该模块，我们必须重塑两层。这是通过下列步骤完成

    
    
    model.AuxLogits.fc = nn.Linear(768, num_classes)
    model.fc = nn.Linear(2048, num_classes)
    

请注意，许多车型也有类似的输出结构，但每次都必须稍有不同的方式处理。此外，检查出重新成形网络的打印模型体系结构，并确保的输出特征的数量是相同的类中的数据集的数目。

    
    
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
    

## 负载数据

现在我们知道输入的内容必须是什么，我们可以初始化数据变换，图像数据集，以及dataloaders。通知时，模型与硬编码正常化值预训练的，如所描述的[这里](https://pytorch.org/docs/master/torchvision/models.html)。

    
    
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
    

## 创建优化

现在，该模型的结构是正确的，对于微调和特征提取的最后一步是创建一个优化，仅更新所需的参数。回想一下，在加载预训练模式后，却重塑之前，如果`
feature_extract =真 `我们手动设置所有参数的`.requires_grad`属性为False。然后，重新初始化层的参数有`
.requires_grad =真 `缺省。所以，现在我们知道， _已.requires_grad = true的参数应优化。
[HTG13接下来，我们做这个名单的SGD算法构造这样的参数和输入的列表。_

为了验证这一点，请查看打印参数学习。微调时，该名单应该是长期的，包括所有的模型参数。然而，当特征提取这个名单应该简短，并且仅包括重塑层的权重和偏见。

    
    
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
    

## 跑训练和验证步骤

最后，最后一步就是设置为模型的损失，然后运行时期的设定次数，培训和验证功能。通知，取决于历元的数目这个步骤可能需要在CPU上一会儿。此外，默认的学习速度是不是最佳的所有车型，所以要实现有必要调整每个型号分别最高的精度。

    
    
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
    

## 与模型对比从头训练有素

只是为了好玩，让我们看看模型如何学习，如果我们不使用迁移学习。微调与特征提取的性能很大程度上取决于数据集，但一般都转移学习方法产生的训练时间和整体精度与从头开始训练的模型方面是有利的结果。

    
    
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
    

![img/sphx_glr_finetuning_torchvision_models_tutorial_001.png](img/sphx_glr_finetuning_torchvision_models_tutorial_001.png)

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

尝试运行一些其他的车型，看看准确度有多好得。此外，请注意特征提取花费较少的时间，因为在落后的过程中，我们没有计算大部分的梯度。有很多地方从这里走。你可以：

  * 运行该代码与较硬的数据集，看迁移学习一些更多的好处
  * 使用这里描述的方法，使用传输学习来更新不同的模式，也许在一个新的领域（即NLP，音频等）
  * 一旦你满意的模型，您可以将其导出为ONNX模型或使用混合前端更快的速度和优化的机会进行跟踪。

**脚本的总运行时间：** （0分钟56.849秒）

[`Download Python source code:
finetuning_torchvision_models_tutorial.py`](../_downloads/64a61387602867f347b7ee35d3215713/finetuning_torchvision_models_tutorial.py)

[`Download Jupyter notebook:
finetuning_torchvision_models_tutorial.ipynb`](../_downloads/df1f5ef1c1a8e1a111e88281b27829fe/finetuning_torchvision_models_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](../intermediate/spatial_transformer_tutorial.html "Spatial
Transformer Networks Tutorial") [![](../_static/images/chevron-right-
orange.svg) Previous](../intermediate/torchvision_tutorial.html "TorchVision
Object Detection Finetuning Tutorial")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。



  * 微调Torchvision模型
    * 输入
    * 辅助函数
      * 型号培训和验证码
      * 设置模型参数.requires_grad属性
    * 初始化和重塑网络
      * RESNET 
      * Alexnet 
      * VGG 
      * Squeezenet 
      * Densenet 
      * 启V3 
    * 负载数据
    * 创建优化
    * 运行训练和验证步骤
    * 与从头经过培训的模型对比
    * 最后的思考和下一步是什么

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)







 
[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)



