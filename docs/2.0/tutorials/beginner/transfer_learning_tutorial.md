# 计算机视觉迁移学习教程 [¶](#transfer-learning-for-computer-vision-tutorial "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/transfer_learning_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>




**作者** 
 :
 [Sasank Chilamkurthy](https://chsasank.github.io)




 在本教程中，您将学习如何使用迁移学习训练
卷积神经网络以进行图像分类。您可以在 [cs231n 笔记](https://cs231n.github.io/transfer-learning/) 阅读有关迁移学习的更多信息




 引用这些注释，




> 
> 
> 
> 在实践中，很少有人从头开始训练整个卷积网络（随机初始化），因为拥有足够大小的数据集相对较少。相反，通常在非常大的数据集（例如 ImageNet，其中包含 120 万张图像，1000 个类别）上预训练 ConvNet，然后使用 ConvNet 作为初始化或固定特征感兴趣的任务的提取器。
> 
> 
> 
> 
>



 这两个主要的迁移学习场景如下所示：



* **微调 ConvNet** 
 ：我们使用预训练的网络（例如在 imagenet 1000 数据集上训练的网络）来初始化网络，而不是随机初始化。训练的其余部分看起来与平常一样。
* **ConvNet 作为固定特征提取器** 
：在这里，我们将冻结除最终完全连接层之外的所有网络的权重
。最后一个完全连接的层被替换为具有随机权重的新层，并且仅训练该层。





```
# License: BSD
# Author: Sasank Chilamkurthy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()   # interactive mode

```






```
<contextlib.ExitStack object at 0x7f12e846ffd0>

```





## 加载数据 [¶](#load-data "此标题的永久链接")




 我们将使用 torchvision 和 torch.utils.data 包来加载
数据。




 我们今天’要解决的问题是训练一个模型来分类
 **蚂蚁** 
 和
 **蜜蜂** 
 。我们有大约 120 个蚂蚁和蜜蜂的训练图像。
每个类别有 75 个验证图像。通常，如果从头开始训练，这是一个非常小的数据集，可以进行泛化。由于我们
正在使用迁移学习，因此我们应该能够
很好地进行合理的概括。




 该数据集是 imagenet 的一个非常小的子集。





 注意




 从
 [此处](https://download.pytorch.org/tutorial/hymenoptera_data.zip) 下载数据并将其解压到当前目录。







```
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose(
        [transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose(
        [transforms.Resize(256),
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

```




### 可视化一些图像 [¶](#visualize-a-few-images "此标题的固定链接")



 让’s 可视化一些训练图像，以便理解数据
augmentations。






```
def imshow(inp, title=None):
 """Display image for Tensor."""
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

```



![['蚂蚁', '蚂蚁', '蚂蚁', '蚂蚁']](https://pytorch.org/tutorials/_images/sphx_glr_transfer_learning_tutorial_001.png)


## 训练模型 [¶](#training-the-model "永久链接到此标题")




 现在，让’s 编写一个通用函数来训练模型。在这里，我们将
说明：



* 调度学习率
* 保存最佳模型



 在下面的内容中，参数
 `scheduler`
 是来自
 `torch.optim.lr_scheduler`
 的 LR 调度程序对象。






```
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
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

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

```




### 可视化模型预测 [¶](#visualizing-the-model-predictions "永久链接到此标题")



 显示一些图像的预测的通用函数






```
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
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

```





## 微调 ConvNet [¶](#finetuning-the-convnet "固定链接到此标题")




 加载预训练模型并重置最终的全连接层。






```
model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

```






```
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /var/lib/jenkins/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth

  0%|          | 0.00/44.7M [00:00<?, ?B/s]
 25%|##4       | 11.1M/44.7M [00:00<00:00, 116MB/s]
 51%|#####     | 22.7M/44.7M [00:00<00:00, 119MB/s]
 76%|#######6  | 34.1M/44.7M [00:00<00:00, 119MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 119MB/s]

```




### 训练和评估 [¶](#train-and-evaluate "永久链接到此标题")



 CPU 大约需要 15-25 分钟。但在 GPU 上，
所需时间不到一分钟。






```
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

```






```
Epoch 0/24
----------
train Loss: 0.4761 Acc: 0.7623
val Loss: 0.2950 Acc: 0.8824

Epoch 1/24
----------
train Loss: 0.5296 Acc: 0.8074
val Loss: 0.6461 Acc: 0.7190

Epoch 2/24
----------
train Loss: 0.4149 Acc: 0.8279
val Loss: 0.3067 Acc: 0.9085

Epoch 3/24
----------
train Loss: 0.6455 Acc: 0.7582
val Loss: 0.3774 Acc: 0.8627

Epoch 4/24
----------
train Loss: 0.3883 Acc: 0.8566
val Loss: 0.2781 Acc: 0.9085

Epoch 5/24
----------
train Loss: 0.5150 Acc: 0.8033
val Loss: 0.2616 Acc: 0.8954

Epoch 6/24
----------
train Loss: 0.3923 Acc: 0.8279
val Loss: 0.3704 Acc: 0.8562

Epoch 7/24
----------
train Loss: 0.4605 Acc: 0.7828
val Loss: 0.2531 Acc: 0.9020

Epoch 8/24
----------
train Loss: 0.2475 Acc: 0.8934
val Loss: 0.2207 Acc: 0.9281

Epoch 9/24
----------
train Loss: 0.2729 Acc: 0.8689
val Loss: 0.1997 Acc: 0.9477

Epoch 10/24
----------
train Loss: 0.3495 Acc: 0.8320
val Loss: 0.1912 Acc: 0.9346

Epoch 11/24
----------
train Loss: 0.3550 Acc: 0.8607
val Loss: 0.2474 Acc: 0.9085

Epoch 12/24
----------
train Loss: 0.2499 Acc: 0.8975
val Loss: 0.2054 Acc: 0.9412

Epoch 13/24
----------
train Loss: 0.3085 Acc: 0.8730
val Loss: 0.1642 Acc: 0.9477

Epoch 14/24
----------
train Loss: 0.2621 Acc: 0.8934
val Loss: 0.2123 Acc: 0.9346

Epoch 15/24
----------
train Loss: 0.3239 Acc: 0.8566
val Loss: 0.2779 Acc: 0.9085

Epoch 16/24
----------
train Loss: 0.2089 Acc: 0.9221
val Loss: 0.1968 Acc: 0.9346

Epoch 17/24
----------
train Loss: 0.2630 Acc: 0.8893
val Loss: 0.1759 Acc: 0.9412

Epoch 18/24
----------
train Loss: 0.2853 Acc: 0.8730
val Loss: 0.1987 Acc: 0.9281

Epoch 19/24
----------
train Loss: 0.2278 Acc: 0.8893
val Loss: 0.1697 Acc: 0.9542

Epoch 20/24
----------
train Loss: 0.2662 Acc: 0.8893
val Loss: 0.1770 Acc: 0.9412

Epoch 21/24
----------
train Loss: 0.2415 Acc: 0.9016
val Loss: 0.2304 Acc: 0.9216

Epoch 22/24
----------
train Loss: 0.3249 Acc: 0.8730
val Loss: 0.1769 Acc: 0.9477

Epoch 23/24
----------
train Loss: 0.2940 Acc: 0.8648
val Loss: 0.1843 Acc: 0.9477

Epoch 24/24
----------
train Loss: 0.3004 Acc: 0.8770
val Loss: 0.1794 Acc: 0.9346

Training complete in 1m 5s
Best val Acc: 0.954248

```






```
visualize_model(model_ft)

```



![预测：蚂蚁，预测：蜜蜂，预测：蚂蚁，预测：蜜蜂，预测：蜜蜂，预测：蚂蚁](https://pytorch.org/tutorials/_images/sphx_glr_transfer_learning_tutorial_002.png)


## ConvNet 作为固定特征提取器 [¶](#convnet-as-fixed-feature-extractor "永久链接到此标题")




 在这里，我们需要冻结除最后一层之外的所有网络。我们需要
设置
 `requires_grad
 

 =
 

 False`
来冻结参数，以便
梯度不会在
 `backward()`中计算
 n.




 您可以在文档中阅读更多相关信息
 [此处](https://pytorch.org/docs/notes/autograd.html#exclusion-subgraphs-from-backward) 
.






```
model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
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

```




### 训练和评估 [¶](#id1 "此标题的永久链接")



 在 CPU 上，与之前的场景相比，这将花费大约一半的时间。
这是预期的，因为
需要为大多数网络计算梯度 don’t。但是，确实需要计算前向。






```
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

```






```
Epoch 0/24
----------
train Loss: 0.6996 Acc: 0.6516
val Loss: 0.2014 Acc: 0.9346

Epoch 1/24
----------
train Loss: 0.4233 Acc: 0.8033
val Loss: 0.2656 Acc: 0.8758

Epoch 2/24
----------
train Loss: 0.4603 Acc: 0.7869
val Loss: 0.1847 Acc: 0.9477

Epoch 3/24
----------
train Loss: 0.3096 Acc: 0.8566
val Loss: 0.1747 Acc: 0.9477

Epoch 4/24
----------
train Loss: 0.4427 Acc: 0.8156
val Loss: 0.1630 Acc: 0.9477

Epoch 5/24
----------
train Loss: 0.5505 Acc: 0.7828
val Loss: 0.1643 Acc: 0.9477

Epoch 6/24
----------
train Loss: 0.3004 Acc: 0.8607
val Loss: 0.1744 Acc: 0.9542

Epoch 7/24
----------
train Loss: 0.4083 Acc: 0.8361
val Loss: 0.1892 Acc: 0.9412

Epoch 8/24
----------
train Loss: 0.4483 Acc: 0.7910
val Loss: 0.1984 Acc: 0.9477

Epoch 9/24
----------
train Loss: 0.3335 Acc: 0.8279
val Loss: 0.1942 Acc: 0.9412

Epoch 10/24
----------
train Loss: 0.2413 Acc: 0.8934
val Loss: 0.2001 Acc: 0.9477

Epoch 11/24
----------
train Loss: 0.3107 Acc: 0.8689
val Loss: 0.1801 Acc: 0.9412

Epoch 12/24
----------
train Loss: 0.3032 Acc: 0.8689
val Loss: 0.1669 Acc: 0.9477

Epoch 13/24
----------
train Loss: 0.3587 Acc: 0.8525
val Loss: 0.1900 Acc: 0.9477

Epoch 14/24
----------
train Loss: 0.2771 Acc: 0.8893
val Loss: 0.2317 Acc: 0.9216

Epoch 15/24
----------
train Loss: 0.3064 Acc: 0.8852
val Loss: 0.1909 Acc: 0.9477

Epoch 16/24
----------
train Loss: 0.4243 Acc: 0.8238
val Loss: 0.2227 Acc: 0.9346

Epoch 17/24
----------
train Loss: 0.3297 Acc: 0.8238
val Loss: 0.1916 Acc: 0.9412

Epoch 18/24
----------
train Loss: 0.4235 Acc: 0.8238
val Loss: 0.1766 Acc: 0.9477

Epoch 19/24
----------
train Loss: 0.2500 Acc: 0.8934
val Loss: 0.2003 Acc: 0.9477

Epoch 20/24
----------
train Loss: 0.2413 Acc: 0.8934
val Loss: 0.1821 Acc: 0.9477

Epoch 21/24
----------
train Loss: 0.3762 Acc: 0.8115
val Loss: 0.1842 Acc: 0.9412

Epoch 22/24
----------
train Loss: 0.3485 Acc: 0.8566
val Loss: 0.2166 Acc: 0.9281

Epoch 23/24
----------
train Loss: 0.3625 Acc: 0.8361
val Loss: 0.1747 Acc: 0.9412

Epoch 24/24
----------
train Loss: 0.3840 Acc: 0.8320
val Loss: 0.1768 Acc: 0.9412

Training complete in 0m 33s
Best val Acc: 0.954248

```






```
visualize_model(model_conv)

plt.ioff()
plt.show()

```



![预测：蜜蜂，预测：蚂蚁，预测：蜜蜂，预测：蜜蜂，预测：蚂蚁，预测：蚂蚁](https://pytorch.org/tutorials/_images/sphx_glr_transfer_learning_tutorial_003.png)


## 自定义图像推断 [¶](#inference-on-custom-images "此标题的固定链接")




 使用经过训练的模型对自定义图像进行预测，并可视化
预测的类标签和图像。






```
def visualize_model_predictions(model,img_path):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = data_transforms'val'
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        imshow(img.cpu().data[0])

        model.train(mode=was_training)

```






```
visualize_model_predictions(
    model_conv,
    img_path='data/hymenoptera_data/val/bees/72100438_73de9f17af.jpg'
)

plt.ioff()
plt.show()

```



![预测：蜜蜂](https://pytorch.org/tutorials/_images/sphx_glr_transfer_learning_tutorial_004.png)


## 进一步学习 [¶](#further-learning "永久链接到此标题")




 如果您想了解有关迁移学习应用的更多信息，
请查看我们的
 [计算机视觉量化迁移学习教程](https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html) 
 。 




**脚本总运行时间:** 
 (1 分 40.858 秒)
