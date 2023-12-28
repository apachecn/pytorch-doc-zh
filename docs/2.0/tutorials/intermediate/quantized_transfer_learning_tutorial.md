


# (测试版)计算机视觉量化迁移学习教程 [¶](#beta-quantized-transfer-learning-for-computer-vision-tutorial "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/quantized_transfer_learning_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html>





 提示




 为了充分利用本教程，我们建议使用此
 [Colab 版本](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/quantized_transfer_learning_tutorial.ipynb ) 
 。
这将允许您尝试下面提供的信息。





**作者** 
 :
 [Zafar Takhirov](https://github.com/z-a-f)




**审阅者** 
 :
 [Raghuraman Krishnamoorthi](https://github.com/raghuramank100)




**编辑者** 
 :
 [Jessica Lin](https://github.com/jlin27)




 本教程基于原始的
 [PyTorch 迁移学习](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) 
 教程，由 [Sasank Chilamkurthy](https://chsasank.github.io/) 
.




 迁移学习是指利用预训练模型
在不同数据集上应用的技术。
迁移学习有两种主要使用方式：



1. **ConvNet 作为固定特征提取器** 
 ：在这里，您
 [“freeze”](https://arxiv.org/abs/1706.04983) 
网络中除最后
几个层(又名“头”，通常是全连接层)之外的所有参数的权重。
这些最后的层被替换为随机初始化的新层
权重并且仅训练这些层。
2. **微调 ConvNet** 
 ：模型不是随机初始化，而是使用预训练网络
进行初始化，然后训练
像往常一样进行，但使用不同的数据集。
通常头部(或其一部分)是
如果输出数量不同，也会在网络中进行替换。
在此方法中，通常将学习率设置为较小的数字。
这样做是因为网络已经经过训练，只需进行微小的更改
需要 “finetune” 将其转换为新数据集。



 您还可以结合上述两种方法：
首先您可以冻结特征提取器，并训练头部。之后，您可以解冻特征提取器(或其一部分)，将学习率设置为较小的值，然后继续训练。




 在此部分中，您将使用第一种方法 – 使用量化模型
提取特征。





## 第 0 部分。先决条件 [¶](#part-0-preventions "永久链接到此标题")




 在深入研究迁移学习之前，让我们回顾一下 “ 先决条件”，
例如安装和数据加载/可视化。






```
# Imports
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import time

plt.ion()

```




### 安装 Nightly Build [¶](#installing-the-nightly-build“永久链接到此标题”)



 因为您将使用 PyTorch 的 Beta 部分，
建议安装最新版本的
 `torch`
 和
 `torchvision`
 。您可以在[此处](https://pytorch.org/get-started/locally/)找到有关本地
安装的最新说明
。
例如，要在没有 GPU 支持的情况下安装：






```
pip install numpy
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
# For CUDA support use https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html

```





### 加载数据 [¶](#load-data "永久链接到此标题")




 注意




 本节与原始迁移学习教程相同。





 我们将使用
 `torchvision`
 和
 `torch.utils.data`
 包来加载
数据。




 您今天要解决的问题是从图像中分类
 **蚂蚁** 
 和
 **蜜蜂** 
。该数据集包含大约 120 个蚂蚁和蜜蜂的训练图像。每个类别有 75 个验证图像。
这被认为是一个非常小的数据集来进行概括。然而，由于
我们正在使用迁移学习，
我们应该能够
很好地进行泛化。




*该数据集是 imagenet 的一个非常小的子集。*





 没有10



 从
 [此处](https://download.pytorch.org/tutorial/hymenoptera_data.zip) 下载数据并将其解压到
 `data`
 目录。







```
import torch
from torchvision import transforms, datasets

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                              shuffle=True, num_workers=8)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

```





### 可视化一些图像 [¶](#visualize-a-few-images "此标题的固定链接")



 让’s 可视化一些训练图像，以便理解数据
augmentations。






```
import torchvision

def imshow(inp, title=None, ax=None, figsize=(5, 5)):
 """Imshow for Tensor."""
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)
  if ax is None:
    fig, ax = plt.subplots(1, figsize=figsize)
  ax.imshow(inp)
  ax.set_xticks([])
  ax.set_yticks([])
  if title is not None:
    ax.set_title(title)

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs, nrow=4)

fig, ax = plt.subplots(1, figsize=(10, 10))
imshow(out, title=[class_names[x] for x in classes], ax=ax)

```





### 模型训练支持函数 [¶](#support-function-for-model-training "永久链接到此标题")



 下面是模型训练的通用函数。
此函数也是



* 安排学习率
* 保存最佳模型





```
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
 """
 Support function for model training.

 Args:
 model: Model to be trained
 criterion: Optimization criterion (loss)
 optimizer: Optimizer to use for training
 scheduler: Instance of ``torch.optim.lr_scheduler``
 num_epochs: Number of epochs
 device: Device to run the training on. Must be 'cpu' or 'cuda'
 """
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

```





### 用于可视化模型预测的支持函数 [¶](#support-function-for-visualizing-the-model-predictions "永久链接到此标题")



 显示一些图像的预测的通用函数






```
def visualize_model(model, rows=3, cols=3):
  was_training = model.training
  model.eval()
  current_row = current_col = 0
  fig, ax = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

  with torch.no_grad():
    for idx, (imgs, lbls) in enumerate(dataloaders['val']):
      imgs = imgs.cpu()
      lbls = lbls.cpu()

      outputs = model(imgs)
      _, preds = torch.max(outputs, 1)

      for jdx in range(imgs.size()[0]):
        imshow(imgs.data[jdx], ax=ax[current_row, current_col])
        ax[current_row, current_col].axis('off')
        ax[current_row, current_col].set_title('predicted: {}'.format(class_names[preds[jdx]]))

        current_col += 1
        if current_col >= cols:
          current_row += 1
          current_col = 0
        if current_row >= rows:
          model.train(mode=was_training)
          return
    model.train(mode=was_training)

```





## 第 1 部分：基于量化特征提取器训练自定义分类器 [¶](#part-1-training-a-custom-classifier-based-on-a-quantized-feature-extractor "此标题的永久链接")




 在本节中，您将使用 “frozen” 量化特征提取器，并
在其顶部训练自定义分类器头。与浮点模型不同，您不需要为量化模型设置requires_grad=False，因为它没有可训练的参数。请参阅
[文档](https://pytorch.org/docs/stable/quantization.html)
了解更多详细信息。




 加载预训练模型：在本练习中，您将使用
 [ResNet-18](https://pytorch.org/hub/pytorch_vision_resnet/) 
.






```
import torchvision.models.quantization as models

# You will need the number of filters in the `fc` for future use.
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_fe = models.resnet18(pretrained=True, progress=True, quantize=True)
num_ftrs = model_fe.fc.in_features

```




 此时需要修改预训练的模型。该模型
在开头和结尾都有量化/反量化块。然而，
因为您只使用特征提取器，所以反量化层
必须移动到线性层(头部)之前。最简单的方法
是将模型包装在
 `nn.Sequential`
 模块中。




 第一步是隔离 ResNet
模型中的特征提取器。尽管在此示例中，您的任务是使用除
 `fc`
 之外的所有层作为特征提取器，但实际上，您可以根据需要使用任意多个部分。如果您还想替换
某些卷积层，这将很有用。





 注意




 将特征提取器与量化模型的其余部分
分离时，您必须手动将量化器/反量化
放置在要保持量化的部分的开头和结尾。





 下面的函数创建一个具有自定义头部的模型。






```
from torch import nn

def create_combined_model(model_fe):
  # Step 1. Isolate the feature extractor.
  model_fe_features = nn.Sequential(
    model_fe.quant,  # Quantize the input
    model_fe.conv1,
    model_fe.bn1,
    model_fe.relu,
    model_fe.maxpool,
    model_fe.layer1,
    model_fe.layer2,
    model_fe.layer3,
    model_fe.layer4,
    model_fe.avgpool,
    model_fe.dequant,  # Dequantize the output
  )

  # Step 2. Create a new "head"
  new_head = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 2),
  )

  # Step 3. Combine, and don't forget the quant stubs.
  new_model = nn.Sequential(
    model_fe_features,
    nn.Flatten(1),
    new_head,
  )
  return new_model

```





!!! warning "警告"

    目前量化模型只能在 CPU 上运行。
但是，可以将模型的非量化部分发送到 GPU。







```
import torch.optim as optim
new_model = create_combined_model(model_fe)
new_model = new_model.to('cpu')

criterion = nn.CrossEntropyLoss()

# Note that we are only training the head.
optimizer_ft = optim.SGD(new_model.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

```




### 训练和评估 [¶](#train-and-evaluate "永久链接到此标题")



 此步骤在 CPU 上大约需要 15-25 分钟。由于量化模型只能在 CPU 上运行，因此无法在 GPU 上运行训练。






```
new_model = train_model(new_model, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=25, device='cpu')

visualize_model(new_model)
plt.tight_layout()

```





## 第 2 部分. 微调可量化模型 [¶](#part-2-finetuning-the-quantible-model "永久链接到此标题")



在这一部分中，我们微调用于迁移学习的特征提取器，并对特征提取器进行量化。请注意，在第 1 部分和第 2 部分中，特征提取器都是量化的。不同之处在于，在第 1 部分中，我们使用预训练的量化模型。在这一部分中，我们在对感兴趣的数据集进行微调后创建了一个量化特征提取器，因此这是一种通过迁移学习获得更高准确性同时具有量化优势的方法。请注意，在我们的具体示例中，训练集非常小(120 个图像)，因此微调整个模型的好处并不明显。不过，
此处显示的过程将提高
较大数据集迁移学习的准确性。




 预训练的特征提取器必须可量化。
要确保其可量化，请执行以下步骤:




> 
> 
> 1. 融合
> `(Conv,
> 
> 
> BN,
> 
> 
> ReLU)`
> ,
> ` (Conv,
> 
> 
> BN)`
> 和
> `(Conv,
> 
> 
> ReLU)`
> 使用
> `torch.quantization.fuse_modules`
>.
> 2. 将特征提取器连接到自定义头。
> 这需要对特征提取器的输出进行反量化。
> 3. 在适当的位置插入伪量化模块
> 在特征提取器中模拟训练过程中的量化。
> 
> 
> 
>



 对于步骤 (1)，我们使用来自
 `torchvision/models/quantization`
 的模型，其中
有一个成员方法
 
 `fuse_model`
 。此函数融合了所有
 `conv`
 、
 `bn`
 和
 `relu`
 模块。对于自定义模型，这需要调用
`torch.quantization.fuse_modules`
 API 以及要手动融合的模块列表。




 步骤 (2) 由上一节中使用的
 `create_combined_model`
 函数执行。




 步骤 (3) 是通过使用
 `torch.quantization.prepare_qat`
 来实现的，其中
插入了伪量化模块。




 作为步骤 (4)，您可以启动“微调” 模型，然后将
转换为完全量化的版本(步骤 5)。




 要将微调模型转换为量化模型，您可以调用
 `torch.quantization.convert`
 函数(在我们的例子中
仅对特征提取器进行量化)。





 注意




 由于随机初始化，您的结果可能与
本教程中显示的结果不同。







```
# notice `quantize=False`
model = models.resnet18(pretrained=True, progress=True, quantize=False)
num_ftrs = model.fc.in_features

# Step 1
model.train()
model.fuse_model()
# Step 2
model_ft = create_combined_model(model)
model_ft[0].qconfig = torch.quantization.default_qat_qconfig  # Use default QAT configuration
# Step 3
model_ft = torch.quantization.prepare_qat(model_ft, inplace=True)

```




### 微调模型 [¶](#finetuning-the-model "永久链接到此标题")



 在当前教程中，对整个模型进行了微调。一般来说，这会带来更高的准确性。然而，由于
此处使用的训练集较小，我们最终会过度拟合训练集。




 步骤 4. 微调模型






```
for param in model_ft.parameters():
  param.requires_grad = True

model_ft.to(device)  # We can fine-tune on GPU if available

criterion = nn.CrossEntropyLoss()

# Note that we are training everything, so the learning rate is lower
# Notice the smaller learning rate
optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.1)

# Decay LR by a factor of 0.3 every several epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.3)

model_ft_tuned = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                             num_epochs=25, device=device)

```




 步骤 5. 转换为量化模型






```
from torch.quantization import convert
model_ft_tuned.cpu()

model_quantized_and_trained = convert(model_ft_tuned, inplace=False)

```




 让我们看看量化模型在一些图像上的表现






```
visualize_model(model_quantized_and_trained)

plt.ioff()
plt.tight_layout()
plt.show()

```










