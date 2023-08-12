# 使用 Captum 进行模型理解

> 译者：[Fadegentle](https://github.com/Fadegentle)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/introyt/captumyt>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/introyt/captumyt.html>

请跟随下面的视频或在 [youtube](https://www.youtube.com/watch?v=Am2EF9CLu-g) 上观看。[在此](https://pytorch-tutorial-assets.s3.amazonaws.com/youtube-series/video7.zip)下载笔记本和相关文件。

<iframe width="560" height="315" src="https://www.youtube.com/embed/Am2EF9CLu-g" title="Model Understanding with Captum" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

[Captum](https://captum.ai/)（拉丁语中的“comprehension〔理解〕”）是一个开源、可扩展的模型解释库，基于 PyTorch 构建。

随着模型复杂性的增加以及由此导致的透明度的缺乏，模型可解释性方法变得越来越重要。模型理解既是一个活跃的研究领域，也是使用机器学习的各行业实际应用的重点。Captum 提供了最先进的算法，包括 Integrated Gradients（积分梯度），为研究人员和开发人员提供了一种简便的方法来了解哪些特征对模型的输出有贡献。

[captum.ai](https://captum.ai/) 网站上可以获取完整的文档、API 参考资料和一系列关于特定主题的教程 。

## 入门

Captum 采用 _归因（Attribution）法_ 来实现模型的可解释性。Captum 提供三种归因：

- **特征归因（Feature Attribution）**：寻求通过输入数据的特征来解释模型的特定输出。例如，通过某些影评中的特定词语来解释影评是正面还是负面的。
- **层归因（Layer Attribution）**：分析模型在特定输入后的隐藏层的活动。例如，分析卷积层在输入图像后的空间映射输出。
- **神经元归因（Neuron Attribution）**：类似于层归因，但集中在单个神经元的活动上。

在这个交互式笔记本中，我们将着重介绍特征归因和层归因。

每种归因类型都有多种相关的**归因算法**。许多归因算法可分为两大类：

- **基于梯度的算法**：这些算法计算模型输出、层输出或神经元激活相对于输入的反向梯度。**Integrated Gradients**（用于特征）、**Layer Gradient * Activation** 和 **Neuron Conductance（神经传导）** 都属此类。
- **基于扰动的算法**：这些算法检查模型输出、层输出或神经元输出相对于输入的变化。输入扰动可以是有指向性的或随机的。**Occlusion（遮挡法）、 Feature Ablation（特征消融）和 Feature Permutation（特征排列）**都属此类。

下面我们要研究这两种类型的算法。

特别是在涉及大模型的情况下，将归因数据与正在检查的输入特征以易关联的方式可视化，是非常有价值的。虽然可以使用 Matplotlib、Plotly 或类似的工具创建自己的可视化，但 Captum 提供了针对其归因的增强工具：

- `captum.attr.visualization` 模块（如下所示导入为 `viz`）提供了用于可视化与图像相关的归因的有用函数。
- **Captum Insights** 是 Captum 上一个易于使用的 API，它提供了一个可视化组件，可为图像、文本和任意模型类型提供现成的可视化。

本笔记本将演示这两种可视化工具集。前几个示例将侧重于计算机视觉用例，但最后的 Captum Insights 部分将演示多模型、视觉问答模型中归因的可视化。

## 安装

在开始之前，您需要一个符合以下内容的 Python 环境：

- Python 3.6 以上版本
- 对于 Captum Insights 示例，Flask 1.1 以上版本及 Flask-Compress（推荐使用最新版本）
- PyTorch 1.2 以上版本（推荐使用最新版本）
- TorchVision 0.6 以上版本（推荐使用最新版本）
- Captum（推荐使用最新版本）
- Matplotlib 3.3.4 版本，因为 Captum 目前使用了 Matplotlib 的一个函数，其参数在后续版本中已更名

要在 Anaconda 或 pip 虚拟环境中安装 Captum，请使用以下适用于您的环境的命令：

用 `conda` 安装：

```shell
conda install pytorch torchvision captum flask-compress matplotlib=3.3.4 -c pytorch
```

用 `pip` 安装：

```shell
pip install torch torchvision captum matplotlib==3.3.4 Flask-Compress
```

在您安装的环境中重启此笔记本，就可以开始了！

## 第一个例子

首先，让我们从一个简单的视觉示例开始。我们将使用在 ImageNet 数据集上预训练的 ResNet 模型。我们将获取一个测试输入，并使用不同的 **Feature Attribution** 算法来检查输入图像如何影响输出，并在一些测试图像上看看有用的输入归因图可视化。

首先，导入一下：

```python
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

import os, sys
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
```

现在，我们要用 TorchVision 模型库下载一个预训练的 ResNet 模型。由于我们不需要训练，我们会将它暂时设置为评估模式。

```python
model = models.resnet18(weights='IMAGENET1K_V1')
model = model.eval()
```

您获取这个交互式笔记本的地方也应该有一个带有 `cat.jpg` 文件的 `img` 文件夹。

```python
test_img = Image.open('img/cat.jpg')
test_img_data = np.asarray(test_img)
plt.imshow(test_img_data)
plt.show()
```

我们的 ResNet 模型是在 ImageNet 数据集上进行训练的，它期望图像具有特定的尺寸，并将通道数据归一化到特定范围的值。我们还将为模型识别的类别，引入人类可读标签列表——这应该也在 `img` 文件夹中。

```python
# model expects 224x224 3-color image
transform = transforms.Compose([
 transforms.Resize(224),
 transforms.CenterCrop(224),
 transforms.ToTensor()
])

# standard ImageNet normalization
transform_normalize = transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )

transformed_img = transform(test_img)
input_img = transform_normalize(transformed_img)
input_img = input_img.unsqueeze(0) # the model requires a dummy batch dimension

labels_path = 'img/imagenet_class_index.json'
with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)
```

现在，我们可以问一个问题：我们的模型认为这张图片代表了什么？

```python
output = model(input_img)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)
pred_label_idx.squeeze_()
predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')
```

我们已经确认了 ResNet 认为我们的猫图片实际上是一只猫。但是模型为什么认为这是一张猫的图片呢？

为了回答这个，我们要看向 Captum。

## 特征归因：Integrated Gradients

**特征归因**将特定的输出归因于输入特征。它使用特定的输入（这里是我们的测试图像）来生成每个输入特征对特定输出特征的相对重要度图。

[Integrated Gradients](https://captum.ai/api/integrated_gradients.html) 是 Captum 提供的特征归因算法之一。Integrated Gradients 通过计算模型输出相对于输入的梯度近似积分，为每个输入特征分配重要度分数。

在我们的案例中，采用输出向量的特定元素，即表示模型对其所选类别置信度的元素，并使用 Integrated Gradients 来了解输入图像的哪些部分对输出产生了贡献。

从 Integrated Gradients 中获得重要度图后，用 Captum 中的可视化工具来展示重要度图。Captum 的 `visualize_image_attr()` 函数提供了多种自定义显示归因数据的选项。在这里，我们传入一个自定义的 Matplotlib 颜色图。

使用 `integrated_gradients.attribute()` 调用运行单元格通常需要一两分钟。

```python
# Initialize the attribution algorithm with the model
integrated_gradients = IntegratedGradients(model)

# Ask the algorithm to attribute our output target to
attributions_ig = integrated_gradients.attribute(input_img, target=pred_label_idx, n_steps=200)

# Show the original image for comparison
_ = viz.visualize_image_attr(None, np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                      method="original_image", title="Original Image")

default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#0000ff'),
                                                  (1, '#0000ff')], N=256)

_ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='positive',
                             title='Integrated Gradients')
```

在上图中，您应该可以看到，Integrated Gradients 给出了图像中猫所在位置周围的最强信号。

## 特征归因：Occlusion

基于梯度的归因方法有助于通过直接计算输出相对于输入的变化来理解模型。_基于扰动的归因_ 方法更直接，通过改变输入来衡量对输出的影响。[Occlusion](https://captum.ai/api/occlusion.html) 就是其中一种方法。它涉及替换输入图像的部分区域，并检查对输出信号的影响。

在下面，我们设置 Occlusion 的归因。就像配置卷积神经网络，您可以指定目标区域的大小和步长，以确定单个测量的间距。我们将使用 `visualize_image_attr_multiple()` 来可视化我们的 Occlusion 归因的输出，显示正负归因的热图，并用用正归因区域遮挡原始图像。通过遮挡，我们可以非常直观地看到模型认为猫的照片中哪些区域最“像猫”。

```python
occlusion = Occlusion(model)

attributions_occ = occlusion.attribute(input_img,
                                       target=pred_label_idx,
                                       strides=(3, 8, 8),
                                       sliding_window_shapes=(3,15, 15),
                                       baselines=0)


_ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map", "heat_map", "masked_image"],
                                      ["all", "positive", "negative", "positive"],
                                      show_colorbar=True,
                                      titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                                      fig_size=(18, 6)
                                     )
```

我们再次看到，图像中包含猫的区域更为重要。

## 层归因：Layer GradCAM

**层归因**可以将模型中隐藏层的活动归因于输入特征。下面，我们将使用层归因算法来检查模型中一个卷积层的活动。

GradCAM 会计算目标输出相对于给定层的梯度，每个输出通道（输出的维度 2）的平均值，并将每个通道的平均梯度乘以层激活。然后将所有通道的结果相加。GradCAM 专为卷积网络而设计，由于卷积层的活动通常空间映射到输入，因此 GradCAM 的归因通常会进行上采样并用于屏蔽输入。

层归因的设置与输入归因类似，但除了模型外，还必须指定模型中要检查的隐藏层。如上所述，在调用 `attribute()` 时，我们要指定感兴趣的目标类别。

```python
layer_gradcam = LayerGradCam(model, model.layer3[1].conv2)
attributions_lgc = layer_gradcam.attribute(input_img, target=pred_label_idx)

_ = viz.visualize_image_attr(attributions_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                             sign="all",
                             title="Layer 3 Block 1 Conv 2")
```

我们将使用 [LayerAttribution](https://captum.ai/api/base_classes.html?highlight=layerattribution#captum.attr.LayerAttribution) 基类中的便捷方法 `interpolate()` 对属性数据进行上采样，以便与输入图像比较。

```python
upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, input_img.shape[2:])

print(attributions_lgc.shape)
print(upsamp_attr_lgc.shape)
print(input_img.shape)

_ = viz.visualize_image_attr_multiple(upsamp_attr_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                                      transformed_img.permute(1,2,0).numpy(),
                                      ["original_image","blended_heat_map","masked_image"],
                                      ["all","positive","positive"],
                                      show_colorbar=True,
                                      titles=["Original", "Positive Attribution", "Masked"],
                                      fig_size=(18, 6))
```

通过这种可视化方式，您可以对隐藏层如何响应输入有新的认识。

## Captum Insights 可视化

Captum Insights 是 Captum 基础上的可解释性可视化部件，可促进对模型的理解。Captum Insights 可跨越图像、文本和其他特征，帮助用户理解特征归因。它允许您可视化多个输入/输出对的属性，并为图像、文本和任意数据提供可视化工具。

在本节笔记本中，我们将使用 Captum Insights 可视化多个图像分类推理。

首先，让我们收集一些图像，看看模型对它们的看法。为了多样化，我们将使用猫、茶壶和三叶虫化石：

```python
imgs = ['img/cat.jpg', 'img/teapot.jpg', 'img/trilobite.jpg']

for img in imgs:
    img = Image.open(img)
    transformed_img = transform(img)
    input_img = transform_normalize(transformed_img)
    input_img = input_img.unsqueeze(0) # the model requires a dummy batch dimension

    output = model(input_img)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()
    predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    print('Predicted:', predicted_label, '/', pred_label_idx.item(), ' (', prediction_score.squeeze().item(), ')')
```

......看起来，我们的模型能够正确识别所有这些内容——当然，我们还想更深入一点。为此，我们将使用 Captum Insights，并通过下面导入的 `AttributionVisualizer` 对象对其进行配置。`AttributionVisualizer` 需要成批的数据，因此我们将使用 Captum 的 `Batch` 辅助类。我们将特别关注图像，因此还要导入 `ImageFeature`。

我们使用以下参数配置 `AttributionVisualizer`：

- 要检查的模型数组（在我们的例子中，就一个模型）
- 一个评分函数，允许 Captum Insights 从一个模型中提取前 k 个预测值
- 一个有序的、人类可读的模型训练类列表
- 一个要探索的特征列表——在我们的例子中，一个 `ImageFeature`
- 一个数据集，是个能返回成批的输入和标签的可迭代对象——就像您在训练时使用的那样

```python
from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature

# Baseline is all-zeros input - this may differ depending on your data
def baseline_func(input):
    return input * 0

# merging our image transforms from above
def full_img_transform(input):
    i = Image.open(input)
    i = transform(i)
    i = transform_normalize(i)
    i = i.unsqueeze(0)
    return i


input_imgs = torch.cat(list(map(lambda i: full_img_transform(i), imgs)), 0)

visualizer = AttributionVisualizer(
    models=[model],
    score_func=lambda o: torch.nn.functional.softmax(o, 1),
    classes=list(map(lambda k: idx_to_labels[k][1], idx_to_labels.keys())),
    features=[
        ImageFeature(
            "Photo",
            baseline_transforms=[baseline_func],
            input_transforms=[],
        )
    ],
    dataset=[Batch(input_imgs, labels=[282,849,69])]
)
```

请注意，和之前的归因不同，运行上述单元格根本不会太久。这是因为 Captum Insights 可让您在可视化组件中配置不同的归因算法，然后计算并显示归因。这一过程只需几分钟。

运行下面的单元格将渲染 Captum Insights 组件。然后，您可以选择归因方法及其参数，根据预测类别或预测正确性筛选模型响应，查看模型带有相关概率的预测，并查看归因与原始图像的热图对比。

```python
visualizer.render()
```