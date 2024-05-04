# TorchVision 对象检测微调教程 [¶](#torchvision-object-detection-finetuning-tutorial "永久链接到此标题")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/torchvision_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html>


> 提示：为了充分利用本教程，我们建议使用此[Colab 版本](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/torchvision_finetuning_instance_segmentation.ipynb)。这将允许您尝试下面提供的信息。

对于本教程，我们将在[Penn-Fudan\用于行人检测和分割的数据库](https://www.cis.upenn.edu/~jshi/ped_html/)。它包含170 个图像和 345 个行人实例，我们将使用它说明如何使用 torchvision 中的新功能，在自定义数据集上训练对象检测和实例分割模型。

> 注意本教程仅适用于 torchvision 版本 >=0.16 或 nightly。如果您的版本 torchvision<=0.15，请按照[本教程](https://github.com/pytorch/教程/blob/d686b662932a380a58b7683425faa00c06bcf502/intermediate_source/torchvision_tutorial.rst)进行操作。

## 定义数据集 [¶](#defining-the-dataset "永久链接到此标题")

用于训练对象检测、实例分割和人物关键点检测的参考脚本可以轻松支持添加新的自定义数据集。数据集应继承自标准 `torch.utils.data.Dataset` 类，并实现 `__len__` 和 `__getitem__` 。

我们需要的唯一特殊的地方是数据集`__getitem__`应返回一个元组：

* __image__：[`torchvision.tv_tensors.Image`](https://pytorch.org/vision/stable/generated/torchvision.tv_tensors.Image.html#torchvision.tv_tensors.Image "(在 Torchvision v0 中.16)")， 形状为 [3, H, W] 的图像，纯张量，或大小为 (H, W) 的 PIL 图像。
* __target__：包含以下字段的字典
    * __boxs__：类型为 [`torchvision.tv_tensors.BoundingBoxes`](https://pytorch.org/vision/stable/generated/torchvision.tv_tensors.BoundingBoxes.html#torchvision.tv_tensors.BoundingBoxes "(在 Torchvision v0.16 中)")。张量形状为 `[N, 4]`；其中N为边界框的坐标数，坐标格式为[x0, y0, x1, y1]，范围从0到W 和0到H。
    * __labels__，类型为 [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(在 PyTorch v2.1)")。 张量形状为 `[N]`；其中N为每个边界框的标签。 0 始终代表背景类别。
    * __image_id__, 类型为`int`；图像标识符。它在数据集中的所有图像之间应该是唯一的，并在eval过程中使用。
    * __area__，类型为 `torch.Tensor(float)` 张量形状为 `[N]`；其中N为边界框的面积。在使用 COCO 度量进行eval时，该区域用于区分小、中、大包围盒的度量得分。
    * __iscrowd__，类型为 `torch.Tensor(uint8)` 张量形状为 `[N]`；在eval过程中将忽略 iscrowd=True 的实例。
    * __masks__（可选，类型为 [`torchvision.tv_tensors.Mask`](https://pytorch.org/vision/stable/generated/torchvision.tv_tensors.Mask.html#torchvision.tv_tensors.Mask "(在 Torchvision v0.16 中)")。张量形状为 `[N，H，W]`；代表每个对象的分割掩码。

如果您的数据集符合上述要求，那么它将适用于参考脚本中的训练和评估代码。评估代码将使用 pycocotools 中的脚本，可以通过 `pip install pycocotools` 安装。

> 注意：对于 Windows，请使用命令从[gautamchitnis](https://github.com/gautamchitnis/cocoapi) 安装 `pycocotools`  
> pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI

关于标签`label`的一点说明。模型将 `0` 类视为背景类。如果您的数据集不包含背景类，那么您的标签中就不应该有 `0`。例如，假设您只有 *猫* 和 *狗* 两个类别，您可以定义 `1`（而不是 0）代表 *猫*，`2` 代表 *狗*。因此，举例来说，如果其中一张图片同时包含两个类别，那么您的标签tensor应该是 `[1, 2]`。

此外，如果您想在训练过程中使用长宽比分组（以便每个批次只包含具有相似长宽比的图像），则建议同时实现 `get_height_and_width` 方法，该方法会返回图像的高度和宽度。如果不提供该方法，我们就会通过 `__getitem__` 查询数据集的所有元素，这样就会在内存中加载图像，速度会比提供自定义方法慢。

### 为 PennFudan 编写自定义数据集 [¶](#writing-a-custom-dataset-for-pennfudan "永久链接到此标题")

让我们为 PennFudan 数据集编写一个数据集。首先，我们下载[数据集](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip)并解压zip文件：

> wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip -P data  
cd data && unzip PennFudanPed.zip

我们将会得到以下文件夹结构:

```
PennFudanPed/
    PedMasks/
        FudanPed00001_mask.png
        FudanPed00002_mask.png
        FudanPed00003_mask.png
        FudanPed00004_mask.png
        ...
    PNGImages/
        FudanPed00001.png
        FudanPed00002.png
        FudanPed00003.png
        FudanPed00004.png
```

这是一对图像和分割掩模的一个示例

```python
import matplotlib.pyplot as plt
from torchvision.io import read_image


image = read_image("data/PennFudanPed/PNGImages/FudanPed00046.png")
mask = read_image("data/PennFudanPed/PedMasks/FudanPed00046_mask.png")

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.title("Image")
plt.imshow(image.permute(1, 2, 0))
plt.subplot(122)
plt.title("Mask")
plt.imshow(mask.permute(1, 2, 0))
```

![https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image01.png](https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image01.png)
![https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image02.png](https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image02.png)

因此，每幅图像都有一个相应的分割掩码，其中每种颜色对应一个不同的实例。让我们为这个数据集编写一个 [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset "(in PyTorch v2.1)") 类。在下面的代码中，我们将图像、边界框和掩码封装到 `torchvision.tv_tensors.TVTensor` 类中，这样我们就可以应用 torchvision 内置的变换（[新的Transforms API](https://pytorch.org/vision/stable/transforms.html)）来完成给定的对象检测和分割任务。也就是说，图像张量将由 [`torchvision.tv_tensors.Image`](https://pytorch.org/vision/stable/generated/torchvision.tv_tensors.Image.html#torchvision.tv_tensors.Image "(in Torchvision v0.16)") 封装，边界框将由 [`torchvision.tv_tensors.BoundingBoxes`](https://pytorch.org/vision/stable/generated/torchvision.tv_tensors.BoundingBoxes.html#torchvision.tv_tensors.BoundingBoxes "(在 Torchvision v0.16 中)") 封装，掩码将由 [`torchvision.tv_tensors.Mask`](https://pytorch.org/vision/stable/generated/torchvision.tv_tensors.Mask.html#torchvision.tv_tensors.Mask "(在 Torchvision v0.16 中)" ) 封装。由于 `torchvision.tv_tensors.TVTensor` 是 [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(在 PyTorch v2 中.1)") 的子类，因此封装对象也是张量，并继承了普通的 [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.1)") API。有关 `torchvision tv_tensors` 的更多信息，请参阅[本文档](https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html#what-are-tvtensors)。

```python
import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

```

这就是数据集的全部内容。现在让我们定义一个可以对此数据集执行预测的模型。

## 定义您的模型 [¶](#defining-your-model "永久链接到此标题")

在本教程中，我们将使用 [Mask R-CNN](https://arxiv.org/abs/1703.06870)，它基于 [Faster R-CNN](https://arxiv.org/abs/1506.01497)。 Faster R-CNN 是一种预测图像中潜在对象的边界框和类别分数的模型。


![https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image03.png](https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image03.png)

Mask R-CNN 在 Faster R-CNN 中添加了一个额外的分支，它还预测每个实例的分割掩模。

![https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image04.png](https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image04.png)

在两种常见情况下，人们可能想要修改 TorchVision Model Zoo 中的可用模型之一。第一种是当我们想要从预先训练的模型开始，并对最后一层进行微调时。另一种是当我们想要用不同的模型替换模型的主干时（例如，为了更快的预测）。

让我们看看在接下来的部分中我们将如何做其中一个或另一个。

### 1 - 从预训练模型进行微调 [¶](#finetuning-from-a-pretrained-model "永久链接到此标题")

假设您想从在 COCO 上预训练的模型开始，并希望针对您的特定类别对其进行微调。这是一种可能的方法：

```python
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

```
结果输出：
Downloading: "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth" to /var/lib/jenkins/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

  0%|          | 0.00/160M [00:00<?, ?B/s]
 21%|##1       | 33.9M/160M [00:00<00:00, 355MB/s]
 43%|####3     | 69.1M/160M [00:00<00:00, 364MB/s]
 65%|######5   | 104M/160M [00:00<00:00, 366MB/s]
 87%|########7 | 140M/160M [00:00<00:00, 367MB/s]
100%|##########| 160M/160M [00:00<00:00, 365MB/s]
```

### 2 - 修改模型以添加不同的主干 [¶](#modifying-the-model-to-add-a-different-backbone "永久链接到此标题")

```python
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# load a pre-trained model for classification and return
# only the features
backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
# ``FasterRCNN`` needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it here
backbone.out_channels = 1280

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'],
    output_size=7,
    sampling_ratio=2,
)

# put the pieces together inside a Faster-RCNN model
model = FasterRCNN(
    backbone,
    num_classes=2,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler,
)
```

```
结果输出：
Downloading: "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth" to /var/lib/jenkins/.cache/torch/hub/checkpoints/mobilenet_v2-7ebf99e0.pth

  0%|          | 0.00/13.6M [00:00<?, ?B/s]
100%|##########| 13.6M/13.6M [00:00<00:00, 348MB/s]
```

### PennFudan 数据集的对象检测和实例分割模型 [¶](#object-detection-and-instance-segmentation-model-for-pennfudan-dataset "永久链接到此标题")

在我们的例子中，我们希望从预训练的模型中进行微调，因为我们的数据集非常小，所以我们将遵循方法 1。

在这里，我们还想计算实例分割掩码，因此我们将使用 Mask R-CNN：

```python
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )

    return model

```

就是这样，这将使`模型`准备好在您的自定义数据集上进行训练和评估。


## 将所有内容放在一起 [¶](#putting-everything-together "永久链接到此标题")

在 `references/detection/` 中，我们有许多辅助函数来简化检测模型的训练和评估。在这里，我们将使用 `references/detection/engine.py` 和 `references/detection/utils.py`。只需将 `/detection` 下的所有内容下载到您的文件夹中，然后在此处使用即可。在 Linux 下，如果有 `wget`，可以使用以下命令下载它们：

```
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py")
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py")
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py")
os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")

```

自 v0.15.0 起，torchvision 提供了[新的 Transforms API](https://pytorch.org/vision/stable/transforms.html)，可以轻松编写用于对象检测和分割任务的数据增强管道。

让我们编写一些用于数据增强/转换的辅助函数：

```python
from torchvision.transforms import v2 as T


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

```

## 测试 `forward()` 方法(可选) [¶](#testing-forward-method-optional "固定链接到此标题")

在迭代数据集之前，最好了解模型在样本数据的训练和推理期间期望什么。

```python
import utils


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn
)

# For Training
images, targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images, targets)  # Returns losses and detections
print(output)

# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)  # Returns predictions
print(predictions[0])
```

```
输出结果：
{'loss_classifier': tensor(0.0820, grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.0278, grad_fn=<DivBackward0>), 'loss_objectness': tensor(0.0027, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.0036, grad_fn=<DivBackward0>)}
{'boxes': tensor([], size=(0, 4), grad_fn=<StackBackward0>), 'labels': tensor([], dtype=torch.int64), 'scores': tensor([], grad_fn=<IndexBackward0>)}

```

现在让我们编写执行训练和验证的主函数：

```python
from engine import train_one_epoch, evaluate

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
dataset_test = PennFudanDataset('data/PennFudanPed', get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=utils.collate_fn
)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it for 5 epochs
num_epochs = 5

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

print("That's it!")
```

```
输出结果：
Epoch: [0]  [ 0/60]  eta: 0:02:43  lr: 0.000090  loss: 2.8181 (2.8181)  loss_classifier: 0.5218 (0.5218)  loss_box_reg: 0.1272 (0.1272)  loss_mask: 2.1324 (2.1324)  loss_objectness: 0.0346 (0.0346)  loss_rpn_box_reg: 0.0022 (0.0022)  time: 2.7332  data: 0.4483  max mem: 1984
Epoch: [0]  [10/60]  eta: 0:00:24  lr: 0.000936  loss: 1.3190 (1.6752)  loss_classifier: 0.4611 (0.4213)  loss_box_reg: 0.2928 (0.3031)  loss_mask: 0.6962 (0.9183)  loss_objectness: 0.0238 (0.0253)  loss_rpn_box_reg: 0.0074 (0.0072)  time: 0.4944  data: 0.0439  max mem: 2762
Epoch: [0]  [20/60]  eta: 0:00:13  lr: 0.001783  loss: 0.9419 (1.2621)  loss_classifier: 0.2171 (0.3037)  loss_box_reg: 0.2906 (0.3064)  loss_mask: 0.4174 (0.6243)  loss_objectness: 0.0190 (0.0210)  loss_rpn_box_reg: 0.0059 (0.0068)  time: 0.2108  data: 0.0042  max mem: 2823
Epoch: [0]  [30/60]  eta: 0:00:08  lr: 0.002629  loss: 0.6349 (1.0344)  loss_classifier: 0.1184 (0.2339)  loss_box_reg: 0.2706 (0.2873)  loss_mask: 0.2276 (0.4897)  loss_objectness: 0.0065 (0.0168)  loss_rpn_box_reg: 0.0059 (0.0067)  time: 0.1650  data: 0.0051  max mem: 2823
Epoch: [0]  [40/60]  eta: 0:00:05  lr: 0.003476  loss: 0.4631 (0.8771)  loss_classifier: 0.0650 (0.1884)  loss_box_reg: 0.1924 (0.2604)  loss_mask: 0.1734 (0.4084)  loss_objectness: 0.0029 (0.0135)  loss_rpn_box_reg: 0.0051 (0.0063)  time: 0.1760  data: 0.0052  max mem: 2823
Epoch: [0]  [50/60]  eta: 0:00:02  lr: 0.004323  loss: 0.3261 (0.7754)  loss_classifier: 0.0368 (0.1606)  loss_box_reg: 0.1424 (0.2366)  loss_mask: 0.1479 (0.3599)  loss_objectness: 0.0022 (0.0116)  loss_rpn_box_reg: 0.0051 (0.0067)  time: 0.1775  data: 0.0052  max mem: 2823
Epoch: [0]  [59/60]  eta: 0:00:00  lr: 0.005000  loss: 0.3261 (0.7075)  loss_classifier: 0.0415 (0.1433)  loss_box_reg: 0.1114 (0.2157)  loss_mask: 0.1573 (0.3316)  loss_objectness: 0.0020 (0.0103)  loss_rpn_box_reg: 0.0052 (0.0066)  time: 0.2064  data: 0.0049  max mem: 2823
Epoch: [0] Total time: 0:00:14 (0.2412 s / it)
creating index...
index created!
Test:  [ 0/50]  eta: 0:00:25  model_time: 0.1576 (0.1576)  evaluator_time: 0.0029 (0.0029)  time: 0.5063  data: 0.3452  max mem: 2823
Test:  [49/50]  eta: 0:00:00  model_time: 0.0335 (0.0701)  evaluator_time: 0.0025 (0.0038)  time: 0.0594  data: 0.0025  max mem: 2823
Test: Total time: 0:00:04 (0.0862 s / it)
Averaged stats: model_time: 0.0335 (0.0701)  evaluator_time: 0.0025 (0.0038)
Accumulating evaluation results...
DONE (t=0.01s).
Accumulating evaluation results...
DONE (t=0.01s).
IoU metric: bbox
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.722
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.987
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.938
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.359
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.752
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.730
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.353
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.762
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.762
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.500
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.775
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.769
IoU metric: segm
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.726
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.993
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.913
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.344
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.593
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.743
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.360
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.760
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.760
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.633
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.662
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.772

...

Epoch: [4]  [ 0/60]  eta: 0:00:32  lr: 0.000500  loss: 0.1593 (0.1593)  loss_classifier: 0.0194 (0.0194)  loss_box_reg: 0.0272 (0.0272)  loss_mask: 0.1046 (0.1046)  loss_objectness: 0.0044 (0.0044)  loss_rpn_box_reg: 0.0037 (0.0037)  time: 0.5369  data: 0.3801  max mem: 3064
Epoch: [4]  [10/60]  eta: 0:00:10  lr: 0.000500  loss: 0.1609 (0.1870)  loss_classifier: 0.0194 (0.0236)  loss_box_reg: 0.0272 (0.0383)  loss_mask: 0.1140 (0.1190)  loss_objectness: 0.0005 (0.0023)  loss_rpn_box_reg: 0.0029 (0.0037)  time: 0.2016  data: 0.0378  max mem: 3064
Epoch: [4]  [20/60]  eta: 0:00:08  lr: 0.000500  loss: 0.1652 (0.1826)  loss_classifier: 0.0224 (0.0242)  loss_box_reg: 0.0286 (0.0374)  loss_mask: 0.1075 (0.1165)  loss_objectness: 0.0003 (0.0016)  loss_rpn_box_reg: 0.0016 (0.0029)  time: 0.1866  data: 0.0044  max mem: 3064
Epoch: [4]  [30/60]  eta: 0:00:06  lr: 0.000500  loss: 0.1676 (0.1884)  loss_classifier: 0.0245 (0.0264)  loss_box_reg: 0.0286 (0.0401)  loss_mask: 0.1075 (0.1175)  loss_objectness: 0.0003 (0.0013)  loss_rpn_box_reg: 0.0018 (0.0030)  time: 0.2106  data: 0.0055  max mem: 3064
Epoch: [4]  [40/60]  eta: 0:00:03  lr: 0.000500  loss: 0.1726 (0.1884)  loss_classifier: 0.0245 (0.0265)  loss_box_reg: 0.0283 (0.0394)  loss_mask: 0.1187 (0.1184)  loss_objectness: 0.0003 (0.0011)  loss_rpn_box_reg: 0.0020 (0.0029)  time: 0.1897  data: 0.0056  max mem: 3064
Epoch: [4]  [50/60]  eta: 0:00:01  lr: 0.000500  loss: 0.1910 (0.1938)  loss_classifier: 0.0273 (0.0280)  loss_box_reg: 0.0414 (0.0418)  loss_mask: 0.1177 (0.1198)  loss_objectness: 0.0003 (0.0010)  loss_rpn_box_reg: 0.0022 (0.0031)  time: 0.1623  data: 0.0056  max mem: 3064
Epoch: [4]  [59/60]  eta: 0:00:00  lr: 0.000500  loss: 0.1732 (0.1888)  loss_classifier: 0.0273 (0.0278)  loss_box_reg: 0.0327 (0.0405)  loss_mask: 0.0993 (0.1165)  loss_objectness: 0.0003 (0.0010)  loss_rpn_box_reg: 0.0023 (0.0030)  time: 0.1732  data: 0.0056  max mem: 3064
Epoch: [4] Total time: 0:00:11 (0.1920 s / it)
creating index...
index created!
Test:  [ 0/50]  eta: 0:00:21  model_time: 0.0589 (0.0589)  evaluator_time: 0.0032 (0.0032)  time: 0.4269  data: 0.3641  max mem: 3064
Test:  [49/50]  eta: 0:00:00  model_time: 0.0515 (0.0521)  evaluator_time: 0.0020 (0.0031)  time: 0.0579  data: 0.0024  max mem: 3064
Test: Total time: 0:00:03 (0.0679 s / it)
Averaged stats: model_time: 0.0515 (0.0521)  evaluator_time: 0.0020 (0.0031)
Accumulating evaluation results...
DONE (t=0.01s).
Accumulating evaluation results...
DONE (t=0.01s).
IoU metric: bbox
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.846
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.997
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.978
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.412
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.689
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.864
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.417
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.876
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.876
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.567
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.750
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.896
IoU metric: segm
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.777
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.997
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.961
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.424
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.631
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.791
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.373
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.814
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.814
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.633
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.713
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.827

That's it!

```

因此，经过一轮训练后，我们获得了 COCO 风格的 mAP > 50，并且掩模 mAP 为 65。

但预测结果是怎样的呢？让我们以数据集中的一张图片为例进行验证

![https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png](https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png)

```python
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

image = read_image("https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png")
eval_transform = get_transform(train=False)

model.eval()
with torch.no_grad():
    x = eval_transform(image)
    # convert RGBA -> RGB and move to device
    x = x[:3, ...].to(device)
    predictions = model([x, ])
    pred = predictions[0]

image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]
pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
pred_boxes = pred["boxes"].long()
output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

masks = (pred["masks"] > 0.7).squeeze(1)
output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))

```

![https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image06.png](https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image06.png)

结果看起来不错！

## 总结 [¶](#wrapping-up "此标题的固定链接")

在本教程中，您学习了如何在自定义数据集上为对象检测模型创建自己的训练管道。为此，您编写了一个 `torch.utils.data.Dataset` 类，该类返回图像以及地面实况框和分割掩码。您还利用了在 COCO train2017 上预训练的 Mask R-CNN 模型，以便在此新数据集上执行迁移学习。

有关包括多机器/多GPU 训练在内的更完整示例，请查看 torchvision 储存库中的 `references/detection/train.py`。

您可以在[此处](https://pytorch.org/tutorials/_static/tv-training-code.py)下载本教程的完整源文件。








