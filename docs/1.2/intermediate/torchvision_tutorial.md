# TorchVision对象检测教程细化和微调

小费

为了得到最本教程中，我们建议使用此[
Colab版本[HTG1。这将让你与下面提供的信息进行试验。](https://colab.research.google.com/github/pytorch/vision/blob/temp-
tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb)

在本教程中，我们将微调预训练[面膜R-CNN ](https://arxiv.org/abs/1703.06870)中的[佩恩 -
复旦数据库行人检测与分割](https://www.cis.upenn.edu/~jshi/ped_html/)模式。它包含170倍的图像与行人345分的情况下，我们将用它来说明如何才能培养上的自定义数据集的实例细分模型使用torchvision的新功能。

## 定义数据集

培训对象检测，例如分割和人关键点检测的参考脚本允许容易地支持添加新的自定义数据集。数据集应该从标准`torch.utils.data.Dataset
`类继承，实施`__len__`和`__getitem__`。

我们需要的唯一的特殊性在于，数据集`__getitem__`应该返回：

  * 图像：尺寸`（H， W）的PIL图像 `
  * 目标：包含以下字段的一个字典
    * `盒 （FloatTensor [N， 4]） `：的`N 的坐标`在包围盒`[X0， Y0， X 1， Y1]`格式中，范围从`0`至`W`和`0`至`H`
    * `标签 （Int64Tensor [N]） `：对于每个边界框的标签
    * `image_id  （Int64Tensor [1]） `：图像标识符。它应该是在数据集中的所有图像之间唯一的，评估过程中使用
    * `面积 （张量[N]） `：将边界框的面积。这是通过COCO度量评估过程中使用，以分离小，中，大箱之间的度量得分。
    * `iscrowd  （UInt8Tensor [N]） `：用iscrowd =真实例将被评估期间忽略。
    * （可选地）`掩模 （UInt8Tensor [N， H， W]） `：本分割掩码的每个其中一个对象
    * （可选地）`关键点 （FloatTensor [N， K， 3]） `：对于每一个中的所述一个N个对象，它包含`K个关键点[X， Y， 能见度]`格式中，定义的对象。能见度= 0表示所述关键点是不可见的。请注意，数据增强，翻转关键点的概念是依赖于数据表示，你可能要适应`引用/检测/ transforms.py`为您的新关键点表示

如果你的模型返回上面的方法，他们将使其成为训练和评估工作，并会使用评估脚本，从`pycocotools  [HTG3。`

此外，如果要训练期间要使用的纵横比的分组（以使每个批次仅包含具有类似的纵横比的图像），则建议也实现`get_height_and_width
`的方法，其中返回的高度和图像的宽度。如果不提供这种方法，我们通过`查询数据集的所有元素__getitem__
`，它加载在存储器中的图像，并且比如果提供一个自定义的方法慢。

### 编写自定义的数据集PennFudan

让我们写的PennFudan数据集的数据集。
[下载并解压缩zip文件](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip)之后，我们有以下文件夹结构：

    
    
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
    

这里是一对图像和分割掩码的一个实例

![intermediate/../../_static/img/tv_tutorial/tv_image01.png](intermediate/../../_static/img/tv_tutorial/tv_image01.png)
![intermediate/../../_static/img/tv_tutorial/tv_image02.png](intermediate/../../_static/img/tv_tutorial/tv_image02.png)

因此，每个图像具有对应的分割掩码，其中每个颜色对应于不同实例。让我们写一个`torch.utils.data.Dataset`类数据集。

    
    
    import os
    import numpy as np
    import torch
    from PIL import Image
    
    
    class PennFudanDataset(object):
        def __init__(self, root, transforms):
            self.root = root
            self.transforms = transforms
            # load all image files, sorting them to
            # ensure that they are aligned
            self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
            self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
    
        def __getitem__(self, idx):
            # load images ad masks
            img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
            mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
            img = Image.open(img_path).convert("RGB")
            # note that we haven't converted the mask to RGB,
            # because each color corresponds to a different instance
            # with 0 being background
            mask = Image.open(mask_path)
            # convert the PIL Image into a numpy array
            mask = np.array(mask)
            # instances are encoded as different colors
            obj_ids = np.unique(mask)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]
    
            # split the color-encoded mask into a set
            # of binary masks
            masks = mask == obj_ids[:, None, None]
    
            # get bounding box coordinates for each mask
            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
    
            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
    
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
    
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
    
            if self.transforms is not None:
                img, target = self.transforms(img, target)
    
            return img, target
    
        def __len__(self):
            return len(self.imgs)
    
    

这是所有的数据集。现在让我们来定义可以在这个数据集进行预测的模型。

## 自定义模型

在本文中，我们将使用[面膜R-CNN ](https://arxiv.org/abs/1703.06870)，它是基于[更快R-CNN
](https://arxiv.org/abs/1506.01497)顶部。更快的R-CNN是预测二者边界框和类得分为图像中的潜在对象的模型。

![intermediate/../../_static/img/tv_tutorial/tv_image03.png](intermediate/../../_static/img/tv_tutorial/tv_image03.png)

面膜R-CNN增加了额外的分支为更快R-CNN，这也预示分割掩码为每个实例。

![intermediate/../../_static/img/tv_tutorial/tv_image04.png](intermediate/../../_static/img/tv_tutorial/tv_image04.png)

有一个地方可能要修改torchvision
modelzoo可用的车型之一两种常见的情况。第一种是当我们想从一个预先训练模型开始，只是微调的最后一层。另一种是当我们想用一个不同的替换机型的骨干（更快的预测，例如）。

让我们去看看我们会怎么做在下面的章节一种或另一种。

### 1 - 从一个预训练的模型微调

让我们假设你想从一个模型上COCO预先训练开始，并希望它微调您的特定类。下面是做这件事的一个可能的方式：

    
    
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    

### 2 - 修改模型添加不同的骨干

    
    
    import torchvision
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280
    
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)
    
    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    
    

### 对于实例分割模型PennFudan数据集

在我们的例子中，我们要微调从预先训练的模式，因为我们的数据是非常小的，所以我们将在下面的方法1号。

在这里，我们也想计算实例分割掩码，所以我们将使用面膜R-CNN：

    
    
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    
    
    def get_model_instance_segmentation(num_classes):
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)
    
        return model
    
    

就是这样，这将使`模型 `准备进行培训，并在您的自定义数据集进行评估。

## 将所有内容放在一起

在`引用/检测/`，我们有一些辅助功能，以简化培训和评估检测模型。在这里，我们将使用`引用/检测/ engine.py`，`引用/检测/
utils.py`和`参考/检测/ transforms.py`。把它们复制到你的文件夹，并在这里使用它们。

让我们来写数据增强/转换了一些辅助功能：

    
    
    import transforms as T
    
    def get_transform(train):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)
    
    

现在，让我们写一个执行训练和验证的主要功能：

    
    
    from engine import train_one_epoch, evaluate
    import utils
    
    
    def main():
        # train on the GPU or on the CPU, if a GPU is not available
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
        # our dataset has two classes only - background and person
        num_classes = 2
        # use our dataset and defined transformations
        dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
        dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
    
        # split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    
        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)
    
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)
    
        # get the model using our helper function
        model = get_model_instance_segmentation(num_classes)
    
        # move model to the right device
        model.to(device)
    
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)
    
        # let's train it for 10 epochs
        num_epochs = 10
    
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=device)
    
        print("That's it!")
    
    

你应该得到作为第一时期的输出：

    
    
    Epoch: [0]  [ 0/60]  eta: 0:01:18  lr: 0.000090  loss: 2.5213 (2.5213)  loss_classifier: 0.8025 (0.8025)  loss_box_reg: 0.2634 (0.2634)  loss_mask: 1.4265 (1.4265)  loss_objectness: 0.0190 (0.0190)  loss_rpn_box_reg: 0.0099 (0.0099)  time: 1.3121  data: 0.3024  max mem: 3485
    Epoch: [0]  [10/60]  eta: 0:00:20  lr: 0.000936  loss: 1.3007 (1.5313)  loss_classifier: 0.3979 (0.4719)  loss_box_reg: 0.2454 (0.2272)  loss_mask: 0.6089 (0.7953)  loss_objectness: 0.0197 (0.0228)  loss_rpn_box_reg: 0.0121 (0.0141)  time: 0.4198  data: 0.0298  max mem: 5081
    Epoch: [0]  [20/60]  eta: 0:00:15  lr: 0.001783  loss: 0.7567 (1.1056)  loss_classifier: 0.2221 (0.3319)  loss_box_reg: 0.2002 (0.2106)  loss_mask: 0.2904 (0.5332)  loss_objectness: 0.0146 (0.0176)  loss_rpn_box_reg: 0.0094 (0.0123)  time: 0.3293  data: 0.0035  max mem: 5081
    Epoch: [0]  [30/60]  eta: 0:00:11  lr: 0.002629  loss: 0.4705 (0.8935)  loss_classifier: 0.0991 (0.2517)  loss_box_reg: 0.1578 (0.1957)  loss_mask: 0.1970 (0.4204)  loss_objectness: 0.0061 (0.0140)  loss_rpn_box_reg: 0.0075 (0.0118)  time: 0.3403  data: 0.0044  max mem: 5081
    Epoch: [0]  [40/60]  eta: 0:00:07  lr: 0.003476  loss: 0.3901 (0.7568)  loss_classifier: 0.0648 (0.2022)  loss_box_reg: 0.1207 (0.1736)  loss_mask: 0.1705 (0.3585)  loss_objectness: 0.0018 (0.0113)  loss_rpn_box_reg: 0.0075 (0.0112)  time: 0.3407  data: 0.0044  max mem: 5081
    Epoch: [0]  [50/60]  eta: 0:00:03  lr: 0.004323  loss: 0.3237 (0.6703)  loss_classifier: 0.0474 (0.1731)  loss_box_reg: 0.1109 (0.1561)  loss_mask: 0.1658 (0.3201)  loss_objectness: 0.0015 (0.0093)  loss_rpn_box_reg: 0.0093 (0.0116)  time: 0.3379  data: 0.0043  max mem: 5081
    Epoch: [0]  [59/60]  eta: 0:00:00  lr: 0.005000  loss: 0.2540 (0.6082)  loss_classifier: 0.0309 (0.1526)  loss_box_reg: 0.0463 (0.1405)  loss_mask: 0.1568 (0.2945)  loss_objectness: 0.0012 (0.0083)  loss_rpn_box_reg: 0.0093 (0.0123)  time: 0.3489  data: 0.0042  max mem: 5081
    Epoch: [0] Total time: 0:00:21 (0.3570 s / it)
    creating index...
    index created!
    Test:  [ 0/50]  eta: 0:00:19  model_time: 0.2152 (0.2152)  evaluator_time: 0.0133 (0.0133)  time: 0.4000  data: 0.1701  max mem: 5081
    Test:  [49/50]  eta: 0:00:00  model_time: 0.0628 (0.0687)  evaluator_time: 0.0039 (0.0064)  time: 0.0735  data: 0.0022  max mem: 5081
    Test: Total time: 0:00:04 (0.0828 s / it)
    Averaged stats: model_time: 0.0628 (0.0687)  evaluator_time: 0.0039 (0.0064)
    Accumulating evaluation results...
    DONE (t=0.01s).
    Accumulating evaluation results...
    DONE (t=0.01s).
    IoU metric: bbox
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.606
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.984
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.780
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.313
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.582
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.612
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.270
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.672
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.672
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.650
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.755
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.664
    IoU metric: segm
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.704
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.979
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.871
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.325
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.488
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.727
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.316
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.748
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.749
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.650
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.673
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.758
    

所以培训一个时期后，我们得到的60.6椰油风格的地图，以及70.4面具地图。

10个时代训练结束后，我得到了以下指标

    
    
    IoU metric: bbox
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.799
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.969
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.935
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.349
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.592
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.831
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.324
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.844
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.844
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.777
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.870
    IoU metric: segm
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.761
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.969
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.919
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.341
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.464
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.788
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.303
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.799
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.799
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.769
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.818
    

但是什么预测样子的？让我们一个图像中的数据集和验证

![intermediate/../../_static/img/tv_tutorial/tv_image05.png](intermediate/../../_static/img/tv_tutorial/tv_image05.png)

训练的模型预测的9种人情况下，这种形象，让我们来看看他们夫妇：

![intermediate/../../_static/img/tv_tutorial/tv_image06.png](intermediate/../../_static/img/tv_tutorial/tv_image06.png)
![intermediate/../../_static/img/tv_tutorial/tv_image07.png](intermediate/../../_static/img/tv_tutorial/tv_image07.png)

结果看起来不错！

## 结束语

在本教程中，你已经学会了如何创建例如分割模型自己训练的管道，在自定义的数据集。对于这一点，你写一个`torch.utils.data.Dataset
`类，返回的图像和地面真值块和分割口罩。您还利用一个面具上COCO train2017 R-CNN模型预先训练，以这个新的数据集进行迁移学习。

对于更完整的示例，其包括多机/多GPU训练，检查`引用/检测/ train.py`，它是存在于torchvision回购。

你可以下载一个完整的源文件本教程[此处[HTG1。](https://pytorch.org/tutorials/_static/tv-training-
code.py)

[Next ![](../_static/images/chevron-right-
orange.svg)](../beginner/finetuning_torchvision_models_tutorial.html
"Finetuning Torchvision Models") [![](../_static/images/chevron-right-
orange.svg) Previous](../beginner/nn_tutorial.html "What is torch.nn really?")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。



  * TorchVision对象检测教程细化和微调
    * 定义数据集
      * 编写自定义数据集PennFudan 
    * 自定义模型
      * 1 - 从一个预训练的模型微调
      * 2 - 修改模型添加不同的骨干
      * [HTG0用于PennFudan数据集的实例分割模型
    * 将所有内容放在一起
    * 结束语

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)







 
[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)



