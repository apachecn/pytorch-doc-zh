# 数据加载和处理教程

**作者** ：[ Sasank Chilamkurthy ](https://chsasank.github.io)

在解决任何机器学习问题的一个很大的功夫去到准备数据。
PyTorch提供了许多工具，使数据加载容易，希望，使你的代码更易读。在本教程中，我们将看到如何从一个不平凡的数据集加载和预处理/增强数据。

要运行本教程中，请确保以下软件包安装：

  * `scikit图像 `：用于图像IO和变换
  * `大熊猫 `：为了方便CSV解析

    
    
    from __future__ import print_function, division
    import os
    import torch
    import pandas as pd
    from skimage import io, transform
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, utils
    
    # Ignore warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    plt.ion()   # interactive mode
    

我们要处理的数据集是面部姿态。这意味着，脸被注释是这样的：

[![../_images/landmarked_face2.png](../_images/landmarked_face2.png)](../_images/landmarked_face2.png)

总体而言，68个不同的标志点被注释为每个面。

Note

从[下载数据集在这里](https://download.pytorch.org/tutorial/faces.zip)，这样的图像是在一个名为“数据/面/”目录。此数据集实际上是由优秀的应用[
DLIB的姿态估计](https://blog.dlib.net/2014/08/real-time-face-pose-
estimation.html)从imagenet标记为“面子”一些图像生成。

数据集配有带注释，看起来像这样一个CSV文件：

    
    
    image_name,part_0_x,part_0_y,part_1_x,part_1_y,part_2_x, ... ,part_67_x,part_67_y
    0805personali01.jpg,27,83,27,98, ... 84,134
    1084239450_e76e00b7e7.jpg,70,236,71,257, ... ,128,312
    

让我们快速读取CSV并获得注释在（N，2）数组，其中N为标志的数量。

    
    
    landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')
    
    n = 65
    img_name = landmarks_frame.iloc[n, 0]
    landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
    landmarks = landmarks.astype('float').reshape(-1, 2)
    
    print('Image name: {}'.format(img_name))
    print('Landmarks shape: {}'.format(landmarks.shape))
    print('First 4 Landmarks: {}'.format(landmarks[:4]))
    

日期：

    
    
    Image name: person-7.jpg
    Landmarks shape: (68, 2)
    First 4 Landmarks: [[32. 65.]
     [33. 76.]
     [34. 86.]
     [34. 97.]]
    

让我们写一个简单的辅助函数来显示的图象和标志性建筑，并用它来显示一个样本。

    
    
    def show_landmarks(image, landmarks):
        """Show image with landmarks"""
        plt.imshow(image)
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
        plt.pause(0.001)  # pause a bit so that plots are updated
    
    plt.figure()
    show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
                   landmarks)
    plt.show()
    

![../_images/sphx_glr_data_loading_tutorial_001.png](../_images/sphx_glr_data_loading_tutorial_001.png)

## 数据集类

`torch.utils.data.Dataset`是表示数据集的抽象类。您的自定义数据集要继承`数据集 `，并覆盖下列方法：

  * `__len__`，使得`LEN（数据集） `返回数据集的大小。
  * `__getitem__`支持索引，使得`数据集[I]`可以被用来获得 \（I \）个样本

让我们创建一个DataSet类为我们的脸地标数据集。我们将读取`__init__`的CSV但留下的图像，以`__getitem__
`读数。因为所有的图像没有存储在存储器中的一次，但根据需要读取，这是记忆效率。

我们的数据集中的样品将是一个字典`{ '图像'： 图像， '标志'： 地标}`。我们的数据集将采取一个可选的参数`变换
`，使得可以在样品被施加任何所需的处理。我们将看到的`用处变换 `在下一节。

    
    
    class FaceLandmarksDataset(Dataset):
        """Face Landmarks dataset."""
    
        def __init__(self, csv_file, root_dir, transform=None):
            """
            Args:
                csv_file (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied
                    on a sample.
            """
            self.landmarks_frame = pd.read_csv(csv_file)
            self.root_dir = root_dir
            self.transform = transform
    
        def __len__(self):
            return len(self.landmarks_frame)
    
        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
    
            img_name = os.path.join(self.root_dir,
                                    self.landmarks_frame.iloc[idx, 0])
            image = io.imread(img_name)
            landmarks = self.landmarks_frame.iloc[idx, 1:]
            landmarks = np.array([landmarks])
            landmarks = landmarks.astype('float').reshape(-1, 2)
            sample = {'image': image, 'landmarks': landmarks}
    
            if self.transform:
                sample = self.transform(sample)
    
            return sample
    

让我们来实例化这个类，并通过数据样本进行迭代。我们将打印首4个样品的尺寸和展示自己的标志性建筑。

    
    
    face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                        root_dir='data/faces/')
    
    fig = plt.figure()
    
    for i in range(len(face_dataset)):
        sample = face_dataset[i]
    
        print(i, sample['image'].shape, sample['landmarks'].shape)
    
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks(**sample)
    
        if i == 3:
            plt.show()
            break
    

![../_images/sphx_glr_data_loading_tutorial_002.png](../_images/sphx_glr_data_loading_tutorial_002.png)

Out:

    
    
    0 (324, 215, 3) (68, 2)
    1 (500, 333, 3) (68, 2)
    2 (250, 258, 3) (68, 2)
    3 (434, 290, 3) (68, 2)
    

## 变换

有一个问题，我们可以从上面看到的是，样本大小相同的不行。大多数的神经网络指望一个固定大小的图像。因此，我们需要编写一些代码prepocessing。让我们创建三个变换：

  * `重新缩放 `：将图像缩放
  * `RandomCrop`：从图像中随机裁剪。这是数据增强。
  * `ToTensor`：转换的numpy的图像焊枪图片（我们需要换轴）。

我们将它们写为可调用的类，而不是简单的功能，这样的变换参数不需要通过每次它叫。对于这一点，我们只需要实现`__call__`方法，如果需要，`
__init__`方法。然后，我们可以使用转换是这样的：

    
    
    tsfm = Transform(params)
    transformed_sample = tsfm(sample)
    

注意下面这些变换怎么过的图像和标志性建筑上应用两者。

    
    
    class Rescale(object):
        """Rescale the image in a sample to a given size.
    
        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size. If int, smaller of image edges is matched
                to output_size keeping aspect ratio the same.
        """
    
        def __init__(self, output_size):
            assert isinstance(output_size, (int, tuple))
            self.output_size = output_size
    
        def __call__(self, sample):
            image, landmarks = sample['image'], sample['landmarks']
    
            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size
    
            new_h, new_w = int(new_h), int(new_w)
    
            img = transform.resize(image, (new_h, new_w))
    
            # h and w are swapped for landmarks because for images,
            # x and y axes are axis 1 and 0 respectively
            landmarks = landmarks * [new_w / w, new_h / h]
    
            return {'image': img, 'landmarks': landmarks}
    
    
    class RandomCrop(object):
        """Crop randomly the image in a sample.
    
        Args:
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
        """
    
        def __init__(self, output_size):
            assert isinstance(output_size, (int, tuple))
            if isinstance(output_size, int):
                self.output_size = (output_size, output_size)
            else:
                assert len(output_size) == 2
                self.output_size = output_size
    
        def __call__(self, sample):
            image, landmarks = sample['image'], sample['landmarks']
    
            h, w = image.shape[:2]
            new_h, new_w = self.output_size
    
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
    
            image = image[top: top + new_h,
                          left: left + new_w]
    
            landmarks = landmarks - [left, top]
    
            return {'image': image, 'landmarks': landmarks}
    
    
    class ToTensor(object):
        """Convert ndarrays in sample to Tensors."""
    
        def __call__(self, sample):
            image, landmarks = sample['image'], sample['landmarks']
    
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            image = image.transpose((2, 0, 1))
            return {'image': torch.from_numpy(image),
                    'landmarks': torch.from_numpy(landmarks)}
    

### 构成变换

现在，我们应用在样品上的变换。

比方说，我们希望将图像的短边重新调整到256，然后随机地从它种植规模224的正方形。即，我们要组成`重新调整 `和`RandomCrop`变换。 `
torchvision.transforms.Compose`是一个简单的可调用的类，它使我们能够做到这一点。

    
    
    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = transforms.Compose([Rescale(256),
                                   RandomCrop(224)])
    
    # Apply each of the above transforms on sample.
    fig = plt.figure()
    sample = face_dataset[65]
    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)
    
        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        show_landmarks(**transformed_sample)
    
    plt.show()
    

![../_images/sphx_glr_data_loading_tutorial_003.png](../_images/sphx_glr_data_loading_tutorial_003.png)

## 通过该数据集迭代

让我们把所有这一切共同创造与由变换的数据集。总之，每一个数据集被采样时间：

  * 图像从上飞文件中读取
  * 变换被应用于所读取的图像上
  * 由于变换之一是随机的，数据被augmentated上采样

我们可以用一个`在创建数据集迭代为 i的 在 范围 `环如前。

    
    
    transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                               root_dir='data/faces/',
                                               transform=transforms.Compose([
                                                   Rescale(256),
                                                   RandomCrop(224),
                                                   ToTensor()
                                               ]))
    
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
    
        print(i, sample['image'].size(), sample['landmarks'].size())
    
        if i == 3:
            break
    

Out:

    
    
    0 torch.Size([3, 224, 224]) torch.Size([68, 2])
    1 torch.Size([3, 224, 224]) torch.Size([68, 2])
    2 torch.Size([3, 224, 224]) torch.Size([68, 2])
    3 torch.Size([3, 224, 224]) torch.Size([68, 2])
    

但是，我们通过使用简单的`对 `循环遍历数据丢失了很多功能。特别是，我们错过了：

  * 配料数据
  * 洗牌的数据
  * 使用`多处理 `工人负载并联的数据。

`torch.utils.data.DataLoader`为提供所有这些功能的迭代器。下面的参数应该是清楚的。感兴趣的一个参数是`
collat​​e_fn  [HTG7。您可以指定样品需要究竟如何使用`collat​​e_fn
`进行批处理。然而，默认的整理应该正常工作对于大多数使用情况。`

    
    
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)
    
    
    # Helper function to show a batch
    def show_landmarks_batch(sample_batched):
        """Show image with landmarks for a batch of samples."""
        images_batch, landmarks_batch = \
                sample_batched['image'], sample_batched['landmarks']
        batch_size = len(images_batch)
        im_size = images_batch.size(2)
        grid_border_size = 2
    
        grid = utils.make_grid(images_batch)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
    
        for i in range(batch_size):
            plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                        landmarks_batch[i, :, 1].numpy() + grid_border_size,
                        s=10, marker='.', c='r')
    
            plt.title('Batch from dataloader')
    
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['landmarks'].size())
    
        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
    

![../_images/sphx_glr_data_loading_tutorial_004.png](../_images/sphx_glr_data_loading_tutorial_004.png)

Out:

    
    
    0 torch.Size([4, 3, 224, 224]) torch.Size([4, 68, 2])
    1 torch.Size([4, 3, 224, 224]) torch.Size([4, 68, 2])
    2 torch.Size([4, 3, 224, 224]) torch.Size([4, 68, 2])
    3 torch.Size([4, 3, 224, 224]) torch.Size([4, 68, 2])
    

## 后记：torchvision

在本教程中，我们已经看到了如何编写和使用的数据集，转换和的DataLoader。 `torchvision
`包提供了一些常见的数据集和变换。你甚至可能没有编写自定义类。一个在torchvision提供更通用的数据集是`ImageFolder
[HTG7。它假定图像通过以下方式进行组织：`

    
    
    root/ants/xxx.png
    root/ants/xxy.jpeg
    root/ants/xxz.png
    .
    .
    .
    root/bees/123.jpg
    root/bees/nsdf3.png
    root/bees/asd932_.png
    

其中，“蚂蚁”，“蜜蜂”等都是一流的标签。其上`操作类似地通用变换PIL.Image`如`RandomHorizo​​ntalFlip`，`
量表 `也可提供。您可以使用这些来写这样的的DataLoader：

    
    
    import torch
    from torchvision import transforms, datasets
    
    data_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
                                               transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                                 batch_size=4, shuffle=True,
                                                 num_workers=4)
    

对于训练代码示例，请参见[ 迁移学习教程 [HTG3。](transfer_learning_tutorial.html)

**脚本的总运行时间：** （0分钟59.213秒）

[`Download Python source code:
data_loading_tutorial.py`](../_downloads/0daab3cdf9be9579bd736e92d8de3917/data_loading_tutorial.py)

[`Download Jupyter notebook:
data_loading_tutorial.ipynb`](../_downloads/21adbaecd47a412f8143afb1c48f05a6/data_loading_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](pytorch_with_examples.html "Learning PyTorch with Examples")
[![](../_static/images/chevron-right-orange.svg)
Previous](blitz/data_parallel_tutorial.html "Optional: Data Parallelism")

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

  * 数据加载和处理教程
    * DataSet类
    * 变换
      * 撰写变换
    * 通过数据集迭代
    * 后记：torchvision 

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

