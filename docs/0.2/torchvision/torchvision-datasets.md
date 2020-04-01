# torchvision.datasets
`torchvision.datasets`中包含了以下数据集

- MNIST
- COCO(用于图像标注和目标检测）(Captioning and Detection)
- LSUN Classification
- ImageFolder
- Imagenet-12
- CIFAR10 and CIFAR100
- STL10

`Datasets` 拥有以下`API`:

`__getitem__`
`__len__`

由于以上`Datasets`都是 `torch.utils.data.Dataset`的子类，所以，他们也可以通过`torch.utils.data.DataLoader`使用多线程(python的多进程）。

举例说明：
`torch.utils.data.DataLoader(coco_cap, batch_size=args.batchSize, shuffle=True, num_workers=args.nThreads)`

在构造函数中，不同的数据集直接的构造函数会有些许不同，但是他们共同拥有 `keyword` 参数。
In the constructor, each dataset has a slightly different API as needed, but they all take the keyword args:
- `transform`： 一个函数，原始图片作为输入，返回一个转换后的图片。(详情请看下面关于`torchvision-tranform`的部分）

- `target_transform` - 一个函数，输入为`target`，输出对其的转换。例子，输入的是图片标注的`string`，输出为`word`的索引。
## MNIST
```python
dset.MNIST(root, train=True, transform=None, target_transform=None, download=False)
```
参数说明：
- root : `processed/training.pt` 和 `processed/test.pt` 的主目录
- train : `True` = 训练集, `False` = 测试集
- download : `True` = 从互联网上下载数据集，并把数据集放在`root`目录下. 如果数据集之前下载过，将处理过的数据(minist.py中有相关函数）放在`processed`文件夹下。

## COCO
需要安装[COCO API](https://github.com/pdollar/coco/tree/master/PythonAPI)

### 图像标注:
```python
dset.CocoCaptions(root="dir where images are", annFile="json annotation file", [transform, target_transform])
```
例子:
```python
import torchvision.datasets as dset
import torchvision.transforms as transforms
cap = dset.CocoCaptions(root = 'dir where images are',
                        annFile = 'json annotation file',
                        transform=transforms.ToTensor())

print('Number of samples: ', len(cap))
img, target = cap[3] # load 4th sample

print("Image Size: ", img.size())
print(target)
```
输出:
```
Number of samples: 82783
Image Size: (3L, 427L, 640L)
[u'A plane emitting smoke stream flying over a mountain.',
u'A plane darts across a bright blue sky behind a mountain covered in snow',
u'A plane leaves a contrail above the snowy mountain top.',
u'A mountain that has a plane flying overheard in the distance.',
u'A mountain view with a plume of smoke in the background']
```
### 检测:
```
dset.CocoDetection(root="dir where images are", annFile="json annotation file", [transform, target_transform])
```
## LSUN
```python
dset.LSUN(db_path, classes='train', [transform, target_transform])
```
参数说明：
- db_path = 数据集文件的根目录
- classes = ‘train’ (所有类别, 训练集), ‘val’ (所有类别, 验证集), ‘test’ (所有类别, 测试集)
[‘bedroom\_train’, ‘church\_train’, …] : a list of categories to load
## ImageFolder
一个通用的数据加载器，数据集中的数据以以下方式组织
```
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
```
```python
dset.ImageFolder(root="root folder path", [transform, target_transform])
```
他有以下成员变量:

- self.classes - 用一个list保存 类名
- self.class_to_idx - 类名对应的 索引
- self.imgs - 保存(img-path, class) tuple的list

## Imagenet-12
This is simply implemented with an ImageFolder dataset.

The data is preprocessed [as described here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset)

[Here is an example](https://github.com/pytorch/examples/blob/27e2a46c1d1505324032b1d94fc6ce24d5b67e97/imagenet/main.py#L48-L62)

## CIFAR
```python
dset.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)

dset.CIFAR100(root, train=True, transform=None, target_transform=None, download=False)
```
参数说明：
- root : `cifar-10-batches-py` 的根目录
- train : `True` = 训练集, `False` = 测试集
- download : `True` = 从互联上下载数据，并将其放在`root`目录下。如果数据集已经下载，什么都不干。
## STL10
```python
dset.STL10(root, split='train', transform=None, target_transform=None, download=False)
```
参数说明：
- root : `stl10_binary`的根目录
- split : 'train' = 训练集, 'test' = 测试集, 'unlabeled' = 无标签数据集, 'train+unlabeled' = 训练 + 无标签数据集 (没有标签的标记为-1)
- download : `True` = 从互联上下载数据，并将其放在`root`目录下。如果数据集已经下载，什么都不干。
