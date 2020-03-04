# torchvision.transforms

> 译者：[@那伊抹微笑](https://github.com/wangyangting)、@dawenzi123、[@LeeGeong](https://github.com/LeeGeong)、@liandongze
> 
> 校对者：[@咸鱼](https://github.com/Watermelon233)

Transforms (变换) 是常见的 image transforms (图像变换) .他们可以使用 `Compose` 类以链在一起来进行操作.

```py
class torchvision.transforms.Compose(transforms)
```

将多个变换组合到一起.

参数：`transforms (Transform 对象列表)` – 要组合的变换列表.


示例：

```py
>>> transforms.Compose([
>>>     transforms.CenterCrop(10),
>>>     transforms.ToTensor(),
>>> ])

```

## PIL Image 上的变换

```py
class torchvision.transforms.Resize(size, interpolation=2)
```

调整输入的 PIL Image 尺寸为给定的 size(尺寸）.

参数：

*   `size (sequence 或 int)` – 期望输出的尺寸. 如果 size(尺寸）是一个像 (h, w) 这样的序列, 则 output size(输出尺寸）将于此匹配. 如果 size(尺寸）是一个 int 类型的数字, 图像较小的边缘将被匹配到该数字. 例如, 如果 height &gt; width, 那么图像将会被重新缩放到 (size * height / width, size). 即按照size/width的比值缩放
*   `interpolation (int, 可选)` – 期望的插值. 默认是 `PIL.Image.BILINEAR`



```py
class torchvision.transforms.Scale(*args, **kwargs)
```

Note: 为了支持 Resize, 该变换已经过时了.

```py
class torchvision.transforms.CenterCrop(size)
```

在中心裁剪指定的 PIL Image.

参数：`size (sequence 或 int)` – 期望裁剪的输出尺寸. 如果 size(尺寸）是 `int` 类型的整数, 而不是像 (h, w) 这样类型的序列, 裁剪出来的图像是 (size, size) 这样的正方形的.


```py
class torchvision.transforms.RandomCrop(size, padding=0)
```

在一个随机位置裁剪指定的 PIL Image.

参数：

*   `size (sequence 或 int)` – 期望输出的裁剪尺寸. 如果 size(尺寸）是 `int` 类型的整数, 而不是像 (h, w) 这样类型的序列, 裁剪出来的图像是 (size, size) 这样的正方形的.
*   `padding (int 或 sequence, 可选)` – 图像的每个边框上的可选填充. 缺省值是 0, 即没有填充. 如果提供长度为 4 的序列, 则分别用于填充左侧, 顶部, 右侧, 底部边界.



```py
class torchvision.transforms.RandomHorizontalFlip
```

以概率0.5随机水平翻转图像

```py
class torchvision.transforms.RandomVerticalFlip
```

以概率0.5随机垂直翻转图像.

```py
class torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
```

将给定的 PIL 图像裁剪为随机大小和纵横比例.

原始高宽比的随机大小(默认: 0.08 到 1.0）和随机宽高比(默认: 3/4 到 4/3）的裁剪. 该裁剪最终会被调整为指定的尺寸.

该操作普遍用于训练 Inception networks.

参数：

*   `size` – 每条边的期望的输出尺寸
*   `scale` – 原始剪裁尺寸大小的范围
*   `ratio` – 原始裁剪纵横比的范围
*   `interpolation` – Default: PIL.Image.BILINEAR



```py
class torchvision.transforms.RandomSizedCrop(*args, **kwargs)
```

Note: 为了支持 RandomResizedCrop, 该变换已经被弃用.

```py
class torchvision.transforms.Grayscale(num_output_channels=1)
```

将图像转换为灰度图像.

参数：`num_output_channels (int)` – (1 or 3) 输出图像所期望的通道数量

返回值：灰度版本的输入. - 如果 num_output_channels == 1 : 返回的图像是 1 通道 - 如果 num_output_channels == 3 : 返回的图像是 3 通道, 并且 r == g == b

返回类型：`PIL Image`

```py
class torchvision.transforms.RandomGrayscale(p=0.1)
```

随机将图像转换为灰度图像, 概率为 p (default 0.1).

参数：`p (float)` – 图像应该被转换成灰度的概率.

返回值：灰度版本的输入图像的概率为 p, 不变的概率为(1-p） - 如果输入图像为1个通道: 则灰度版本是 1 通道 - 如果输入图像为3个通道: 则灰度版本是 3 通道, 并且 r == g == b

返回类型：`PIL Image`

```py
class torchvision.transforms.FiveCrop(size)
```

将给定的 PIL Image 裁剪成四个角落和中心裁剪

注解：

该变换返回一个图像元组, 并且数据集返回的输入和目标的数量可能不匹配. 请参阅下面的例子来处理这个问题.

参数：`size (sequence 或 int)` – 期望输出的裁剪尺寸. 如果 `size` 是 `int` 类型的整数, 而不是像 `(h, w)` 这样类型的序列, 裁剪出来的图像是 `(size, size)` 这样的正方形的..


示例：

```py
>>> transform = Compose([
>>>    FiveCrop(size), # 一个 PIL Images 的列表
>>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # 返回一个4D Tensor
>>> ])
>>> #在你的测试循环可以如下操作:
>>> input, target = batch # 输入是5DTensor,输出是2D
>>> bs, ncrops, c, h, w = input.size()
>>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
>>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops

```

```py
class torchvision.transforms.TenCrop(size, vertical_flip=False)
```

将给定的 PIL Image 裁剪成四个角, 中心裁剪, 并加上这些的翻转版本(默认使用水平翻转）

注解：

该变换返回一个图像元组, 并且数据集返回的输入和目标的数量可能不匹配. 请参阅下面的例子来处理这个问题.

参数：

*   `size (sequence 或 int)` – 期望输出的裁剪尺寸. 如果 size(尺寸）是 `int` 类型的整数, 而不是像 (h, w) 这样类型的序列, 裁剪出来的图像是 (size, size) 这样的正方形的.
*   `vertical_flip (bool)` – 使用垂直翻转而不是水平的方式



示例：

```py
>>> transform = Compose([
>>>    TenCrop(size), # this is a list of PIL Images
>>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
>>> ])
>>> #In your test loop you can do the following:
>>> input, target = batch # input is a 5d tensor, target is 2d
>>> bs, ncrops, c, h, w = input.size()
>>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
>>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops

```

```py
class torchvision.transforms.Pad(padding, fill=0)
```

用指定的 “pad” 值填充指定的 PIL image.

参数：

*   `padding (int 或 tuple)` – 填充每个边框. 如果提供了一个 int 型的整数, 则用于填充所有边界. 如果提供长度为 2 的元组, 则这是分别在 左/右 和 上/下 的填充. 如果提供长度为 4 的元组, 则这是分别用于 左, 上, 右 和 下 部边界的填充.
*   `fill` – 像素填充. 默认值为 0\. 如果长度为 3 的元组, 分别用于填充 R, G, B 通道.



```py
class torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
```

随机更改图像的亮度, 对比度和饱和度.

参数：

*   `brightness (float)` – 亮度改变的范围. brightness_factor 从 [max(0, 1 - brightness), 1 + brightness]的范围中一致选择.
*   `contrast (float)` – 对比度改变的范围. contrast_factor 从 [max(0, 1 - contrast), 1 + contrast]的范围中一致选择.
*   `saturation (float)` – 饱和度改变的范围. saturation_factor 从[max(0, 1 - saturation), 1 + saturation]的范围中一致选择.
*   `hue (float)` – 色调改变的范围. hue_factor 从 [-hue, hue]的范围中一致选择. 应该 &gt;=0 且 &lt;= 0.5.



## torch.*Tensor 上的变换

```py
class torchvision.transforms.Normalize(mean, std)
```

用均值和标准偏差对张量图像进行归一化. 给定均值: `(M1,...,Mn)` 和标准差: `(S1,..,Sn)` 用于 `n` 个通道, 该变换将标准化输入 `torch.*Tensor` 的每一个通道. 例如: `input[channel] = (input[channel] - mean[channel]) / std[channel]`

参数：

*   `mean (sequence)` – 每一个通道的均值序列.
*   `std (sequence)` – 每一个通道的标准差序列.



```py
__call__(tensor)
```

参数：`tensor (Tensor)` – 需要被归一化的大小为 (C, H, W)Tensor image.

返回值：归一化后的 Tensor image.

返回类型：`Tensor`

## 转换类型的变换

```py
class torchvision.transforms.ToTensor
```

转换一个 `PIL Image` 或 `numpy.ndarray` 为 tensor(张量）.

将范围 [0, 255] 中的 PIL Image 或 numpy.ndarray (H x W x C) 转换形状为 (C x H x W) , 值范围为 [0.0, 1.0] 的 torch.FloatTensor.

```py
__call__(pic)
```

参数：`pic (PIL Image 或 numpy.ndarray)` – 将要被转换为 tensor 的 Image.

返回值：转换后的 image.

返回类型：`Tensor`

```py
class torchvision.transforms.ToPILImage(mode=None)
```

转换一个 tensor 或 ndarray 为 PIL Image.

转换一个形状为(C x H x W) 的 torch.*Tensor 或一个形状为(H x W x C )的numpy ndarray 至一个 PIL Image ,同时保留值范围.

参数：`mode (PIL.Image 模式)` – 输入数据的色域和像素深度 (可选). 如果 `mode` 为 `None` (默认) ,这里对输入数据有一些假设: 1\. 如果输入有3个通道, `mode` 假设为 `RGB`. 2\. 如果输入有4个通道, `mode` 假设为 `RGBA`. 3\. 如果输入有1个通道, `mode` 根据数据类型确定 (i,e, `int`, `float`, `short`).


```py
__call__(pic)
```

参数：`pic (Tensor 或 numpy.ndarray)` – 要转换为PIL Image的图像.

返回值：转换为PIL Image的图像.

返回类型：`PIL Image`

## 通用的变换

```py
class torchvision.transforms.Lambda(lambd)
```

应用一个用户定义的 Lambda 作为变换.

参数：`lambd (function)` – Lambda/function 以用于 transform.
