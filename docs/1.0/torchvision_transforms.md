

# torchvision.transforms

> 译者：[BXuan694](https://github.com/BXuan694)

transforms包含了一些常用的图像变换，这些变换能够用[`Compose`](#torchvision.transforms.Compose "torchvision.transforms.Compose")串联组合起来。另外，torchvision提供了[`torchvision.transforms.functional`](#module-torchvision.transforms.functional "torchvision.transforms.functional")模块。functional可以提供了一些更加精细的变换，用于搭建复杂的变换流水线(例如分割任务）。

```py
class torchvision.transforms.Compose(transforms)
```

用于把一系列变换组合到一起。

参数：

*   **transforms**(list或`Transform`对象）- 一系列需要进行组合的变换。

示例：

```py
>>> transforms.Compose([
>>>     transforms.CenterCrop(10),
>>>     transforms.ToTensor(),
>>> ])
```

## 对PIL图片的变换

```py
class torchvision.transforms.CenterCrop(size)
```

在中心处裁剪PIL图片。

 
参数：

*  **size**(_序列_ _或_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")）– 需要裁剪出的形状。如果size是int，将会裁剪成正方形；如果是形如(h, w)的序列，将会裁剪成矩形。 

```py
class torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
```

随机改变图片的亮度、对比度和饱和度。

 
参数： 

*   **brightness**([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_或_ _float类型元组(min, max)_）– 亮度的扰动幅度。brightness_factor从[max(0, 1 - brightness), 1 + brightness]中随机采样产生。应当是非负数。
*   **contrast**([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_或_ _float类型元组(min, max)_）– 对比度扰动幅度。contrast_factor从[max(0, 1 - contrast), 1 + contrast]中随机采样产生。应当是非负数。
*   **saturation**([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_或_ _float类型元组(min, max)_）– 饱和度扰动幅度。saturation_factor从[max(0, 1 - saturation), 1 + saturation]中随机采样产生。应当是非负数。
*   **hue**([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_或_ _float类型元组(min, max)_）– 色相扰动幅度。hue_factor从[-hue, hue]中随机采样产生，其值应当满足0<= hue <= 0.5或-0.5 <= min <= max <= 0.5


```py
class torchvision.transforms.FiveCrop(size)
```

从四角和中心裁剪PIL图片。

注意：

该变换返回图像元组，可能会导致图片在网络中传导后和数据集给出的标签等信息不能匹配。处理方法见下面的示例。

参数：

*   **size**(_序列_ _或_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")）– 需要裁剪出的形状。如果size是int，将会裁剪成正方形；如果是序列，如(h, w)，将会裁剪成矩形。 

示例：

```py
>>> transform = Compose([
>>>    FiveCrop(size), # 这里产生了一个PIL图像列表
>>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # 返回4维张量
>>> ])
>>> # 在测试阶段你可以这样做：
>>> input, target = batch # input是5维张量，target是2维张量。
>>> bs, ncrops, c, h, w = input.size()
>>> result = model(input.view(-1, c, h, w)) # 把batch size和ncrops融合在一起
>>> result_avg = result.view(bs, ncrops, -1).mean(1) # crops上的平均值

```

```py
class torchvision.transforms.Grayscale(num_output_channels=1)
```

把图片转换为灰阶。

参数：

*   **num_output_channels**([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")，1或3）– 希望得到的图片通道数。

返回：
*   输入图片的灰阶版本。 - 如果num_output_channels == 1：返回单通道图像；- 如果num_output_channels == 3：返回3通道图像，其中r == g == b。

返回类型：
*   PIL图像。

```py
class torchvision.transforms.Pad(padding, fill=0, padding_mode='constant')
```

对PIL图像的各条边缘进行扩展。

 
参数： 

*   **padding**([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _或_ [_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")）– 在每条边上展开的宽度。如果传入的是单个int，就在所有边展开。如果传入长为2的元组，则指定左右和上下的展开宽度。如果传入长为4的元组，则依次指定为左、上、右、下的展开宽度。
*   **fill**([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _或_ [_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")） – 像素填充值。默认是0。如果指定长度为3的元组，表示分别填充R, G, B通道。这个参数仅在padding_mode是‘constant’时指定有效。
*  **padding_mode**([_str_](https://docs.python.org/3/library/functions.html#func-str "(in Python v3.7)")）– 展开类型。应当是‘constant’，‘edge’，‘reflect’或‘symmetric’之一。默认为‘constant’。
   * constant：用常数扩展，这个值由fill参数指定。
   * edge：用图像边缘上的指填充。
   * reflect：以边缘为对称轴进行轴对称填充(边缘值不重复）。
   &gt; 例如，在[1, 2, 3, 4]的两边填充2个元素会得到[3, 2, 1, 2, 3, 4, 3, 2]。
   * symmetric：用图像边缘的反转进行填充(图像的边缘值需要重复）。
   &gt; 例如，在[1, 2, 3, 4]的两边填充2个元素会得到[2, 1, 1, 2, 3, 4, 4, 3]。



```py
class torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0)
```

保持像素的分布中心不变，对图像做随机仿射变换。

 
参数： 

*   **degrees**(_序列_ _或_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _或_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")）– 旋转角度的筛选范围。如果是序列(min, max），从中随机均匀采样；如果是数字，则从(-degrees, +degrees）中采样。如果不需要旋转，那么设置为0。
*   **translate**([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _可选_）– 元组，元素值是水平和垂直平移变换的最大绝对值。例如translate=(a, b)时，水平位移值从 -img_width * a &lt; dx &lt; img_width * a中随机采样得到，垂直位移值从-img_height * b &lt; dy &lt; img_height * b中随机采样得到。默认不做平移变换。
*   **scale**([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _可选_）– 尺度放缩因子的内区间，如[a, b]，放缩因子scale的随机采样区间为：a &lt;= scale &lt;= b。默认不进行尺度放缩变换。
*   **shear**(_序列_ _或_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _或_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选_）– 扭曲角度的筛选范围。如果是序列(min, max），从中随机均匀采样；如果是数字，则从(-degrees, +degrees）中采样。默认不会进行扭曲操作。
*   **resample**(_{PIL.Image.NEAREST_ _,_ _PIL.Image.BILINEAR_ _,_ _PIL.Image.BICUBIC}_ _,_ _可选_）– 可选的重采样滤波器，见[filters](http://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters)。如果没有该选项，或者图片模式是“1”或“P”，设置为PIL.Image.NEAREST。
*   **fillcolor**([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")）– 在输出图片的变换外区域可选地填充颜色。(Pillow&gt;=5.0.0）。


```py
class torchvision.transforms.RandomApply(transforms, p=0.5)
```

对transforms中的各变换以指定的概率决定是否选择。
 
参数： 

*   **transforms**([_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)") _or_ [_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")）– 变换的集合。
*   **p**([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")）– 概率。

```py
class torchvision.transforms.RandomChoice(transforms)
```

从列表中随机选择一种变换。

```py
class torchvision.transforms.RandomCrop(size, padding=0, pad_if_needed=False)
```

对给出的PIL图片在随机位置处进行裁剪。

 
参数： 

*   **size**(_序列_ _或_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")）– 想要裁剪出的图片的形状。如果size是int，按照正方形(size, size）裁剪； 如果size是序列(h, w），裁剪为矩形。
*   **padding**([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _或_ _序列_ _,_ _可选_）– 在图像的边缘进行填充，默认0，即不做填充。如果指定长为4的序列，则分别指定左、上、右、下的填充宽度。
*   **pad_if_needed**(_boolean_）– 如果设置为True，若图片小于目标形状，将进行填充以避免报异常。
*   **fill**([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _或_ [_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")） – 像素填充值。默认是0。如果指定长度为3的元组，表示分别填充R, G, B通道。这个参数仅在padding_mode是‘constant’时指定有效。
*  **padding_mode**([_str_](https://docs.python.org/3/library/functions.html#func-str "(in Python v3.7)")）– 展开类型。应当是‘constant’，‘edge’，‘reflect’或‘symmetric’之一。默认为‘constant’。
   * constant：用常数扩展，这个值由fill参数指定。
   * edge：用图像边缘上的指填充。
   * reflect：以边缘为对称轴进行轴对称填充(边缘值不重复）。
   &gt; 例如，在[1, 2, 3, 4]的两边填充2个元素会得到[3, 2, 1, 2, 3, 4, 3, 2]。
   * symmetric：用图像边缘的反转进行填充(图像的边缘值需要重复）。
   &gt; 例如，在[1, 2, 3, 4]的两边填充2个元素会得到[2, 1, 1, 2, 3, 4, 4, 3]。


```py
class torchvision.transforms.RandomGrayscale(p=0.1)
```

以概率p(默认0.1）将图片随机转化为灰阶图片。

参数：

*   **p**([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")）–图像转化为灰阶的概率。

返回：

*   以概率p转换为灰阶，以概率(1-p）不做变换。如果输入图像为1通道，则灰阶版本也是1通道。如果输入图像为3通道，则灰阶版本是3通道，r == g == b。

返回类型：

*   PIL图像。


```py
class torchvision.transforms.RandomHorizontalFlip(p=0.5)
```

以给定的概率随机水平翻折PIL图片。

参数：

*   **p**([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")）– 翻折图片的概率。默认0.5。

```py
class torchvision.transforms.RandomOrder(transforms)
```

以随机的顺序对图片做变换。

```py
class torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
```

以随机的形状和长宽比裁剪图片。

以随机的形状(默认从原始图片的0.08到1.0) 和随机长宽比(默认从3/4到4/3）裁剪图片。然后调整到指定形状。这一变换通常用于训练Inception网络。

 
参数： 

*   **size** – 每条边的期望输出形状。
*   **scale** – 裁剪原始图片出的形状范围。
*   **ratio** – 原始长宽比裁剪出的目标范围。
*   **interpolation** – 默认：PIL.Image.BILINEAR。



```py
class torchvision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None)
```

以指定的角度选装图片。

 
参数： 

*   **degrees**(_序列_ _或_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")）– 旋转角度的随机选取范围。如果degrees是序列(min, max），则从中随机选取；如果是数字，则选择范围是(-degrees, +degrees）。
*   **resample**(_{PIL.Image.NEAREST_ _,_ _PIL.Image.BILINEAR_ _,_ _PIL.Image.BICUBIC}_ _,_ _可选_) – 可选的重采样滤波器，见[filters](http://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters)。如果该选项忽略，或图片模式是“1”或者“P”则设置为PIL.Image.NEAREST。
*   **expand**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_）– 可选的扩展标志。如果设置为True, 将输出扩展到足够大从而能容纳全图。如果设置为False或不设置，输出图片将和输入同样大。注意expand标志要求 flag assumes rotation around the center and no translation。
*   **center**(_2-tuple_ _,_ _可选_）– 可选的旋转中心坐标。以左上角为原点计算。默认是图像中心。



```py
class torchvision.transforms.RandomSizedCrop(*args, **kwargs)
```

注意：该变换已被弃用，可用RandomResizedCrop代替。

```py
class torchvision.transforms.RandomVerticalFlip(p=0.5)
```

以给定的概率随机垂直翻折PIL图片。

参数：

*   **p** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – 翻折图片的概率。默认0.5。

```py
class torchvision.transforms.Resize(size, interpolation=2)
```

将输入PIL图片调整大小到给定形状。

 
参数： 

*   **size**(_序列_ _或_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")）– 期望输出形状。如果size形如(h, w），输出就以该形状。 如果size是int更短的边将调整为int，即如果高&gt;宽，那么图片将调整为(size * 高 / 宽，size）。
*   **interpolation**([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选_）– 插值方式。默认采用`PIL.Image.BILINEAR`。



```py
class torchvision.transforms.Scale(*args, **kwargs)
```

注意：该变换已被弃用，可用Resize代替。

```py
class torchvision.transforms.TenCrop(size, vertical_flip=False)
```

将PIL图片以四角和中心裁剪，同时加入翻折版本。(默认以水平的方式翻折）

注意：

该变换返回图像元组，可能会导致图片在网络中传导后和数据集给出的标签等信息不能匹配。处理方法见下面的示例。

 
参数： 

*   **size**(_序列_ _或_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")）– 期望裁剪输出的形状。需要裁剪出的形状。如果size是int，将会裁剪成正方形；如果是序列，如(h, w)，将会裁剪成矩形。
*   **vertical_flip**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")）– 是否用垂直翻折。

示例：

```py
>>> transform = Compose([
>>>    TenCrop(size), # 这里产生了一个PIL图像列表
>>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) 返回4维张量
>>> ])
>>> # 在测试阶段你可以这样做：
>>> input, target = batch # input是5维张量, target是2维张量
>>> bs, ncrops, c, h, w = input.size()
>>> result = model(input.view(-1, c, h, w)) # 把batch size和ncrops融合在一起
>>> result_avg = result.view(bs, ncrops, -1).mean(1) # crops上的平均值

```

## torch.*Tensor上的变换

```py
class torchvision.transforms.LinearTransformation(transformation_matrix)
```

用一个预先准备好的变换方阵对图片张量做变换。

torch.*Tensor会被transformation_matrix拉平，和变换矩阵做点积后调整到原始张量的形状。

应用：

- 白化：将数据的分布中心处理到0，计算数据的协方差矩阵。
   - 用np.dot(X.T, X)可以处理到[D x D]的形状，对此做奇异值分解然后传给transformation_matrix即可。

 
参数：

*   **transformation_matrix**([_Tensor_](../tensors.html#torch.Tensor "torch.Tensor")）– [D x D]的张量，D = C x H x W。

```py
class torchvision.transforms.Normalize(mean, std)
```

用平均值和标准差标准化输入图片。给定`n`个通道的平均值`(M1,...,Mn)`和标准差`(S1,..,Sn)`，这一变换会在`torch.*Tensor`的每一个通道上进行标准化，即`input[channel] = (input[channel] - mean[channel]) / std[channel]`。


参数： 

*   **mean**(_序列_）– 序列，包含各通道的平均值。
*   **std**(_序列_）– 序列，包含各通道的标准差。

```py
__call__(tensor)
```
 
参数：

*   **tensor** ([_Tensor_](../tensors.html#torch.Tensor "torch.Tensor")) – 需要标准化的图像Tensor，形状须为(C, H, W)。

返回：

*   标准化之后的图片Tensor

返回类型：

*   [Tensor](../tensors.html#torch.Tensor "torch.Tensor")。

## 格式变换

```py
class torchvision.transforms.ToPILImage(mode=None)
```

把张量或ndarray转化为PIL图像。

把形状为C x H x W的torch.*Tensor或者形状为H x W x C的numpy矩阵ndarray转化为PIL图像，保留值的上下界。

参数 ：

*   **mode** ([PIL.Image mode](http://pillow.readthedocs.io/en/latest/handbook/concepts.html#modes)) – 输入数据的颜色空间或者像素深度(可选）。 如果`mode`设置为`None`(默认），按照下面的规则进行处理：
1. 如果输入3通道，`mode`会设置为`RGB`。
2. 如果输入4通道，`mode`会设置为`RGBA`。
3. 如果输入1通道，`mode`由数据类型决定(即`int`，`float`，`short`)。

```py
__call__(pic)
```

参数：

*   **pic** ([_Tensor_](../tensors.html#torch.Tensor "torch.Tensor") _或_ [_numpy.ndarray_](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v1.15)")）– 要转化成PIL图像的图片。

返回：

*   转化后的PIL图像。

返回类型：

*   PIL图像。

```py
class torchvision.transforms.ToTensor
```

将`PIL Image`或`numpy.ndarray`转化成张量。

把PIL图像或[0, 255]范围内的numpy.ndarray(形状(H x W x C)）转化成torch.FloatTensor，张量形状(C x H x W)，范围在[0.0, 1.0]中。输入应是是PIL图像且是模式(L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1）中的一种，或输入是numpy.ndarray且类型为np.uint8。

```py
__call__(pic)
```

参数：
*   **pic** (_PIL图像_ _或_ [_numpy.ndarray_](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v1.15)")) – 要转化成张量的图片。

返回：

*   转化后的图像。

返回类型：

*   [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 。

## 通用变换

```py
class torchvision.transforms.Lambda(lambd)
```

将用户定义的函数用作变换。

参数：

*   **lambd** (_函数_) – 用于变换的Lambda函数或函数名。

## Functional变换

functional可以提供了一些更加精细的变换，用于搭建复杂的变换流水线。和前面的变换相反，函数变换的参数不包含随机数种子生成器。这意味着你必须指定所有参数的值，但是你可以自己引入随机数。比如，对一组图片使用如下的functional变换：

```py
import torchvision.transforms.functional as TF
import random

def my_segmentation_transforms(image, segmentation):
    if random.random() > 5:
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle)
        segmentation = TF.rotate(segmentation, angle)
    # 更多变换 ...
    return image, segmentation

```

```py
torchvision.transforms.functional.adjust_brightness(img, brightness_factor)
```

调整图像亮度。
 
参数： 

*   **img**(_PIL图像_）– 要调整的PIL图像。
*   **brightness_factor**([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")）– 亮度的调整值，可以是任意非负整数。0表示黑色图像，1表示原始图像，2表示增加到2倍亮度。

返回：

*   调整亮度后的图像。

返回类型：

*   PIL图像。

```py
torchvision.transforms.functional.adjust_contrast(img, contrast_factor)
```

调整图像对比度。
 
参数： 

*   **img**(_PIL图像_）– 要调整的PIL图像。
*   **contrast_factor**([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")）– 对比度的调整幅度，可以是任意非负数。0表示灰阶图片，1表示原始图片，2表示对比度增加到2倍。


返回：

*   调整对比度之后的图像。

返回类型：

*   PIL图像。

```py
torchvision.transforms.functional.adjust_gamma(img, gamma, gain=1)
```

对图像进行伽马矫正。

又称幂率变换。RGB模式下的强度按照下面的等式进行调整：

$$I_{out} = 255 \times gain \times (\dfrac{I_{in}}{255})^\gamma$$

更多细节见[伽马矫正](https://en.wikipedia.org/wiki/Gamma_correction)。

 
参数： 

*   **img**(_PIL图像_）– PIL要调整的PIL图像。
*   **gamma**([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")）– 非负实数，公式中的 $$\gamma$$。gamma大于1时令暗区更暗，gamma小于1时使得暗区更亮。
*   **gain**([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")）– 常数乘数。


```py
torchvision.transforms.functional.adjust_hue(img, hue_factor)
```

调整图像色相。

调整时，先把图像转换到HSV空间，然后沿着色相轴(H轴）循环移动。最后切换回图像原始模式。

`hue_factor`是H通道的偏移量，必须在`[-0.5, 0.5]`的范围内。

更多细节见[色相](https://en.wikipedia.org/wiki/Hue)。

 
参数： 

*   **img**(_PIL图像_）– 要调整的PIL图像。
*   **hue_factor**([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")）– H通道的偏移量应该在[-0.5, 0.5]的范围内。0.5和-0.5分别表示在HSV空间的H轴上沿正、负方向进行移动，0表示不偏移。因此，-0.5和0.5都能表示补色，0表示原图。


返回：

*   色相调整后的图像。

返回类型：

*   PIL图像。

```py
torchvision.transforms.functional.adjust_saturation(img, saturation_factor)
```

调整图像的颜色饱和度。

 
参数： 

*   **img**(_PIL图像_）– 要调整的PIL图像。
*   **saturation_factor**([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")）– 饱和度调整值。0表示纯黑白图像，1表示原始图像，2表示增加到原来的2倍。

返回：

*   调整饱和度之后的图像。

返回类型：

*   PIL图像。

```py
torchvision.transforms.functional.affine(img, angle, translate, scale, shear, resample=0, fillcolor=None)
```
保持图片像素分布中心不变，进行仿射变换。

参数： 

*   **img**(_PIL图像_）– 要旋转的PIL图像。
*   **angle**(_{python:float_ _或_ _int}_）– 旋转角度，应在时钟方向的-180到180度之间。
*   **translate**([_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)") _或_ _整形数元组_）– 水平和垂直变换(旋转之后）
*   **scale**([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")）– 尺度变换。
*   **shear**([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")）– 扭曲角度，应在时钟方向的-180到180度之间。
*   **resample**(_`PIL.Image.NEAREST`_ _或_ _`PIL.Image.BILINEAR`_ _或_ _`PIL.Image.BICUBIC`_ _,_ _可选_）– 可选的重采样滤波器，见[滤波器](http://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters)。如果不设置该选项，或者图像模式是“1”或“P”，设置为`PIL.Image.NEAREST`。
*   **fillcolor**([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")）– 可选在输出图片的变换外区域可选地填充颜色。(Pillow&gt;=5.0.0）


```py
torchvision.transforms.functional.crop(img, i, j, h, w)
```

裁剪指定PIL图像。

 
参数： 

*   **img**(_PIL图像_）– 要裁剪的图像。
*   **i** – 最上侧像素的坐标。
*   **j** – 最左侧像素的坐标。
*   **h** – 要裁剪出的高度。
*   **w** – 要裁剪出的宽度。


返回：

*   裁剪出的图像。

返回类型：

*   PIL图像。

```py
torchvision.transforms.functional.five_crop(img, size)
```

在四角和中心处裁剪图片。

注意：

该变换返回图像元组，可能会导致图片在网络中传导后和你的`Dataset`给出的标签等信息不能匹配。

 
参数：

*   **size**(_序列_ _或_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")）– 希望得到的裁剪输出。如果size是序列(h, w)，输出矩形；如果是int ，输出形状为(size, size)的正方形。

返回：

*   元组(tl, tr, bl, br, center)，分别表示左上、右上、左下、右下。

返回类型：

*   [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") 

```py
torchvision.transforms.functional.hflip(img)
```

将指定图像水平翻折。

 
参数：

*   **img**(_PIL图像_）– 要翻折的图像。

返回：

*   水平翻折后的图像。

返回类型：

*   PIL图像。

```py
torchvision.transforms.functional.normalize(tensor, mean, std)
```

用均值和方差将图像标准化。

更多细节见[`Normalize`](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Normalize)。

 
参数： 

*   **tensor**([_Tensor_](../tensors.html#torch.Tensor "torch.Tensor")）– 需要标准化的图像Tensor，形状应是(C, H, W)。
*   **mean**(_序列_）– 各通道的均值。
*   **std**(_序列_）– 各通道的标准差。

返回：

*   标准化之后的图像Tensor。

返回类型：

*   [Tensor](../tensors.html#torch.Tensor "torch.Tensor")。

```py
torchvision.transforms.functional.pad(img, padding, fill=0, padding_mode='constant')
```

用指定的填充模式和填充值填充PIL图像。

 
参数： 

*   **img**(_PIL图像_）– 要填充的图像。
*   **padding**([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _或_ [_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")）– 各边的填充可宽度。如果指定为int，表示所有边都按照此宽度填充。如果指定为长为2的元组，表示左右和上下边的填充宽度。如果指定为长为4的元组，分别表示左、上、右、下的填充宽度。
*   **fill** – 要填充的像素值，默认是0。如果指定为长为3的元组，表示RGB三通道的填充值。这个选项仅在padding_mode是constant时有用。
*  **padding_mode** – 填充类型，应当为：constant，edge，reflect或symmetric。默认是constant。
   * constant：用常数填充，该常数值由fill指定。 
   * edge：用边上的值填充。
   * reflect： 以边为对称轴进行填充。(不重复边上的值）

      * 在reflect模式中，在两边分别用2个元素填充[1, 2, 3, 4]将会得到[3, 2, 1, 2, 3, 4, 3, 2]。
   * symmetric：以边为对称轴进行填充。(重复边上的值）

       * 在symmetric模式中，在两边分别用2个元素填充[1, 2, 3, 4]将会得到[2, 1, 1, 2, 3, 4, 4, 3]。


返回：

*   填充后的图像。

返回类型：

*   PIL图像

```py
torchvision.transforms.functional.resize(img, size, interpolation=2)
```

将原是PIL图像重新调整到指定形状。

 
参数： 

*   **img**(_PIL图像_）– 要调整形状的图像。
*   **size**(_序列_ _或_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")）– 输出图像的形状。如果size指定为序列(h, w)，输出矩形。如果size指定为int图片的短边将调整为这个数，长边按照相同的长宽比进行调整。即，如果高度&gt;宽度，则图片形状将调整为 $$(size\times\frac{高度}{宽度}, size)$$
*   **interpolation**([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选_）– 插值方式，默认是`PIL.Image.BILINEAR`。


返回：

*   调整大小之后的图片。

返回类型：

*   PIL图像。

```py
torchvision.transforms.functional.resized_crop(img, i, j, h, w, size, interpolation=2)
```

裁剪PIL并调整到目标形状。

注意：在[RandomResizedCrop](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomResizedCrop)被调用。
 
参数： 

*   **img**(_PIL图像_）– 要裁剪的图像。
*   **i** – 最上侧的像素坐标。
*   **j** – 最左侧的像素坐标。
*   **h** – 裁剪出的图像高度。
*   **w** – 裁剪出的图像宽度。
*   **size**(_序列_ _或_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")）– 要输出的图像形状，同`scale`。
*   **interpolation**([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选_）– 插值方式，默认是 `PIL.Image.BILINEAR`。


返回：

*   裁剪后的图片。

返回类型：

*   PIL图像。

```py
torchvision.transforms.functional.rotate(img, angle, resample=False, expand=False, center=None)
```

旋转图片。
 
参数： 

*   **img**(_PIL图像_）– 要旋转的PIL图像。
*   **angle**(_[float](https://docs.python.org/3/library/functions.html#float)_ _或_ _[int](https://docs.python.org/3/library/functions.html#int)}_）– 顺时针旋转角度。
*   **resample**(_`PIL.Image.NEAREST`_ _或_ _`PIL.Image.BILINEAR`_ _或_ _`PIL.Image.BICUBIC`_ _,_ _可选_） – 可选的重采样滤波器，见[滤波器](http://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters)。如果该选项不设置，或者图像模式是“1”或“P”，将被设置为PIL.Image.NEAREST。
*   **expand**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_）– 可选的扩展选项。如果设置为True，使输出足够大，从而包含了所有像素。如果设置为False或不设置，则输出应和输入形状相同。注意expand选项假定旋转中心是center且不做平移。
*   **center**(_2-tuple_ _,_ _可选_）– 可选的旋转中心。原点在左上角。默认以图片中心为旋转中心。

```py
torchvision.transforms.functional.ten_crop(img, size, vertical_flip=False)
```

将图片在四角和中心处裁剪，同时返回它们翻折后的图片。(默认水平翻折）

注意：

*   该变换返回图像元组，可能会导致图片在网络中传导后和你的`Dataset`给出的标签等信息不能匹配。

参数：

*   **size**(_序列_ _或_ _[int](https://docs.python.org/3/library/functions.html#int)_）- 裁剪后输出的形状。如果size是int，输出(size, size)的正方形；如果size是序列，输出矩形。
*   **vertical_flip**(_[bool](https://docs.python.org/3/library/functions.html#bool)_）- 使用垂直翻折。

返回：

* **元组(tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip）** - 对应的左上、右上、左下、右下、中心裁剪图片和水平翻折后的图片。

返回类型：

* 元组

```py
torchvision.transforms.functional.to_grayscale(img, num_output_channels=1)
```

将图像输出成灰阶版本。

参数：

*   **img**(_PIL图像_）– 要转化成灰阶图像的图片。

返回：

*   灰阶版本的图像。如果num_output_channels == 1：返回单通道图像；如果num_output_channels == 3：返回三通道图像，其中r == g == b。

返回类型：

*   PIL图像。

```py
torchvision.transforms.functional.to_pil_image(pic, mode=None)
```

将张量或ndarray转化为PIL图像。

更多细节见`ToPIlImage`。
 
参数： 

*   **pic**([_Tensor_](../tensors.html#torch.Tensor "torch.Tensor") _或_ [_numpy.ndarray_](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v1.15)")）– 要转化成PIL的图片。
*   **mode**([PIL.Image mode](http://pillow.readthedocs.io/en/latest/handbook/concepts.html#modes)）– 输入数据的色彩空间和像素深度。(可选）

 
返回：

*   要转化成PIL图像的数据。

返回类型：

*   PIL图像。

```py
torchvision.transforms.functional.to_tensor(pic)
```

将`PIL Image`或`numpy.ndarray`转化成张量。

更多细节见[`ToTensor`](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToPILImage)。

 参数：
 *   **pic** (_PIL图像_ _或_ [_numpy.ndarray_](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v1.15)")) – 要转化成张量的图片。
 
 返回：
 
 *   转化后的图片。
 
 返回类型：
 
 *   [Tensor](../tensors.html#torch.Tensor "torch.Tensor")。

```py
torchvision.transforms.functional.vflip(img)
```

垂直翻折PIL图像。

参数：

*   **img**(_PIL图像_）– 要翻折的图像。

返回：

*   垂直翻折后的图像。

返回类型：

*   PIL图像。