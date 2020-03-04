# pytorch torchvision transform

## 对PIL.Image进行变换
### class torchvision.transforms.Compose(transforms)
将多个`transform`组合起来使用。

`transforms`： 由`transform`构成的列表.
例子：
```python
transforms.Compose([
     transforms.CenterCrop(10),
     transforms.ToTensor(),
 ])
 ```


### class torchvision.transforms.Scale(size, interpolation=2)

将输入的`PIL.Image`重新改变大小成给定的`size`，`size`是最小边的边长。举个例子，如果原图的`height>width`,那么改变大小后的图片大小是`(size*height/width, size)`。
**用例:**
```python
from torchvision import transforms
from PIL import Image
crop = transforms.Scale(12)
img = Image.open('test.jpg')

print(type(img))
print(img.size)

croped_img=crop(img)
print(type(croped_img))
print(croped_img.size)
```
```
<class 'PIL.PngImagePlugin.PngImageFile'>
(10, 10)
<class 'PIL.Image.Image'>
(12, 12)
```

### class torchvision.transforms.CenterCrop(size)
将给定的`PIL.Image`进行中心切割，得到给定的`size`，`size`可以是`tuple`，`(target_height, target_width)`。`size`也可以是一个`Integer`，在这种情况下，切出来的图片的形状是正方形。

### class torchvision.transforms.RandomCrop(size, padding=0)
切割中心点的位置随机选取。`size`可以是`tuple`也可以是`Integer`。

### class torchvision.transforms.RandomHorizontalFlip
随机水平翻转给定的`PIL.Image`,概率为`0.5`。即：一半的概率翻转，一半的概率不翻转。

### class torchvision.transforms.RandomSizedCrop(size, interpolation=2)
先将给定的`PIL.Image`随机切，然后再`resize`成给定的`size`大小。
### class torchvision.transforms.Pad(padding, fill=0)
将给定的`PIL.Image`的所有边用给定的`pad value`填充。
`padding：`要填充多少像素
`fill：`用什么值填充
例子：
```python
from torchvision import transforms
from PIL import Image
padding_img = transforms.Pad(padding=10, fill=0)
img = Image.open('test.jpg')

print(type(img))
print(img.size)

padded_img=padding(img)
print(type(padded_img))
print(padded_img.size)
```
```
<class 'PIL.PngImagePlugin.PngImageFile'>
(10, 10)
<class 'PIL.Image.Image'>
(30, 30) #由于上下左右都要填充10个像素，所以填充后的size是(30,30)
```

## 对Tensor进行变换
### class torchvision.transforms.Normalize(mean, std)
给定均值：`(R,G,B)` 方差：`(R，G，B）`，将会把`Tensor`正则化。即：`Normalized_image=(image-mean)/std`。

## Conversion Transforms

### class torchvision.transforms.ToTensor
把一个取值范围是`[0,255]`的`PIL.Image`或者`shape`为`(H,W,C)`的`numpy.ndarray`，转换成形状为`[C,H,W]`，取值范围是`[0,1.0]`的`torch.FloadTensor`
```python
data = np.random.randint(0, 255, size=300)
img = data.reshape(10,10,3)
print(img.shape)
img_tensor = transforms.ToTensor()(img) # 转换成tensor
print(img_tensor)
```

### class torchvision.transforms.ToPILImage
将`shape`为`(C,H,W)`的`Tensor`或`shape`为`(H,W,C)`的`numpy.ndarray`转换成`PIL.Image`，值不变。

## 通用变换
### class torchvision.transforms.Lambda(lambd)
使用`lambd`作为转换器。
