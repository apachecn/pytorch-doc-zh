# torch.nn.functional

> 译者：[hijkzzz](https://github.com/hijkzzz)

## 卷积函数

### conv1d

```py
torch.nn.functional.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) → Tensor
```

对由多个输入平面组成的输入信号进行一维卷积. 

有关详细信息和输出形状, 请参见[`Conv1d`](#torch.nn.Conv1d "torch.nn.Conv1d"). 

注意

在某些情况下, 当使用CUDA后端与CuDNN时, 该操作符可能会选择不确定性算法来提高性能. 如果这不是您希望的, 您可以通过设置`torch.backends.cudn .deterministic = True`来尝试使操作具有确定性(可能会以性能为代价). 请参阅关于 [Reproducibility](notes/randomness.html) 了解背景.

 
参数:

*   **input** – 输入tensor, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bminibatch%7D%20%5Ctimes%20%5Ctext%7Bin%5C_channels%7D%20%5Ctimes%20iW))
*   **weight** – 卷积核, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bout%5C_channels%7D%20%5Ctimes%20%5Cfrac%7B%5Ctext%7Bin%5C_channels%7D%7D%7B%5Ctext%7Bgroups%7D%7D%20%5Ctimes%20kW))
*   **bias** – 可选的偏置, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bout%5C_channels%7D)). 默认值: `None`
*   **stride** – 卷积核的步幅, 可以是单个数字或一个元素元组`(sW,)`. 默认值: 1
*   **padding** – 在输入的两边隐式加零. 可以是单个数字或一个元素元组`(padW, )`. 默认值:  0
*   **dilation** – 核元素之间的空洞. 可以是单个数字或单元素元组`(dW,)`. 默认值:  1
*   **groups** – 将输入分成组, ![](http://latex.codecogs.com/gif.latex?%5Ctext%7Bin%5C_channels%7D) 应该可以被组的数目整除. 默认值:  1

 


例子:

```py
>>> filters = torch.randn(33, 16, 3)
>>> inputs = torch.randn(20, 16, 50)
>>> F.conv1d(inputs, filters)

```

### conv2d

```py
torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) → Tensor
```

对由多个输入平面组成的输入图像应用二维卷积.

有关详细信息和输出形状, 请参见[`Conv2d`](#torch.nn.Conv2d "torch.nn.Conv2d").

注意

在某些情况下, 当使用CUDA后端与CuDNN时, 该操作符可能会选择不确定性算法来提高性能. 如果这不是您希望的, 您可以通过设置`torch.backends.cudn .deterministic = True`来尝试使操作具有确定性(可能会以性能为代价). 请参阅关于 [Reproducibility](notes/randomness.html) 了解背景.

 
参数:

*   **input** – 输入tensor, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bminibatch%7D%20%5Ctimes%20%5Ctext%7Bin%5C_channels%7D%20%5Ctimes%20iH%20%5Ctimes%20iW))
*   **weight** – 卷积核, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bout%5C_channels%7D%20%5Ctimes%20%5Cfrac%7B%5Ctext%7Bin%5C_channels%7D%7D%7B%5Ctext%7Bgroups%7D%7D%20%5Ctimes%20kH%20%5Ctimes%20kW))
*   **bias** – 可选的偏置, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bout%5C_channels%7D)). 默认值:  `None`
*   **stride** – 卷积核的步幅, 可以是单个数字或一个元素元组 `(sH, sW)`. 默认值:  1
*   **padding** – 在输入的两边隐式加零. 可以是单个数字或一个元素元组 `(padH, padW)`. 默认值:  0
*   **dilation** – 核元素之间的空洞. 可以是单个数字或单元素元组 `(dH, dW)`. 默认值:  1
*   **groups** – 将输入分成组, ![](http://latex.codecogs.com/gif.latex?%5Ctext%7Bin%5C_channels%7D) 应该可以被组的数目整除. 默认值:  1

 


例子:

```py
>>> # With square kernels and equal stride
>>> filters = torch.randn(8,4,3,3)
>>> inputs = torch.randn(1,4,5,5)
>>> F.conv2d(inputs, filters, padding=1)

```

### conv3d

```py
torch.nn.functional.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) → Tensor
```

对由多个输入平面组成的输入图像应用三维卷积.

有关详细信息和输出形状, 请参见 [`Conv3d`](#torch.nn.Conv3d "torch.nn.Conv3d").

注意

在某些情况下, 当使用CUDA后端与CuDNN时, 该操作符可能会选择不确定性算法来提高性能. 如果这不是您希望的, 您可以通过设置`torch.backends.cudn .deterministic = True`来尝试使操作具有确定性(可能会以性能为代价). 请参阅关于 [Reproducibility](notes/randomness.html) 了解背景.

 
参数:

*   **input** – 输入tensor, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bminibatch%7D%20%5Ctimes%20%5Ctext%7Bin%5C_channels%7D%20%5Ctimes%20iT%20%5Ctimes%20iH%20%5Ctimes%20iW))
*   **weight** – 卷积核, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bout%5C_channels%7D%20%5Ctimes%20%5Cfrac%7B%5Ctext%7Bin%5C_channels%7D%7D%7B%5Ctext%7Bgroups%7D%7D%20%5Ctimes%20kT%20%5Ctimes%20kH%20%5Ctimes%20kW))
*   **bias** – 可选的偏置, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bout%5C_channels%7D)). 默认值:  None
*   **stride** – 卷积核的步幅, 可以是单个数字或一个元素元组 `(sT, sH, sW)`. 默认值:  1
*   **padding** – 在输入的两边隐式加零. 可以是单个数字或一个元素元组 `(padT, padH, padW)`. 默认值:  0
*   **dilation** – 核元素之间的空洞. 可以是单个数字或单元素元组 `(dT, dH, dW)`. 默认值:  1
*   **groups** – 将输入分成组, ![](http://latex.codecogs.com/gif.latex?%5Ctext%7Bin%5C_channels%7D) 应该可以被组的数目整除. 默认值:  1

 


例子:

```py
>>> filters = torch.randn(33, 16, 3, 3, 3)
>>> inputs = torch.randn(20, 16, 50, 10, 20)
>>> F.conv3d(inputs, filters)

```

### conv_transpose1d

```py
torch.nn.functional.conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) → Tensor
```

对由多个输入平面组成的输入信号应用一维转置卷积算子, 有时也称为反卷积. 

有关详细信息和输出形状, 请参见 [`ConvTranspose1d`](#torch.nn.ConvTranspose1d "torch.nn.ConvTranspose1d") 

注意

在某些情况下, 当使用CUDA后端与CuDNN时, 该操作符可能会选择不确定性算法来提高性能. 如果这不是您希望的, 您可以通过设置`torch.backends.cudn .deterministic = True`来尝试使操作具有确定性(可能会以性能为代价). 请参阅关于 [Reproducibility](notes/randomness.html) 了解背景.

 
参数:

*   **input** – 输入tensor, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bminibatch%7D%20%5Ctimes%20%5Ctext%7Bin%5C_channels%7D%20%5Ctimes%20iW))
*   **weight** – 卷积核, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bin%5C_channels%7D%20%5Ctimes%20%5Cfrac%7B%5Ctext%7Bout%5C_channels%7D%7D%7B%5Ctext%7Bgroups%7D%7D%20%5Ctimes%20kW))
*   **bias** – 可选的偏置, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bout%5C_channels%7D)). 默认值:  None
*   **stride** – 卷积核的步幅, 可以是单个数字或一个元素元组 `(sW,)`. 默认值:  1
*   **padding** – 输入中的每个维度的两边都将添加零填充`kernel_size - 1 - padding`. 可以是单个数字或元组 `(padW,)`. 默认值:  0
*   **output_padding** – 添加到输出形状中每个维度的一侧的额外大小. 可以是单个数字或元组 `(out_padW)`. 默认值:  0
*   **groups** – 将输入分成组, ![](http://latex.codecogs.com/gif.latex?%5Ctext%7Bin%5C_channels%7D) 应该可以被组的数目整除. 默认值:  1
*   **dilation** – 核元素之间的空洞. 可以是单个数字或单元素元组 `(dW,)`. 默认值:  1

 


例子:

```py
>>> inputs = torch.randn(20, 16, 50)
>>> weights = torch.randn(16, 33, 5)
>>> F.conv_transpose1d(inputs, weights)

```

### conv_transpose2d

```py
torch.nn.functional.conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) → Tensor
```

对由多个输入平面组成的输入图像应用二维转置卷积算子, 有时也称为反卷积.


有关详细信息和输出形状, 请参见 [`ConvTranspose2d`](#torch.nn.ConvTranspose2d "torch.nn.ConvTranspose2d").

注意

在某些情况下, 当使用CUDA后端与CuDNN时, 该操作符可能会选择不确定性算法来提高性能. 如果这不是您希望的, 您可以通过设置`torch.backends.cudn .deterministic = True`来尝试使操作具有确定性(可能会以性能为代价). 请参阅关于 [Reproducibility](notes/randomness.html) 了解背景.

 
参数:

*   **input** – 输入tensor, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bminibatch%7D%20%5Ctimes%20%5Ctext%7Bin%5C_channels%7D%20%5Ctimes%20iH%20%5Ctimes%20iW))
*   **weight** – 卷积核, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bin%5C_channels%7D%20%5Ctimes%20%5Cfrac%7B%5Ctext%7Bout%5C_channels%7D%7D%7B%5Ctext%7Bgroups%7D%7D%20%5Ctimes%20kH%20%5Ctimes%20kW))
*   **bias** –可选的偏置, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bout%5C_channels%7D)). 默认值:  None
*   **stride** – 卷积核的步幅, 可以是单个数字或一个元素元组 `(sH, sW)`. 默认值:  1
*   **padding** – 输入中的每个维度的两边都将添加零填充`kernel_size - 1 - padding`. 可以是单个数字或元组 `(padH, padW)`. 默认值:  0
*   **output_padding** – 添加到输出形状中每个维度的一侧的额外大小. 可以是单个数字或元组 `(out_padH, out_padW)`. 默认值:  0
*   **groups** – 将输入分成组, ![](http://latex.codecogs.com/gif.latex?%5Ctext%7Bin%5C_channels%7D) 应该可以被组的数目整除. 默认值:  1
*   **dilation** – 核元素之间的空洞. 可以是单个数字或单元素元组 `(dH, dW)`. 默认值:  1

 


例子:

```py
>>> # With square kernels and equal stride
>>> inputs = torch.randn(1, 4, 5, 5)
>>> weights = torch.randn(4, 8, 3, 3)
>>> F.conv_transpose2d(inputs, weights, padding=1)

```

### conv_transpose3d

```py
torch.nn.functional.conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) → Tensor
```

对由多个输入平面组成的输入图像应用一个三维转置卷积算子, 有时也称为反卷积

有关详细信息和输出形状, 请参见 [`ConvTranspose3d`](#torch.nn.ConvTranspose3d "torch.nn.ConvTranspose3d").

注意

在某些情况下, 当使用CUDA后端与CuDNN时, 该操作符可能会选择不确定性算法来提高性能. 如果这不是您希望的, 您可以通过设置`torch.backends.cudn .deterministic = True`来尝试使操作具有确定性(可能会以性能为代价). 请参阅关于 [Reproducibility](notes/randomness.html) 了解背景.

 
参数:

*   **input** – 输入tensor, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bminibatch%7D%20%5Ctimes%20%5Ctext%7Bin%5C_channels%7D%20%5Ctimes%20iT%20%5Ctimes%20iH%20%5Ctimes%20iW))
*   **weight** – 卷积核, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bin%5C_channels%7D%20%5Ctimes%20%5Cfrac%7B%5Ctext%7Bout%5C_channels%7D%7D%7B%5Ctext%7Bgroups%7D%7D%20%5Ctimes%20kT%20%5Ctimes%20kH%20%5Ctimes%20kW))
*   **bias** –可选的偏置, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bout%5C_channels%7D)). 默认值:  None
*   **stride** – 卷积核的步幅, 可以是单个数字或一个元素元组 `(sT, sH, sW)`. 默认值:  1
*   **padding** – 输入中的每个维度的两边都将添加零填充`kernel_size - 1 - padding`. 可以是单个数字或元组 `(padT, padH, padW)`. 默认值:  0
*   **output_padding** – 添加到输出形状中每个维度的一侧的额外大小. 可以是单个数字或元组 `(out_padT, out_padH, out_padW)`. 默认值:  0
*   **groups** – 将输入分成组, ![](http://latex.codecogs.com/gif.latex?%5Ctext%7Bin%5C_channels%7D) 应该可以被组的数目整除. 默认值:  1
*   **dilation** – 核元素之间的空洞. 可以是单个数字或单元素元组 `(dT, dH, dW)`. 默认值:  1

 


例子:

```py
>>> inputs = torch.randn(20, 16, 50, 10, 20)
>>> weights = torch.randn(16, 33, 3, 3, 3)
>>> F.conv_transpose3d(inputs, weights)

```

### unfold

```py
torch.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)
```

从批量的`input`tensor中提取滑动局部块.

警告

目前, 仅支持 4-D input tensors (如批量的图像 tensors).

细节请参阅 [`torch.nn.Unfold`](#torch.nn.Unfold "torch.nn.Unfold")

### fold

```py
torch.nn.functional.fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)
```

将一组滑动局部块数组合成一个大的tensor.

警告

目前, 仅支持 4-D input tensors (如批量的图像 tensors).

细节请参阅 [`torch.nn.Fold`](#torch.nn.Fold "torch.nn.Fold") 

## 池化函数

### avg_pool1d

```py
torch.nn.functional.avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) → Tensor
```

对由多个输入平面组成的输入信号应用一维平均池化.

有关详细信息和输出形状, 请参见 [`AvgPool1d`](#torch.nn.AvgPool1d "torch.nn.AvgPool1d").

 
参数:

*   **input** – 输入tensor, 形状为 ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bminibatch%7D%20%5Ctimes%20%5Ctext%7Bin%5C_channels%7D%20%5Ctimes%20iW))
*   **kernel_size** – 窗口的大小. 可以是单个数字或元组 ![](http://latex.codecogs.com/gif.latex?(kW%2C))
*   **stride** – 窗户的步幅. 可以是单个数字或元组 `(sW,)`. 默认值:  `kernel_size`
*   **padding** – 在输入的两边隐式加零. 可以是单个数字或一个元素元组 `(padW,)`. 默认值:  0
*   **ceil_mode** – 如果 `True`, 将用 `ceil` 代替 `floor`计算输出形状. 默认值:  `False`
*   **count_include_pad** – 如果 `True`, 将在平均计算中包括零填充. 默认值:  `True`

 


```py
例子::
```

```py
>>> # pool of square window of size=3, stride=2
>>> input = torch.tensor([[[1,2,3,4,5,6,7]]])
>>> F.avg_pool1d(input, kernel_size=3, stride=2)
tensor([[[ 2.,  4.,  6.]]])

```

### avg_pool2d

```py
torch.nn.functional.avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) → Tensor
```

应用2D平均池化操作于 ![](http://latex.codecogs.com/gif.latex?kH%20%5Ctimes%20kW) 区域, 步幅为 ![](http://latex.codecogs.com/gif.latex?sH%20%5Ctimes%20sW) . 输出特征的数量等于输入平面的数量.

有关详细信息和输出形状, 请参见 [`AvgPool2d`](#torch.nn.AvgPool2d "torch.nn.AvgPool2d").


参数:

*   **input** – input tensor ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bminibatch%7D%20%5Ctimes%20%5Ctext%7Bin%5C_channels%7D%20%5Ctimes%20iH%20%5Ctimes%20iW))
*   **kernel_size** – 池化区域的大小, 可以是一个数字或者元组 ![](http://latex.codecogs.com/gif.latex?(kH%20%5Ctimes%20kW))
*   **stride** – 池化步幅, 可以是一个数字或者元组 `(sH, sW)`. 默认值:  `kernel_size`
*   **padding** – 在输入的两边隐式加零. 可以是单个数字或一个元素元组 `(padH, padW)`. 默认值:  0
*   **ceil_mode** – 如果 `True`, 将用 `ceil` 代替 `floor`计算输出形状. 默认值:  `False`
*   **count_include_pad** – 如果 `True`, 将在平均计算中包括零填充. 默认值:  `True`

 


### avg_pool3d

```py
torch.nn.functional.avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) → Tensor
```

应用3D平均池化操作于 ![](http://latex.codecogs.com/gif.latex?kT%20%5Ctimes%20kH%20%5Ctimes%20kW) 区域, 步幅为 ![](http://latex.codecogs.com/gif.latex?sT%20%5Ctimes%20sH%20%5Ctimes%20sW) . 输出特征的数量等于 ![](http://latex.codecogs.com/gif.latex?%5Clfloor%5Cfrac%7B%5Ctext%7Binput%20planes%7D%7D%7BsT%7D%5Crfloor).

有关详细信息和输出形状, 请参见  [`AvgPool3d`](#torch.nn.AvgPool3d "torch.nn.AvgPool3d").

 
参数:

*   **input** – input tensor ![](http://latex.codecogs.com/gif.latex?(%5Ctext%7Bminibatch%7D%20%5Ctimes%20%5Ctext%7Bin%5C_channels%7D%20%5Ctimes%20iT%20%5Ctimes%20iH%20%5Ctimes%20iW))
*   **kernel_size** – 池化区域的大小, 可以是一个数字或者元组 ![](http://latex.codecogs.com/gif.latex?(kT%20%5Ctimes%20kH%20%5Ctimes%20kW))
*   **stride** – 池化步幅, 可以是一个数字或者元组 `(sT, sH, sW)`. 默认值:  `kernel_size`
*   **padding** – 在输入的两边隐式加零. 可以是单个数字或一个元素元组 `(padT, padH, padW)`, 默认值:  0
*   **ceil_mode** – 如果 `True`, 将用 `ceil` 代替 `floor`计算输出形状. 默认值:  `False`
*   **count_include_pad** – 如果 `True`, 将在平均计算中包括零填充. 默认值:  `True`

 


### max_pool1d

```py
torch.nn.functional.max_pool1d(*args, **kwargs)
```

在由多个输入平面组成的输入信号上应用1D 最大池化.

详情见 [`MaxPool1d`](#torch.nn.MaxPool1d "torch.nn.MaxPool1d").

### max_pool2d

```py
torch.nn.functional.max_pool2d(*args, **kwargs)
```

在由多个输入平面组成的输入信号上应用2D 最大池化.

详情见 [`MaxPool2d`](#torch.nn.MaxPool2d "torch.nn.MaxPool2d").

### max_pool3d

```py
torch.nn.functional.max_pool3d(*args, **kwargs)
```

在由多个输入平面组成的输入信号上应用3D 最大池化.

详情见 [`MaxPool3d`](#torch.nn.MaxPool3d "torch.nn.MaxPool3d").

### max_unpool1d

```py
torch.nn.functional.max_unpool1d(input, indices, kernel_size, stride=None, padding=0, output_size=None)
```

计算逆 `MaxPool1d`.

请参见 [`MaxUnpool1d`](#torch.nn.MaxUnpool1d "torch.nn.MaxUnpool1d").

### max_unpool2d

```py
torch.nn.functional.max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None)
```

计算逆 `MaxPool2d`.

详情见 [`MaxUnpool2d`](#torch.nn.MaxUnpool2d "torch.nn.MaxUnpool2d").

### max_unpool3d

```py
torch.nn.functional.max_unpool3d(input, indices, kernel_size, stride=None, padding=0, output_size=None)
```

计算逆 `MaxPool3d`.

详情见 [`MaxUnpool3d`](#torch.nn.MaxUnpool3d "torch.nn.MaxUnpool3d").

### lp_pool1d

```py
torch.nn.functional.lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False)
```

在由多个输入平面组成的输入信号上应用1D幂平均池化. 如果所有输入的p次方的和为零, 梯度也为零. 

详情见 [`LPPool1d`](#torch.nn.LPPool1d "torch.nn.LPPool1d").

### lp_pool2d

```py
torch.nn.functional.lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False)
```

在由多个输入平面组成的输入信号上应用2D幂平均池化. 如果所有输入的p次方的和为零, 梯度也为零. 

详情见 [`LPPool2d`](#torch.nn.LPPool2d "torch.nn.LPPool2d").

### adaptive_max_pool1d

```py
torch.nn.functional.adaptive_max_pool1d(*args, **kwargs)
```

在由多个输入平面组成的输入信号上应用1D自适应最大池.

请参见 [`AdaptiveMaxPool1d`](#torch.nn.AdaptiveMaxPool1d "torch.nn.AdaptiveMaxPool1d")和输出形状.

 
参数:

*   **output_size** – 输出目标大小(单个整数)
*   **return_indices** – 是否返回池化索引. 默认值:  `False`

 


### adaptive_max_pool2d

```py
torch.nn.functional.adaptive_max_pool2d(*args, **kwargs)
```

在由多个输入平面组成的输入信号上应用2D自适应最大池.

请参见 [`AdaptiveMaxPool2d`](#torch.nn.AdaptiveMaxPool2d "torch.nn.AdaptiveMaxPool2d")  关于详情和输出形状.

 
参数:

*   **output_size** – 输出目标大小(单个整数 or 双整数元组)
*   **return_indices** – 是否返回池化索引. 默认值:  `False`

 


### adaptive_max_pool3d

```py
torch.nn.functional.adaptive_max_pool3d(*args, **kwargs)
```

在由多个输入平面组成的输入信号上应用3D自适应最大池.

请参见 [`AdaptiveMaxPool3d`](#torch.nn.AdaptiveMaxPool3d "torch.nn.AdaptiveMaxPool3d")和输出形状.

 
参数:

*   **output_size** – 输出目标大小(单个整数 或者 三整数元组)
*   **return_indices** – 是否返回池化索引. 默认值:  `False`

 


### adaptive_avg_pool1d

```py
torch.nn.functional.adaptive_avg_pool1d(input, output_size) → Tensor
```

在由多个输入平面组成的输入信号上应用1D自适应平均池化.

请参见 [`AdaptiveAvgPool1d`](#torch.nn.AdaptiveAvgPool1d "torch.nn.AdaptiveAvgPool1d")  关于详情和输出形状.

 
| 参数:| **output_size** – 输出目标大小(单个整数) 


### adaptive_avg_pool2d

```py
torch.nn.functional.adaptive_avg_pool2d(input, output_size)
```

在由多个输入平面组成的输入信号上应用2D自适应平均池化.

请参见 [`AdaptiveAvgPool2d`](#torch.nn.AdaptiveAvgPool2d "torch.nn.AdaptiveAvgPool2d")  关于详情和输出形状.

 
| 参数:| **output_size** – 输出目标大小(单个整数 或者双整数元组) 


### adaptive_avg_pool3d

```py
torch.nn.functional.adaptive_avg_pool3d(input, output_size)
```

在由多个输入平面组成的输入信号上应用3D自适应平均池化.

请参见 [`AdaptiveAvgPool3d`](#torch.nn.AdaptiveAvgPool3d "torch.nn.AdaptiveAvgPool3d")  关于详情和输出形状.

 
| 参数:| **output_size** – 输出目标大小(单个整数 或者三整数元组) 


## 非线性激活函数

### threshold

```py
torch.nn.functional.threshold(input, threshold, value, inplace=False)
```

为 input Tensor 的每个元素设置阈值.

请参见 [`Threshold`](#torch.nn.Threshold "torch.nn.Threshold").

```py
torch.nn.functional.threshold_(input, threshold, value) → Tensor
```

原地版本的 [`threshold()`](#torch.nn.functional.threshold "torch.nn.functional.threshold").

### relu

```py
torch.nn.functional.relu(input, inplace=False) → Tensor
```

逐元素应用整流线性单元函数. 请参见 [`ReLU`](#torch.nn.ReLU "torch.nn.ReLU").

```py
torch.nn.functional.relu_(input) → Tensor
```

原地版本的 [`relu()`](#torch.nn.functional.relu "torch.nn.functional.relu").

### hardtanh

```py
torch.nn.functional.hardtanh(input, min_val=-1., max_val=1., inplace=False) → Tensor
```

逐元素应用hardtanh函数. 请参见 [`Hardtanh`](#torch.nn.Hardtanh "torch.nn.Hardtanh").

```py
torch.nn.functional.hardtanh_(input, min_val=-1., max_val=1.) → Tensor
```

原地版本的 [`hardtanh()`](#torch.nn.functional.hardtanh "torch.nn.functional.hardtanh").

### relu6

```py
torch.nn.functional.relu6(input, inplace=False) → Tensor
```

逐元素应用函数 ![](http://latex.codecogs.com/gif.latex?%5Ctext%7BReLU6%7D(x)%20%3D%20%5Cmin(%5Cmax(0%2Cx)%2C%206)).

请参见 [`ReLU6`](#torch.nn.ReLU6 "torch.nn.ReLU6").

### elu

```py
torch.nn.functional.elu(input, alpha=1.0, inplace=False)
```

逐元素应用, ![](http://latex.codecogs.com/gif.latex?%5Ctext%7BELU%7D(x)%20%3D%20%5Cmax(0%2Cx)%20%2B%20%5Cmin(0%2C%20%5Calpha%20*%20(%5Cexp(x)%20-%201))).

请参见 [`ELU`](#torch.nn.ELU "torch.nn.ELU").

```py
torch.nn.functional.elu_(input, alpha=1.) → Tensor
```

原地版本的 [`elu()`](#torch.nn.functional.elu "torch.nn.functional.elu").

### selu

```py
torch.nn.functional.selu(input, inplace=False) → Tensor
```

逐元素应用, ![](http://latex.codecogs.com/gif.latex?%5Ctext%7BSELU%7D(x)%20%3D%20scale%20*%20(%5Cmax(0%2Cx)%20%2B%20%5Cmin(0%2C%20%5Calpha%20*%20(%5Cexp(x)%20-%201)))), with ![](http://latex.codecogs.com/gif.latex?%5Calpha%3D1.6732632423543772848170429916717) and ![](http://latex.codecogs.com/gif.latex?scale%3D1.0507009873554804934193349852946).

请参见 [`SELU`](#torch.nn.SELU "torch.nn.SELU").

### celu

```py
torch.nn.functional.celu(input, alpha=1., inplace=False) → Tensor
```

逐元素应用, ![](http://latex.codecogs.com/gif.latex?%5Ctext%7BCELU%7D(x)%20%3D%20%5Cmax(0%2Cx)%20%2B%20%5Cmin(0%2C%20%5Calpha%20*%20(%5Cexp(x%2F%5Calpha)%20-%201))).

请参见 [`CELU`](#torch.nn.CELU "torch.nn.CELU").

### leaky_relu

```py
torch.nn.functional.leaky_relu(input, negative_slope=0.01, inplace=False) → Tensor
```

逐元素应用, ![](http://latex.codecogs.com/gif.latex?%5Ctext%7BLeakyReLU%7D(x)%20%3D%20%5Cmax(0%2C%20x)%20%2B%20%5Ctext%7Bnegative%5C_slope%7D%20*%20%5Cmin(0%2C%20x))

请参见 [`LeakyReLU`](#torch.nn.LeakyReLU "torch.nn.LeakyReLU").

```py
torch.nn.functional.leaky_relu_(input, negative_slope=0.01) → Tensor
```

原地版本的 [`leaky_relu()`](#torch.nn.functional.leaky_relu "torch.nn.functional.leaky_relu").

### prelu

```py
torch.nn.functional.prelu(input, weight) → Tensor
```

逐元素应用函数 ![](http://latex.codecogs.com/gif.latex?%5Ctext%7BPReLU%7D(x)%20%3D%20%5Cmax(0%2Cx)%20%2B%20%5Ctext%7Bweight%7D%20*%20%5Cmin(0%2Cx)) where weight is a learnable parameter.

请参见 [`PReLU`](#torch.nn.PReLU "torch.nn.PReLU").

### rrelu

```py
torch.nn.functional.rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False) → Tensor
```

随机的 leaky ReLU.

请参见 [`RReLU`](#torch.nn.RReLU "torch.nn.RReLU").

```py
torch.nn.functional.rrelu_(input, lower=1./8, upper=1./3, training=False) → Tensor
```

原地版本的 [`rrelu()`](#torch.nn.functional.rrelu "torch.nn.functional.rrelu").

### glu

```py
torch.nn.functional.glu(input, dim=-1) → Tensor
```

门控线性单元. 计算:

![](http://latex.codecogs.com/gif.latex?%0D%0AH%20%3D%20A%20%5Ctimes%20%5Csigma(B))

其中`inpuy`沿`dim`分成两半, 形成`A`和`B`. 

见 [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083).

 
参数:

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – input tensor
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 维度用于分割 input

 


### logsigmoid

```py
torch.nn.functional.logsigmoid(input) → Tensor
```

逐元素应用 ![](http://latex.codecogs.com/gif.latex?%5Ctext%7BLogSigmoid%7D(x)%20%3D%20%5Clog%20%5Cleft(%5Cfrac%7B1%7D%7B1%20%2B%20%5Cexp(-x_i)%7D%5Cright))

请参见 [`LogSigmoid`](#torch.nn.LogSigmoid "torch.nn.LogSigmoid").

### hardshrink

```py
torch.nn.functional.hardshrink(input, lambd=0.5) → Tensor
```

逐元素应用hardshrink函数

请参见 [`Hardshrink`](#torch.nn.Hardshrink "torch.nn.Hardshrink").

### tanhshrink

```py
torch.nn.functional.tanhshrink(input) → Tensor
```

逐元素应用, ![](http://latex.codecogs.com/gif.latex?%5Ctext%7BTanhshrink%7D(x)%20%3D%20x%20-%20%5Ctext%7BTanh%7D(x))

请参见 [`Tanhshrink`](#torch.nn.Tanhshrink "torch.nn.Tanhshrink").

### softsign

```py
torch.nn.functional.softsign(input) → Tensor
```

逐元素应用, the function ![](http://latex.codecogs.com/gif.latex?%5Ctext%7BSoftSign%7D(x)%20%3D%20%5Cfrac%7Bx%7D%7B1%20%2B%20%7Cx%7C%7D)

请参见 [`Softsign`](#torch.nn.Softsign "torch.nn.Softsign").

### softplus

```py
torch.nn.functional.softplus(input, beta=1, threshold=20) → Tensor
```

### softmin

```py
torch.nn.functional.softmin(input, dim=None, _stacklevel=3, dtype=None)
```

应用 softmin 函数.

注意 ![](http://latex.codecogs.com/gif.latex?%5Ctext%7BSoftmin%7D(x)%20%3D%20%5Ctext%7BSoftmax%7D(-x)). See softmax definition for mathematical formula.

请参见 [`Softmin`](#torch.nn.Softmin "torch.nn.Softmin").

 
参数:

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – input
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 计算softmin的维度(因此dim上的每个切片的和为1).
*   **dtype** (`torch.dtype`, 可选的) – 返回tenosr的期望数据类型.

 


:param如果指定, 输入张量在执行::param操作之前被转换为`dtype`. 这对于防止数据类型溢出非常有用. 默认值:  None.

### softmax

```py
torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)
```

应用 softmax 函数.

Softmax定义为:

![](http://latex.codecogs.com/gif.latex?%5Ctext%7BSoftmax%7D(x_%7Bi%7D)%20%3D%20%5Cfrac%7Bexp(x_i)%7D%7B%5Csum_j%20exp(x_j)%7D)

它应用于dim上的所有切片, 并将对它们进行重新缩放, 使元素位于`(0,1)`范围内, 和为1.

请参见 [`Softmax`](#torch.nn.Softmax "torch.nn.Softmax").

 
参数:

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – input
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 将计算softmax的维度.
*   **dtype** (`torch.dtype`, 可选的) – 返回tenosr的期望数据类型.

 


:param如果指定, 输入张量在执行::param操作之前被转换为`dtype`. 这对于防止数据类型溢出非常有用. 默认值:  None.

注意

此函数不能直接用于`NLLLoss`, `NLLLoss`期望在`Softmax`和它自己之间计算对数. 改用`log_softmax` (它速度更快, 数值性能更好).

### softshrink

```py
torch.nn.functional.softshrink(input, lambd=0.5) → Tensor
```

逐元素应用 soft shrinkage 函数

请参见 [`Softshrink`](#torch.nn.Softshrink "torch.nn.Softshrink").

### gumbel_softmax

```py
torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10)
```

采样自Gumbel-Softmax分布, 并可选地离散化.

 
参数:

*   **logits** – `[batch_size, num_features]` 非规范化对数概率
*   **tau** – 非负的温度标量
*   **hard** – 如果 `True`, 返回的样本将会离散为 one-hot 向量, 但将是可微分的似乎它是autograd中的soft sample

 

| Returns: | 从 Gumbel-Softmax 分布采样的 tensor, 形状为 `batch_size x num_features` . 如果 `hard=True`, 返回值是 one-hot 编码, 否则, 它们就是特征和为1的概率分布 


约束:

*   目前仅支持 2D 输入 `logits` tensor , 形状为 `batch_size x num_features`

基于 [https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb](https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb) , (MIT license)

### log_softmax

```py
torch.nn.functional.log_softmax(input, dim=None, _stacklevel=3, dtype=None)
```

应用 softmax 跟随对数运算.

虽然在数学上等价于log(softmax(x)), 但分别执行这两个操作比较慢, 而且在数值上不稳定. 这个函数使用另一种公式来正确计算输出和梯度.

请参见 [`LogSoftmax`](#torch.nn.LogSoftmax "torch.nn.LogSoftmax").

 
参数:

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – input
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – A dimension along which log_softmax will be computed.
*   **dtype** (`torch.dtype`, 可选的) – 返回tenosr的期望数据类型.

 


:param如果指定, 输入张量在执行::param操作之前被转换为`dtype`. 这对于防止数据类型溢出非常有用. 默认值:  None.

### tanh

```py
torch.nn.functional.tanh(input) → Tensor
```

逐元素应用, ![](http://latex.codecogs.com/gif.latex?%5Ctext%7BTanh%7D(x)%20%3D%20%5Ctanh(x)%20%3D%20%5Cfrac%7B%5Cexp(x)%20-%20%5Cexp(-x)%7D%7B%5Cexp(x)%20%2B%20%5Cexp(-x)%7D)

请参见 [`Tanh`](#torch.nn.Tanh "torch.nn.Tanh").

### sigmoid

```py
torch.nn.functional.sigmoid(input) → Tensor
```

逐元素应用函数 ![](http://latex.codecogs.com/gif.latex?%5Ctext%7BSigmoid%7D(x)%20%3D%20%5Cfrac%7B1%7D%7B1%20%2B%20%5Cexp(-x)%7D)

请参见 [`Sigmoid`](#torch.nn.Sigmoid "torch.nn.Sigmoid").

## 规范化函数

### batch_norm

```py
torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)
```

对一批数据中的每个通道应用批量标准化.

请参见 [`BatchNorm1d`](#torch.nn.BatchNorm1d "torch.nn.BatchNorm1d"), [`BatchNorm2d`](#torch.nn.BatchNorm2d "torch.nn.BatchNorm2d"), [`BatchNorm3d`](#torch.nn.BatchNorm3d "torch.nn.BatchNorm3d").

### instance_norm

```py
torch.nn.functional.instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-05)
```

对批中每个数据样本中的每个通道应用实例规范化.

请参见 [`InstanceNorm1d`](#torch.nn.InstanceNorm1d "torch.nn.InstanceNorm1d"), [`InstanceNorm2d`](#torch.nn.InstanceNorm2d "torch.nn.InstanceNorm2d"), [`InstanceNorm3d`](#torch.nn.InstanceNorm3d "torch.nn.InstanceNorm3d").

### layer_norm

```py
torch.nn.functional.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05)
```

对最后特定数量的维度应用layer规范化.

请参见 [`LayerNorm`](#torch.nn.LayerNorm "torch.nn.LayerNorm").

### local_response_norm

```py
torch.nn.functional.local_response_norm(input, size, alpha=0.0001, beta=0.75, k=1.0)
```

对由多个输入平面组成的输入信号进行局部响应归一化, 其中通道占据第二维. 跨通道应用标准化.

请参见 [`LocalResponseNorm`](#torch.nn.LocalResponseNorm "torch.nn.LocalResponseNorm").

### normalize

```py
torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12, out=None)
```

执行 ![](http://latex.codecogs.com/gif.latex?L_p) 规范化对于指定维度.

对于一个 tensor `input` 尺寸为 ![](http://latex.codecogs.com/gif.latex?(n_0%2C%20...%2C%20n_%7Bdim%7D%2C%20...%2C%20n_k)), 每一 ![](http://latex.codecogs.com/gif.latex?n_%7Bdim%7D) -元素向量![](http://latex.codecogs.com/gif.latex?v) 沿着维度 `dim` 被转换为

![](http://latex.codecogs.com/gif.latex?%0D%0Av%20%3D%20%5Cfrac%7Bv%7D%7B%5Cmax(%5ClVert%20v%20%5CrVert_p%2C%20%5Cepsilon)%7D.%0D%0A%0D%0A)

对于默认参数，它使用沿维度![](http://latex.codecogs.com/gif.latex?1)的欧几里得范数进行标准化.

 
参数:

*   **input** – 任意形状的输入 tensor
*   **p** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – 范数公式中的指数值. 默认值:  2
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 进行规约的维度. 默认值:  1
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – 避免除以零的小值. 默认值:  1e-12
*   **out** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _可选的_) – 输出 tensor. 如果 `out` 被设置, 此操作不可微分.

 


## 线性函数

### linear

```py
torch.nn.functional.linear(input, weight, bias=None)
```

对传入数据应用线性转换: ![](http://latex.codecogs.com/gif.latex?y%20%3D%20xA%5ET%20%2B%20b).

形状:

> *   Input: ![](http://latex.codecogs.com/gif.latex?(N%2C%20*%2C%20in%5C_features))  `*` 表示任意数量的附加维度
> *   Weight: ![](http://latex.codecogs.com/gif.latex?(out%5C_features%2C%20in%5C_features))
> *   Bias: ![](http://latex.codecogs.com/gif.latex?(out%5C_features))
> *   Output: ![](http://latex.codecogs.com/gif.latex?(N%2C%20*%2C%20out%5C_features))

### bilinear

```py
torch.nn.functional.bilinear(input1, input2, weight, bias=None)
```

## Dropout 函数

### dropout

```py
torch.nn.functional.dropout(input, p=0.5, training=True, inplace=False)
```

在训练过程中，使用伯努利分布的样本，随机地用概率`p`将输入张量的一些元素归零.

请参见 [`Dropout`](#torch.nn.Dropout "torch.nn.Dropout").

 
参数:

*   **p** – 清零概率. 默认值:  0.5
*   **training** – 如果 `True` 使用 dropout. 默认值:  `True`
*   **inplace** – 如果设置为 `True`, 将会原地操作. 默认值:  `False`

 


### alpha_dropout

```py
torch.nn.functional.alpha_dropout(input, p=0.5, training=False, inplace=False)
```

应用 alpha dropout.

请参见 [`AlphaDropout`](#torch.nn.AlphaDropout "torch.nn.AlphaDropout").

### dropout2d

```py
torch.nn.functional.dropout2d(input, p=0.5, training=True, inplace=False)
```

随机归零input tensor的整个通道 (一个通道是一个 2D 特征图, 例如, ![](http://latex.codecogs.com/gif.latex?j)-th channel of the ![](http://latex.codecogs.com/gif.latex?i)-th sample in the batched input is a 2D tensor ![](http://latex.codecogs.com/gif.latex?%5Ctext%7Binput%7D%5Bi%2C%20j%5D))). 每次前向传递时，每个信道都将被独立清零. 用概率 `p` 从 Bernoulli 分布采样.

请参见 [`Dropout2d`](#torch.nn.Dropout2d "torch.nn.Dropout2d").

 
参数:

*   **p** – 通道清零的概率. 默认值:  0.5
*   **training** – 使用 dropout 如果设为 `True`. 默认值:  `True`
*   **inplace** – 如果设置为 `True`, 将会做原地操作. 默认值:  `False`

 


### dropout3d

```py
torch.nn.functional.dropout3d(input, p=0.5, training=True, inplace=False)
```

随机归零input tensor的整个通道 (一个通道是一个 3D 特征图, 例如, the ![](http://latex.codecogs.com/gif.latex?j)-th channel of the ![](http://latex.codecogs.com/gif.latex?i)-th sample in the batched input is a 3D tensor ![](http://latex.codecogs.com/gif.latex?%5Ctext%7Binput%7D%5Bi%2C%20j%5D)). 每次前向传递时，每个信道都将被独立清零. 用概率 `p` 从 Bernoulli 分布采样.

请参见 [`Dropout3d`](#torch.nn.Dropout3d "torch.nn.Dropout3d").

 
参数:

*   **p** – 通道清零的概率. 默认值:  0.5
*   **training** – 使用 dropout 如果设为 `True`. 默认值:  `True`
*   **inplace** – 如果设置为 `True`, 将会做原地操作. 默认值:  `False`

 


## 稀疏函数

### embedding

```py
torch.nn.functional.embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
```

一个简单的查找表，查找固定字典中的embedding(嵌入)内容和大小.

 这个模块通常用于使用索引检索单词嵌入。模块的输入是索引列表和嵌入矩阵，输出是相应的单词嵌入.

请参见 [`torch.nn.Embedding`](#torch.nn.Embedding "torch.nn.Embedding").

 
参数:

*   **input** (_LongTensor_) –  包含嵌入矩阵中的索引的tensor
*   **weight** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 嵌入矩阵的行数等于可能的最大索引数+ 1，列数等于嵌入大小
*   **padding_idx** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) –  如果给定，每当遇到索引时，在`padding_idx` (初始化为零)用嵌入向量填充输出.
*   **max_norm** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _可选的_) – 如果给定，则将范数大于`max_norm`的每个嵌入向量重新规范化，得到范数`max_norm`。注意:这将修改适当的`weight`.
*   **norm_type** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _可选的_) – 用于计算`max_norm`选项的p范数的p. 默认 `2`.
*   **scale_grad_by_freq** (_boolean__,_ _可选的_) – 如果给定，这将通过小批处理中单词频率的倒数来缩放梯度. 默认 `False`.
*   **sparse** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选的_) – If `True`, 梯度 w.r.t. `weight` 将会是一个稀疏 tensor. 请看 [`torch.nn.Embedding`](#torch.nn.Embedding "torch.nn.Embedding")有关稀疏梯度的更多详细信息.

 


形状:

>*   Input:  任意形状LongTensor, 包含要提取的索引的
>*   Weight: 浮点型嵌入矩阵, 形状为 (V, embedding_dim),
>    V = maximum index + 1 并且 embedding_dim = the embedding size
>*   Output: `(*, embedding_dim)`,  `*` 是输入形状

例子:

```py
>>> # a batch of 2 samples of 4 indices each
>>> input = torch.tensor([[1,2,4,5],[4,3,2,9]])
>>> # an embedding matrix containing 10 tensors of size 3
>>> embedding_matrix = torch.rand(10, 3)
>>> F.embedding(input, embedding_matrix)
tensor([[[ 0.8490,  0.9625,  0.6753],
 [ 0.9666,  0.7761,  0.6108],
 [ 0.6246,  0.9751,  0.3618],
 [ 0.4161,  0.2419,  0.7383]],

 [[ 0.6246,  0.9751,  0.3618],
 [ 0.0237,  0.7794,  0.0528],
 [ 0.9666,  0.7761,  0.6108],
 [ 0.3385,  0.8612,  0.1867]]])

>>> # example with padding_idx
>>> weights = torch.rand(10, 3)
>>> weights[0, :].zero_()
>>> embedding_matrix = weights
>>> input = torch.tensor([[0,2,0,5]])
>>> F.embedding(input, embedding_matrix, padding_idx=0)
tensor([[[ 0.0000,  0.0000,  0.0000],
 [ 0.5609,  0.5384,  0.8720],
 [ 0.0000,  0.0000,  0.0000],
 [ 0.6262,  0.2438,  0.7471]]])

```

### embedding_bag

```py
torch.nn.functional.embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, mode='mean', sparse=False)
```

计算嵌入`bags`的和、平均值或最大值，而不实例化中间嵌入。.

请参见 [`torch.nn.EmbeddingBag`](#torch.nn.EmbeddingBag "torch.nn.EmbeddingBag")

 
参数:

*   **input** (_LongTensor_) – 包含索引`bags`的张量
*   **weight** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 嵌入矩阵的行数等于可能的最大索引数+ 1，列数等于嵌入大小
*   **offsets** (_LongTensor__,_ _可选的_) – 仅当`input`为1D时使用。`offsets`确定输入中每个`bag`(序列)的起始索引位置
*   **max_norm** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _可选的_) –  如果给定，范数大于`max_norm`的每个嵌入向量将被重新规格化为范数`max_norm`。注意:这将就地修改`weight`
*   **norm_type** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _可选的_) – The `p` in the `p`-norm to compute for the `max_norm` option. 默认 `2`.
*   **scale_grad_by_freq** (_boolean__,_ _可选的_) – 如果给定，这将通过小批处理中单词频率的倒数来缩放梯度。默认假。注意:当`mode="max"`时不支持此选项.
*   **mode** (_string__,_ _可选的_) – `"sum"`, `"mean"` or `"max"`. 指定reduce`bag`的方法. 默认值:  `"mean"`
*   **sparse** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选的_) – 如果`True`，梯度w.r.t.权值就是一个稀疏张量.请参见 [`torch.nn.Embedding`](#torch.nn.Embedding "torch.nn.Embedding") 关于稀疏梯度. 注意: 此选项不支持 `mode="max"`.



形状:

> *   `input` (LongTensor) 和 `offsets` (LongTensor, 可选的)
>     
>     
>     *   如果 `input` 是 2D的, 形状为 `B x N`,
>         
>         
>         
>         它将被视为每个固定长度`N`的`B`个bag(序列)，这将根据模式以某种方式返回`B`个聚合值。在本例中，`offsets`被忽略，并且要求为`None`
>         
>         
>     *   如果 `input` 是 1D的, 形状为 `N`,
>         
>         
>         
>         it will be treated as a concatenation of multiple bags (sequences). `offsets` is required to be a 1D tensor containing the starting index positions of each bag in `input`. Therefore, for `offsets` of shape `B`, `input` will be viewed as having `B` bags. Empty bags (i.e., having 0-length) will have returned vectors filled by zeros.
>         它将被视为多个`bag`(序列)的串联。`offsets`必须是一个一维tensor，其中包含`input`中每个`bag`的起始索引位置。因此，对于形状`B`的偏移量，输入将被视为有`B`个bag。空bags( 即，具有0长度)将返回由0填充的向量
>         
> *   `weight` (Tensor): 模块的可学习权重, 形状 `(num_embeddings x embedding_dim)`
>     
>     
> *   `output`: 聚合的嵌入值, 形状 `B x embedding_dim`

例子:

```py
>>> # an Embedding module containing 10 tensors of size 3
>>> embedding_matrix = torch.rand(10, 3)
>>> # a batch of 2 samples of 4 indices each
>>> input = torch.tensor([1,2,4,5,4,3,2,9])
>>> offsets = torch.tensor([0,4])
>>> F.embedding_bag(embedding_matrix, input, offsets)
tensor([[ 0.3397,  0.3552,  0.5545],
 [ 0.5893,  0.4386,  0.5882]])

```

## 距离函数

### pairwise_distance

```py
torch.nn.functional.pairwise_distance(x1, x2, p=2.0, eps=1e-06, keepdim=False)
```

请参见 [`torch.nn.PairwiseDistance`](#torch.nn.PairwiseDistance "torch.nn.PairwiseDistance")

### cosine_similarity

```py
torch.nn.functional.cosine_similarity(x1, x2, dim=1, eps=1e-8) → Tensor
```

 返回x1和x2之间的余弦相似度，沿dim计算

![](http://latex.codecogs.com/gif.latex?%0D%0A%5Ctext%7Bsimilarity%7D%20%3D%20%5Cdfrac%7Bx_1%20%5Ccdot%20x_2%7D%7B%5Cmax(%5CVert%20x_1%20%5CVert%20_2%20%5Ccdot%20%5CVert%20x_2%20%5CVert%20_2%2C%20%5Cepsilon)%7D%0D%0A%0D%0A)

 
参数:

*   **x1** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 第一个input.
*   **x2** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 第二个input (大小和 x1 匹配).
*   **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) – 维度. 默认值:  1
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _可选的_) – 非常小的值避免除以0. 默认值:  1e-8

 


```py
形状:
```

*   Input: ![](http://latex.codecogs.com/gif.latex?(%5Cast_1%2C%20D%2C%20%5Cast_2)) where D is at position `dim`.
*   Output: ![](http://latex.codecogs.com/gif.latex?(%5Cast_1%2C%20%5Cast_2)) where 1 is at position `dim`.

Example:

```py
>>> input1 = torch.randn(100, 128)
>>> input2 = torch.randn(100, 128)
>>> output = F.cosine_similarity(input1, input2)
>>> print(output)

```

### pdist

```py
torch.nn.functional.pdist(input, p=2) → Tensor
```

计算输入中每​​对行向量之间的p范数距离。 这与`torch.norm(input[:, None] - input, dim=2, p=p)`的上三角形部分（不包括对角线）相同。 如果行是连续的，则此函数将更快

如果输入具有形状 ![](http://latex.codecogs.com/gif.latex?N%20%5Ctimes%20M) 则输出将具有形状 ![](http://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B2%7D%20N%20(N%20-%201)).

这个函数相当于 `scipy.spatial.distance.pdist(input, ‘minkowski’, p=p)` 如果 ![](http://latex.codecogs.com/gif.latex?p%20%5Cin%20(0%2C%20%5Cinfty)). 当 ![](http://latex.codecogs.com/gif.latex?p%20%3D%200) 它等价于 `scipy.spatial.distance.pdist(input, ‘hamming’) * M`. 当 ![](http://latex.codecogs.com/gif.latex?p%20%3D%20%5Cinfty), 最相近的scipy函数是 `scipy.spatial.distance.pdist(xn, lambda x, y: np.abs(x - y).max())`.

 
参数:

*   **input** – 输入tensor, 形状为 ![](http://latex.codecogs.com/gif.latex?N%20%5Ctimes%20M).
*   **p** – 计算每个向量对之间的p范数距离的p值 ![](http://latex.codecogs.com/gif.latex?%5Cin%20%5B0%2C%20%5Cinfty%5D).

 


## 损失函数

### binary_cross_entropy

```py
torch.nn.functional.binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean')
```

测量目标和输出之间二进制交叉熵的函数.

请参见 [`BCELoss`](#torch.nn.BCELoss "torch.nn.BCELoss").

 
参数:

*   **input** – 任意形状的tensor
*   **target** – 与输入形状相同的tensor
*   **weight** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _可选的_) – 手动重新调整weight, 如果提供, 它重复来匹配输入张量的形状
*   **size_average** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选的_) – 废弃的 (见 `reduction`). 默认情况下，批处理中的每个损失元素的平均损失。注意，对于某些损失，每个样本有多个元素。如果`size_average`设置为`False`，则对每个小批的损失进行汇总。reduce为False时忽略. 默认值:  `True`
*   **reduce** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选的_) – 废弃的 (见 `reduction`). 默认情况下，根据size_average，对每个小批量的观察结果的损失进行平均或求和。 当reduce为False时，返回每批元素的损失并忽略`size_average`. 默认值:  `True`
*   **reduction** (_string__,_ _可选的_) – 指定要应用于输出的`reduction`：'none'| 'mean'| 'sum'。 'none'：没有reduction，'mean'：输出的总和将除以输出中的元素数量 'sum'：输出将被求和。 注意：`size_average`和`reduce`正在被弃用，同时，指定这两个args中的任何一个都将覆盖reduce。 默认值：'mean', 默认值:  ‘mean’
 

例子:

```py
>>> input = torch.randn((3, 2), requires_grad=True)
>>> target = torch.rand((3, 2), requires_grad=False)
>>> loss = F.binary_cross_entropy(F.sigmoid(input), target)
>>> loss.backward()

```

### binary_cross_entropy_with_logits

```py
torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
```

Function that measures Binary Cross Entropy between target and output logits.

请参见 [`BCEWithLogitsLoss`](#torch.nn.BCEWithLogitsLoss "torch.nn.BCEWithLogitsLoss").

 
参数:

*   **input** – Tensor of arbitrary shape
*   **target** – Tensor of the same shape as input
*   **weight** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _可选的_) – a manual rescaling weight if provided it’s repeated to match input tensor shape
*   **size_average** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选的_) – Deprecated (see `reduction`). By 默认, the losses are averaged over each loss element in the batch. Note that for some losses, there multiple elements per sample. If the field `size_average` is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. 默认值:  `True`
*   **reduce** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选的_) – Deprecated (see `reduction`). By 默认, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce` is `False`, returns a loss per batch element instead and ignores `size_average`. 默认值:  `True`
*   **reduction** (_string__,_ _可选的_) – Specifies the reduction to apply to the output: ‘none’ &#124; ‘mean’ &#124; ‘sum’. ‘none’: no reduction will be applied, ‘mean’: the sum of the output will be divided by the number of elements in the output, ‘sum’: the output will be summed. Note: `size_average` and `reduce` are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. 默认值:  ‘mean’
*   **pos_weight** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _可选的_) – a weight of positive examples. Must be a vector with length equal to the number of classes.

 


例子:

```py
>>> input = torch.randn(3, requires_grad=True)
>>> target = torch.empty(3).random_(2)
>>> loss = F.binary_cross_entropy_with_logits(input, target)
>>> loss.backward()

```

### poisson_nll_loss

```py
torch.nn.functional.poisson_nll_loss(input, target, log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')
```

Poisson negative log likelihood loss.

请参见 [`PoissonNLLLoss`](#torch.nn.PoissonNLLLoss "torch.nn.PoissonNLLLoss").

 
参数:

*   **input** – expectation of underlying Poisson distribution.
*   **target** – random sample ![](http://latex.codecogs.com/gif.latex?target%20%5Csim%20%5Ctext%7BPoisson%7D(input)).
*   **log_input** – if `True` the loss is computed as ![](http://latex.codecogs.com/gif.latex?%5Cexp(%5Ctext%7Binput%7D)%20-%20%5Ctext%7Btarget%7D%20*%20%5Ctext%7Binput%7D), if `False` then loss is ![](http://latex.codecogs.com/gif.latex?%5Ctext%7Binput%7D%20-%20%5Ctext%7Btarget%7D%20*%20%5Clog(%5Ctext%7Binput%7D%2B%5Ctext%7Beps%7D)). 默认值:  `True`
*   **full** – whether to compute full loss, i. e. to add the Stirling approximation term. 默认值:  `False` ![](http://latex.codecogs.com/gif.latex?%5Ctext%7Btarget%7D%20*%20%5Clog(%5Ctext%7Btarget%7D)%20-%20%5Ctext%7Btarget%7D%20%2B%200.5%20*%20%5Clog(2%20*%20%5Cpi%20*%20%5Ctext%7Btarget%7D)).
*   **size_average** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选的_) – Deprecated (see `reduction`). By 默认, the losses are averaged over each loss element in the batch. Note that for some losses, there multiple elements per sample. If the field `size_average` is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. 默认值:  `True`
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _可选的_) – Small value to avoid evaluation of ![](http://latex.codecogs.com/gif.latex?%5Clog(0)) when `log_input`=``False``. 默认值:  1e-8
*   **reduce** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选的_) – Deprecated (see `reduction`). By 默认, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce` is `False`, returns a loss per batch element instead and ignores `size_average`. 默认值:  `True`
*   **reduction** (_string__,_ _可选的_) – Specifies the reduction to apply to the output: ‘none’ &#124; ‘mean’ &#124; ‘sum’. ‘none’: no reduction will be applied, ‘mean’: the sum of the output will be divided by the number of elements in the output, ‘sum’: the output will be summed. Note: `size_average` and `reduce` are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. 默认值:  ‘mean’

 


### cosine_embedding_loss

```py
torch.nn.functional.cosine_embedding_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean') → Tensor
```

请参见 [`CosineEmbeddingLoss`](#torch.nn.CosineEmbeddingLoss "torch.nn.CosineEmbeddingLoss").

### cross_entropy

```py
torch.nn.functional.cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
```

This criterion combines `log_softmax` and `nll_loss` in a single function.

请参见 [`CrossEntropyLoss`](#torch.nn.CrossEntropyLoss "torch.nn.CrossEntropyLoss").

 
参数:

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – ![](http://latex.codecogs.com/gif.latex?(N%2C%20C)) where `C = number of classes` or ![](http://latex.codecogs.com/gif.latex?(N%2C%20C%2C%20H%2C%20W)) in case of 2D Loss, or ![](http://latex.codecogs.com/gif.latex?(N%2C%20C%2C%20d_1%2C%20d_2%2C%20...%2C%20d_K)) where ![](http://latex.codecogs.com/gif.latex?K%20%3E%201) in the case of K-dimensional loss.
*   **target** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – ![](http://latex.codecogs.com/gif.latex?(N)) where each value is ![](http://latex.codecogs.com/gif.latex?0%20%5Cleq%20%5Ctext%7Btargets%7D%5Bi%5D%20%5Cleq%20C-1), or ![](http://latex.codecogs.com/gif.latex?(N%2C%20d_1%2C%20d_2%2C%20...%2C%20d_K)) where ![](http://latex.codecogs.com/gif.latex?K%20%5Cgeq%201) for K-dimensional loss.
*   **weight** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _可选的_) – a manual rescaling weight given to each class. If given, has to be a Tensor of size `C`
*   **size_average** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选的_) – Deprecated (see `reduction`). By 默认, the losses are averaged over each loss element in the batch. Note that for some losses, there multiple elements per sample. If the field `size_average` is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. 默认值:  `True`
*   **ignore_index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) – Specifies a target value that is ignored and does not contribute to the input gradient. When `size_average` is `True`, the loss is averaged over non-ignored targets. 默认值:  -100
*   **reduce** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选的_) – Deprecated (see `reduction`). By 默认, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce` is `False`, returns a loss per batch element instead and ignores `size_average`. 默认值:  `True`
*   **reduction** (_string__,_ _可选的_) – Specifies the reduction to apply to the output: ‘none’ &#124; ‘mean’ &#124; ‘sum’. ‘none’: no reduction will be applied, ‘mean’: the sum of the output will be divided by the number of elements in the output, ‘sum’: the output will be summed. Note: `size_average` and `reduce` are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. 默认值:  ‘mean’

 


例子:

```py
>>> input = torch.randn(3, 5, requires_grad=True)
>>> target = torch.randint(5, (3,), dtype=torch.int64)
>>> loss = F.cross_entropy(input, target)
>>> loss.backward()

```

### ctc_loss

```py
torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean')
```

The Connectionist Temporal Classification loss.

请参见 [`CTCLoss`](#torch.nn.CTCLoss "torch.nn.CTCLoss").

注意

在某些情况下, 当使用CUDA后端与CuDNN时, 该操作符可能会选择不确定性算法来提高性能. 如果这不是您希望的, 您可以通过设置`torch.backends.cudn .deterministic = True`来尝试使操作具有确定性(可能会以性能为代价). 请参阅关于 [Reproducibility](notes/randomness.html) 了解背景.

注意

When using the CUDA backend, this operation may induce nondeterministic behaviour in be backward that is not easily switched off. Please see the notes on [Reproducibility](notes/randomness.html) for background.

 
参数:

*   **log_probs** – ![](http://latex.codecogs.com/gif.latex?(T%2C%20N%2C%20C)) where `C = number of characters in alphabet including blank`, `T = input length`, and `N = batch size`. The logarithmized probabilities of the outputs (e.g. obtained with [`torch.nn.functional.log_softmax()`](#torch.nn.functional.log_softmax "torch.nn.functional.log_softmax")).
*   **targets** – ![](http://latex.codecogs.com/gif.latex?(N%2C%20S)) or `(sum(target_lengths))`. Targets (cannot be blank). In the second form, the targets are assumed to be concatenated.
*   **input_lengths** – ![](http://latex.codecogs.com/gif.latex?(N)). Lengths of the inputs (must each be ![](http://latex.codecogs.com/gif.latex?%5Cleq%20T))
*   **target_lengths** – ![](http://latex.codecogs.com/gif.latex?(N)). Lengths of the targets
*   **blank** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) – Blank label. 默认 ![](http://latex.codecogs.com/gif.latex?0).
*   **reduction** (_string__,_ _可选的_) – Specifies the reduction to apply to the output: ‘none’ &#124; ‘mean’ &#124; ‘sum’. ‘none’: no reduction will be applied, ‘mean’: the output losses will be divided by the target lengths and then the mean over the batch is taken. 默认值:  ‘mean’

 


Example:

```py
>>> log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
>>> targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
>>> input_lengths = torch.full((16,), 50, dtype=torch.long)
>>> target_lengths = torch.randint(10,30,(16,), dtype=torch.long)
>>> loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
>>> loss.backward()

```

### hinge_embedding_loss

```py
torch.nn.functional.hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction='mean') → Tensor
```

请参见 [`HingeEmbeddingLoss`](#torch.nn.HingeEmbeddingLoss "torch.nn.HingeEmbeddingLoss").

### kl_div

```py
torch.nn.functional.kl_div(input, target, size_average=None, reduce=None, reduction='mean')
```

The [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback-Leibler_divergence) Loss.

请参见 [`KLDivLoss`](#torch.nn.KLDivLoss "torch.nn.KLDivLoss").

 
参数:

*   **input** – Tensor of arbitrary shape
*   **target** – Tensor of the same shape as input
*   **size_average** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选的_) – Deprecated (see `reduction`). By 默认, the losses are averaged over each loss element in the batch. Note that for some losses, there multiple elements per sample. If the field `size_average` is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. 默认值:  `True`
*   **reduce** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选的_) – Deprecated (see `reduction`). By 默认, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce` is `False`, returns a loss per batch element instead and ignores `size_average`. 默认值:  `True`
*   **reduction** (_string__,_ _可选的_) – Specifies the reduction to apply to the output: ‘none’ &#124; ‘batchmean’ &#124; ‘sum’ &#124; ‘mean’. ‘none’: no reduction will be applied ‘batchmean’: the sum of the output will be divided by the batchsize ‘sum’: the output will be summed ‘mean’: the output will be divided by the number of elements in the output 默认值:  ‘mean’

 


:param .. note:: `size_average` and `reduce` are in the process of being deprecated,: and in the meantime, specifying either of those two args will override `reduction`. :param .. note:: `reduction=’mean’` doesn’t return the true kl divergence value, please use: `reduction=’batchmean’` which aligns with KL math definition.

> In the next major release, ‘mean’ will be changed to be the same as ‘batchmean’.

### l1_loss

```py
torch.nn.functional.l1_loss(input, target, size_average=None, reduce=None, reduction='mean') → Tensor
```

Function that takes the mean element-wise absolute value difference.

请参见 [`L1Loss`](#torch.nn.L1Loss "torch.nn.L1Loss").

### mse_loss

```py
torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='mean') → Tensor
```

Measures the element-wise mean squared error.

请参见 [`MSELoss`](#torch.nn.MSELoss "torch.nn.MSELoss").

### margin_ranking_loss

```py
torch.nn.functional.margin_ranking_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean') → Tensor
```

请参见 [`MarginRankingLoss`](#torch.nn.MarginRankingLoss "torch.nn.MarginRankingLoss").

### multilabel_margin_loss

```py
torch.nn.functional.multilabel_margin_loss(input, target, size_average=None, reduce=None, reduction='mean') → Tensor
```

请参见 [`MultiLabelMarginLoss`](#torch.nn.MultiLabelMarginLoss "torch.nn.MultiLabelMarginLoss").

### multilabel_soft_margin_loss

```py
torch.nn.functional.multilabel_soft_margin_loss(input, target, weight=None, size_average=None) → Tensor
```

请参见 [`MultiLabelSoftMarginLoss`](#torch.nn.MultiLabelSoftMarginLoss "torch.nn.MultiLabelSoftMarginLoss").

### multi_margin_loss

```py
torch.nn.functional.multi_margin_loss(input, target, p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')
```

```py
multi_margin_loss(input, target, p=1, margin=1, weight=None, size_average=None,
```

reduce=None, reduction=’mean’) -> Tensor

请参见 [`MultiMarginLoss`](#torch.nn.MultiMarginLoss "torch.nn.MultiMarginLoss").

### nll_loss

```py
torch.nn.functional.nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
```

The negative log likelihood loss.

请参见 [`NLLLoss`](#torch.nn.NLLLoss "torch.nn.NLLLoss").

 
参数:

*   **input** – ![](http://latex.codecogs.com/gif.latex?(N%2C%20C)) where `C = number of classes` or ![](http://latex.codecogs.com/gif.latex?(N%2C%20C%2C%20H%2C%20W)) in case of 2D Loss, or ![](http://latex.codecogs.com/gif.latex?(N%2C%20C%2C%20d_1%2C%20d_2%2C%20...%2C%20d_K)) where ![](http://latex.codecogs.com/gif.latex?K%20%3E%201) in the case of K-dimensional loss.
*   **target** – ![](http://latex.codecogs.com/gif.latex?(N)) where each value is ![](http://latex.codecogs.com/gif.latex?0%20%5Cleq%20%5Ctext%7Btargets%7D%5Bi%5D%20%5Cleq%20C-1), or ![](http://latex.codecogs.com/gif.latex?(N%2C%20d_1%2C%20d_2%2C%20...%2C%20d_K)) where ![](http://latex.codecogs.com/gif.latex?K%20%5Cgeq%201) for K-dimensional loss.
*   **weight** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _可选的_) – a manual rescaling weight given to each class. If given, has to be a Tensor of size `C`
*   **size_average** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选的_) – Deprecated (see `reduction`). By 默认, the losses are averaged over each loss element in the batch. Note that for some losses, there multiple elements per sample. If the field `size_average` is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. 默认值:  `True`
*   **ignore_index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选的_) – Specifies a target value that is ignored and does not contribute to the input gradient. When `size_average` is `True`, the loss is averaged over non-ignored targets. 默认值:  -100
*   **reduce** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选的_) – Deprecated (see `reduction`). By 默认, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce` is `False`, returns a loss per batch element instead and ignores `size_average`. 默认值:  `True`
*   **reduction** (_string__,_ _可选的_) – Specifies the reduction to apply to the output: ‘none’ &#124; ‘mean’ &#124; ‘sum’. ‘none’: no reduction will be applied, ‘mean’: the sum of the output will be divided by the number of elements in the output, ‘sum’: the output will be summed. Note: `size_average` and `reduce` are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. 默认值:  ‘mean’

 


Example:

```py
>>> # input is of size N x C = 3 x 5
>>> input = torch.randn(3, 5, requires_grad=True)
>>> # each element in target has to have 0 <= value < C
>>> target = torch.tensor([1, 0, 4])
>>> output = F.nll_loss(F.log_softmax(input), target)
>>> output.backward()

```

### smooth_l1_loss

```py
torch.nn.functional.smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean')
```

Function that uses a squared term if the absolute element-wise error falls below 1 and an L1 term otherwise.

请参见 [`SmoothL1Loss`](#torch.nn.SmoothL1Loss "torch.nn.SmoothL1Loss").

### soft_margin_loss

```py
torch.nn.functional.soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean') → Tensor
```

请参见 [`SoftMarginLoss`](#torch.nn.SoftMarginLoss "torch.nn.SoftMarginLoss").

### triplet_margin_loss

```py
torch.nn.functional.triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')
```

请参见 [`TripletMarginLoss`](#torch.nn.TripletMarginLoss "torch.nn.TripletMarginLoss")

## Vision functions

### pixel_shuffle

```py
torch.nn.functional.pixel_shuffle()
```

Rearranges elements in a tensor of shape ![](http://latex.codecogs.com/gif.latex?(*%2C%20C%20%5Ctimes%20r%5E2%2C%20H%2C%20W)) to a tensor of shape ![](http://latex.codecogs.com/gif.latex?(C%2C%20H%20%5Ctimes%20r%2C%20W%20%5Ctimes%20r)).

请参见 [`PixelShuffle`](#torch.nn.PixelShuffle "torch.nn.PixelShuffle").

 
参数:

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **upscale_factor** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – factor to increase spatial resolution by

 


例子:

```py
>>> input = torch.randn(1, 9, 4, 4)
>>> output = torch.nn.functional.pixel_shuffle(input, 3)
>>> print(output.size())
torch.Size([1, 1, 12, 12])

```

### pad

```py
torch.nn.functional.pad(input, pad, mode='constant', value=0)
```

Pads tensor.

```py
Pading size:
```

The number of dimensions to pad is ![](http://latex.codecogs.com/gif.latex?%5Cleft%5Clfloor%5Cfrac%7B%5Ctext%7Blen(pad)%7D%7D%7B2%7D%5Cright%5Crfloor) and the dimensions that get padded begins with the last dimension and moves forward. For example, to pad the last dimension of the input tensor, then <cite>pad</cite> has form <cite>(padLeft, padRight)</cite>; to pad the last 2 dimensions of the input tensor, then use <cite>(padLeft, padRight, padTop, padBottom)</cite>; to pad the last 3 dimensions, use <cite>(padLeft, padRight, padTop, padBottom, padFront, padBack)</cite>.

```py
Padding mode:
```

请参见 [`torch.nn.ConstantPad2d`](#torch.nn.ConstantPad2d "torch.nn.ConstantPad2d"), [`torch.nn.ReflectionPad2d`](#torch.nn.ReflectionPad2d "torch.nn.ReflectionPad2d"), and [`torch.nn.ReplicationPad2d`](#torch.nn.ReplicationPad2d "torch.nn.ReplicationPad2d") for concrete examples on how each of the padding modes works. Constant padding is implemented for arbitrary dimensions. Replicate padding is implemented for padding the last 3 dimensions of 5D input tensor, or the last 2 dimensions of 4D input tensor, or the last dimension of 3D input tensor. Reflect padding is only implemented for padding the last 2 dimensions of 4D input tensor, or the last dimension of 3D input tensor.

注意

When using the CUDA backend, this operation may induce nondeterministic behaviour in be backward that is not easily switched off. Please see the notes on [Reproducibility](notes/randomness.html) for background.

 
参数:

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – `Nd` tensor
*   **pad** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")) – m-elem tuple, where ![](http://latex.codecogs.com/gif.latex?%5Cfrac%7Bm%7D%7B2%7D%20%5Cleq) input dimensions and ![](http://latex.codecogs.com/gif.latex?m) is even.
*   **mode** – ‘constant’, ‘reflect’ or ‘replicate’. 默认值:  ‘constant’
*   **value** – fill value for ‘constant’ padding. 默认值:  0

 


例子:

```py
>>> t4d = torch.empty(3, 3, 4, 2)
>>> p1d = (1, 1) # pad last dim by 1 on each side
>>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
>>> print(out.data.size())
torch.Size([3, 3, 4, 4])
>>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
>>> out = F.pad(t4d, p2d, "constant", 0)
>>> print(out.data.size())
torch.Size([3, 3, 8, 4])
>>> t4d = torch.empty(3, 3, 4, 2)
>>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
>>> out = F.pad(t4d, p3d, "constant", 0)
>>> print(out.data.size())
torch.Size([3, 9, 7, 3])

```

### interpolate

```py
torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
```

Down/up samples the input to either the given `size` or the given `scale_factor`

The algorithm used for interpolation is determined by `mode`.

Currently temporal, spatial and volumetric sampling are supported, i.e. expected inputs are 3-D, 4-D or 5-D in shape.

The input dimensions are interpreted in the form: `mini-batch x channels x [可选的 depth] x [可选的 height] x width`.

The modes available for resizing are: `nearest`, `linear` (3D-only), `bilinear` (4D-only), `trilinear` (5D-only), `area`

 
参数:

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _or_ _Tuple__[_[_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_] or_ _Tuple__[_[_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_] or_ _Tuple__[_[_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_]_) – output spatial size.
*   **scale_factor** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ _Tuple__[_[_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_]_) – multiplier for spatial size. Has to match input size if it is a tuple.
*   **mode** (_string_) – algorithm used for upsampling: ‘nearest’ &#124; ‘linear’ &#124; ‘bilinear’ &#124; ‘trilinear’ &#124; ‘area’. 默认值:  ‘nearest’
*   **align_corners** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选的_) – if True, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels. This only has effect when `mode` is `linear`, `bilinear`, or `trilinear`. 默认值:  False

 


警告

With `align_corners = True`, the linearly interpolating modes (`linear`, `bilinear`, and `trilinear`) don’t proportionally align the output and input pixels, and thus the output values can depend on the input size. This was the 默认 behavior for these modes up to version 0.3.1\. Since then, the 默认 behavior is `align_corners = False`. 请参见 [`Upsample`](#torch.nn.Upsample "torch.nn.Upsample") for concrete examples on how this affects the outputs.

注意

When using the CUDA backend, this operation may induce nondeterministic behaviour in be backward that is not easily switched off. Please see the notes on [Reproducibility](notes/randomness.html) for background.

### upsample

```py
torch.nn.functional.upsample(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
```

Upsamples the input to either the given `size` or the given `scale_factor`

警告

This function is deprecated in favor of [`torch.nn.functional.interpolate()`](#torch.nn.functional.interpolate "torch.nn.functional.interpolate"). This is equivalent with `nn.functional.interpolate(...)`.

注意

When using the CUDA backend, this operation may induce nondeterministic behaviour in be backward that is not easily switched off. Please see the notes on [Reproducibility](notes/randomness.html) for background.

The algorithm used for upsampling is determined by `mode`.

Currently temporal, spatial and volumetric upsampling are supported, i.e. expected inputs are 3-D, 4-D or 5-D in shape.

The input dimensions are interpreted in the form: `mini-batch x channels x [可选的 depth] x [可选的 height] x width`.

The modes available for upsampling are: `nearest`, `linear` (3D-only), `bilinear` (4D-only), `trilinear` (5D-only)

 
参数:

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor
*   **size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _or_ _Tuple__[_[_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_] or_ _Tuple__[_[_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_] or_ _Tuple__[_[_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_]_) – output spatial size.
*   **scale_factor** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – multiplier for spatial size. Has to be an integer.
*   **mode** (_string_) – algorithm used for upsampling: ‘nearest’ &#124; ‘linear’ &#124; ‘bilinear’ &#124; ‘trilinear’. 默认值:  ‘nearest’
*   **align_corners** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选的_) – if True, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels. This only has effect when `mode` is `linear`, `bilinear`, or `trilinear`. 默认值:  False

 


警告

With `align_corners = True`, the linearly interpolating modes (`linear`, `bilinear`, and `trilinear`) don’t proportionally align the output and input pixels, and thus the output values can depend on the input size. This was the 默认 behavior for these modes up to version 0.3.1\. Since then, the 默认 behavior is `align_corners = False`. 请参见 [`Upsample`](#torch.nn.Upsample "torch.nn.Upsample") for concrete examples on how this affects the outputs.

### upsample_nearest

```py
torch.nn.functional.upsample_nearest(input, size=None, scale_factor=None)
```

Upsamples the input, using nearest neighbours’ pixel values.

警告

This function is deprecated in favor of [`torch.nn.functional.interpolate()`](#torch.nn.functional.interpolate "torch.nn.functional.interpolate"). This is equivalent with `nn.functional.interpolate(..., mode='nearest')`.

Currently spatial and volumetric upsampling are supported (i.e. expected inputs are 4 or 5 dimensional).

 
参数:

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – input
*   **size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _or_ _Tuple__[_[_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_] or_ _Tuple__[_[_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_]_) – output spatia size.
*   **scale_factor** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – multiplier for spatial size. Has to be an integer.

 


注意

When using the CUDA backend, this operation may induce nondeterministic behaviour in be backward that is not easily switched off. Please see the notes on [Reproducibility](notes/randomness.html) for background.

### upsample_bilinear

```py
torch.nn.functional.upsample_bilinear(input, size=None, scale_factor=None)
```

Upsamples the input, using bilinear upsampling.

警告

This function is deprecated in favor of [`torch.nn.functional.interpolate()`](#torch.nn.functional.interpolate "torch.nn.functional.interpolate"). This is equivalent with `nn.functional.interpolate(..., mode='bilinear', align_corners=True)`.

Expected inputs are spatial (4 dimensional). Use `upsample_trilinear` fo volumetric (5 dimensional) inputs.

 
参数:

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – input
*   **size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _or_ _Tuple__[_[_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_]_) – output spatial size.
*   **scale_factor** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _or_ _Tuple__[_[_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_]_) – multiplier for spatial size

 


注意

When using the CUDA backend, this operation may induce nondeterministic behaviour in be backward that is not easily switched off. Please see the notes on [Reproducibility](notes/randomness.html) for background.

### grid_sample

```py
torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros')
```

Given an `input` and a flow-field `grid`, computes the `output` using `input` values and pixel locations from `grid`.

Currently, only spatial (4-D) and volumetric (5-D) `input` are supported.

In the spatial (4-D) case, for `input` with shape ![](http://latex.codecogs.com/gif.latex?(N%2C%20C%2C%20H_%5Ctext%7Bin%7D%2C%20W_%5Ctext%7Bin%7D)) and `grid` with shape ![](http://latex.codecogs.com/gif.latex?(N%2C%20H_%5Ctext%7Bout%7D%2C%20W_%5Ctext%7Bout%7D%2C%202)), the output will have shape ![](http://latex.codecogs.com/gif.latex?(N%2C%20C%2C%20H_%5Ctext%7Bout%7D%2C%20W_%5Ctext%7Bout%7D)).

For each output location `output[n, :, h, w]`, the size-2 vector `grid[n, h, w]` specifies `input` pixel locations `x` and `y`, which are used to interpolate the output value `output[n, :, h, w]`. In the case of 5D inputs, `grid[n, d, h, w]` specifies the `x`, `y`, `z` pixel locations for interpolating `output[n, :, d, h, w]`. `mode` argument specifies `nearest` or `bilinear` interpolation method to sample the input pixels.

`grid` should have most values in the range of `[-1, 1]`. This is because the pixel locations are normalized by the `input` spatial dimensions. For example, values `x = -1, y = -1` is the left-top pixel of `input`, and values `x = 1, y = 1` is the right-bottom pixel of `input`.

If `grid` has values outside the range of `[-1, 1]`, those locations are handled as defined by `padding_mode`. Options are

> *   `padding_mode="zeros"`: use `0` for out-of-bound values,
> *   `padding_mode="border"`: use border values for out-of-bound values,
> *   `padding_mode="reflection"`: use values at locations reflected by the border for out-of-bound values. For location far away from the border, it will keep being reflected until becoming in bound, e.g., (normalized) pixel location `x = -3.5` reflects by `-1` and becomes `x' = 2.5`, then reflects by border `1` and becomes `x'' = -0.5`.

注意

This function is often used in building Spatial Transformer Networks.

注意

When using the CUDA backend, this operation may induce nondeterministic behaviour in be backward that is not easily switched off. Please see the notes on [Reproducibility](notes/randomness.html) for background.

 
参数:

*   **input** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – input of shape ![](http://latex.codecogs.com/gif.latex?(N%2C%20C%2C%20H_%5Ctext%7Bin%7D%2C%20W_%5Ctext%7Bin%7D)) (4-D case) or ![](http://latex.codecogs.com/gif.latex?(N%2C%20C%2C%20D_%5Ctext%7Bin%7D%2C%20H_%5Ctext%7Bin%7D%2C%20W_%5Ctext%7Bin%7D)) (5-D case)
*   **grid** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – flow-field of shape ![](http://latex.codecogs.com/gif.latex?(N%2C%20H_%5Ctext%7Bout%7D%2C%20W_%5Ctext%7Bout%7D%2C%202)) (4-D case) or ![](http://latex.codecogs.com/gif.latex?(N%2C%20D_%5Ctext%7Bout%7D%2C%20H_%5Ctext%7Bout%7D%2C%20W_%5Ctext%7Bout%7D%2C%203)) (5-D case)
*   **mode** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")) – interpolation mode to calculate output values ‘bilinear’ &#124; ‘nearest’. 默认值:  ‘bilinear’
*   **padding_mode** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")) – padding mode for outside grid values ‘zeros’ &#124; ‘border’ &#124; ‘reflection’. 默认值:  ‘zeros’

 

| Returns: | output Tensor 

| Return type: | output ([Tensor](tensors.html#torch.Tensor "torch.Tensor")) 


### affine_grid

```py
torch.nn.functional.affine_grid(theta, size)
```

Generates a 2d flow field, given a batch of affine matrices `theta` Generally used in conjunction with [`grid_sample()`](#torch.nn.functional.grid_sample "torch.nn.functional.grid_sample") to implement Spatial Transformer Networks.

 
参数:

*   **theta** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – input batch of affine matrices (![](http://latex.codecogs.com/gif.latex?N%20%5Ctimes%202%20%5Ctimes%203))
*   **size** (_torch.Size_) – the target output image size (![](http://latex.codecogs.com/gif.latex?N%20%5Ctimes%20C%20%5Ctimes%20H%20%5Ctimes%20W)) Example: torch.Size((32, 3, 24, 24))

 

| Returns: | output Tensor of size (![](http://latex.codecogs.com/gif.latex?N%20%5Ctimes%20H%20%5Ctimes%20W%20%5Ctimes%202)) 

| Return type: | output ([Tensor](tensors.html#torch.Tensor "torch.Tensor")) 


## DataParallel functions (multi-GPU, distributed)

### data_parallel

```py
torch.nn.parallel.data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None)
```

Evaluates module(input) in parallel across the GPUs given in device_ids.

This is the functional version of the DataParallel module.

 
参数:

*   **module** ([_Module_](#torch.nn.Module "torch.nn.Module")) – the module to evaluate in parallel
*   **inputs** (_tensor_) – inputs to the module
*   **device_ids** (_list of python:int_ _or_ [_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device")) – GPU ids on which to replicate module
*   **output_device** (_list of python:int_ _or_ [_torch.device_](tensor_attributes.html#torch.torch.device "torch.torch.device")) – GPU location of the output Use -1 to indicate the CPU. (默认值:  device_ids[0])

 

| Returns: | a Tensor containing the result of module(input) located on output_device 
