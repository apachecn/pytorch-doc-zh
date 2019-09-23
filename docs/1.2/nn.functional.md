# torch.nn.functional

## 卷积函数

###  conv1d 

`torch.nn.functional.``conv1d`( _input_ , _weight_ , _bias=None_ , _stride=1_
, _padding=0_ , _dilation=1_ , _groups=1_ ) → Tensor

    

施加1D卷积在几个输入平面组成的输入信号。

参见[ `Conv1d`](nn.html#torch.nn.Conv1d "torch.nn.Conv1d")的详细信息和输出形状。

注意

在使用CUDA后端与CuDNN当某些情况下，这种操作者可以选择不确定性的算法来提高性能。如果这是不可取的，你可以尝试通过设置`
torch.backends.cudnn.deterministic  =  真[使操作确定性（可能以性能为代价） HTG6] `。请参阅[ 重复性
](notes/randomness.html)为背景的音符。

Parameters

    

  * **输入** \- 的输入张量形状 （ minibatch  ， in_channels  ， i的 W  ） （\文本{minibatch}，\文本{在\ _channels}，IW） （ minibatch  ， in_channels  ， i的 W  ）

  * **重量** \- 的过滤器形状 （ out_channels  ， in_channels  基团 ， K  W  ） （\文本{出\ _channels}，\压裂{\文本{在\ _channels}} {\文本{基}}，千瓦） （ out_channels  ， 基团 in_channels  [HTG9 3]  ， K  W  ）

  * **偏压** \- 形状 （ out_channels  ）[的可选偏置HTG13 ]  （\文本{出\ _channels}） （ out_channels  ） 。默认值：`无 `

  * **步幅** \- 在卷积内核的步幅。可以是单一的数或一个元素的元组（SW）。默认值：1

  * **填充** \- 对输入的两侧隐填补处理。可以是单一的数或一个元素的元组（padW，）。默认值：0

  * **扩张** \- 内核元件之间的间隔。可以是单一的数或一个元素的元组（DW）。默认值：1

  * **基团** \- 分裂输入成组， in_channels  \文本{在\ _channels }  in_channels  应该是组数整除。默认值：1

例子：

    
    
    >>> filters = torch.randn(33, 16, 3)
    >>> inputs = torch.randn(20, 16, 50)
    >>> F.conv1d(inputs, filters)
    

###  conv2d 

`torch.nn.functional.``conv2d`( _input_ , _weight_ , _bias=None_ , _stride=1_
, _padding=0_ , _dilation=1_ , _groups=1_ ) → Tensor

    

施加二维卷积在几个输入平面构成的输入图像。

参见[ `Conv2d`](nn.html#torch.nn.Conv2d "torch.nn.Conv2d")的详细信息和输出形状。

Note

In some circumstances when using the CUDA backend with CuDNN, this operator
may select a nondeterministic algorithm to increase performance. If this is
undesirable, you can try to make the operation deterministic (potentially at a
performance cost) by setting `torch.backends.cudnn.deterministic = True`.
Please see the notes on [Reproducibility](notes/randomness.html) for
background.

Parameters

    

  * **输入** \- 的输入张量形状 （ minibatch  ， in_channels  ， i的 H  ， i的 W  ） （\文本{minibatch}，\文本{在\ _channels}，1H，IW） （ minibatch  ， in_channels  ， i的 H  ， i的 W  ）

  * **重量** \- 的过滤器形状 （ out_channels  ， in_channels  基团 ， K  H  ， K  W  ） （\文本{出\ _channels}，\压裂{\文本{在\ _channels}} {\文本{基}}，KH，千瓦） （ out_channels  ， 基团 in_channels  ， K  H  ， K  W  ）

  * **偏压** \- 形状 （ out_channels  ）[的可选偏置张量HTG13]  （\文本{出\ _channels}） （ out_channels  ） 。默认值：`无 `

  * **步幅** \- 在卷积内核的步幅。可以是单一的数或一个元组（SH，SW）。默认值：1

  * **填充** \- 对输入的两侧隐填补处理。可以是单一的数或一个元组（PADH，padW）。默认值：0

  * **扩张** \- 内核元件之间的间隔。可以是单一的数或一个元组（DH，一页）。默认值：1

  * **基团** \- 分裂输入成组， in_channels  \文本{在\ _channels }  in_channels  应该是组数整除。默认值：1

Examples:

    
    
    >>> # With square kernels and equal stride
    >>> filters = torch.randn(8,4,3,3)
    >>> inputs = torch.randn(1,4,5,5)
    >>> F.conv2d(inputs, filters, padding=1)
    

###  conv3d 

`torch.nn.functional.``conv3d`( _input_ , _weight_ , _bias=None_ , _stride=1_
, _padding=0_ , _dilation=1_ , _groups=1_ ) → Tensor

    

应用了3D卷积在数输入面构成的输入图像。

参见[ `Conv3d`](nn.html#torch.nn.Conv3d "torch.nn.Conv3d")的详细信息和输出形状。

Note

In some circumstances when using the CUDA backend with CuDNN, this operator
may select a nondeterministic algorithm to increase performance. If this is
undesirable, you can try to make the operation deterministic (potentially at a
performance cost) by setting `torch.backends.cudnn.deterministic = True`.
Please see the notes on [Reproducibility](notes/randomness.html) for
background.

Parameters

    

  * **输入** \- 的输入张量形状 （ minibatch  ， in_channels  ， i的 T  ， i的 H  ， i的 W  ） （\文本{minibatch}，\文本{在\ _channels}，这，1H，IW） （ minibatch  ， in_channels  ， i的 T  ， i的 H  ， i的 W  ）

  * **重量** \- 的过滤器形状 （ out_channels  ， in_channels  基团 ， K  T  ， K  H  ， K  W  ） （\文本{出\ _channels}，\压裂{\文本{在\ _channels}} {\文本{基}}，KT，KH，千瓦） （ out_channels  ， 基团 in_channels  ， K  T  ， K  H  ， K  W  ）

  * **偏压** \- 形状 （ out_channels  ）[的可选偏置张量HTG13]  （\文本{出\ _channels}） （ out_channels  ） 。默认值：无

  * **步幅** \- 在卷积内核的步幅。可以是单一的数或一个元组（ST，SH，SW）。默认值：1

  * **填充** \- 对输入的两侧隐填补处理。可以是单一的数或一个元组（PADT，PADH，padW）。默认值：0

  * **扩张** \- 内核元件之间的间隔。可以是单一的数或一个元组（DT，DH，一页）。默认值：1

  * **groups** – split input into groups, in_channels\text{in\\_channels}in_channels should be divisible by the number of groups. Default: 1

Examples:

    
    
    >>> filters = torch.randn(33, 16, 3, 3, 3)
    >>> inputs = torch.randn(20, 16, 50, 10, 20)
    >>> F.conv3d(inputs, filters)
    

###  conv_transpose1d 

`torch.nn.functional.``conv_transpose1d`( _input_ , _weight_ , _bias=None_ ,
_stride=1_ , _padding=0_ , _output_padding=0_ , _groups=1_ , _dilation=1_ ) →
Tensor

    

适用过的几个输入飞机组成的输入信号，有时也称为“反卷积”一维的转置卷积运算。

参见[ `ConvTranspose1d`](nn.html#torch.nn.ConvTranspose1d
"torch.nn.ConvTranspose1d")的详细信息和输出形状。

Note

In some circumstances when using the CUDA backend with CuDNN, this operator
may select a nondeterministic algorithm to increase performance. If this is
undesirable, you can try to make the operation deterministic (potentially at a
performance cost) by setting `torch.backends.cudnn.deterministic = True`.
Please see the notes on [Reproducibility](notes/randomness.html) for
background.

Parameters

    

  * **input** – input tensor of shape (minibatch,in_channels,iW)(\text{minibatch} , \text{in\\_channels} , iW)(minibatch,in_channels,iW)

  * **重量** \- 的过滤器形状 （ in_channels  ， out_channels  基团 ， K  W  ） （\文本{在\ _channels}，\压裂{\文本{出\ _channels}} {\文本{基}}，千瓦） （ in_channels  ， 基团 out_channels  [HTG9 3]  ， K  W  ）

  * **偏压** \- 形状 （ out_channels  ）[的可选偏置HTG13 ]  （\文本{出\ _channels}） （ out_channels  ） 。默认值：无

  * **步幅** \- 在卷积内核的步幅。可以是单一的数或一个元组`（SW） `。默认值：1

  * **填充** \- `扩张 *  （kernel_size  -  1） -  填充 `零填充将被添加到输入中的每个维度的两侧。可以是单一的数或一个元组`（padW，） `。默认值：0

  * **output_padding** \- 附加大小添加到在输出形状的每个维度的一侧。可以是单一的数或一个元组`（out_padW） `。默认值：0

  * **groups** – split input into groups, in_channels\text{in\\_channels}in_channels should be divisible by the number of groups. Default: 1

  * **扩张** \- 内核元件之间的间隔。可以是单一的数或一个元组`（DW） `。默认值：1

Examples:

    
    
    >>> inputs = torch.randn(20, 16, 50)
    >>> weights = torch.randn(16, 33, 5)
    >>> F.conv_transpose1d(inputs, weights)
    

###  conv_transpose2d 

`torch.nn.functional.``conv_transpose2d`( _input_ , _weight_ , _bias=None_ ,
_stride=1_ , _padding=0_ , _output_padding=0_ , _groups=1_ , _dilation=1_ ) →
Tensor

    

应用在多个输入平面构成的输入图像的2D转卷积运算，有时也被称为“反卷积”。

参见[ `ConvTranspose2d`](nn.html#torch.nn.ConvTranspose2d
"torch.nn.ConvTranspose2d")的详细信息和输出形状。

Note

In some circumstances when using the CUDA backend with CuDNN, this operator
may select a nondeterministic algorithm to increase performance. If this is
undesirable, you can try to make the operation deterministic (potentially at a
performance cost) by setting `torch.backends.cudnn.deterministic = True`.
Please see the notes on [Reproducibility](notes/randomness.html) for
background.

Parameters

    

  * **input** – input tensor of shape (minibatch,in_channels,iH,iW)(\text{minibatch} , \text{in\\_channels} , iH , iW)(minibatch,in_channels,iH,iW)

  * **重量** \- 的过滤器形状 （ in_channels  ， out_channels  基团 ， K  H  ， K  W  ） （\文本{在\ _channels}，\压裂{\文本{出\ _channels}} {\文本{基}}，KH，千瓦） （ in_channels  ， 基团 out_channels  ， K  H  ， K  W  ）

  * **bias** – optional bias of shape (out_channels)(\text{out\\_channels})(out_channels) . Default: None

  * **步幅** \- 在卷积内核的步幅。可以是 单数或一个元组`（SH， SW）。默认值：1`

  * **填充** \- `扩张 *  （kernel_size  -  1） -  填充 `零填充将被添加到输入中的每个维度的两侧。可以是单一的数或一个元组`（PADH， padW） `。默认值：0

  * **output_padding** \- 附加大小添加到在输出形状的每个维度的一侧。可以是单一的数或一个元组`（out_padH， out_padW） `。默认值：0

  * **groups** – split input into groups, in_channels\text{in\\_channels}in_channels should be divisible by the number of groups. Default: 1

  * **扩张** \- 内核元件之间的间隔。可以是单一的数或一个元组`（DH， 一页） `。默认值：1

Examples:

    
    
    >>> # With square kernels and equal stride
    >>> inputs = torch.randn(1, 4, 5, 5)
    >>> weights = torch.randn(4, 8, 3, 3)
    >>> F.conv_transpose2d(inputs, weights, padding=1)
    

###  conv_transpose3d 

`torch.nn.functional.``conv_transpose3d`( _input_ , _weight_ , _bias=None_ ,
_stride=1_ , _padding=0_ , _output_padding=0_ , _groups=1_ , _dilation=1_ ) →
Tensor

    

应用在多个输入平面构成的输入图像的3D换位卷积运算，有时也被称为“解卷积”

参见[ `ConvTranspose3d`](nn.html#torch.nn.ConvTranspose3d
"torch.nn.ConvTranspose3d")的详细信息和输出形状。

Note

In some circumstances when using the CUDA backend with CuDNN, this operator
may select a nondeterministic algorithm to increase performance. If this is
undesirable, you can try to make the operation deterministic (potentially at a
performance cost) by setting `torch.backends.cudnn.deterministic = True`.
Please see the notes on [Reproducibility](notes/randomness.html) for
background.

Parameters

    

  * **input** – input tensor of shape (minibatch,in_channels,iT,iH,iW)(\text{minibatch} , \text{in\\_channels} , iT , iH , iW)(minibatch,in_channels,iT,iH,iW)

  * **重量** \- 的过滤器形状 （ in_channels  ， out_channels  基团 ， K  T  ， K  H  ， K  W  ） （\文本{在\ _channels}，\压裂{\文本{出\ _channels}} {\文本{基}}，KT，KH，千瓦） （ in_channels  ， 基团 out_channels  ， K  T  ， K  H  ， K  W  ）

  * **bias** – optional bias of shape (out_channels)(\text{out\\_channels})(out_channels) . Default: None

  * **步幅** \- 在卷积内核的步幅。可以是（  SH ST， SW） 单数或一个元组`。默认值：1`

  * **填充** \- `扩张 *  （kernel_size  -  1） -  填充 `零填充将被添加到输入中的每个维度的两侧。可以是 单数或一个元组`（PADT， PADH， padW）。默认值：0`

  * **output_padding** \- 附加大小添加到在输出形状的每个维度的一侧。可以是（  out_padH out_padT， out_padW）单个数字或一个元组``。默认值：0

  * **groups** – split input into groups, in_channels\text{in\\_channels}in_channels should be divisible by the number of groups. Default: 1

  * **dilation** – the spacing between kernel elements. Can be a single number or a tuple (dT, dH, dW). Default: 1

Examples:

    
    
    >>> inputs = torch.randn(20, 16, 50, 10, 20)
    >>> weights = torch.randn(16, 33, 3, 3, 3)
    >>> F.conv_transpose3d(inputs, weights)
    

### 展开

`torch.nn.functional.``unfold`( _input_ , _kernel_size_ , _dilation=1_ ,
_padding=0_ , _stride=1_
)[[source]](_modules/torch/nn/functional.html#unfold)

    

从提取的批量输入张量滑动局部块。

警告

目前，只有4-d的输入张量（成批图像样张量）的支持。

Warning

展开的张量的多于一个的元件可指代单个存储器位置。其结果是，就地操作（特别是那些有量化的）可能会导致不正确的行为。如果你需要写张，请先克隆。

参见[ `torch.nn.Unfold`](nn.html#torch.nn.Unfold "torch.nn.Unfold")详细内容

### 倍

`torch.nn.functional.``fold`( _input_ , _output_size_ , _kernel_size_ ,
_dilation=1_ , _padding=0_ , _stride=1_
)[[source]](_modules/torch/nn/functional.html#fold)

    

结合滑动局部块到大量含有张量的阵列。

Warning

目前，只有4-d输出张量（成批图像样张量）的支持。

参见[ `torch.nn.Fold`](nn.html#torch.nn.Fold "torch.nn.Fold")详细内容

## 汇集功能

###  avg_pool1d 

`torch.nn.functional.``avg_pool1d`( _input_ , _kernel_size_ , _stride=None_ ,
_padding=0_ , _ceil_mode=False_ , _count_include_pad=True_ ) → Tensor

    

适用在几个输入平面组成的输入信号的平均1D池。

参见[ `AvgPool 1D`](nn.html#torch.nn.AvgPool1d
"torch.nn.AvgPool1d")的细节和输出形状。

Parameters

    

  * **input** – input tensor of shape (minibatch,in_channels,iW)(\text{minibatch} , \text{in\\_channels} , iW)(minibatch,in_channels,iW)

  * **kernel_size** \- 窗口的大小。可以是单一的数或一个元组（千瓦）

  * **步幅** \- 窗口的步幅。可以是单一的数或一个元组（SW）。默认值：`kernel_size`

  * **填充** \- 对输入的两侧隐含零个填补处理。可以是单一的数或一个元组（padW，）。默认值：0

  * **ceil_mode** \- 真时，将使用小区而非地板来计算输出形状。默认值：`假 `

  * **count_include_pad** \- 真时，将包括在平均计算的零填充。默认值：`真 `

Examples:

    
    
    >>> # pool of square window of size=3, stride=2
    >>> input = torch.tensor([[[1, 2, 3, 4, 5, 6, 7]]], dtype=torch.float32)
    >>> F.avg_pool1d(input, kernel_size=3, stride=2)
    tensor([[[ 2.,  4.,  6.]]])
    

###  avg_pool2d 

`torch.nn.functional.``avg_pool2d`( _input_ , _kernel_size_ , _stride=None_ ,
_padding=0_ , _ceil_mode=False_ , _count_include_pad=True_ ,
_divisor_override=None_ ) → Tensor

    

适用于 K  H  × ķ2D平均池操作 W  的kH \倍千瓦 K  H  × K  W¯¯  由步长HTG48]  S  H [HTG57区域] × S
W  SH \倍SW  S  H  × S  W  步骤。的输出特征的数量等于输入平面的数量。

参见[ `AvgPool2d`](nn.html#torch.nn.AvgPool2d
"torch.nn.AvgPool2d")的详细信息和输出形状。

Parameters

    

  * **输入** \- 输入张量 （ minibatch  ， in_channels  ， i的 H  ， i的 W  ） （\文本{minibatch}，\文本{在\ _channels}，1H，IW） （ minibatch  ， in_channels  ， i的 H  ， i的 W  ）

  * **kernel_size** \- 汇集区域的大小。可以是单一的数或一个元组（KH，千瓦）

  * **步幅** \- 汇集操作步幅。可以是单一的数或一个元组（SH，SW）。默认值：`kernel_size`

  * **填充** \- 对输入的两侧隐含零个填补处理。可以是单一的数或一个元组（PADH，padW）。默认值：0

  * **ceil_mode** \- 真时，将使用小区而非地板式中来计算输出形状。默认值：`假 `

  * **count_include_pad** – when True, will include the zero-padding in the averaging calculation. Default: `True`

  * **divisor_override** \- 如果指定的话，它将被用作除数时，池区的尺寸，否则将被使用。默认值：无

###  avg_pool3d 

`torch.nn.functional.``avg_pool3d`( _input_ , _kernel_size_ , _stride=None_ ,
_padding=0_ , _ceil_mode=False_ , _count_include_pad=True_ ,
_divisor_override=None_ ) → Tensor

    

适用于 K  T  × ķ3D平均池操作 H  × K  W  KT \倍的kH \倍千瓦 K  T  × K  H  × K  W  通过步骤大小的区域
S  T  × S  H  × S  W  ST \倍SH \倍SW  S  T  × S  H  × S  W  步骤。的输出特征的数量等于 ⌊ 输入平面
S  T  ⌋ \ lfloor \压裂{\文本{输入平面}} {ST} \ rfloor  ⌊ S  T  输入平面 ⌋ 。

参见[ `AvgPool3d`](nn.html#torch.nn.AvgPool3d
"torch.nn.AvgPool3d")的详细信息和输出形状。

Parameters

    

  * **输入** \- 输入张量 （ minibatch  ， in_channels  ， i的 T  × i的 H  ， i的 W  ） （\文本{minibatch}，\文本{在\ _channels}，这\倍IH，IW） （ minibatch  ， in_channels  ， i的 T  × i的 H  i的 W  ） [HTG9 5]

  * **kernel_size** \- 汇集区域的大小。可以是单一的数或一个元组（KT，KH，千瓦）

  * **步幅** \- 汇集操作步幅。可以是单一的数或一个元组（ST，SH，SW）。默认值：`kernel_size`

  * **填充** \- 对输入的两侧隐含零个填补处理。可以是单一的数或一个元组（PADT，PADH，padW），默认值：0

  * **ceil_mode** \- 真时，将使用小区而非地板式中来计算输出形状

  * **count_include_pad** \- 真时，将包括在平均计算补零

  * **divisor_override** – if specified, it will be used as divisor, otherwise size of the pooling region will be used. Default: None

###  max_pool1d 

`torch.nn.functional.``max_pool1d`( _*args_ , _**kwargs_ )

    

应用于一维的最大汇集了多个输入飞机组成的输入信号。

参见[ `MaxPool1d`](nn.html#torch.nn.MaxPool1d "torch.nn.MaxPool1d")了解详情。

###  max_pool2d 

`torch.nn.functional.``max_pool2d`( _*args_ , _**kwargs_ )

    

施加最大的2D汇集在几个输入平面组成的输入信号。

参见[ `MaxPool2d`](nn.html#torch.nn.MaxPool2d "torch.nn.MaxPool2d")了解详情。

###  max_pool3d 

`torch.nn.functional.``max_pool3d`( _*args_ , _**kwargs_ )

    

应用了3D最大汇集了多个输入飞机组成的输入信号。

参见[ `MaxPool3d`](nn.html#torch.nn.MaxPool3d "torch.nn.MaxPool3d")了解详情。

###  max_unpool1d 

`torch.nn.functional.``max_unpool1d`( _input_ , _indices_ , _kernel_size_ ,
_stride=None_ , _padding=0_ , _output_size=None_
)[[source]](_modules/torch/nn/functional.html#max_unpool1d)

    

计算的`MaxPool1d`的局部逆。

参见[ `MaxUnpool1d`](nn.html#torch.nn.MaxUnpool1d
"torch.nn.MaxUnpool1d")了解详情。

###  max_unpool2d 

`torch.nn.functional.``max_unpool2d`( _input_ , _indices_ , _kernel_size_ ,
_stride=None_ , _padding=0_ , _output_size=None_
)[[source]](_modules/torch/nn/functional.html#max_unpool2d)

    

计算的`MaxPool2d`的局部逆。

参见[ `MaxUnpool2d`](nn.html#torch.nn.MaxUnpool2d
"torch.nn.MaxUnpool2d")了解详情。

###  max_unpool3d 

`torch.nn.functional.``max_unpool3d`( _input_ , _indices_ , _kernel_size_ ,
_stride=None_ , _padding=0_ , _output_size=None_
)[[source]](_modules/torch/nn/functional.html#max_unpool3d)

    

计算的`MaxPool3d`的局部逆。

参见[ `MaxUnpool3d`](nn.html#torch.nn.MaxUnpool3d
"torch.nn.MaxUnpool3d")了解详情。

###  lp_pool1d 

`torch.nn.functional.``lp_pool1d`( _input_ , _norm_type_ , _kernel_size_ ,
_stride=None_ , _ceil_mode=False_
)[[source]](_modules/torch/nn/functional.html#lp_pool1d)

    

适用在几个输入平面组成的输入信号的功率1D平均池。如果所有输入至p的的功率之和为零时，梯度被设置为零。

参见[ `LPPool1d`](nn.html#torch.nn.LPPool1d "torch.nn.LPPool1d")了解详情。

###  lp_pool2d 

`torch.nn.functional.``lp_pool2d`( _input_ , _norm_type_ , _kernel_size_ ,
_stride=None_ , _ceil_mode=False_
)[[source]](_modules/torch/nn/functional.html#lp_pool2d)

    

适用在几个输入平面组成的输入信号的2D功率平均池。如果所有输入至p的的功率之和为零时，梯度被设置为零。

参见[ `LPPool2d`](nn.html#torch.nn.LPPool2d "torch.nn.LPPool2d")了解详情。

###  adaptive_max_pool1d 

`torch.nn.functional.``adaptive_max_pool1d`( _*args_ , _**kwargs_ )

    

适用在几个输入平面组成的输入信号的1D自适应最大池。

参见[ `AdaptiveMaxPool1d`](nn.html#torch.nn.AdaptiveMaxPool1d
"torch.nn.AdaptiveMaxPool1d")的详细信息和输出形状。

Parameters

    

  * **output_size** \- 目标输出大小（单整数）

  * **return_indices** \- 是否返回池指数。默认值：`假 `

###  adaptive_max_pool2d 

`torch.nn.functional.``adaptive_max_pool2d`( _*args_ , _**kwargs_ )

    

适用在几个输入平面组成的输入信号的2D自适应最大池。

参见[ `AdaptiveMaxPool2d`](nn.html#torch.nn.AdaptiveMaxPool2d
"torch.nn.AdaptiveMaxPool2d")的详细信息和输出形状。

Parameters

    

  * **output_size** \- 目标输出大小（单整数或双整数元组）

  * **return_indices** – whether to return pooling indices. Default: `False`

###  adaptive_max_pool3d 

`torch.nn.functional.``adaptive_max_pool3d`( _*args_ , _**kwargs_ )

    

适用在几个输入平面组成的输入信号的3D自适应最大池。

参见[ `[HTG2自适应MaxPool3d `](nn.html#torch.nn.AdaptiveMaxPool3d
"torch.nn.AdaptiveMaxPool3d")的细节和输出形状。

Parameters

    

  * **output_size** \- 目标输出大小（单整数或三元组整数）

  * **return_indices** – whether to return pooling indices. Default: `False`

###  adaptive_avg_pool1d 

`torch.nn.functional.``adaptive_avg_pool1d`( _input_ , _output_size_ ) →
Tensor

    

适用在几个输入平面组成的输入信号的1D自适应平均池。

参见[ `[HTG2自适应AvgPool 1D `](nn.html#torch.nn.AdaptiveAvgPool1d
"torch.nn.AdaptiveAvgPool1d")的详细信息和输出形状。

Parameters

    

**output_size** – the target output size (single integer)

###  adaptive_avg_pool2d 

`torch.nn.functional.``adaptive_avg_pool2d`( _input_ , _output_size_
)[[source]](_modules/torch/nn/functional.html#adaptive_avg_pool2d)

    

适用在几个输入平面组成的输入信号的2D自适应平均池。

参见[ `AdaptiveAvgPool2d`](nn.html#torch.nn.AdaptiveAvgPool2d
"torch.nn.AdaptiveAvgPool2d")的详细信息和输出形状。

Parameters

    

**output_size** – the target output size (single integer or double-integer
tuple)

###  adaptive_avg_pool3d 

`torch.nn.functional.``adaptive_avg_pool3d`( _input_ , _output_size_
)[[source]](_modules/torch/nn/functional.html#adaptive_avg_pool3d)

    

适用在几个输入平面组成的输入信号的3D自适应平均池。

参见[ `AdaptiveAvgPool3d`](nn.html#torch.nn.AdaptiveAvgPool3d
"torch.nn.AdaptiveAvgPool3d")的详细信息和输出形状。

Parameters

    

**output_size** – the target output size (single integer or triple-integer
tuple)

## 非线性激活函数

### 阈

`torch.nn.functional.``threshold`( _input_ , _threshold_ , _value_ ,
_inplace=False_ )[[source]](_modules/torch/nn/functional.html#threshold)

    

阈值输入张量的每个元素。

参见[ `阈值 `](nn.html#torch.nn.Threshold "torch.nn.Threshold")了解更多详情。

`torch.nn.functional.``threshold_`( _input_ , _threshold_ , _value_ ) →
Tensor

    

就地版本的 `阈值（） `。

###  RELU 

`torch.nn.functional.``relu`( _input_ , _inplace=False_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#relu)

    

施加整流的线性单元函数逐元素。参见[ `RELU`](nn.html#torch.nn.ReLU "torch.nn.ReLU")了解更多详情。

`torch.nn.functional.``relu_`( _input_ ) → Tensor

    

就地版本的 `RELU（） `。

###  hardtanh 

`torch.nn.functional.``hardtanh`( _input_ , _min_val=-1._ , _max_val=1._ ,
_inplace=False_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#hardtanh)

    

应用HardTanh功能件明智的。参见[ `Hardtanh`](nn.html#torch.nn.Hardtanh
"torch.nn.Hardtanh")了解更多详情。

`torch.nn.functional.``hardtanh_`( _input_ , _min_val=-1._ , _max_val=1._ ) →
Tensor

    

就地版本的 `hardtanh（） `。

###  relu6 

`torch.nn.functional.``relu6`( _input_ , _inplace=False_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#relu6)

    

适用逐元素函数 ReLU6  （ × ） =  分钟HTG17] ⁡ （ MAX  ⁡ （ 0  ， × ） ， 6  ） \文本{ReLU6}（X）=
\分钟（\ MAX（0，x）时，6） ReLU6  （ × ） =  分钟HTG73] （ MAX  （ 0  ， × ） ， 6  ） [HT G98]
。

参见[ `ReLU6`](nn.html#torch.nn.ReLU6 "torch.nn.ReLU6")了解更多详情。

###  ELU 

`torch.nn.functional.``elu`( _input_ , _alpha=1.0_ , _inplace=False_
)[[source]](_modules/torch/nn/functional.html#elu)

    

适用逐元素， ELU  （ × ） =  MAX  ⁡ （ 0  ， × ） \+  分钟HTG33] ⁡ （ 0  ， α *  （ EXP  ⁡ （ ×
） \-  1  ） ） \文本{ELU}（X）= \最大（0，x）的+ \分钟（0，\阿尔法*（\ EXP（X） - 1）） ELU  （ × ） =
MAX  （ 0  ， × ） \+  分钟HTG121] （ 0  α *  （ 实验值 （ × ） \-  1  ） ） 。

参见[ `ELU`](nn.html#torch.nn.ELU "torch.nn.ELU")了解更多详情。

`torch.nn.functional.``elu_`( _input_ , _alpha=1._ ) → Tensor

    

就地版本的 `ELU（） `。

### 九色鹿

`torch.nn.functional.``selu`( _input_ , _inplace=False_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#selu)

    

适用逐元素， 九色鹿 （ × ） =  S  C  一 L  E  *  （ MAX  ⁡ （ 0  ， × ） \+  分钟HTG47] ⁡ （ 0  ，
α *  （ EXP  ⁡ （ × ） \-  1  ） ） ） \文本{九色鹿} （X）=比例*（\ MAX（0，x）的+ \分钟（0，\阿尔法*（\
EXP（X） - 1））） 九色鹿 （ × ） =  S  C  一 L  E  *  （ MAX  （ 0  ， × ） \+  分钟HTG159] （
0  ， α *  （ EXP  （ × ） [HT G191]  \-  1  ） ） ） ，其中 α =
1.6732632423543772848170429916717  \阿尔法= 1.6732632423543772848170429916717  α
=  1  。  6  7  3  2  6  3  2  4  2  3  ​​ 5  4  3  7  7  2  8  4  8  1  7  0
4  2  9  9  1  6  7  1  7  和 S  C  一 L  E  =
1.0507009873554804934193349852946  规模= 1.050700987355480493419334985 2946  S
C  一 升 E  =  1  。  0  5  0  7  0  0  9  8  7  3  5  5  4  8  0  4  9  3  4  1
9  3  3  4  9  8  5  2  9  4  6  。

参见[ `九色鹿 `](nn.html#torch.nn.SELU "torch.nn.SELU")了解更多详情。

###  celu 

`torch.nn.functional.``celu`( _input_ , _alpha=1._ , _inplace=False_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#celu)

    

适用逐元素， CELU  （ × ） =  MAX  ⁡ （ 0  ， × ） \+  分钟HTG33] ⁡ （ 0  ， α *  （ EXP  ⁡ （
× /  α ） \-  1  ） ） \文本{CELU}（X）= \ MAX（0，x）的+ \分钟（0，\阿尔法*（\ EXP（X / \阿尔法） -
1）） CELU  （ × ） =  MAX  （ 0  ， × ） \+  分钟HTG125] （ 0  ， α *  （ EXP  （ × /  α ）
\-  1  ） ） 。

参见[ `CELU`](nn.html#torch.nn.CELU "torch.nn.CELU")了解更多详情。

###  leaky_relu 

`torch.nn.functional.``leaky_relu`( _input_ , _negative_slope=0.01_ ,
_inplace=False_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#leaky_relu)

    

适用逐元素， LeakyReLU  （ × ） =  MAX  ⁡ （ 0  ， × ） \+  negative_slope  *  分钟HTG37] ⁡
（ 0  ， × ） \文本{LeakyReLU}（X）= \ MAX（0，X）+ \文本{负\ _slope} * \分钟（0，x）的 LeakyReLU
（ × ） =  MAX  （ 0  ， × ） \+  negative_slope  *  分钟HTG119] （ 0  × ）

参见[ `LeakyReLU`](nn.html#torch.nn.LeakyReLU "torch.nn.LeakyReLU")了解更多详情。

`torch.nn.functional.``leaky_relu_`( _input_ , _negative_slope=0.01_ ) →
Tensor

    

就地版本的 `leaky_relu（） `。

###  prelu 

`torch.nn.functional.``prelu`( _input_ , _weight_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#prelu)

    

适用逐元素的函数 PReLU  （ × ） =  MAX  ⁡ （ 0  ， × ） \+  重量 *  分钟HTG37] ⁡ （ 0  ， × ）
\文本{PReLU}（X）= \ MAX（0，x）的+ \文本{重量} * \分钟（0，x）的 PReLU  （ × ） =  MAX  （ 0  ， ×
） \+  重量 *  分钟HTG119] （ 0  ， × ） 其中权重是可学习参数。

参见[ `PReLU`](nn.html#torch.nn.PReLU "torch.nn.PReLU")了解更多详情。

###  rrelu 

`torch.nn.functional.``rrelu`( _input_ , _lower=1./8_ , _upper=1./3_ ,
_training=False_ , _inplace=False_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#rrelu)

    

随机漏RELU。

参见[ `RReLU`](nn.html#torch.nn.RReLU "torch.nn.RReLU")了解更多详情。

`torch.nn.functional.``rrelu_`( _input_ , _lower=1./8_ , _upper=1./3_ ,
_training=False_ ) → Tensor

    

就地版本的 `rrelu（） `。

###  glu的

`torch.nn.functional.``glu`( _input_ , _dim=-1_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#glu)

    

门控线性单元。计算：

GLU(a,b)=a⊗σ(b)\text{GLU}(a, b) = a \otimes \sigma(b) GLU(a,b)=a⊗σ(b)

其中，输入被分成两半沿暗淡，以形成一和 B ， σ \西格玛 σ 为S形函数和 ⊗ \ otimes  ⊗ 为逐元素矩阵之间的产物。

参见[带门卷积网络](https://arxiv.org/abs/1612.08083)语言模型。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入张量

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 在其上分割输入维数。默认值：-1

### 格鲁

`torch.nn.functional.``gelu`( _input_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#gelu)

    

适用逐元素的函数 格鲁 （ × ） =  × *  Φ （ × ） \文本{格鲁}（X）= X * \披（X） 格鲁 （ × ） =  × *  Φ （ ×
）

其中 Φ （ × ） \披（X） Φ （ ×  ） 为高斯分布的累积分布函数。

参见[高斯误差线性单位（GELUs）](https://arxiv.org/abs/1606.08415)。

###  logsigmoid 

`torch.nn.functional.``logsigmoid`( _input_ ) → Tensor

    

适用逐元素 LogSigmoid  （ × i的 ） =  日志 ⁡ （ 1  1  \+  EXP  ⁡ （ \-  × i的 ） ）
\文本{LogSigmoid}（X_I）= \ LOG \左（\压裂{1} {1 + \ EXP（-x_i）} \右） LogSigmoid  （ × i的
[ H TG96]  ） =  LO  G  （ 1  \+  EXP  （ \-  × i的 ） 1  ）

参见[ `LogSigmoid`](nn.html#torch.nn.LogSigmoid
"torch.nn.LogSigmoid")了解更多详情。

###  hardshrink 

`torch.nn.functional.``hardshrink`( _input_ , _lambd=0.5_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#hardshrink)

    

适用的硬收缩函数逐元素

参见[ `Hardshrink`](nn.html#torch.nn.Hardshrink
"torch.nn.Hardshrink")了解更多详情。

###  tanhshrink 

`torch.nn.functional.``tanhshrink`( _input_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#tanhshrink)

    

适用逐元素， Tanhshrink  （ × ） =  × \-  双曲正切 （ × ） \文本{Tanhshrink}（X）= X -
\文本{双曲正切}（X） Tanhshrink  （ × ） =  × \-  双曲正切 （ × ）

参见[ `Tanhshrink`](nn.html#torch.nn.Tanhshrink
"torch.nn.Tanhshrink")了解更多详情。

###  softsign 

`torch.nn.functional.``softsign`( _input_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#softsign)

    

适用逐元素，函数 SoftSign  （ × ）  =  × 1  \+  |  × |  \文本{SoftSign}（X）= \压裂{X} {1 + |
X |}  SoftSign  （ × ） =  1  \+  |  × |  ×

参见[ `Softsign`](nn.html#torch.nn.Softsign "torch.nn.Softsign")了解更多详情。

###  softplus 

`torch.nn.functional.``softplus`( _input_ , _beta=1_ , _threshold=20_ ) →
Tensor

    

###  softmin 

`torch.nn.functional.``softmin`( _input_ , _dim=None_ , __stacklevel=3_ ,
_dtype=None_ )[[source]](_modules/torch/nn/functional.html#softmin)

    

应用一个softmin功能。

注意， Softmin  （ × ） =  使用SoftMax  （ \-  × ） \文本{Softmin }（X）= \文本{使用SoftMax}（ -
x）的 Softmin  （ × ） =  使用SoftMax  （ \-  × ） 。见数学公式定义添加Softmax。

参见[ `Softmin`](nn.html#torch.nn.Softmin "torch.nn.Softmin")了解更多详情。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 沿其softmin将被计算的尺寸（因此沿暗淡每片将总结为1）。

  * **DTYPE** （`torch.dtype`，可选） - 所需的数据返回张量的类型。如果指定，输入张量浇铸到`在执行操作之前D型细胞 `。这是为了防止数据溢出型有用。默认值：无。

###  SOFTMAX 

`torch.nn.functional.``softmax`( _input_ , _dim=None_ , __stacklevel=3_ ,
_dtype=None_ )[[source]](_modules/torch/nn/functional.html#softmax)

    

应用一个SOFTMAX功能。

使用SoftMax定义为：

使用SoftMax  （ × i的 ） =  E  × p  （ × i的 ） Σ [HTG43：J  E  × p  （ × [HTG57：J  ）
\文本{使用SoftMax}（X_ {I}）= \压裂{EXP（X_I）} {\ sum_j EXP（x_j）}  使用SoftMax  （ × i的 ）
=  Σ [HTG145：J  E  × p  （ × [HTG183：J  [HT G199] ） E  × p  （ × i的 ） ​​

它适用于沿暗淡所有切片，并且将重新缩放它们使得元件位于在范围 [0,1] 和总和为1。

参见[ `SOFTMAX`](nn.html#torch.nn.Softmax "torch.nn.Softmax")了解更多详情。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – input

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 沿其SOFTMAX将被计算的尺寸。

  * **dtype** (`torch.dtype`, optional) – the desired data type of returned tensor. If specified, the input tensor is casted to `dtype`before the operation is performed. This is useful for preventing data type overflows. Default: None.

Note

此功能不与NLLLoss，其预计使用SoftMax和自身之间要计算日志直接工作。使用log_softmax代替（它的速度更快，具有更好的数值属性）。

###  softshrink 

`torch.nn.functional.``softshrink`( _input_ , _lambd=0.5_ ) → Tensor

    

应用软收缩功能的elementwise

参见[ `Softshrink`](nn.html#torch.nn.Softshrink
"torch.nn.Softshrink")了解更多详情。

###  gumbel_softmax 

`torch.nn.functional.``gumbel_softmax`( _logits_ , _tau=1_ , _hard=False_ ,
_eps=1e-10_ , _dim=-1_
)[[source]](_modules/torch/nn/functional.html#gumbel_softmax)

    

从冈贝尔-使用SoftMax分布（[链接1 ](https://arxiv.org/abs/1611.00712) [链路2
](https://arxiv.org/abs/1611.01144)）和任选的离散化的样品。

Parameters

    

  * **logits** \-  [...，NUM_FEATURES] 非标准化数概率

  * **tau蛋白** \- 非负标量温度

  * **硬** \- 如果`真 `，返回的样品将被离散化作为一热载体，但将被区分为如果是在autograd软样品

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 沿其SOFTMAX将被计算的尺寸。缺省值：-1。

Returns

    

相同的形状，从冈贝尔-使用SoftMax分布logits 的采样张量。如果`硬=真 `，返回的样品将一热的，否则会概率分布其总和为1所有暗淡。

Note

此功能是在这里遗留原因，可以从nn.Functional在将来被移除。

Note

对于主特技硬是做 y_hard - y_soft.detach（）+ y_soft

它实现了两件事情： - 使输出值完全独热（因为我们添加然后减去y_soft值） - 使梯度等于y_soft梯度（因为我们去除所有其他梯度）

Examples::

    
    
    
    >>> logits = torch.randn(20, 32)
    >>> # Sample soft categorical using reparametrization trick:
    >>> F.gumbel_softmax(logits, tau=1, hard=False)
    >>> # Sample hard categorical using "Straight-through" trick:
    >>> F.gumbel_softmax(logits, tau=1, hard=True)
    

###  log_softmax 

`torch.nn.functional.``log_softmax`( _input_ , _dim=None_ , __stacklevel=3_ ,
_dtype=None_ )[[source]](_modules/torch/nn/functional.html#log_softmax)

    

应用一个SOFTMAX接着是对数。

虽然数学上等同于登录（SOFTMAX（X）），分别在做这两个操作是比较慢，并且数值上是不稳定的。该函数使用一个替代的制剂来计算输出和正确梯度。

参见[ `LogSoftmax`](nn.html#torch.nn.LogSoftmax
"torch.nn.LogSoftmax")了解更多详情。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – input

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 甲沿log_softmax将被计算尺寸。

  * **dtype** (`torch.dtype`, optional) – the desired data type of returned tensor. If specified, the input tensor is casted to `dtype`before the operation is performed. This is useful for preventing data type overflows. Default: None.

### 的tanh 

`torch.nn.functional.``tanh`( _input_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#tanh)

    

适用逐元素， 双曲正切 （ × ） =  的tanh  ⁡ （ × ） =  实验值 ⁡ （ × ） \-  EXP  ⁡ （ \-  × ） EXP  ⁡
（ × ） \+  EXP  ⁡ （ \-  × ） \文本{双曲正切}（x）的= \的tanh（x）= \压裂{\ EXP（X） - \ EXP（-x）}
{\ EXP（X）+ \ EXP（-x）}  双曲正切 （ × ） =  的tanh  （ × ） =  实验值 （ × ） \+  EXP  （ \-
× ） EXP  （ × ） \-  EXP  （ \-  × ）

参见[ `双曲正切 `](nn.html#torch.nn.Tanh "torch.nn.Tanh")了解更多详情。

### 乙状结肠

`torch.nn.functional.``sigmoid`( _input_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#sigmoid)

    

适用逐元素函数 乙状结肠 （ × ） =  1  1  \+  实验值 ⁡ （ \-  × ） \文本{乙状结肠}（X）= \压裂{1} {1个+ \
EXP（-x）}  乙状结肠 （ × ） =  1  \+  实验值 （ \-  × ） [H T G98]  1

参见[ `乙状结肠 `](nn.html#torch.nn.Sigmoid "torch.nn.Sigmoid")了解更多详情。

## 归一化函数

###  batch_norm 

`torch.nn.functional.``batch_norm`( _input_ , _running_mean_ , _running_var_ ,
_weight=None_ , _bias=None_ , _training=False_ , _momentum=0.1_ , _eps=1e-05_
)[[source]](_modules/torch/nn/functional.html#batch_norm)

    

适用批标准化用于在批量数据的每个信道。

参见[ `BatchNorm1d`](nn.html#torch.nn.BatchNorm1d "torch.nn.BatchNorm1d")，[
`BatchNorm2d`](nn.html#torch.nn.BatchNorm2d "torch.nn.BatchNorm2d")，[ `
BatchNorm3d`](nn.html#torch.nn.BatchNorm3d "torch.nn.BatchNorm3d")了解详情。

###  instance_norm 

`torch.nn.functional.``instance_norm`( _input_ , _running_mean=None_ ,
_running_var=None_ , _weight=None_ , _bias=None_ , _use_input_stats=True_ ,
_momentum=0.1_ , _eps=1e-05_
)[[source]](_modules/torch/nn/functional.html#instance_norm)

    

适用实例标准化为在间歇的每个数据样本中的每一个通道。

参见[ `InstanceNorm1d`](nn.html#torch.nn.InstanceNorm1d
"torch.nn.InstanceNorm1d")，[ `InstanceNorm2d`
](nn.html#torch.nn.InstanceNorm2d "torch.nn.InstanceNorm2d")，[ `
InstanceNorm3d`](nn.html#torch.nn.InstanceNorm3d
"torch.nn.InstanceNorm3d")了解详情。

###  layer_norm 

`torch.nn.functional.``layer_norm`( _input_ , _normalized_shape_ ,
_weight=None_ , _bias=None_ , _eps=1e-05_
)[[source]](_modules/torch/nn/functional.html#layer_norm)

    

适用于过去的某些维数层正常化。

参见[ `LayerNorm`](nn.html#torch.nn.LayerNorm "torch.nn.LayerNorm")了解详情。

###  local_response_norm 

`torch.nn.functional.``local_response_norm`( _input_ , _size_ , _alpha=0.0001_
, _beta=0.75_ , _k=1.0_
)[[source]](_modules/torch/nn/functional.html#local_response_norm)

    

适用在几个输入平面，其中信道占用所述第二维组成的输入信号响应的本地归一化。适用跨渠道正常化。

参见[ `LocalResponseNorm`](nn.html#torch.nn.LocalResponseNorm
"torch.nn.LocalResponseNorm")了解详情。

### 正常化

`torch.nn.functional.``normalize`( _input_ , _p=2_ , _dim=1_ , _eps=1e-12_ ,
_out=None_ )[[source]](_modules/torch/nn/functional.html#normalize)

    

执行 L  P  L_P  L  p  超过规定尺寸的输入归一化。

对于张量`输入 `大小的 （ n的 0  ， 。 。 。 ， n的 d  i的 M  ， 。 。 。 ， n的 K  ） （N_0，...，N_
{暗淡}，...，n_k） （ n的 0  ， [HTG1 02]。  。  。  ， n的 d  i的 M  ， 。  。  。  ， n的 K  ）
，各 n的 d  i的 M  N_ {暗淡}  n的 d  i的 M  \- 元素向量 [HTG266】V  [HTG269】v ​​  [HTG278】v
沿着维度`暗淡 `被变换为

v=vmax⁡(∥v∥p,ϵ).v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.
v=max(∥v∥p​,ϵ)v​.

用默认的参数，它使用欧几里得范数超过矢量沿着维度 1  1  1  [HTG23用于归一化。

Parameters

    

  * **输入** \- 任何形状的输入张量

  * **P** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 在常态制剂中的指数值。默认值：2

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的尺寸，以减少。默认值：1

  * **EPS** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 小的值，以避免被零除。默认值：1E-12

  * **OUT** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 输出张量。如果`OUT`时，此操作将不会被微分的。

## 线性函数

### 线性

`torch.nn.functional.``linear`( _input_ , _weight_ , _bias=None_
)[[source]](_modules/torch/nn/functional.html#linear)

    

适用的线性变换，以将输入数据： Y  =  × A  T  \+  b  Y = XA ^ T + b  Y  =  × A  T  \+  b  。

形状：

>   * 输入： （ N  ， *  ， i的 n的 _  F  E  一 T  U  R  E  S  ） （N，*，在\ _features） （ N
， *  ， i的 n的 _  [HTG70 F的 E  一 T  U  R  E  S  ） 其中 * 是指任何数量的附加维度的

>

>   * 重量： （ O  U  T  _  F  E  一 T  U  R  E  S  ， i的 n的 _  F  ë  一 T  U  R  E
S  ） （下\ _features，在\ _features） （ O  U  T  _  F  E  一 T  U  R  E  S  ， i的 n的
_  F  E  一 T  U  R  E  S  ）

>

>   * 偏压： （ O  U  T  _  F  E  一 T  U  R  E  S  ） （下\ _features） （ O  U  T  _
F  E  一 T  U  R  E  S  ）

>

>   * 输出： （ N  ， *  ， O  U  T  _  F  E  一个 T  U  R  E  S  ） （N，*，出\ _features）
（ N  ， *  ， O  U  吨 _  F  E  一 T  U  R  E  S  ）

>

>

### 双线性

`torch.nn.functional.``bilinear`( _input1_ , _input2_ , _weight_ , _bias=None_
)[[source]](_modules/torch/nn/functional.html#bilinear)

    

## 差函数

### 差

`torch.nn.functional.``dropout`( _input_ , _p=0.5_ , _training=True_ ,
_inplace=False_ )[[source]](_modules/torch/nn/functional.html#dropout)

    

在训练期间，随机归零一些输入张量与概率`P`使用样品从贝努利分布元件。

参见[ `降 `](nn.html#torch.nn.Dropout "torch.nn.Dropout")了解详情。

Parameters

    

  * **P** \- 元素的概率将被归零。默认值：0.5

  * **训练** \- 如果是`真 `申请退学。默认值：`真 `

  * **就地** \- 如果设置为`真 `，会做此操作就地。默认值：`假 `

###  alpha_dropout 

`torch.nn.functional.``alpha_dropout`( _input_ , _p=0.5_ , _training=False_ ,
_inplace=False_ )[[source]](_modules/torch/nn/functional.html#alpha_dropout)

    

适用的α-差向输入。

参见[ `AlphaDropout`](nn.html#torch.nn.AlphaDropout
"torch.nn.AlphaDropout")了解详情。

###  dropout2d 

`torch.nn.functional.``dropout2d`( _input_ , _p=0.5_ , _training=True_ ,
_inplace=False_ )[[source]](_modules/torch/nn/functional.html#dropout2d)

    

随机零出整个信道（信道是2D特征映射，例如， [HTG6：J  [HTG9：J  [HTG18：J  的第信道 i的 i的 i的 在成批输入第样品是二维张量
输入 [ i的 ， [HTG62：J  \文本{输入} [I，J]  输入 [  i的 ， [HTG88：J  [H输入张量的TG91]
））。每个信道将与使用的样品从一个伯努利分布概率`P`独立地置零在每一个前向呼叫。

参见[ `Dropout2d`](nn.html#torch.nn.Dropout2d "torch.nn.Dropout2d")了解详情。

Parameters

    

  * **P** \- 一个信道的概率将被归零。默认值：0.5

  * **training** – apply dropout if is `True`. Default: `True`

  * **inplace** – If set to `True`, will do this operation in-place. Default: `False`

###  dropout3d 

`torch.nn.functional.``dropout3d`( _input_ , _p=0.5_ , _training=True_ ,
_inplace=False_ )[[source]](_modules/torch/nn/functional.html#dropout3d)

    

随机零出整个信道（信道是3D特征地图，例如， [HTG6：J  [HTG9：J  [HTG18：J  的第信道 i的 i的 i的 在成批输入第样品是三维张量
输入 [ i的 ， [HTG62：J  \文本{输入} [I，J]  输入 [  i的 ， [HTG88：J  [H输入张量的TG91]
））。每个信道将与使用的样品从一个伯努利分布概率`P`独立地置零在每一个前向呼叫。

参见[ `Dropout3d`](nn.html#torch.nn.Dropout3d "torch.nn.Dropout3d")了解详情。

Parameters

    

  * **p** – probability of a channel to be zeroed. Default: 0.5

  * **training** – apply dropout if is `True`. Default: `True`

  * **inplace** – If set to `True`, will do this operation in-place. Default: `False`

## 稀疏功能

### 嵌入

`torch.nn.functional.``embedding`( _input_ , _weight_ , _padding_idx=None_ ,
_max_norm=None_ , _norm_type=2.0_ , _scale_grad_by_freq=False_ ,
_sparse=False_ )[[source]](_modules/torch/nn/functional.html#embedding)

    

简单的查找表中查找在一个固定字典和大小的嵌入。

该模块经常被用来获取字的嵌入使用索引。输入到模块是指数列表，和嵌入基质，并且输出是对应的字的嵌入。

参见[ `torch.nn.Embedding`](nn.html#torch.nn.Embedding
"torch.nn.Embedding")了解更多详情。

Parameters

    

  * **输入** （ _LongTensor_ ） - 张量包含索引到嵌入基质

  * **重量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 与行数等于最大可能的索引+ 1，并等于嵌入尺寸的列数的包埋基质

  * **padding_idx** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 如果给定的，垫在与嵌入矢量输出`padding_idx`（初始化为零）每当遇到的索引。

  * **max_norm** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 如果给定的，具有范数大于各嵌入矢量`max_norm`被重新归一化，以具有规范`max_norm`。注意：这将修改`重量 `原地。

  * **norm_type** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 的p范数的p来计算用于`max_norm`选项。默认`2`。

  * **scale_grad_by_freq** （ _布尔_ _，_ _可选_ ） - 如果给出，这将通过的话频率在微型逆扩展梯度批量。默认的`假 [HTG11。`

  * **稀疏** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，梯度WRT `重量 `将是一个稀疏张量。参见下[ `笔记torch.nn.Embedding`[HTG23对于有关稀疏梯度的更多细节。](nn.html#torch.nn.Embedding "torch.nn.Embedding")

Shape:

    

  * 输入：包含索引来提取任意形状的LongTensor

  * Weight: Embedding matrix of floating point type with shape (V, embedding_dim),
    

其中V =最大索引+ 1和embedding_dim =嵌入尺寸

  * 输出：（*，embedding_dim），其中 * 是输入形状

Examples:

    
    
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
    

###  embedding_bag 

`torch.nn.functional.``embedding_bag`( _input_ , _weight_ , _offsets=None_ ,
_max_norm=None_ , _norm_type=2_ , _scale_grad_by_freq=False_ , _mode='mean'_ ,
_sparse=False_ , _per_sample_weights=None_
)[[source]](_modules/torch/nn/functional.html#embedding_bag)

    

计算总和，装置或马克塞斯袋【HTG1]的嵌入的的，没有实例化的嵌入中间。

参见[ `torch.nn.EmbeddingBag`](nn.html#torch.nn.EmbeddingBag
"torch.nn.EmbeddingBag")了解更多详情。

Note

当使用CUDA后端，该操作可以诱导向后非确定性的行为是不容易断开。请参阅[ 重复性 ](notes/randomness.html)为背景的音符。

Parameters

    

  * **输入** （ _LongTensor_ ） - 张量含有指数的袋装入嵌入基质

  * **weight** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – The embedding matrix with number of rows equal to the maximum possible index + 1, and number of columns equal to the embedding size

  * **偏移** （ _LongTensor_ _，_ _可选_ ） - 仅当`输入 `是1D使用。 `偏移 `确定`输入 `的每个袋（序列）的起始索引位置。

  * **max_norm** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – If given, each embedding vector with norm larger than `max_norm`is renormalized to have norm `max_norm`. Note: this will modify `weight`in-place.

  * **norm_type** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 在`P`在`p`范数来计算的`max_norm`选项。默认`2`。

  * **scale_grad_by_freq** （ _布尔_ _，_ _可选_ ） - 如果给定的，这将通过的话频率在微型逆扩展梯度批量。默认的`假 [HTG11。注意：不支持此选项时`模式= “MAX” [HTG15。``

  * **模式** （ _串_ _，_ _可选_ ） - `“总和” `，`“的意思是” `或`“最大” `。指定要降低袋的方式。默认值：`“的意思是” `

  * **稀疏** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，梯度WRT `重量 `将是一个稀疏张量。参见下[ `笔记torch.nn.Embedding`[HTG23对于有关稀疏梯度的更多细节。注意：不支持此选项时`模式= “MAX” [HTG27。`](nn.html#torch.nn.Embedding "torch.nn.Embedding")

  * **per_sample_weights** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 浮动/双权重的张量，或无来表示所有权重应取为1，如果指定，`per_sample_weights`必须具有完全相同的形状作为输入，被视为具有相同的`偏移 `，如果这些都不是无。

Shape:

>   * `输入 `（LongTensor）和`偏移 `（LongTensor，可选）

>

>     * 如果`输入 `是形状的2D （B，N）

>

> 它会被视为`B`袋（序列）各固定长度的`N`，这将返回`B`值在某种程度上取决于`模式 `聚合。 `偏移 `被忽略，并且需要为`
无 `在这种情况下。

>

>     * 如果`输入 `是形状的1D （N）

>

> 它会被视为多个袋（序列）的级联。 `偏移 `需要为含有`输入 `每个袋子的起始索引位置的1D张量。因此，对于`偏移 `形状的（B），`输入
`将被视为具有`B`袋。空袋通过零填充（即，具有长度为0的）将已经返回向量。

>

>   * `重量 `（张量）：形状（num_embeddings，embedding_dim）的模块的可学习权重

>

>   * `per_sample_weights`（张量，可选）。具有相同的形状为`输入 `。

>

>   * `输出 `：聚集嵌入形状的值（B，embedding_dim）

>

>

Examples:

    
    
    >>> # an Embedding module containing 10 tensors of size 3
    >>> embedding_matrix = torch.rand(10, 3)
    >>> # a batch of 2 samples of 4 indices each
    >>> input = torch.tensor([1,2,4,5,4,3,2,9])
    >>> offsets = torch.tensor([0,4])
    >>> F.embedding_bag(embedding_matrix, input, offsets)
    tensor([[ 0.3397,  0.3552,  0.5545],
            [ 0.5893,  0.4386,  0.5882]])
    

###  one_hot 

`torch.nn.functional.``one_hot`( _tensor_ , _num_classes=-1_ ) → LongTensor

    

注意到LongTensor具有形状的指标值`（*） `和返回的形状`（*， num_classes）的张量
`具有零到处除非最后一维的索引输入张量，在这种情况下这将是1的相应值相匹配。

参见[维基百科](https://en.wikipedia.org/wiki/One-hot)一热。

Parameters

    

  * **张量** （ _LongTensor_ ） - 的任何形状的类值。

  * **num_classes** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 类的总数。如果设置为-1，班数将被推断为一个比输入张量最大的一类值。

Returns

    

LongTensor具有与最后一维的由输入指定的指数值1和0其他地方多了一个维度。

例子

    
    
    >>> F.one_hot(torch.arange(0, 5) % 3)
    tensor([[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]])
    >>> F.one_hot(torch.arange(0, 5) % 3, num_classes=5)
    tensor([[1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]])
    >>> F.one_hot(torch.arange(0, 6).view(3,2) % 3)
    tensor([[[1, 0, 0],
             [0, 1, 0]],
            [[0, 0, 1],
             [1, 0, 0]],
            [[0, 1, 0],
             [0, 0, 1]]])
    

## 距离函数

###  pairwise_distance 

`torch.nn.functional.``pairwise_distance`( _x1_ , _x2_ , _p=2.0_ , _eps=1e-06_
, _keepdim=False_
)[[source]](_modules/torch/nn/functional.html#pairwise_distance)

    

参见[ `torch.nn.PairwiseDistance`](nn.html#torch.nn.PairwiseDistance
"torch.nn.PairwiseDistance")详细内容

###  cosine_similarity 

`torch.nn.functional.``cosine_similarity`( _x1_ , _x2_ , _dim=1_ , _eps=1e-8_
) → Tensor

    

返回x1和x2之间的余弦相似，沿着昏暗的计算。

similarity=x1⋅x2max⁡(∥x1∥2⋅∥x2∥2,ϵ)\text{similarity} = \dfrac{x_1 \cdot
x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}
similarity=max(∥x1​∥2​⋅∥x2​∥2​,ϵ)x1​⋅x2​​

Parameters

    

  * **X1** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 第一输入端。

  * **X2** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 第二输入（的大小匹配X1）。

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 矢量的维数。默认值：1

  * **EPS** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 小值由零避免分裂。默认值：1E-8

Shape:

    

  * 输入： （ *  1  d  ， *  2  ） （\ ast_1，d，\ ast_2） （ *  1  ， d  ， *  2  [H TG105] ） 其中d是在位置暗淡。

  * 输出： （ *  1  *  2  ） （\ ast_1，\ ast_2） （ *  1  ， *  2  ） 其中，1是在位置暗淡。

例：

    
    
    >>> input1 = torch.randn(100, 128)
    >>> input2 = torch.randn(100, 128)
    >>> output = F.cosine_similarity(input1, input2)
    >>> print(output)
    

###  pdist 

`torch.nn.functional.``pdist`( _input_ , _p=2_ ) → Tensor

    

计算在输入每对行向量之间的p-范数的距离。这是相同的上三角部分，不包括对角线，的 torch.norm（输入[:,无] - 输入，暗淡= 2，P =
p）的。如果行是连续的，此功能会更快。

如果输入具有形状 N  × M  n的\倍M  N  × M  然后输出将具有形状 1  2  N  （ N  \-  1  ） \压裂{1} {2}
N（N - 1） 2  [HTG10 0]  1  N  （ N  \-  1  ） 。

该函数等于 scipy.spatial.distance.pdist（输入， '明可夫斯基'，p值= P）如果 p  ∈ （ 0  ， ∞ ） 在（0，\
infty）  p  [ p \ HTG36]∈ （ 0  ， ∞ ） 。当 P  =  0  p = 0时 p  =  0  它相当于
scipy.spatial.distance.pdist（输入， '汉明' ）* M 。当 P  =  ∞ P = \ infty  p  =  ∞
，最接近SciPy的函数为 scipy.spatial.distance.pdist（XN，拉姆达X，Y：np.abs（X - Y）的.max（））。

Parameters

    

  * **输入** \- 形状的输入张量 N  × M  n的\倍M  N  × M  。

  * **P** \- 为对p范数距离p值，以各矢量对之间计算 ∈ [ 0  ， ∞ \ [0，\ infty]  ∈ [ 0  ， ∞ 。

## 损失函数

###  binary_cross_entropy 

`torch.nn.functional.``binary_cross_entropy`( _input_ , _target_ ,
_weight=None_ , _size_average=None_ , _reduce=None_ , _reduction='mean'_
)[[source]](_modules/torch/nn/functional.html#binary_cross_entropy)

    

该功能可以测量目标与输出之间的二进制交叉熵。

参见[ `BCELoss`](nn.html#torch.nn.BCELoss "torch.nn.BCELoss")了解详情。

Parameters

    

  * **输入** \- 任意形状的张量

  * **目标** \- 相同的形状作为输入的张量

  * **重量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 如果提供手动重新缩放重量它重复输入张量形状匹配

  * **size_average** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 已过时（见`还原 `）。默认情况下，损失平均超过批中每个元素的损失。注意：每个样本，对于一些损失，有多个元素。如果该字段`size_average`被设定为`假 `时，损失代替求和每个minibatch。当减少是`假 `忽略。默认值：`真 `

  * **减少** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 已过时（见`还原 `）。默认情况下，损耗进行平均或求和观测为视`size_average`每个minibatch。当`减少 `是`假 `，返回每批元件的损耗，而不是并忽略`size_average`。默认值：`真 `

  * **还原** （ _串_ _，_ _可选_ ） - 指定还原应用到输出：`'无' `| `'的意思是' `| `'和' `。 `'无' `：不降低将被应用，`'意味' `：将输出的总和将通过的数量来划分在输出中，`'和' `元素：输出将被累加。注意：`size_average`和`减少 `处于被淘汰，并且在此同时，指定是这两个参数的个数将覆盖`还原 `。默认值：`'平均' `

Examples:

    
    
    >>> input = torch.randn((3, 2), requires_grad=True)
    >>> target = torch.rand((3, 2), requires_grad=False)
    >>> loss = F.binary_cross_entropy(F.sigmoid(input), target)
    >>> loss.backward()
    

###  binary_cross_entropy_with_logits 

`torch.nn.functional.``binary_cross_entropy_with_logits`( _input_ , _target_ ,
_weight=None_ , _size_average=None_ , _reduce=None_ , _reduction='mean'_ ,
_pos_weight=None_
)[[source]](_modules/torch/nn/functional.html#binary_cross_entropy_with_logits)

    

功能测量目标和输出logits之间的二进制交叉熵。

参见[ `BCEWithLogitsLoss`](nn.html#torch.nn.BCEWithLogitsLoss
"torch.nn.BCEWithLogitsLoss")了解详情。

Parameters

    

  * **input** – Tensor of arbitrary shape

  * **target** – Tensor of the same shape as input

  * **weight** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – a manual rescaling weight if provided it’s repeated to match input tensor shape

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

  * **pos_weight** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 正例的权重。必须与长度等于类的数量的矢量。

Examples:

    
    
    >>> input = torch.randn(3, requires_grad=True)
    >>> target = torch.empty(3).random_(2)
    >>> loss = F.binary_cross_entropy_with_logits(input, target)
    >>> loss.backward()
    

###  poisson_nll_loss 

`torch.nn.functional.``poisson_nll_loss`( _input_ , _target_ ,
_log_input=True_ , _full=False_ , _size_average=None_ , _eps=1e-08_ ,
_reduce=None_ , _reduction='mean'_
)[[source]](_modules/torch/nn/functional.html#poisson_nll_loss)

    

泊松负对数似然的损失。

参见[ `PoissonNLLLoss`](nn.html#torch.nn.PoissonNLLLoss
"torch.nn.PoissonNLLLoss")了解详情。

Parameters

    

  * **输入** \- 底层泊松分布的期望。

  * **目标** \- 随机样品 T  一 R  G  E  T  〜 泊松 （ i的 n的 p  U  T  ） 目标\ SIM \文本{泊松}（输入）  T  一 R  克 E  T  〜 泊松 （ i的 n的 p  U  T  ） 。

  * **log_input** \- 如果`真 `损耗被计算为 EXP  ⁡ （ 输入 ） \-  目标 *  输入 \ EXP（\文本{输入}） - \文本{目标} * \文本{输入}  实验值 （ 输入 ） \-  目标 *  输入 时，如果`假 `然后损失是 [HTG9 2]输入  \-  目标 *  日志 ⁡ （ 输入 \+  EPS  ） \文本{输入} - \文本{目标} * \日志（\文本{输入} + \文本{EPS}） 输入 \-  目标 *  LO  G  （ 输入 \+  EPS  ） 。默认值：`真 `

  * **全** \- 是否计算全部损失，我。即添加的斯特林近似术语。默认值：`假 `目标 *  登录 ⁡ （ 目标 ） \-  目标 \+  0.5  *  日志 ⁡ （ 2  *  π *  目标 ） \文本{目标} * \日志（\文本{目标} ） - \文本{目标} + 0.5 * \日志（2 * \ PI * \文本{靶}） 目标 *  LO 克 （ 目标 ） \-  目标 \+  0  。  5  *  LO  G  （ 2  *  π *  目标 ） 。

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **EPS** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 小值，以避免的 [评价HTG12]  日志 ⁡ （ 0  ） \日志（0） LO  G  （ 0  ） 时`log_input`=``FALSE``。默认值：1E-8

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

###  cosine_embedding_loss 

`torch.nn.functional.``cosine_embedding_loss`( _input1_ , _input2_ , _target_
, _margin=0_ , _size_average=None_ , _reduce=None_ , _reduction='mean'_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#cosine_embedding_loss)

    

参见[ `CosineEmbeddingLoss`](nn.html#torch.nn.CosineEmbeddingLoss
"torch.nn.CosineEmbeddingLoss")了解详情。

###  cross_entropy 

`torch.nn.functional.``cross_entropy`( _input_ , _target_ , _weight=None_ ,
_size_average=None_ , _ignore_index=-100_ , _reduce=None_ , _reduction='mean'_
)[[source]](_modules/torch/nn/functional.html#cross_entropy)

    

该标准结合 log_softmax 和 nll_loss 在一个单一的功能。

参见[ `CrossEntropyLoss`](nn.html#torch.nn.CrossEntropyLoss
"torch.nn.CrossEntropyLoss")了解详情。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） -  （ N  ， C  ） （N，C） （ N  ， C  ） 其中 C =类或 数（ N  ， C  ， H  ， W  ） （N，C，H，W） （ N  ， C  ， H  ， W  ） 在2D损失的情况下，或 （ N  ， C  ， d  1  ， d  2  ， 。  。  。  ， d  K  ） （N，C，D_1， D_2，...，d_K） （ N  C  ， d  1  ， d  2  ， 。  。  。  ， d  K ​​  ） 其中 K  ≥ 1  ķ\ GEQ 1  ķ  ≥ 1  在K维损失的情况下。

  * **目标** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） -  （ N  ） （N） （ N  ） 其中每个值是 0  ≤ 目标 [ i的 ≤ C  \-  1  0 \当量\文本{目标} [I] \当量C-1  0  ≤ 目标 [ i的 ≤ C  \-  1  或 （ N  ， d  1  ， d  2  ， 。  。  。  ， d  K  ） （N，D_1，D_2， ...，d_K） （ N  ， d  1  ， d  2  ， 。 [HTG252  。  ， d  K  ） 其中 K  ≥ 1  ķ\ GEQ 1  ķ  ≥ 1  为K维损失。

  * **重量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 给每个类的手动重新缩放权重。如果给定的，必须是尺寸℃的张量

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **ignore_index** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 指定将被忽略，并且不向目标值输入梯度。当`size_average`是`真 `，损失平均超过非忽略的目标。默认值：-100

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

Examples:

    
    
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> target = torch.randint(5, (3,), dtype=torch.int64)
    >>> loss = F.cross_entropy(input, target)
    >>> loss.backward()
    

###  ctc_loss 

`torch.nn.functional.``ctc_loss`( _log_probs_ , _targets_ , _input_lengths_ ,
_target_lengths_ , _blank=0_ , _reduction='mean'_ , _zero_infinity=False_
)[[source]](_modules/torch/nn/functional.html#ctc_loss)

    

该联结颞分类损失。

参见[ `CTCLoss`](nn.html#torch.nn.CTCLoss "torch.nn.CTCLoss")了解详情。

Note

In some circumstances when using the CUDA backend with CuDNN, this operator
may select a nondeterministic algorithm to increase performance. If this is
undesirable, you can try to make the operation deterministic (potentially at a
performance cost) by setting `torch.backends.cudnn.deterministic = True`.
Please see the notes on [Reproducibility](notes/randomness.html) for
background.

Note

When using the CUDA backend, this operation may induce nondeterministic
behaviour in be backward that is not easily switched off. Please see the notes
on [Reproducibility](notes/randomness.html) for background.

Parameters

    

  * **log_probs** \-  （ T  ， N  ， C  ） （T，N，C） （ T  ， N  C  ） 其中 C =字符的字母数，包括空白， T =输入长度和 N =批量大小。的输出的取对数概率（例如，用 `获得torch.nn.functional.log_softmax（） `）。

  * **目标** \-  （ N  ， S  ） （N，S） （  N  ， S  ） 或（总和（target_lengths））。目标不能为空。在第二种形式中，目标被认为是连接在一起。

  * **input_lengths** \-  （ N  ） （N） （ N  ） 。的输入端（长度每一个都必须 ≤ T  \当量T  ≤ T  ）

  * **target_lengths** \-  （ N  ） （N） （ N  ） 。目标的长度

  * **空白** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 空白标签。默认 0  0  0  。

  * **还原** （ _串_ _，_ _可选_ ） - 指定还原应用到输出：`'无' `| `'的意思是' `| `'和' `。 `'无' `：不降低将被应用，`'意味' `：输出损耗将由目标长度，然后被划分平均过的批料采取，`'和' `：输出将被累加。默认值：`'平均' `

  * **zero_infinity** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 是否为零无限损失和相关联的梯度。默认值：`假 `主要是当输入太短，无法对准目标出现无限损失。

Example:

    
    
    >>> log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
    >>> targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
    >>> input_lengths = torch.full((16,), 50, dtype=torch.long)
    >>> target_lengths = torch.randint(10,30,(16,), dtype=torch.long)
    >>> loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
    >>> loss.backward()
    

###  hinge_embedding_loss 

`torch.nn.functional.``hinge_embedding_loss`( _input_ , _target_ ,
_margin=1.0_ , _size_average=None_ , _reduce=None_ , _reduction='mean'_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#hinge_embedding_loss)

    

参见[ `HingeEmbeddingLoss`](nn.html#torch.nn.HingeEmbeddingLoss
"torch.nn.HingeEmbeddingLoss")了解详情。

###  kl_div 

`torch.nn.functional.``kl_div`( _input_ , _target_ , _size_average=None_ ,
_reduce=None_ , _reduction='mean'_
)[[source]](_modules/torch/nn/functional.html#kl_div)

    

的 `的Kullback-Leibler距离divergence`_  损失。

参见[ `KLDivLoss`](nn.html#torch.nn.KLDivLoss "torch.nn.KLDivLoss")了解详情。

Parameters

    

  * **input** – Tensor of arbitrary shape

  * **target** – Tensor of the same shape as input

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **还原** （ _串_ _，_ _可选_ ） - 指定还原应用到输出：`'无' `| `'batchmean' `| `'和' `| `'意味' `。 `'无' `：不降低将被应用`'batchmean' `：将输出的总和将由BATCHSIZE [HTG32划分]  '和' ：输出将被累加`'平均' `：输出将通过在输出的默认元素的数目可分为： `'意味' `

Note

`size_average`和`减少 `处于被淘汰，并且在此同时，指定是这两个参数的个数将覆盖`还原 `。

Note

：ATTR：`还原 `= `'平均' `不回真正的KL散度值，请使用：ATTR：`还原 `= `'batchmean'
`与KL数学定义对齐。在接下来的主要版本，`'的意思是' `将变更为是相同的“batchmean”。

###  l1_loss 

`torch.nn.functional.``l1_loss`( _input_ , _target_ , _size_average=None_ ,
_reduce=None_ , _reduction='mean'_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#l1_loss)

    

函数，它的平均元素方面的绝对差值。

参见[ `L1Loss`](nn.html#torch.nn.L1Loss "torch.nn.L1Loss")了解详情。

###  mse_loss 

`torch.nn.functional.``mse_loss`( _input_ , _target_ , _size_average=None_ ,
_reduce=None_ , _reduction='mean'_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#mse_loss)

    

措施逐元素均方误差。

参见[ `MSELoss`](nn.html#torch.nn.MSELoss "torch.nn.MSELoss")了解详情。

###  margin_ranking_loss 

`torch.nn.functional.``margin_ranking_loss`( _input1_ , _input2_ , _target_ ,
_margin=0_ , _size_average=None_ , _reduce=None_ , _reduction='mean'_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#margin_ranking_loss)

    

参见[ `MarginRankingLoss`](nn.html#torch.nn.MarginRankingLoss
"torch.nn.MarginRankingLoss")了解详情。

###  multilabel_margin_loss 

`torch.nn.functional.``multilabel_margin_loss`( _input_ , _target_ ,
_size_average=None_ , _reduce=None_ , _reduction='mean'_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#multilabel_margin_loss)

    

参见[ `MultiLabelMarginLoss`](nn.html#torch.nn.MultiLabelMarginLoss
"torch.nn.MultiLabelMarginLoss")了解详情。

###  multilabel_soft_margin_loss 

`torch.nn.functional.``multilabel_soft_margin_loss`( _input_ , _target_ ,
_weight=None_ , _size_average=None_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#multilabel_soft_margin_loss)

    

参见[ `MultiLabelSoftMarginLoss`](nn.html#torch.nn.MultiLabelSoftMarginLoss
"torch.nn.MultiLabelSoftMarginLoss")了解详情。

###  multi_margin_loss 

`torch.nn.functional.``multi_margin_loss`( _input_ , _target_ , _p=1_ ,
_margin=1.0_ , _weight=None_ , _size_average=None_ , _reduce=None_ ,
_reduction='mean'_
)[[source]](_modules/torch/nn/functional.html#multi_margin_loss)

    

multi_margin_loss(input, target, p=1, margin=1, weight=None,
size_average=None,

    

减少=无，减少=”平均”） - & GT ;张量

参见[ `MultiMarginLoss`](nn.html#torch.nn.MultiMarginLoss
"torch.nn.MultiMarginLoss")了解详情。

###  nll_loss 

`torch.nn.functional.``nll_loss`( _input_ , _target_ , _weight=None_ ,
_size_average=None_ , _ignore_index=-100_ , _reduce=None_ , _reduction='mean'_
)[[source]](_modules/torch/nn/functional.html#nll_loss)

    

负对数似然的损失。

参见[ `NLLLoss`](nn.html#torch.nn.NLLLoss "torch.nn.NLLLoss")了解详情。

Parameters

    

  * **输入** \-  （ N  ， C  ） （N，C） （  N  ， C  ） 其中 C =班数或 （ N  ， C  ， H  ， W  ） （ N，C，H，W） （ N  C  ， H  ， W  [H TG102]） 在2D损失的情况下，或 （ N  ， C  ， d  1  ， d  2  ， 。  。  。  ， d  K  ） （N，C，D_1， D_2，...，d_K） （ N  C  ， d  1  ， d  2  ， 。  。  。  ， d  K  ​​  ） 其中 K  ≥ 1  ķ\ GEQ 1  ķ  ≥ 1  在K维损失的情况下。

  * **目标** \-  （ N  ） （N） （ N  ） 其中每个值是 0  ≤ 目标 [ i的 ≤ C  \-  1  0 \当量\文本{目标} [I] \当量C-1  0  ≤ 目标 [ i的 ≤ [HTG9 9]  C  \-  1  或 （ N  ， d  1  ， d  2  ， 。  。  。  ， d  K  ） （N，D_1，D_2， ...，d_K） （ N  ， d  1  ， d  2  ， 。 [HTG248  。  ， d  K  ​​  ） 其中 K  ≥ 1  ķ\ GEQ 1  ķ  ≥ 1  为K维损失。

  * **weight** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – a manual rescaling weight given to each class. If given, has to be a Tensor of size C

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **ignore_index** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _,_ _optional_ ) – Specifies a target value that is ignored and does not contribute to the input gradient. When `size_average`is `True`, the loss is averaged over non-ignored targets. Default: -100

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

Example:

    
    
    >>> # input is of size N x C = 3 x 5
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> # each element in target has to have 0 <= value < C
    >>> target = torch.tensor([1, 0, 4])
    >>> output = F.nll_loss(F.log_softmax(input), target)
    >>> output.backward()
    

###  smooth_l1_loss 

`torch.nn.functional.``smooth_l1_loss`( _input_ , _target_ ,
_size_average=None_ , _reduce=None_ , _reduction='mean'_
)[[source]](_modules/torch/nn/functional.html#smooth_l1_loss)

    

使用的平方项，如果绝对逐元素误差低于1和L1术语否则功能。

参见[ `SmoothL1Loss`](nn.html#torch.nn.SmoothL1Loss
"torch.nn.SmoothL1Loss")了解详情。

###  soft_margin_loss 

`torch.nn.functional.``soft_margin_loss`( _input_ , _target_ ,
_size_average=None_ , _reduce=None_ , _reduction='mean'_ ) →
Tensor[[source]](_modules/torch/nn/functional.html#soft_margin_loss)

    

参见[ `SoftMarginLoss`](nn.html#torch.nn.SoftMarginLoss
"torch.nn.SoftMarginLoss")了解详情。

###  triplet_margin_loss 

`torch.nn.functional.``triplet_margin_loss`( _anchor_ , _positive_ ,
_negative_ , _margin=1.0_ , _p=2_ , _eps=1e-06_ , _swap=False_ ,
_size_average=None_ , _reduce=None_ , _reduction='mean'_
)[[source]](_modules/torch/nn/functional.html#triplet_margin_loss)

    

参见[ `TripletMarginLoss`](nn.html#torch.nn.TripletMarginLoss
"torch.nn.TripletMarginLoss")详细内容

## 视觉功能

###  pixel_shuffle 

`torch.nn.functional.``pixel_shuffle`()

    

重新排列的元件在形状 （ *  ， ℃的张量 × R  2  ， H  ， W  ） （*，C \倍R ^ 2，H，W） （ *  ， C  × R  2
， H  ， W  ） 到的张量定型 （ *  ， C  H  × R  ， W  × R  ） （*，C，H \倍R，W \次数R） （ *  ， C
， H  × R  W  × R  ） 。

参见[ `PixelShuffle`](nn.html#torch.nn.PixelShuffle
"torch.nn.PixelShuffle")了解详情。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入张量

  * **upscale_factor** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 因子以增加由空间分辨率

Examples:

    
    
    >>> input = torch.randn(1, 9, 4, 4)
    >>> output = torch.nn.functional.pixel_shuffle(input, 3)
    >>> print(output.size())
    torch.Size([1, 1, 12, 12])
    

### 垫

`torch.nn.functional.``pad`( _input_ , _pad_ , _mode='constant'_ , _value=0_
)[[source]](_modules/torch/nn/functional.html#pad)

    

垫张量。

Padding size:

    

填充大小，通过该垫输入的`一些尺寸 `从最后一维起始和向前移动进行说明。  ⌊ LEN（垫） 2  ⌋ \左\ lfloor
\压裂{\文本{LEN（垫）}} {2} \右\ rfloor  ⌊ 2  LEN（垫） ⌋ 输入的`尺寸
`将被填充。例如，为了垫只有输入张量的最后一维，然后 `垫 `的形式为 （ padding_left  ， padding_right  ）
（\文本{填充\ _Left}，\文本{填充\ _right}） （ padding_left  ， padding_right  ）
;到垫输入张量的最后2米的尺寸，然后用 （ padding_left  ， padding_right  ， （\ {文本填充\ _Left}，
\文本{填充\ _right}， （ padding_left  ， padding_right  ， padding_top  ，
padding_bottom  ） \文本{填充\ _top}，\文本{填充\ _bottom}） padding_top  ，
padding_bottom  ） ;到垫的最后3个维度，用 （ padding_left [HTG24 7] ， padding_right  ，
（\文本{填充\ _Left}，\文本{填充\ _right}， （ padding_left  ​​， padding_right  ，
padding_top  ， padding_bottom  \文本{填充\ _top}，\文本{填充\ _bottom}  padding_top  ，
padding_bottom  padding_front  ， padding_back  ） \文本{填充\ _front}，\文本{填充\
_back}） padding_front  ， padding_back  ） 。

Padding mode:

    

参见[ `torch.nn.ConstantPad2d`](nn.html#torch.nn.ConstantPad2d
"torch.nn.ConstantPad2d")，[ `torch.nn.ReflectionPad2d`
](nn.html#torch.nn.ReflectionPad2d "torch.nn.ReflectionPad2d")，和[ `
torch.nn.ReplicationPad2d`](nn.html#torch.nn.ReplicationPad2d
"torch.nn.ReplicationPad2d")关于如何每个填充模式中的工作原理的具体例子。恒定填充为了任意尺寸来实现。复制填充被用于填充5D输入张量的最后3个维度，或最后2个维度4D输入张量的，或3D输入张量的最后一维实现。反映填充仅用于填充的最后2个维度4D输入张量，或3D输入张量的最后一维的实现。

Note

When using the CUDA backend, this operation may induce nondeterministic
behaviour in be backward that is not easily switched off. Please see the notes
on [Reproducibility](notes/randomness.html) for background.

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - N维张量

  * **垫** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")） - 间 - 元素的元组，其中 M  2  ≤ \压裂{米} {2} \当量 2  M  ≤ 输入的尺寸和 M  M  [HTG10 1]  M  是偶数。

  * **模式** \- `'恒定' `，`'反映' `，`'复制' `或`'循环' `。默认值：`'恒定' `

  * **值** \- 填补`'恒定' `填充值。默认值：`0`

Examples:

    
    
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
    

### 内插

`torch.nn.functional.``interpolate`( _input_ , _size=None_ ,
_scale_factor=None_ , _mode='nearest'_ , _align_corners=None_
)[[source]](_modules/torch/nn/functional.html#interpolate)

    

向下/向上采样输入要么给定`大小 `或给定的`scale_factor`

用于内插的算法由`模式 `确定。

目前的时间，空间和体积采样都被支持，即预期输入是3-d，4-d或在形状5- d。

输入尺寸解释的形式：迷你批次X通道×[可选深度]×[可选高度]×宽度。

可用于调整大小的模式是：最近，线性（3D-只），双线性，双三次（4D-只），三线性（5D-只），面积

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **大小** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ _元组_ _[_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _]或_ _元组_ _[_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _]或_ _元组_ _[_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") __ ） - 输出空间大小。

  * **scale_factor** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ _元组_ _[_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") __ ） - 乘法器，用于空间尺寸。有，如果它是一个元组匹配输入的内容。

  * **模式** （[ _STR_ ](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)")） - 用于上采样算法：`'最近' `| `'线性' `| `'双线性' `| `'双三次' `| `'三线性' `| `'区域' `。默认值：`'最近' `

  * **align_corners** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 几何上，我们考虑的输入和输出作为平方的像素而不是点。如果设置为`真 `，输入和输出张量由它们的拐角像素的中心点对齐，在拐角处的像素保持的值。如果设置为`假 `，输入和输出张量由它们的拐角像素的角点对准，并且内插使用边缘值填充为外的边界值，使这操作 _独立_ 输入尺寸的时`scale_factor`保持相同。这仅具有效果时`模式 `是`'线性' `，`'双线性' `，`'双三次' `或`'三线性' `。默认值：`假 `

Note

与`模式=“双三次”`，有可能引起过冲，换句话说，它可以产生负的值或值大于255的图像。显式调用`result.clamp（分钟= 0，
[HTG7最大= 255） `如果要显示图像时减小过冲。

Warning

与`align_corners  =  真
`时，线性地内插模式（线性，双线性和三线性）不按比例对齐的输出和输入的像素，和因此输出值可以依赖于输入的大小。这是这些模式可支持高达0.3.1版本的默认行为。此后，缺省行为是`
align_corners  =  假 `。参见[ `上采样 `](nn.html#torch.nn.Upsample
"torch.nn.Upsample")关于这如何影响输出的具体例子。

Note

When using the CUDA backend, this operation may induce nondeterministic
behaviour in be backward that is not easily switched off. Please see the notes
on [Reproducibility](notes/randomness.html) for background.

### 上采样

`torch.nn.functional.``upsample`( _input_ , _size=None_ , _scale_factor=None_
, _mode='nearest'_ , _align_corners=None_
)[[source]](_modules/torch/nn/functional.html#upsample)

    

上采样输入要么给定`大小 `或给定的`scale_factor`

Warning

此功能有利于弃用`torch.nn.functional.interpolate（） `。这相当于与`
nn.functional.interpolate（......） `。

Note

When using the CUDA backend, this operation may induce nondeterministic
behaviour in be backward that is not easily switched off. Please see the notes
on [Reproducibility](notes/randomness.html) for background.

用于上采样的算法由`模式 `确定。

目前的时间，空间和体积上采样都被支持，即预期输入是3-d，4-d或在形状5- d。

The input dimensions are interpreted in the form: mini-batch x channels x
[optional depth] x [optional height] x width.

可用于上采样的模式是：最近，线性（3D-只），双线性，双三次（4D-只），三线性（5D-只）

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **size** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_ _Tuple_ _[_[ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _] or_ _Tuple_ _[_[ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _,_[ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _] or_ _Tuple_ _[_[ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _,_[ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _,_[ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _]_ ) – output spatial size.

  * **scale_factor** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ _元组_ _[_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") __ ） - 乘法器，用于空间尺寸。必须是一个整数。

  * **模式** （ _串_ ） - 算法用于上采样：`'最近' `| `'线性' `| `'双线性' `| `'双三次' `| `'三线性' `。默认值：`'最近' `

  * **align_corners** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Geometrically, we consider the pixels of the input and output as squares rather than points. If set to `True`, the input and output tensors are aligned by the center points of their corner pixels, preserving the values at the corner pixels. If set to `False`, the input and output tensors are aligned by the corner points of their corner pixels, and the interpolation uses edge value padding for out-of-boundary values, making this operation _independent_ of input size when `scale_factor`is kept the same. This only has an effect when `mode`is `'linear'`, `'bilinear'`, `'bicubic'`or `'trilinear'`. Default: `False`

Note

With `mode='bicubic'`, it’s possible to cause overshoot, in other words it can
produce negative values or values greater than 255 for images. Explicitly call
`result.clamp(min=0, max=255)`if you want to reduce the overshoot when
displaying the image.

Warning

With `align_corners = True`, the linearly interpolating modes (linear,
bilinear, and trilinear) don’t proportionally align the output and input
pixels, and thus the output values can depend on the input size. This was the
default behavior for these modes up to version 0.3.1. Since then, the default
behavior is `align_corners = False`. See
[`Upsample`](nn.html#torch.nn.Upsample "torch.nn.Upsample") for concrete
examples on how this affects the outputs.

###  upsample_nearest 

`torch.nn.functional.``upsample_nearest`( _input_ , _size=None_ ,
_scale_factor=None_
)[[source]](_modules/torch/nn/functional.html#upsample_nearest)

    

上采样的输入，使用最近邻居的像素值。

Warning

此功能有利于弃用`torch.nn.functional.interpolate（） `。这相当于与`
nn.functional.interpolate（...， 模式= '最近'） `。

目前空间和体积上采样的支持（即预期输入是4或5维）。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – input

  * **大小** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ _元组_ _[_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _]或_ _元组_ _[_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") __ ） - 输出spatia大小。

  * **scale_factor** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 用于空间大小乘数。必须是一个整数。

Note

When using the CUDA backend, this operation may induce nondeterministic
behaviour in be backward that is not easily switched off. Please see the notes
on [Reproducibility](notes/randomness.html) for background.

###  upsample_bilinear 

`torch.nn.functional.``upsample_bilinear`( _input_ , _size=None_ ,
_scale_factor=None_
)[[source]](_modules/torch/nn/functional.html#upsample_bilinear)

    

上采样输入，使用双线性采样。

Warning

此功能有利于弃用`torch.nn.functional.interpolate（） `。这相当于与`
nn.functional.interpolate（...， 模式= '双线性'， align_corners =真） `。

预期输入是空间（4维）。使用 upsample_trilinear  FO体积（5维）输入。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – input

  * **大小** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ _元组_ _[_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") __ ） - 输出空间大小。

  * **scale_factor** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ _元组_ _[_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") __ ） - 乘法器，用于空间尺寸

Note

When using the CUDA backend, this operation may induce nondeterministic
behaviour in be backward that is not easily switched off. Please see the notes
on [Reproducibility](notes/randomness.html) for background.

###  grid_sample 

`torch.nn.functional.``grid_sample`( _input_ , _grid_ , _mode='bilinear'_ ,
_padding_mode='zeros'_
)[[source]](_modules/torch/nn/functional.html#grid_sample)

    

给定一个`输入 `和流场`格 `时，`输出 `使用计算`从`格输入 `值和像素位置 `。

目前，只有空间（4- d）和体积（5-d）`输入 `的支持。

在空间（4- d）情况下，对于`输入 `具有形状 （ N  ， C  ， H  在 ， W  在 ） （N，C，H_ \文本{IN} ，W_
\文本{在}） （ N  C  ， H  在 ， W  [HT G100]  在 ） 和`格 `具有形状 （ N  ， H  OUT  ， W  OUT
， 2  ） （N，H_ \文本{出}，W_ \文本{出}，2） （ N  ， H  [H TG192]  OUT  ， W  OUT  ， 2  ）
，输出将具有形状 （ N  ​​， C  ， H  OUT  ， W  OUT  ） （N，C，H_ \文本{出}，W_ \文本{出}） （ N  ， C
， H  OUT  ， W  OUT  [HTG38 0]） 。

对于每个输出位置`输出[N， ： H， W]`，尺寸-2载体`格[N， H， W]`指定`输入 `的像素位置`× `和`Y
`，其被用于内插的输出值`输出[N， ： H， W]`。在图5D的输入的情况下，`格[N， d  H， W]`指定`× `，`Y`，`
Z`用于内插的像素位置`输出[N， ： d  H， W]`。 `模式 `参数指定`最近 `或`双线性 `的内插方法来采样输入像素。

`格 `指定由`输入 `空间尺寸归一化的采样像素位置。因此，应该具有在`的范围内最值[-1， 1]`。例如，值`× =  -1， Y  =
-1-`被输入的`的左上端的像素 `和值`× =  1，  Y  =  1`是`输入 `右下角的像素。

如果`格 `具有`的范围之外的值[-1， 1]`，相应的输出处理如由`padding_mode`中定义。选项

>   * `padding_mode = “0” `：使用`0  [HTG7用于结合外的网格位置，`

>

>   * `padding_mode = “边界” `：使用边界值外的结合网格位置，

>

>   * `padding_mode =“反射”
`：在由边界为外的结合网格位置反射的地点使用的值。用于位置远离边界，它会保持被反射直到成为结合的，例如，（归一化）的像素位置`× =  -3.5
`由边界`-1-`反射并变成`×”  =  1.5`，然后通过边界`1`反射并变成`× '' =  -0.5`。

>

>

Note

该功能通常在建设[空间变压器网络部](https://arxiv.org/abs/1506.02025)使用。

Note

When using the CUDA backend, this operation may induce nondeterministic
behaviour in be backward that is not easily switched off. Please see the notes
on [Reproducibility](notes/randomness.html) for background.

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 形状 的输入（ N  ， C  ， H  在 ， W  在 ） （N，C，H_ \文本{IN} ，W_ \文本{在}） （ N  C  ， H  在 ， W  在 ） （4- d情况下）或 （ N  ， C  d  在 ， H  在 ， W  在 ） （N，C，D_ \文本{在}，H_ \文本{IN}，W_ \文本{在}） （ N  ， C  d  在 ， H  在 ， W  ​​  在 ） （5- d情况下）

  * **格** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的形状 流场（  N  ， H  OUT  ， W  OUT  ， 2  ） （N，H_ \文本{出} ，W_ \文本{出}，2） （ N  ， H  OUT  ， W  OUT  ， 2  ） （4- d情况下）或 （ N  ， d  OUT  ， H  OUT  ， W  OUT  ， 3  ） （N，D_ \文本{出}，H_ \文本{出}，W_ \文本{出}，3） （ N  ， [HTG1 91] d  OUT  ， H  OUT  ， W  ​​ OUT  ， 3  ） （5- d情况下）

  * **模式** （[ _STR_ ](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)")） - 内插模式来计算输出值`'双线性' `| `'最近' `。默认值：`'双线性' `

  * **padding_mode** （[ _STR_ ](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)")） - 填充模式用于外部电网的值`'零' `| `'边界' `| `'反射' `。默认值：`'零' `

Returns

    

输出张量

Return type

    

输出（[张量](tensors.html#torch.Tensor "torch.Tensor")）

###  affine_grid 

`torch.nn.functional.``affine_grid`( _theta_ , _size_
)[[source]](_modules/torch/nn/functional.html#affine_grid)

    

产生二维流场，在给定批次的仿射矩阵`THETA`的。与 `结合通常使用grid_sample（） `实施空间变换器网络。

Parameters

    

  * **THETA** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 仿射矩阵的输入批次（ N  × 2  × 3  n的\倍2 \倍3  N  × 2  × 3  ）

  * **大小** （ _torch.Size_ ） - 目标输出图像尺寸（ N  × C  × H  × W  n的\ C时代\倍ħ\倍W  N  × C  × H  × W¯¯  ）。例如：torch.Size（（32，3，24，24））

Returns

    

大小的输出张量（ N  × H  × W  × 2  n的\倍ħ\倍数W \倍2  N  × H  × W  × 2  ）

Return type

    

output ([Tensor](tensors.html#torch.Tensor "torch.Tensor"))

## 数据并行功能（多GPU，分布式）

###  data_parallel 

`torch.nn.parallel.``data_parallel`( _module_ , _inputs_ , _device_ids=None_ ,
_output_device=None_ , _dim=0_ , _module_kwargs=None_
)[[source]](_modules/torch/nn/parallel/data_parallel.html#data_parallel)

    

评估跨在给定device_ids所述GPU并行模块（输入）。

这是该数据并行模块的功能版本。

Parameters

    

  * **模块** （[ _模块_ ](nn.html#torch.nn.Module "torch.nn.Module")） - 模块并行评估

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入到模块

  * **device_ids** （ _蟒的列表：INT_ _或_ [ _torch.device_ ](tensor_attributes.html#torch.torch.device "torch.torch.device")） - 在其上GPU IDS复制模块

  * **output_device** （ _蟒的列表：INT_ _或_ [ _torch.device_ ](tensor_attributes.html#torch.torch.device "torch.torch.device")） - 输出使用GPU位置 - 1，以指示CPU。 （默认值：device_ids [0]）

Returns

    

含模块（输入）的结果的张量位于output_device

[Next ![](_static/images/chevron-right-orange.svg)](nn.init.html
"torch.nn.init") [![](_static/images/chevron-right-orange.svg)
Previous](nn.html "torch.nn")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * torch.nn.functional 
    * 卷积函数
      * conv1d 
      * conv2d 
      * conv3d 
      * conv_transpose1d 
      * conv_transpose2d 
      * conv_transpose3d 
      * 展开
      * 倍
    * 池功能
      * avg_pool1d 
      * avg_pool2d 
      * avg_pool3d 
      * max_pool1d 
      * max_pool2d 
      * max_pool3d 
      * max_unpool1d 
      * max_unpool2d 
      * max_unpool3d 
      * lp_pool1d 
      * lp_pool2d 
      * adaptive_max_pool1d 
      * adaptive_max_pool2d 
      * adaptive_max_pool3d 
      * adaptive_avg_pool1d 
      * adaptive_avg_pool2d 
      * adaptive_avg_pool3d 
    * 非线性激活函数
      * 阈
      * RELU 
      * hardtanh 
      * relu6 
      * ELU 
      * 九色鹿
      * celu 
      * leaky_relu 
      * prelu 
      * rrelu 
      * glu的
      * 格鲁
      * logsigmoid 
      * hardshrink 
      * tanhshrink 
      * softsign 
      * softplus 
      * softmin 
      * SOFTMAX 
      * softshrink 
      * gumbel_softmax 
      * log_softmax 
      * 的tanh 
      * 乙状结肠
    * 归一化函数
      * batch_norm 
      * instance_norm 
      * layer_norm 
      * local_response_norm 
      * 正常化
    * 线性函数
      * 线性
      * 双线性
    * 降函数
      * 差
      * alpha_dropout 
      * dropout2d 
      * dropout3d 
    * 稀疏功能
      * 嵌入
      * embedding_bag 
      * one_hot 
    * 距离函数
      * pairwise_distance 
      * cosine_similarity 
      * pdist 
    * 损失函数
      * binary_cross_entropy 
      * binary_cross_entropy_with_logits 
      * poisson_nll_loss 
      * cosine_embedding_loss 
      * cross_entropy 
      * ctc_loss 
      * hinge_embedding_loss 
      * kl_div 
      * l1_loss 
      * mse_loss 
      * margin_ranking_loss 
      * multilabel_margin_loss 
      * multilabel_soft_margin_loss 
      * multi_margin_loss 
      * nll_loss 
      * smooth_l1_loss 
      * soft_margin_loss 
      * triplet_margin_loss 
    * 视觉功能
      * pixel_shuffle 
      * 垫
      * 内插
      * 上采样
      * upsample_nearest 
      * upsample_bilinear 
      * grid_sample 
      * affine_grid 
    * 数据并行功能（多GPU，分布式）
      * data_parallel 

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

![](_static/images/pytorch-x.svg)

[](https://pytorch.org/)

  * 入门
  * 特点
  * 生态系统
  * [博客](https://pytorch.org/blog/)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [资源](https://pytorch.org/resources)
  * [ Github的](https://github.com/pytorch/pytorch)

