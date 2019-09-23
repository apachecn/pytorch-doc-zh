# torch.Tensor

A`torch.Tensor`是包含单一数据类型的元素的多维矩阵。

Torch 定义了9种CPU类型和九种GPU张量类型：

数据类型

|

D型

|

CPU张量

|

GPU张量  
  
---|---|---|---  
  
32位浮点

|

`torch.float32`或`torch.float`

|

`torch.FloatTensor`

|

`torch.cuda.FloatTensor` 
  
64位浮点

|

`torch.float64`或`torch.double`

|

`torch.DoubleTensor`

|

`torch.cuda.DoubleTensor` 
  
16位浮点

|

`torch.float16`或`torch.half`

|

`torch.HalfTensor`

|

`torch.cuda.HalfTensor` 
  
8位整数（无符号）

|

`torch.uint8`

|

`torch.ByteTensor`

|

`torch.cuda.ByteTensor` 
  
8位整数（签名）

|

`torch.int8`

|

`torch.CharTensor`

|

`torch.cuda.CharTensor` 
  
16位整数（签名）

|

`torch.int16`或`torch.short`

|

`torch.ShortTensor`

|

`torch.cuda.ShortTensor` 
  
32位整数（签名）

|

`torch.int32`或`torch.int`

|

`torch.IntTensor`

|

`torch.cuda.IntTensor` 
  
64位整数（签名）

|

`torch.int64`或`torch.long`

|

`torch.LongTensor`

|

`torch.cuda.LongTensor` 
  
布尔

|

`torch.bool`

|

`torch.BoolTensor`

|

`torch.cuda.BoolTensor` 
  
`torch.Tensor`是默认张量类型的别名（`torch.FloatTensor`）。

张量可以从一个Python [ `列表 `](https://docs.python.org/3/library/stdtypes.html#list
"\(in Python v3.7\)")或序列使用[ `torch.tensor（） [HTG10被构造]
`](torch.html#torch.tensor "torch.tensor")构造：

    
    
    >>> torch.tensor([[1., -1.], [1., -1.]])
    tensor([[ 1.0000, -1.0000],
            [ 1.0000, -1.0000]])
    >>> torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    tensor([[ 1,  2,  3],
            [ 4,  5,  6]])
    

警告

[ `torch.tensor（） `](torch.html#torch.tensor "torch.tensor")总是副本`数据
`。如果你有一个张量`数据 `，只是想改变它的`requires_grad`标志，使用 `requires_grad_ （） `或 `分离（）
`，以避免副本。如果你有一个numpy的阵列，并希望避免拷贝，使用[ `torch.as_tensor（） `
[HTG35。](torch.html#torch.as_tensor "torch.as_tensor")

特定数据类型的张量可以通过使[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype
"torch.torch.dtype")和/或[ `torch.device构建 `
](tensor_attributes.html#torch.torch.device "torch.torch.device")对构造或张量创建OP：

    
    
    >>> torch.zeros([2, 4], dtype=torch.int32)
    tensor([[ 0,  0,  0,  0],
            [ 0,  0,  0,  0]], dtype=torch.int32)
    >>> cuda0 = torch.device('cuda:0')
    >>> torch.ones([2, 4], dtype=torch.float64, device=cuda0)
    tensor([[ 1.0000,  1.0000,  1.0000,  1.0000],
            [ 1.0000,  1.0000,  1.0000,  1.0000]], dtype=torch.float64, device='cuda:0')
    

张量的内容可以使用Python的索引和切片符号来访问和修改：

    
    
    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    >>> print(x[1][2])
    tensor(6)
    >>> x[0][1] = 8
    >>> print(x)
    tensor([[ 1,  8,  3],
            [ 4,  5,  6]])
    

使用 `torch.Tensor.item（） `以获得从含有单个值的张量一个Python数：

    
    
    >>> x = torch.tensor([[1]])
    >>> x
    tensor([[ 1]])
    >>> x.item()
    1
    >>> x = torch.tensor(2.5)
    >>> x
    tensor(2.5000)
    >>> x.item()
    2.5
    

甲张量可以用`requires_grad =真 `，使得[ `torch.autograd`](autograd.html#module-
torch.autograd "torch.autograd")对它们的记录操作的自动创建分化。

    
    
    >>> x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
    >>> out = x.pow(2).sum()
    >>> out.backward()
    >>> x.grad
    tensor([[ 2.0000, -2.0000],
            [ 2.0000,  2.0000]])
    

各张量具有相关联的`torch.Storage
`，其保持它的数据。张量类提供多维的，[存储的跨距](https://en.wikipedia.org/wiki/Stride_of_an_array)视图并在其上限定的数值的操作。

注意

有关[ `的更多信息torch.dtype`](tensor_attributes.html#torch.torch.dtype
"torch.torch.dtype")，[ `torch.device`
](tensor_attributes.html#torch.torch.device "torch.torch.device")，和[ `
torch.layout`](tensor_attributes.html#torch.torch.layout
"torch.torch.layout")的属性的 `torch.Tensor`参见[ 张量属性
](tensor_attributes.html#tensor-attributes-doc)。

Note

其中一个突变的方法张标有下划线的后缀。例如，`torch.FloatTensor.abs_（） `计算就地绝对值，并返回改性张量，而`
torch.FloatTensor.abs（） `计算结果在一个新的张量。

Note

要更改现有的张量的[ `torch.device`](tensor_attributes.html#torch.torch.device
"torch.torch.device")和/或[ `torch.dtype`
](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") ，可以考虑使用 `至（）
`关于张量的方法。

Warning

的 `torch.Tensor`
当前实现引入了内存开销，从而可能导致在许多微小的张量的应用出乎意料的高内存使用情况。如果您遇到这种情况，可以考虑使用一个大的结构。

_class_`torch.``Tensor`

    

有两种创建一个张量，这取决于你的使用情况的几个主要途径。

  * 要创建具有预先存在的数据的张量，用[ `torch.tensor（） `](torch.html#torch.tensor "torch.tensor")。

  * 要创建具有特定大小的张量，使用`Torch 。*`张量创建OPS（见[ 创建行动 ](torch.html#tensor-creation-ops)）。

  * 要创建具有相同的尺寸（以及类似的类型）作为另一张张量，使用`Torch 。* _像 `张量创建OPS（见[ 创建行动 ](torch.html#tensor-creation-ops)）。

  * 要创建具有相似类型但不同大小的另一张张量，使用`tensor.new_ *`创建欢声笑语。

`new_tensor`( _data_ , _dtype=None_ , _device=None_ , _requires_grad=False_ )
→ Tensor

    

返回与`数据 `作为张量数据的新的张量。默认情况下，返回的张量具有相同的[ `torch.dtype`
](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")和[ `
torch.device`](tensor_attributes.html#torch.torch.device
"torch.torch.device")作为本张量。

Warning

`new_tensor（） `总是副本`数据 `。如果你有一个张量`数据 `，并希望避免拷贝，使用 `
torch.Tensor.requires_grad_（） `或 `torch.Tensor.detach（） `
。如果你有一个numpy的阵列，并希望避免拷贝，使用[ `torch.from_numpy（） `
[HTG31。](torch.html#torch.from_numpy "torch.from_numpy")

Warning

当数据是张量×， `new_tensor（） `读出从不管它是通过 '数据'，并构造一个叶变量。因此`tensor.new_tensor（X）
`等于`x.clone（）。分离（） `和`tensor.new_tensor（X， requires_grad =真） `等于`
x.clone（）。分离（）。requires_grad_（真） `。使用`克隆（） `和`分离（）的当量的建议`。

Parameters

    

  * **数据** （ _array_like_ ） - 返回的张量份`数据 `。

  * **DTYPE** （[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")，可选） - 返回的张量的所期望的类型。默认值：如果无，相同[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")如这个张量。

  * **装置** （[ `torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device")，可选） - 返回的张量的所需的设备。默认值：如果无，相同[ `torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device")如这个张量。

  * **requires_grad** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果autograd应返回的记录张操作。默认值：`假 [HTG13。`

例：

    
    
    >>> tensor = torch.ones((2,), dtype=torch.int8)
    >>> data = [[0, 1], [2, 3]]
    >>> tensor.new_tensor(data)
    tensor([[ 0,  1],
            [ 2,  3]], dtype=torch.int8)
    

`new_full`( _size_ , _fill_value_ , _dtype=None_ , _device=None_ ,
_requires_grad=False_ ) → Tensor

    

返回大小 `大小的张量填充`与`fill_value`。默认情况下，返回的张量具有相同的[ `torch.dtype`
](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")和[ `
torch.device`](tensor_attributes.html#torch.torch.device
"torch.torch.device")作为本张量。

Parameters

    

  * **fill_value** （ _标量_ ） - 的数量来填充与输出张量。

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired type of returned tensor. Default: if None, same [`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") as this tensor.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if None, same [`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device") as this tensor.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> tensor = torch.ones((2,), dtype=torch.float64)
    >>> tensor.new_full((3, 4), 3.141592)
    tensor([[ 3.1416,  3.1416,  3.1416,  3.1416],
            [ 3.1416,  3.1416,  3.1416,  3.1416],
            [ 3.1416,  3.1416,  3.1416,  3.1416]], dtype=torch.float64)
    

`new_empty`( _size_ , _dtype=None_ , _device=None_ , _requires_grad=False_ ) →
Tensor

    

返回大小 `大小填充`与未初始化的数据的张量。默认情况下，返回的张量具有相同的[ `torch.dtype`
](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")和[ `
torch.device`](tensor_attributes.html#torch.torch.device
"torch.torch.device")作为本张量。

Parameters

    

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired type of returned tensor. Default: if None, same [`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") as this tensor.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if None, same [`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device") as this tensor.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> tensor = torch.ones(())
    >>> tensor.new_empty((2, 3))
    tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
            [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])
    

`new_ones`( _size_ , _dtype=None_ , _device=None_ , _requires_grad=False_ ) →
Tensor

    

返回大小 填充有`1``大小`的张量。默认情况下，返回的张量具有相同的[ `torch.dtype`
](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")和[ `
torch.device`](tensor_attributes.html#torch.torch.device
"torch.torch.device")作为本张量。

Parameters

    

  * **大小** （ _INT ..._ ） - 列表，元组，或`torch.Size`定义输出张量的形状的整数。

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired type of returned tensor. Default: if None, same [`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") as this tensor.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if None, same [`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device") as this tensor.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> tensor = torch.tensor((), dtype=torch.int32)
    >>> tensor.new_ones((2, 3))
    tensor([[ 1,  1,  1],
            [ 1,  1,  1]], dtype=torch.int32)
    

`new_zeros`( _size_ , _dtype=None_ , _device=None_ , _requires_grad=False_ ) →
Tensor

    

返回大小 填充有`0``大小`的张量。默认情况下，返回的张量具有相同的[ `torch.dtype`
](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")和[ `
torch.device`](tensor_attributes.html#torch.torch.device
"torch.torch.device")作为本张量。

Parameters

    

  * **size** ( _int..._ ) – a list, tuple, or `torch.Size`of integers defining the shape of the output tensor.

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired type of returned tensor. Default: if None, same [`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") as this tensor.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if None, same [`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device") as this tensor.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> tensor = torch.tensor((), dtype=torch.float64)
    >>> tensor.new_zeros((2, 3))
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]], dtype=torch.float64)
    

`is_cuda`

    

为`真 [HTG3如果张量存储在GPU，`假 [HTG7否则。``

`device`

    

是[ `torch.device`[HTG5如果本张量。](tensor_attributes.html#torch.torch.device
"torch.torch.device")

`grad`

    

该属性是`无 `缺省和成为张量在第一时间 `呼叫向后（） `计算梯度为`自 `。然后，属性将包含计算出的梯度，并 `向后（） `
将积累（添加）梯度到它将来的呼叫。

`ndim`

    

别名 `暗淡（） `

`T`

    

这是张量与它的尺寸逆转。

如果`n的 `是尺寸在`× `的数量，`XT`等价于`x.permute第（n-1， N-2， ...， 0） `。

`abs`() → Tensor

    

参见[ `torch.abs（） `](torch.html#torch.abs "torch.abs")

`abs_`() → Tensor

    

就地版本 `的ABS（） `

`acos`() → Tensor

    

参见[ `torch.acos（） `](torch.html#torch.acos "torch.acos")

`acos_`() → Tensor

    

就地版本 `ACOS的（） `

`add`( _value_ ) → Tensor

    

添加（值= 1，其他） - & GT ;张量

参见[ `torch.add（） `](torch.html#torch.add "torch.add")

`add_`( _value_ ) → Tensor

    

add_（值= 1，其他） - & GT ;张量

就地版本的 `添加（） `

`addbmm`( _beta=1_ , _alpha=1_ , _batch1_ , _batch2_ ) → Tensor

    

参见[ `torch.addbmm（） `](torch.html#torch.addbmm "torch.addbmm")

`addbmm_`( _beta=1_ , _alpha=1_ , _batch1_ , _batch2_ ) → Tensor

    

就地版本 `addbmm的（） `

`addcdiv`( _value=1_ , _tensor1_ , _tensor2_ ) → Tensor

    

参见[ `torch.addcdiv（​​） `](torch.html#torch.addcdiv "torch.addcdiv")

`addcdiv_`( _value=1_ , _tensor1_ , _tensor2_ ) → Tensor

    

就地版本的 `addcdiv（​​） `

`addcmul`( _value=1_ , _tensor1_ , _tensor2_ ) → Tensor

    

参见[ `torch.addcmul（） `](torch.html#torch.addcmul "torch.addcmul")

`addcmul_`( _value=1_ , _tensor1_ , _tensor2_ ) → Tensor

    

就地版本 `addcmul的（） `

`addmm`( _beta=1_ , _alpha=1_ , _mat1_ , _mat2_ ) → Tensor

    

参见[ `torch.addmm（） `](torch.html#torch.addmm "torch.addmm")

`addmm_`( _beta=1_ , _alpha=1_ , _mat1_ , _mat2_ ) → Tensor

    

就地版本 `addmm的（） `

`addmv`( _beta=1_ , _alpha=1_ , _mat_ , _vec_ ) → Tensor

    

参见[ `torch.addmv（） `](torch.html#torch.addmv "torch.addmv")

`addmv_`( _beta=1_ , _alpha=1_ , _mat_ , _vec_ ) → Tensor

    

就地版本 `addmv的（） `

`addr`( _beta=1_ , _alpha=1_ , _vec1_ , _vec2_ ) → Tensor

    

参见[ `torch.addr（） `](torch.html#torch.addr "torch.addr")

`addr_`( _beta=1_ , _alpha=1_ , _vec1_ , _vec2_ ) → Tensor

    

就地版本 `的addr（） `

`allclose`( _other_ , _rtol=1e-05_ , _atol=1e-08_ , _equal_nan=False_ ) →
Tensor

    

参见[ `torch.allclose（） `](torch.html#torch.allclose "torch.allclose")

`apply_`( _callable_ ) → Tensor

    

适用的函数`可调用 `，在张量的每个元件，与由`可调用 `返回的值替换每个元件。

Note

此功能只适用于CPU张量，不应该在要求高性能的代码段中使用。

`argmax`( _dim=None_ , _keepdim=False_ ) → LongTensor

    

参见[ `torch.argmax（） `](torch.html#torch.argmax "torch.argmax")

`argmin`( _dim=None_ , _keepdim=False_ ) → LongTensor

    

参见[ `torch.argmin（） `](torch.html#torch.argmin "torch.argmin")

`argsort`( _dim=-1_ , _descending=False_ ) → LongTensor

    

参见：FUNC： torch.argsort

`asin`() → Tensor

    

参见[ `torch.asin（） `](torch.html#torch.asin "torch.asin")

`asin_`() → Tensor

    

就地版本的 `ASIN（） `

`as_strided`( _size_ , _stride_ , _storage_offset=0_ ) → Tensor

    

参见[ `torch.as_strided（） `](torch.html#torch.as_strided "torch.as_strided")

`atan`() → Tensor

    

参见[ `torch.atan（） `](torch.html#torch.atan "torch.atan")

`atan2`( _other_ ) → Tensor

    

参见[ `torch.atan2（） `](torch.html#torch.atan2 "torch.atan2")

`atan2_`( _other_ ) → Tensor

    

就地版本 `ATAN2的（） `

`atan_`() → Tensor

    

就地版本 `ATAN的（） `

`backward`( _gradient=None_ , _retain_graph=None_ , _create_graph=False_
)[[source]](_modules/torch/tensor.html#Tensor.backward)

    

计算当前张量w.r.t.的梯度图叶。

该图是使用链式法则区分。如果张量是非标量（即，其数据具有一个以上的元素），并且需要的梯度，所述函数另外需要指定`梯度
`。它应该是匹配的类型和位置的张量，包含有区别的功能w.r.t.的梯度`自 `。

此功能聚集在叶梯度 - 你可能需要调用它之前为零它们。

Parameters

    

  * **梯度** （ _张量_ _或_ [ _无_ ](https://docs.python.org/3/library/constants.html#None "\(in Python v3.7\)")） - 梯度w.r.t.张量。如果它是一个张量，它将被自动转换为不需要研究所除非`create_graph`为True张量。无值可以用于标量张量或那些不要求毕业生指定。如果没有值是可以接受的，然后这种说法是可选的。

  * **retain_graph** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`假 `，用于计算梯度的图表将被释放。请注意，在几乎所有情况下的设置则不需要此选项设置为True，往往可以以更有效的方式围绕工作。默认为`create_graph`的值。

  * **create_graph** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，所述衍生物的图形将被构建，从而允许计算高阶衍生产品。默认为`假 [HTG17。`

`baddbmm`( _beta=1_ , _alpha=1_ , _batch1_ , _batch2_ ) → Tensor

    

参见[ `torch.baddbmm（） `](torch.html#torch.baddbmm "torch.baddbmm")

`baddbmm_`( _beta=1_ , _alpha=1_ , _batch1_ , _batch2_ ) → Tensor

    

就地版本 `baddbmm的（） `

`bernoulli`( _*_ , _generator=None_ ) → Tensor

    

返回一个结果张量，其中每个 导致[I]  \ texttt {结果[I]}  导致[I]  从独立地取样 伯努利 （ 自[I]  ）  \文本{伯努利}（\
texttt {自[I]}） 伯努利 （ 自[I]  ） 。 `自 `必须浮点`DTYPE`，结果将具有相同的`DTYPE`。

参见[ `torch.bernoulli（） `](torch.html#torch.bernoulli "torch.bernoulli")

`bernoulli_`()

    

`bernoulli_`( _p=0.5_ , _*_ , _generator=None_ ) → Tensor

    

填充的`自 `每个位置与一个独立的样品从 伯努利 （ p  ） \文本{伯努利}（\ texttt {p}） 伯努利 （ p  ） 。 `自
`可以具有积分`DTYPE`。

`bernoulli_`( _p_tensor_ , _*_ , _generator=None_ ) → Tensor

    

`p_tensor`应该是被用于绘制二进制随机数包含概率的张量。

的 i的 T  H  \文本{I} ^ {第}  I  T  H  的`自 `张量元件将被设置为从 伯努利 取样的值（ p_tensor [I]  ）
\文本{伯努利}（\ texttt {p \ _tensor [I]}） 伯努利 （ p_tensor [I]  ） 。

`自 `可以有积分`D类 `，而`p_tensor`必须浮点`DTYPE`。

另请参见 `伯努利（） `和[ `torch.bernoulli（） `](torch.html#torch.bernoulli
"torch.bernoulli")

`bfloat16`() → Tensor

    

`self.bfloat16（） `等于`self.to（torch.bfloat16） `。参见 `至（） `。

`bincount`( _weights=None_ , _minlength=0_ ) → Tensor

    

参见[ `torch.bincount（） `](torch.html#torch.bincount "torch.bincount")

`bitwise_not`() → Tensor

    

参见[ `torch.bitwise_not（） `](torch.html#torch.bitwise_not
"torch.bitwise_not")

`bitwise_not_`() → Tensor

    

就地版本的 `bitwise_not（） `

`bmm`( _batch2_ ) → Tensor

    

参见[ `torch.bmm（） `](torch.html#torch.bmm "torch.bmm")

`bool`() → Tensor

    

`self.bool（） `等于`self.to（torch.bool） `。参见 `至（） `。

`byte`() → Tensor

    

`self.byte（） `等于`self.to（torch.uint8） `。参见 `至（） `。

`cauchy_`( _median=0_ , _sigma=1_ , _*_ , _generator=None_ ) → Tensor

    

填充与柯西分布中奖号码的张量：

f(x)=1πσ(x−median)2+σ2f(x) = \dfrac{1}{\pi} \dfrac{\sigma}{(x -
\text{median})^2 + \sigma^2}f(x)=π1​(x−median)2+σ2σ​

`ceil`() → Tensor

    

参见[ `torch.ceil（） `](torch.html#torch.ceil "torch.ceil")

`ceil_`() → Tensor

    

就地版本 `小区的（） `

`char`() → Tensor

    

`self.char（） `等于`self.to（torch.int8） `。参见 `至（） `。

`cholesky`( _upper=False_ ) → Tensor

    

参见[ `torch.cholesky（） `](torch.html#torch.cholesky "torch.cholesky")

`cholesky_inverse`( _upper=False_ ) → Tensor

    

参见[ `torch.cholesky_inverse（） `](torch.html#torch.cholesky_inverse
"torch.cholesky_inverse")

`cholesky_solve`( _input2_ , _upper=False_ ) → Tensor

    

参见[ `torch.cholesky_solve（） `](torch.html#torch.cholesky_solve
"torch.cholesky_solve")

`chunk`( _chunks_ , _dim=0_ ) → List of Tensors

    

参见[ `torch.chunk（） `](torch.html#torch.chunk "torch.chunk")

`clamp`( _min_ , _max_ ) → Tensor

    

参见[ `torch.clamp（） `](torch.html#torch.clamp "torch.clamp")

`clamp_`( _min_ , _max_ ) → Tensor

    

就地版本 `夹具（） `

`clone`() → Tensor

    

返回`自 `张的副本。使副本具有相同的大小和数据类型为`自 `。

Note

不像 copy_（），该函数被记录在计算图。梯度传播到克隆的张量将传播到原来的张量。

`contiguous`() → Tensor

    

返回包含相同的数据`自 `张量的连续张量。如果`自 `张量是连续的，则该函数返回`自 `张量。

`copy_`( _src_ , _non_blocking=False_ ) → Tensor

    

拷贝从`的元素的src`到`自 `张量并返回`自 `。

的`SRC`张量必须是[ broadcastable  ](notes/broadcasting.html#broadcasting-
semantics)与`自 `张量。它可以是不同的数据类型的或驻留在不同设备上。

Parameters

    

  * **SRC** （ _张量_ ） - 源张量从复制

  * **non_blocking** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 如果`真 `，并将该副本是CPU和GPU之间，可能会出现复制异步相对于主机。对于其他情况，这种说法没有任何效果。

`cos`() → Tensor

    

参见[ `torch.cos（） `](torch.html#torch.cos "torch.cos")

`cos_`() → Tensor

    

就地版本 `COS的（） `

`cosh`() → Tensor

    

参见[ `torch.cosh（） `](torch.html#torch.cosh "torch.cosh")

`cosh_`() → Tensor

    

就地版本 `COSH的（） `

`cpu`() → Tensor

    

返回此对象的CPU内存拷贝。

如果该对象已在CPU内存和正确的设备上，则没有执行复制操作，并返回原来的对象。

`cross`( _other_ , _dim=-1_ ) → Tensor

    

参见[ `torch.cross（） `](torch.html#torch.cross "torch.cross")

`cuda`( _device=None_ , _non_blocking=False_ ) → Tensor

    

返回此对象的CUDA内存的副本。

如果该对象已在CUDA内存和正确的设备上，则没有执行复制操作，并返回原来的对象。

Parameters

    

  * **装置** （[ `torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device")） - 目标GPU设备。默认为当前CUDA设备。

  * **non_blocking** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 如果`真 `和源极被固定存储器，复制将是异步相对于主机。另外，参数没有任何影响。默认值：`假 [HTG13。`

`cumprod`( _dim_ , _dtype=None_ ) → Tensor

    

参见[ `torch.cumprod（） `](torch.html#torch.cumprod "torch.cumprod")

`cumsum`( _dim_ , _dtype=None_ ) → Tensor

    

参见[ `torch.cumsum（） `](torch.html#torch.cumsum "torch.cumsum")

`data_ptr`() → int

    

返回的`自 `张量的第一个元素的地址。

`dequantize`() → Tensor

    

给定一个量化张量，去量化它，并返回去量化的浮动张量。

`det`() → Tensor

    

参见[ `torch.det（） `](torch.html#torch.det "torch.det")

`dense_dim`() → int

    

如果`自 `是一个稀疏COO张量（即，与`torch.sparse_coo`布局），它返回致密的维数。否则，这将引发一个错误。

另请参见 `Tensor.sparse_dim（） `。

`detach`()

    

返回一个新的张量，从当前图形分离。

其结果将永远不需要梯度。

Note

回到张量股与原来相同的存储。就地对它们中的修改可以看出，并可能引发正确性检查错误。重要提示：以前，就地尺寸/步幅/存储的变化（如 resize_  /
resize_as_  /  SET_  /  transpose_
）来返回的张量也更新原有的张量。现在，这些就地变化将不再更新原有的张量，而会触发一个错误。对于稀疏张量：就地索引/值的变化（如 zero_  /
copy_  /  add_ ）发送到返回张量将不再更新原始张量，而会触发一个错误。

`detach_`()

    

分离从创建它，使其成为一个叶子图表中的张量。意见不能就地分离。

`diag`( _diagonal=0_ ) → Tensor

    

参见[ `torch.diag（） `](torch.html#torch.diag "torch.diag")

`diag_embed`( _offset=0_ , _dim1=-2_ , _dim2=-1_ ) → Tensor

    

参见[ `torch.diag_embed（） `](torch.html#torch.diag_embed "torch.diag_embed")

`diagflat`( _offset=0_ ) → Tensor

    

参见[ `torch.diagflat（） `](torch.html#torch.diagflat "torch.diagflat")

`diagonal`( _offset=0_ , _dim1=0_ , _dim2=1_ ) → Tensor

    

参见[ `torch.diagonal（） `](torch.html#torch.diagonal "torch.diagonal")

`fill_diagonal_`( _fill_value_ , _wrap=False_ ) → Tensor

    

填充有至少2维的张量的主对角线。当变暗& GT ; 2，输入的所有尺寸必须相等的长度。这个函数修改就地输入张量，并返回输入张量。

Parameters

    

  * **fill_value** （[HTG2标量） - 填充值

  * **包裹** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 对角N列高层矩阵后“包裹”。

Example:

    
    
    >>> a = torch.zeros(3, 3)
    >>> a.fill_diagonal_(5)
    tensor([[5., 0., 0.],
            [0., 5., 0.],
            [0., 0., 5.]])
    >>> b = torch.zeros(7, 3)
    >>> b.fill_diagonal_(5)
    tensor([[5., 0., 0.],
            [0., 5., 0.],
            [0., 0., 5.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]])
    >>> c = torch.zeros(7, 3)
    >>> c.fill_diagonal_(5, wrap=True)
    tensor([[5., 0., 0.],
            [0., 5., 0.],
            [0., 0., 5.],
            [0., 0., 0.],
            [5., 0., 0.],
            [0., 5., 0.],
            [0., 0., 5.]])
    

`digamma`() → Tensor

    

参见[ `torch.digamma（） `](torch.html#torch.digamma "torch.digamma")

`digamma_`() → Tensor

    

就地版本 `digamma的（） `

`dim`() → int

    

返回的`自 `张量的维数。

`dist`( _other_ , _p=2_ ) → Tensor

    

参见[ `torch.dist（） `](torch.html#torch.dist "torch.dist")

`div`( _value_ ) → Tensor

    

参见[ `torch.div（） `](torch.html#torch.div "torch.div")

`div_`( _value_ ) → Tensor

    

就地版本 `的div（） `

`dot`( _tensor2_ ) → Tensor

    

参见[ `torch.dot（） `](torch.html#torch.dot "torch.dot")

`double`() → Tensor

    

`self.double（） `等于`self.to（torch.float64） `。参见 `至（） `。

`eig`( _eigenvectors=False) - > (Tensor_, _Tensor_ )

    

参见[ `torch.eig（） `](torch.html#torch.eig "torch.eig")

`element_size`() → int

    

返回单个元素的字节大小。

Example:

    
    
    >>> torch.tensor([]).element_size()
    4
    >>> torch.tensor([], dtype=torch.uint8).element_size()
    1
    

`eq`( _other_ ) → Tensor

    

参见[ `torch.eq（） `](torch.html#torch.eq "torch.eq")

`eq_`( _other_ ) → Tensor

    

就地版本 `当量的（） `

`equal`( _other_ ) → bool

    

参见[ `torch.equal（） `](torch.html#torch.equal "torch.equal")

`erf`() → Tensor

    

参见[ `torch.erf（） `](torch.html#torch.erf "torch.erf")

`erf_`() → Tensor

    

就地版本 `ERF的（） `

`erfc`() → Tensor

    

参见[ `torch.erfc（） `](torch.html#torch.erfc "torch.erfc")

`erfc_`() → Tensor

    

就地版本 `ERFC的（） `

`erfinv`() → Tensor

    

参见[ `torch.erfinv（） `](torch.html#torch.erfinv "torch.erfinv")

`erfinv_`() → Tensor

    

就地版本 `erfinv的（） `

`exp`() → Tensor

    

参见[ `torch.exp（） `](torch.html#torch.exp "torch.exp")

`exp_`() → Tensor

    

就地版本 `EXP的（） `

`expm1`() → Tensor

    

参见[ `torch.expm1（） `](torch.html#torch.expm1 "torch.expm1")

`expm1_`() → Tensor

    

就地版本 `的expm1的（） `

`expand`( _*sizes_ ) → Tensor

    

返回`自 `张量具有扩展到更大尺寸单尺寸的新视图。

传递-1作为大小为一个尺寸是指在不改变其尺寸的大小。

张量，也可以扩大到尺寸的数量较多，而新的将在前面追加。对于新的尺寸，大小不能设置为-1。

扩大的张量不分配新的内存，但是仅创建其中大小为一的尺寸是通过设置`步幅
[HTG3到0扩展到一个更大的尺寸上的现有张量的新视图。尺寸1的任何尺寸可扩展到任意值而不分配新的内存。`

Parameters

    

***的大小** （ _torch.Size_ _或_ _INT ..._ ） - 所需的扩展大小

Warning

膨胀张量的多于一个的元件可指代单个存储器位置。其结果是，就地操作（特别是那些有量化的）可能会导致不正确的行为。如果你需要写张量，请先克隆它们。

Example:

    
    
    >>> x = torch.tensor([[1], [2], [3]])
    >>> x.size()
    torch.Size([3, 1])
    >>> x.expand(3, 4)
    tensor([[ 1,  1,  1,  1],
            [ 2,  2,  2,  2],
            [ 3,  3,  3,  3]])
    >>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
    tensor([[ 1,  1,  1,  1],
            [ 2,  2,  2,  2],
            [ 3,  3,  3,  3]])
    

`expand_as`( _other_ ) → Tensor

    

展开，这个张量，以相同的尺寸为`其他 `。 `self.expand_as（其他）​​ `等于`self.expand（other.size（））
`。

请参阅 `扩大（） `关于`更多信息展开 [HTG9。`

Parameters

    

**其他** （ `torch.Tensor`） - 结果张量具有相同的大小为`其他 [ HTG11。`

`exponential_`( _lambd=1_ , _*_ , _generator=None_ ) → Tensor

    

填充`自 `张量与从指数分布绘制的元素：

f(x)=λe−λxf(x) = \lambda e^{-\lambda x}f(x)=λe−λx

`fft`( _signal_ndim_ , _normalized=False_ ) → Tensor

    

参见[ `torch.fft（） `](torch.html#torch.fft "torch.fft")

`fill_`( _value_ ) → Tensor

    

填充具有指定值`自 `张量。

`flatten`( _input_ , _start_dim=0_ , _end_dim=-1_ ) → Tensor

    

见[ `torch.flatten（） `](torch.html#torch.flatten "torch.flatten")

`flip`( _dims_ ) → Tensor

    

参见[ `torch.flip（） `](torch.html#torch.flip "torch.flip")

`float`() → Tensor

    

`self.float（） `等于`self.to（torch.float32） `。参见 `至（） `。

`floor`() → Tensor

    

参见[ `torch.floor（） `](torch.html#torch.floor "torch.floor")

`floor_`() → Tensor

    

就地版本 `地板（） `

`fmod`( _divisor_ ) → Tensor

    

参见[ `torch.fmod（） `](torch.html#torch.fmod "torch.fmod")

`fmod_`( _divisor_ ) → Tensor

    

就地版本 `FMOD的（） `

`frac`() → Tensor

    

参见[ `torch.frac（） `](torch.html#torch.frac "torch.frac")

`frac_`() → Tensor

    

就地版本 `压裂的（） `

`gather`( _dim_ , _index_ ) → Tensor

    

参见[ `torch.gather（） `](torch.html#torch.gather "torch.gather")

`ge`( _other_ ) → Tensor

    

参见[ `torch.ge（） `](torch.html#torch.ge "torch.ge")

`ge_`( _other_ ) → Tensor

    

就地版本的Ge`（） `

`gels`( _A_ )[[source]](_modules/torch/tensor.html#Tensor.gels)

    

参见[ `torch.lstsq（） `](torch.html#torch.lstsq "torch.lstsq")

`geometric_`( _p_ , _*_ , _generator=None_ ) → Tensor

    

填充`自 `张量与来自几何分布绘制的元素：

f(X=k)=pk−1(1−p)f(X=k) = p^{k - 1} (1 - p)f(X=k)=pk−1(1−p)

`geqrf`( _) - > (Tensor_, _Tensor_ )

    

参见[ `torch.geqrf（） `](torch.html#torch.geqrf "torch.geqrf")

`ger`( _vec2_ ) → Tensor

    

参见[ `torch.ger（） `](torch.html#torch.ger "torch.ger")

`get_device`( _) - > Device ordinal (Integer_)

    

对于CUDA张量，此函数返回在其上驻留张量GPU的设备序号。对于CPU张量，则会引发错误。

Example:

    
    
    >>> x = torch.randn(3, 4, 5, device='cuda:0')
    >>> x.get_device()
    0
    >>> x.cpu().get_device()  # RuntimeError: get_device is not implemented for type torch.FloatTensor
    

`gt`( _other_ ) → Tensor

    

参见[ `torch.gt（） `](torch.html#torch.gt "torch.gt")

`gt_`( _other_ ) → Tensor

    

就地版本 `GT的（） `

`half`() → Tensor

    

`self.half（） `等于`self.to（torch.float16） `。参见 `至（） `。

`hardshrink`( _lambd=0.5_ ) → Tensor

    

参见[ `torch.nn.functional.hardshrink（） `
](nn.functional.html#torch.nn.functional.hardshrink
"torch.nn.functional.hardshrink")

`histc`( _bins=100_ , _min=0_ , _max=0_ ) → Tensor

    

参见[ `torch.histc（） `](torch.html#torch.histc "torch.histc")

`ifft`( _signal_ndim_ , _normalized=False_ ) → Tensor

    

参见[ `torch.ifft（） `](torch.html#torch.ifft "torch.ifft")

`index_add_`( _dim_ , _index_ , _tensor_ ) → Tensor

    

积累的[ `的元素张量 `](torch.html#torch.tensor "torch.tensor")到`自 `张量通过增加在给定的顺序的索引`
索引 `。例如，如果`暗淡 ==  0`和`索引[I]  = =  [HTG27：J `，则`i的 `次的[ `行张量 `
](torch.html#torch.tensor "torch.tensor")被添加到`[HTG41：J `次的`自 `行。

的 `暗淡 `次的[ `维张量 `](torch.html#torch.tensor "torch.tensor")必须具有尺寸为长度相同的`索引
`（它必须是一个矢量），和所有其他的尺寸必须相符`自 `，或错误将被提高。

Note

当使用CUDA后端，该操作可以诱导非确定性的行为是不容易断开。请参阅[ 重复性 ](notes/randomness.html)为背景的音符。

Parameters

    

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 维沿着该索引

  * **索引** （ _LongTensor_ ） - 的[ `索引张量 `](torch.html#torch.tensor "torch.tensor")从选择

  * **张量** （ _张量_ ） - 包含值张量来添加

Example:

    
    
    >>> x = torch.ones(5, 3)
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    >>> index = torch.tensor([0, 4, 2])
    >>> x.index_add_(0, index, t)
    tensor([[  2.,   3.,   4.],
            [  1.,   1.,   1.],
            [  8.,   9.,  10.],
            [  1.,   1.,   1.],
            [  5.,   6.,   7.]])
    

`index_add`( _dim_ , _index_ , _tensor_ ) → Tensor

    

外的地方的 `版本torch.Tensor.index_add_（） `

`index_copy_`( _dim_ , _index_ , _tensor_ ) → Tensor

    

的[ `拷贝的元素张量 `](torch.html#torch.tensor "torch.tensor")成通过在[HTG10给定的顺序选择指数的`
自 `张量] 索引 。例如，如果`暗淡 ==  0`和`索引[I]  = =  [HTG27：J `，则`i的 `次的[ `行张量 `
](torch.html#torch.tensor "torch.tensor")复制到`[HTG41：J `次的`自行 `。

The `dim`th dimension of [`tensor`](torch.html#torch.tensor "torch.tensor")
must have the same size as the length of `index`(which must be a vector), and
all other dimensions must match `self`, or an error will be raised.

Parameters

    

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – dimension along which to index

  * **index** ( _LongTensor_ ) – indices of [`tensor`](torch.html#torch.tensor "torch.tensor") to select from

  * **张量** （ _张量_ ） - 包含值张量来复制

Example:

    
    
    >>> x = torch.zeros(5, 3)
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    >>> index = torch.tensor([0, 4, 2])
    >>> x.index_copy_(0, index, t)
    tensor([[ 1.,  2.,  3.],
            [ 0.,  0.,  0.],
            [ 7.,  8.,  9.],
            [ 0.,  0.,  0.],
            [ 4.,  5.,  6.]])
    

`index_copy`( _dim_ , _index_ , _tensor_ ) → Tensor

    

外的地方的 `版本torch.Tensor.index_copy_（） `

`index_fill_`( _dim_ , _index_ , _val_ ) → Tensor

    

填充`自的元素 `张量与值`VAL`通过在`索引给定的顺序选择所述索引 `。

Parameters

    

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – dimension along which to index

  * **索引** （ _LongTensor_ ） - 的`指数自 `张量，以填补在

  * **VAL** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 以填补的值

Example::

    
    
    
    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    >>> index = torch.tensor([0, 2])
    >>> x.index_fill_(1, index, -1)
    tensor([[-1.,  2., -1.],
            [-1.,  5., -1.],
            [-1.,  8., -1.]])
    

`index_fill`( _dim_ , _index_ , _value_ ) → Tensor

    

外的地方的 `版本torch.Tensor.index_fill_（） `

`index_put_`( _indices_ , _value_ , _accumulate=False_ ) → Tensor

    

从张量`值 `把值代入张量`自 `使用中指定的索引`指数 `（这是张量的元组）。表达式`tensor.index_put_（指数， 值） `等于`
张量[指数]  =  值 `。返回`自 [HTG31。`

如果`积累 `是`真 `，元素在[ `张量 `](torch.html#torch.tensor "torch.tensor")添加到`自
`。如果累积为`假 `，行为是不确定如果索引包含重复的元素。

Parameters

    

  * **指数** （ _LongTensor_ 的元组） - 用于索引到自张量。

  * **的值** （ _张量_ ） - 相同类型的张量为自。

  * **积累** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 是否积累到自

`index_put`( _indices_ , _value_ , _accumulate=False_ ) → Tensor

    

外的地方版本的 `index_put_（） `

`index_select`( _dim_ , _index_ ) → Tensor

    

参见[ `torch.index_select（） `](torch.html#torch.index_select
"torch.index_select")

`indices`() → Tensor

    

如果`自 `是一个稀疏COO张量（即，与`torch.sparse_coo`布局），它返回所包含的索引张量的图。否则，这将引发一个错误。

另请参见 `Tensor.values（） `。

Note

这种方法只能在聚结的稀疏张量被调用。参见`对于细节Tensor.coalesce（） `。

`int`() → Tensor

    

`self.int（） `等于`self.to（torch.int32） `。参见 `至（） `。

`int_repr`() → Tensor

    

给定一个量化的张量，`self.int_repr（） `返回与uint8_t作为存储给定的张量的底层uint8_t值数据类型的CPU张量。

`inverse`() → Tensor

    

参见[ `torch.inverse（） `](torch.html#torch.inverse "torch.inverse")

`irfft`( _signal_ndim_ , _normalized=False_ , _onesided=True_ ,
_signal_sizes=None_ ) → Tensor

    

参见[ `torch.irfft（） `](torch.html#torch.irfft "torch.irfft")

`is_contiguous`() → bool

    

返回true如果`自 `张量是在用C顺序存储器中连续。

`is_floating_point`() → bool

    

返回true的`自 `的数据类型是一个浮点数据类型。

`is_leaf`()

    

有所有的张量 `requires_grad`是`假 `将叶按约定张量。

对于张量具有 `requires_grad`，它是`真 `，他们将叶张量如果他们被创建用户。这意味着它们不是一个操作的结果等`grad_fn
`是无。

仅叶张量人员在 `GRAD`至 `向后（在呼叫期间填充） `。为了得到 `毕业 `填充的无叶张量，你可以使用 `retain_grad（）
`[ HTG23。

Example:

    
    
    >>> a = torch.rand(10, requires_grad=True)
    >>> a.is_leaf
    True
    >>> b = torch.rand(10, requires_grad=True).cuda()
    >>> b.is_leaf
    False
    # b was created by the operation that cast a cpu Tensor into a cuda Tensor
    >>> c = torch.rand(10, requires_grad=True) + 2
    >>> c.is_leaf
    False
    # c was created by the addition operation
    >>> d = torch.rand(10).cuda()
    >>> d.is_leaf
    True
    # d does not require gradients and so has no operation creating it (that is tracked by the autograd engine)
    >>> e = torch.rand(10).cuda().requires_grad_()
    >>> e.is_leaf
    True
    # e requires gradients and has no operations creating it
    >>> f = torch.rand(10, requires_grad=True, device="cuda")
    >>> f.is_leaf
    True
    # f requires grad, has no operation creating it
    

`is_pinned`()[[source]](_modules/torch/tensor.html#Tensor.is_pinned)

    

如果此张驻留在固定的内存，则返回true

`is_set_to`( _tensor_ ) → bool

    

如果该对象是指从Torch C API作为给定的张量相同`THTensor`对象返回真。

`is_shared`()[[source]](_modules/torch/tensor.html#Tensor.is_shared)

    

检查，如果张量是在共享存储器中。

这始终是`真 [HTG3对于CUDA张量。`

`is_signed`() → bool

    

如果`自 `数据类型是有符号的数据类型，则返回True。

`is_sparse`()

    

`item`() → number

    

返回此张量作为标准Python数的值。这仅适用于一个元素张量。对于其它情况，参见 `tolist（） `。

此操作不可微。

Example:

    
    
    >>> x = torch.tensor([1.0])
    >>> x.item()
    1.0
    

`kthvalue`( _k_ , _dim=None_ , _keepdim=False) - > (Tensor_, _LongTensor_ )

    

参见[ `torch.kthvalue（） `](torch.html#torch.kthvalue "torch.kthvalue")

`le`( _other_ ) → Tensor

    

参见[ `torch.le（） `](torch.html#torch.le "torch.le")

`le_`( _other_ ) → Tensor

    

就地版本的 `乐（） `

`lerp`( _end_ , _weight_ ) → Tensor

    

参见[ `torch.lerp（） `](torch.html#torch.lerp "torch.lerp")

`lerp_`( _end_ , _weight_ ) → Tensor

    

就地版本 `线性插值的（） `

`log`() → Tensor

    

参见[ `torch.log（） `](torch.html#torch.log "torch.log")

`log_`() → Tensor

    

就地版本 `日志（） `

`logdet`() → Tensor

    

参见[ `torch.logdet（） `](torch.html#torch.logdet "torch.logdet")

`log10`() → Tensor

    

参见[ `torch.log10（） `](torch.html#torch.log10 "torch.log10")

`log10_`() → Tensor

    

就地版本 `LOG10的（） `

`log1p`() → Tensor

    

参见[ `torch.log1p（） `](torch.html#torch.log1p "torch.log1p")

`log1p_`() → Tensor

    

就地版本 `log1p的（） `

`log2`() → Tensor

    

参见[ `torch.log2（） `](torch.html#torch.log2 "torch.log2")

`log2_`() → Tensor

    

就地版本 `的log 2的（） `

`log_normal_`( _mean=1_ , _std=2_ , _*_ , _generator=None_ )

    

填充`自 `张量与从所述给定参数的对数正态分布的数字样本意味着 μ \亩 μ 和标准偏差 σ \西格玛 σ 。注意，[ `意味着 `
](torch.html#torch.mean "torch.mean")和[ `STD`](torch.html#torch.std
"torch.std")是平均值和底层的标准偏差正态分布，而不是返回的分布：

f(x)=1xσ2π e−(ln⁡x−μ)22σ2f(x) = \dfrac{1}{x \sigma \sqrt{2\pi}}\
e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}f(x)=xσ2π​1​ e−2σ2(lnx−μ)2​

`logsumexp`( _dim_ , _keepdim=False_ ) → Tensor

    

参见[ `torch.logsumexp（） `](torch.html#torch.logsumexp "torch.logsumexp")

`long`() → Tensor

    

`self.long（） `等于`self.to（torch.int64） `。参见 `至（） `。

`lstsq`( _A) - > (Tensor_, _Tensor_ )

    

See [`torch.lstsq()`](torch.html#torch.lstsq "torch.lstsq")

`lt`( _other_ ) → Tensor

    

参见[ `torch.lt（） `](torch.html#torch.lt "torch.lt")

`lt_`( _other_ ) → Tensor

    

就地版本 `LT的（） `

`lu`( _pivot=True_ , _get_infos=False_
)[[source]](_modules/torch/tensor.html#Tensor.lu)

    

参见[ `torch.lu（） `](torch.html#torch.lu "torch.lu")

`lu_solve`( _LU_data_ , _LU_pivots_ ) → Tensor

    

参见[ `torch.lu_solve（） `](torch.html#torch.lu_solve "torch.lu_solve")

`map_`( _tensor_ , _callable_ )

    

适用`可调用 `在`自 `张量的每个元件和给定[ `张量 `](torch.html#torch.tensor
"torch.tensor")，并将结果存储在`自 `张量。 `自 `张量和给定的[ `张量 `](torch.html#torch.tensor
"torch.tensor")必须[ broadcastable  ](notes/broadcasting.html#broadcasting-
semantics)。

在`调用 `应具备的特征：

    
    
    def callable(a, b) -> number
    

`masked_scatter_`( _mask_ , _source_ )

    

从`源 `复制内容纳入`自 `在位置张量，其中`掩模 `为真。的`形状掩模 `必须[ broadcastable
](notes/broadcasting.html#broadcasting-semantics)与下面的张量的形状。的`源
`应当具有至少一样多的元素的那些中`数掩模 `

Parameters

    

  * **掩模** （ _BoolTensor_ ） - 布尔掩码

  * **源** （ _张量_ ） - 张量从复制

Note

的`掩模 `操作上的`自 `张量，而不是在给定的`源 `张量。

`masked_scatter`( _mask_ , _tensor_ ) → Tensor

    

外的地方的 `版本torch.Tensor.masked_scatter_（） `

`masked_fill_`( _mask_ , _value_ )

    

填充的`自 `张量元素与`值 `其中`掩模 `为True。的`形状掩模 `必须[ broadcastable
](notes/broadcasting.html#broadcasting-semantics)与下面的张量的形状。

Parameters

    

  * **mask** ( _BoolTensor_) – the boolean mask

  * **值** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 的值与填

`masked_fill`( _mask_ , _value_ ) → Tensor

    

外的地方的 `版本torch.Tensor.masked_fill_（） `

`masked_select`( _mask_ ) → Tensor

    

参见[ `torch.masked_select（） `](torch.html#torch.masked_select
"torch.masked_select")

`matmul`( _tensor2_ ) → Tensor

    

参见[ `torch.matmul（） `](torch.html#torch.matmul "torch.matmul")

`matrix_power`( _n_ ) → Tensor

    

参见[ `torch.matrix_power（） `](torch.html#torch.matrix_power
"torch.matrix_power")

`max`( _dim=None_ , _keepdim=False) - > Tensor or (Tensor_, _Tensor_ )

    

参见[ `torch.max（） `](torch.html#torch.max "torch.max")

`mean`( _dim=None_ , _keepdim=False) - > Tensor or (Tensor_, _Tensor_ )

    

参见[ `torch.mean（） `](torch.html#torch.mean "torch.mean")

`median`( _dim=None_ , _keepdim=False) - > (Tensor_, _LongTensor_ )

    

参见[ `torch.median（） `](torch.html#torch.median "torch.median")

`min`( _dim=None_ , _keepdim=False) - > Tensor or (Tensor_, _Tensor_ )

    

参见[ `torch.min（） `](torch.html#torch.min "torch.min")

`mm`( _mat2_ ) → Tensor

    

参见[ `torch.mm（） `](torch.html#torch.mm "torch.mm")

`mode`( _dim=None_ , _keepdim=False) - > (Tensor_, _LongTensor_ )

    

参见[ `torch.mode（） `](torch.html#torch.mode "torch.mode")

`mul`( _value_ ) → Tensor

    

参见[ `torch.mul（） `](torch.html#torch.mul "torch.mul")

`mul_`( _value_ )

    

就地版本 `MUL的（） `

`multinomial`( _num_samples_ , _replacement=False_ , _*_ , _generator=None_ )
→ Tensor

    

参见[ `torch.multinomial（） `](torch.html#torch.multinomial
"torch.multinomial")

`mv`( _vec_ ) → Tensor

    

参见[ `torch.mv（） `](torch.html#torch.mv "torch.mv")

`mvlgamma`( _p_ ) → Tensor

    

参见[ `torch.mvlgamma（） `](torch.html#torch.mvlgamma "torch.mvlgamma")

`mvlgamma_`( _p_ ) → Tensor

    

就地版本 `mvlgamma的（） `

`narrow`( _dimension_ , _start_ , _length_ ) → Tensor

    

参见[ `torch.narrow（） `](torch.html#torch.narrow "torch.narrow")

Example:

    
    
    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> x.narrow(0, 0, 2)
    tensor([[ 1,  2,  3],
            [ 4,  5,  6]])
    >>> x.narrow(1, 1, 2)
    tensor([[ 2,  3],
            [ 5,  6],
            [ 8,  9]])
    

`narrow_copy`( _dimension_ , _start_ , _length_ ) → Tensor

    

同 `Tensor.narrow（） `，除了返回一个副本，而不是共享存储。这主要是为稀疏张量，其不具有共享存储窄方法。主叫``narrow_copy
`与``dimemsion  & GT ;  self.sparse_dim（） ``会返回缩小的相关密集维度副本，``self.shape`
`相应更新。

`ndimension`() → int

    

Alias for `dim()`

`ne`( _other_ ) → Tensor

    

参见[ `torch.ne（） `](torch.html#torch.ne "torch.ne")

`ne_`( _other_ ) → Tensor

    

就地版本 `NE的（） `

`neg`() → Tensor

    

参见[ `torch.neg（） `](torch.html#torch.neg "torch.neg")

`neg_`() → Tensor

    

就地版本 `NEG的（） `

`nelement`() → int

    

别名 `numel（） `

`nonzero`() → LongTensor

    

参见[ `torch.nonzero（） `](torch.html#torch.nonzero "torch.nonzero")

`norm`( _p='fro'_ , _dim=None_ , _keepdim=False_ , _dtype=None_
)[[source]](_modules/torch/tensor.html#Tensor.norm)

    

参见[ `torch.norm（） `](torch.html#torch.norm "torch.norm")

`normal_`( _mean=0_ , _std=1_ , _*_ , _generator=None_ ) → Tensor

    

填充`自 `与元素样品张量从正态分布由[ `参数化的意思是 `](torch.html#torch.mean "torch.mean")和[ `
STD`](torch.html#torch.std "torch.std")。

`numel`() → int

    

参见[ `torch.numel（） `](torch.html#torch.numel "torch.numel")

`numpy`() → numpy.ndarray

    

返回`自 `张量作为NumPy的`ndarray`。这个张量和返回`ndarray`共享同一基础存储。为`自变化 `张量将反映在`
ndarray`，反之亦然。

`orgqr`( _input2_ ) → Tensor

    

参见[ `torch.orgqr（） `](torch.html#torch.orgqr "torch.orgqr")

`ormqr`( _input2_ , _input3_ , _left=True_ , _transpose=False_ ) → Tensor

    

参见[ `torch.ormqr（） `](torch.html#torch.ormqr "torch.ormqr")

`permute`( _*dims_ ) → Tensor

    

置换，这个张量的尺寸。

Parameters

    

***变暗** （ _INT ..._ ） - 尺寸的所需排序

例

    
    
    >>> x = torch.randn(2, 3, 5)
    >>> x.size()
    torch.Size([2, 3, 5])
    >>> x.permute(2, 0, 1).size()
    torch.Size([5, 2, 3])
    

`pin_memory`() → Tensor

    

复制张量固定的内存，如果它不是已经固定。

`pinverse`() → Tensor

    

参见[ `torch.pinverse（） `](torch.html#torch.pinverse "torch.pinverse")

`pow`( _exponent_ ) → Tensor

    

参见[ `torch.pow（） `](torch.html#torch.pow "torch.pow")

`pow_`( _exponent_ ) → Tensor

    

就地版本 `POW的（） `

`prod`( _dim=None_ , _keepdim=False_ , _dtype=None_ ) → Tensor

    

参见[ `torch.prod（） `](torch.html#torch.prod "torch.prod")

`put_`( _indices_ , _tensor_ , _accumulate=False_ ) → Tensor

    

拷贝从[ `的元素张量 `](torch.html#torch.tensor "torch.tensor")到由索引指定的位置。用于索引的目的，`自
`张量就好像它是一个1-d张量处理。

If `accumulate`is `True`, the elements in [`tensor`](torch.html#torch.tensor
"torch.tensor") are added to `self`. If accumulate is `False`, the behavior is
undefined if indices contain duplicate elements.

Parameters

    

  * **指数** （ _LongTensor_ ） - 索引为自

  * **张量** （ _张量_ ） - 包含值从复制张量

  * **accumulate** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether to accumulate into self

Example:

    
    
    >>> src = torch.tensor([[4, 3, 5],
                            [6, 7, 8]])
    >>> src.put_(torch.tensor([1, 3]), torch.tensor([9, 10]))
    tensor([[  4,   9,   5],
            [ 10,   7,   8]])
    

`qr`( _some=True) - > (Tensor_, _Tensor_ )

    

参见[ `torch.qr（） `](torch.html#torch.qr "torch.qr")

`qscheme`() → torch.qscheme

    

返回给定QTensor的量化方案。

`q_scale`() → float

    

鉴于线性（仿射）量化量化的张量，返回底层量化器的规模（）。

`q_zero_point`() → int

    

鉴于线性（仿射）量化量化的张量，返回底层量化器的zero_point（）。

`random_`( _from=0_ , _to=None_ , _*_ , _generator=None_ ) → Tensor

    

填充`自 `超过`[从离散均匀分布采样数张量从 至 -  1]`。如果未指定，则这些值通常仅由`自
`张量的数据类型的限制。然而，对于浮点类型，如果未指定，范围将是`[0， 2 ^尾数]`，以确保每一个值可表示。例如，
torch.tensor（1，D型细胞= torch.double）.random_（）将均匀的`[0， 2 ^ 53]`。

`reciprocal`() → Tensor

    

参见[ `torch.reciprocal（） `](torch.html#torch.reciprocal "torch.reciprocal")

`reciprocal_`() → Tensor

    

就地版本的 `倒数（） `

`register_hook`( _hook_
)[[source]](_modules/torch/tensor.html#Tensor.register_hook)

    

寄存器向后钩。

钩将被称为每相对于该张量的梯度被计算的时间。钩子应该具有以下特征：

    
    
    hook(grad) -> Tensor or None
    

钩不应该修改它的参数，但它可以任选地返回一个新的梯度，这将代替 `GRAD`一起使用。

该函数返回与方法`handle.remove手柄（） `，其去除从所述模块的钩。

Example:

    
    
    >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
    >>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
    >>> v.backward(torch.tensor([1., 2., 3.]))
    >>> v.grad
    
     2
     4
     6
    [torch.FloatTensor of size (3,)]
    
    >>> h.remove()  # removes the hook
    

`remainder`( _divisor_ ) → Tensor

    

参见[ `torch.remainder（） `](torch.html#torch.remainder "torch.remainder")

`remainder_`( _divisor_ ) → Tensor

    

就地版本 `剩余的（） `

`renorm`( _p_ , _dim_ , _maxnorm_ ) → Tensor

    

参见[ `torch.renorm（） `](torch.html#torch.renorm "torch.renorm")

`renorm_`( _p_ , _dim_ , _maxnorm_ ) → Tensor

    

就地版本 `重归一化的（） `

`repeat`( _*sizes_ ) → Tensor

    

重复沿着指定的尺寸，这个张量。

不同于 `扩大（） `，该函数将张量的数据。

Warning

`torch.repeat（） `从[ numpy.repeat
](https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html)行为不同，但是更类似于[
numpy.tile
](https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html)。对于类似于操作者numpy.repeat
参见[ `torch.repeat_interleave（） `](torch.html#torch.repeat_interleave
"torch.repeat_interleave")。

Parameters

    

**尺寸** （ _torch.Size_ _或_ _INT ..._ ） - 的次数重复此张量沿着每个维度

Example:

    
    
    >>> x = torch.tensor([1, 2, 3])
    >>> x.repeat(4, 2)
    tensor([[ 1,  2,  3,  1,  2,  3],
            [ 1,  2,  3,  1,  2,  3],
            [ 1,  2,  3,  1,  2,  3],
            [ 1,  2,  3,  1,  2,  3]])
    >>> x.repeat(4, 2, 1).size()
    torch.Size([4, 2, 3])
    

`repeat_interleave`( _repeats_ , _dim=None_ ) → Tensor

    

参见[ `torch.repeat_interleave（） `](torch.html#torch.repeat_interleave
"torch.repeat_interleave")。

`requires_grad`()

    

是`真 [HTG3如果梯度需要计算该张量，`假 [HTG7否则。``

Note

该梯度需要计算的张量，这一事实并不意味着 `毕业 `属性将被填充，请参阅 `is_leaf`的更多细节。

`requires_grad_`( _requires_grad=True_ ) → Tensor

    

改变，如果autograd应在此张记录操作：设置这个张量的 `requires_grad`属性原地。返回此张量。

`require_grad_（）的 `主要使用情形是告诉autograd到开始记录的操作的张量`张量 `。如果`张量 `具有`
requires_grad =假 `（因为它被通过的DataLoader获得，或者需要预处理或初始化），`tensor.requires_grad_（）
`使得它使autograd将开始记录操作上`张 [HTG23。`

Parameters

    

**requires_grad** （[ _布尔_
](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")）
- 如果autograd应在此张记录操作。默认值：`真 [HTG9。`

Example:

    
    
    >>> # Let's say we want to preprocess some saved weights and use
    >>> # the result as new weights.
    >>> saved_weights = [0.1, 0.2, 0.3, 0.25]
    >>> loaded_weights = torch.tensor(saved_weights)
    >>> weights = preprocess(loaded_weights)  # some function
    >>> weights
    tensor([-0.5503,  0.4926, -2.1158, -0.8303])
    
    >>> # Now, start to record operations done to weights
    >>> weights.requires_grad_()
    >>> out = weights.pow(2).sum()
    >>> out.backward()
    >>> weights.grad
    tensor([-1.1007,  0.9853, -4.2316, -1.6606])
    

`reshape`( _*shape_ ) → Tensor

    

返回与元素`自 `但具有指定形状的相同的数据和数字的张量。如果`定型 `是与当前的形状相容的此方法返回的图。参见 `
torch.Tensor.view（） `上时，它可以返回的图。

参见[ `torch.reshape（） `](torch.html#torch.reshape "torch.reshape")

Parameters

    

**定型** （ _蟒的元组：整数_ _或_ _INT ..._ ） - 所需的形状

`reshape_as`( _other_ ) → Tensor

    

返回该张量为相同的形状，`其他 `。 `self.reshape_as（其他）​​ `等于`self.reshape（other.sizes（））
`。如果`other.sizes（） `是与当前的形状相容的此方法返回的图。参见 `torch.Tensor.view（） `上时，它可以返回的图。

请参阅[ `重塑（） `](torch.html#torch.reshape "torch.reshape")关于`更多信息重塑 [HTG9。`

Parameters

    

**其他** （ `torch.Tensor`） - 结果张量具有相同的形状`其他 [ HTG11。`

`resize_`( _*sizes_ ) → Tensor

    

调整大小`自
`张量，以指定的大小。如果元件的数量大于当前存储大小，则底层存储被调整大小，以适应元件的新号码。如果元件的数目较小时，底层的存储不被改变。存在的元素将会保留，但任何新的内存未初始化。

Warning

这是一个低级别的方法。存储被重新解释为C-连续的，忽略当前进展（除非目标大小等于电流的大小，在这种情况下，张量保持不变）。在大多数情况下，你反而要使用 `
视图（） `，其检查连续性，或 `重塑（）`，其复制数据如果需要的话。若要更改就地自定义步幅大小，请参阅 `SET_（） `。

Parameters

    

**尺寸** （ _torch.Size_ _或_ _INT ..._ ） - 所需的大小

Example:

    
    
    >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]])
    >>> x.resize_(2, 2)
    tensor([[ 1,  2],
            [ 3,  4]])
    

`resize_as_`( _tensor_ ) → Tensor

    

调整大小的`自 `张量是大小相同的指定[ `张量 `](torch.html#torch.tensor "torch.tensor")。这等同于`
self.resize_（tensor.size（）） `。

`retain_grad`()[[source]](_modules/torch/tensor.html#Tensor.retain_grad)

    

启用了非叶张量.grad属性。

`rfft`( _signal_ndim_ , _normalized=False_ , _onesided=True_ ) → Tensor

    

参见[ `torch.rfft（） `](torch.html#torch.rfft "torch.rfft")

`roll`( _shifts_ , _dims_ ) → Tensor

    

参见[ `torch.roll（） `](torch.html#torch.roll "torch.roll")

`rot90`( _k_ , _dims_ ) → Tensor

    

参见[ `torch.rot90（） `](torch.html#torch.rot90 "torch.rot90")

`round`() → Tensor

    

参见[ `torch.round（） `](torch.html#torch.round "torch.round")

`round_`() → Tensor

    

就地版本的 `轮（） `

`rsqrt`() → Tensor

    

参见[ `torch.rsqrt（） `](torch.html#torch.rsqrt "torch.rsqrt")

`rsqrt_`() → Tensor

    

就地版本 `rsqrt的（） `

`scatter`( _dim_ , _index_ , _source_ ) → Tensor

    

外的地方的 `版本torch.Tensor.scatter_（） `

`scatter_`( _dim_ , _index_ , _src_ ) → Tensor

    

写入所有值从张量`SRC`到`自 `在`索引 [指定的索引HTG11]张量。对于SRC `，它的输出索引是由它的指数`SRC`为`
维[HTG22指定在`每个值] ！=  暗淡 `和在`索引 `为`维 [相应的值HTG35] = `暗淡 `。

对于3-d张量，`自 `被更新为：

    
    
    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
    

这是在 `聚集（） `所描述的方式相反的操作。

`自 `，`索引 `和`SRC`（如果它是一个张量）应具有相同的维数。它也要求`index.size（d） & LT ; =
src.size（d） [HTG19用于所有维度`d`，以及`index.size（d） & LT ; =  self.size（d）
`针对所有维度`d  ！=  暗淡 `。`

此外，对于 `聚集（） `，`索引的值 `必须介于`0`和`self.size（DIM） -  1`以下，并且所有在一排值一起指定的尺寸
`暗淡 `必须是唯一的。

Parameters

    

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的轴，沿着该索引

  * **索引** （ _LongTensor_ ） - 元件以分散的指数，可以是空的或SRC的相同的尺寸。当空，操作返回身份

  * **SRC** （ _张量_ ） - 源元素（一个或多个）以散射，柜面值未指定被

  * **值** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 源元素（一个或多个）以散射，柜面 SRC 未指定

Example:

    
    
    >>> x = torch.rand(2, 5)
    >>> x
    tensor([[ 0.3992,  0.2908,  0.9044,  0.4850,  0.6004],
            [ 0.5735,  0.9006,  0.6797,  0.4152,  0.1732]])
    >>> torch.zeros(3, 5).scatter_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
    tensor([[ 0.3992,  0.9006,  0.6797,  0.4850,  0.6004],
            [ 0.0000,  0.2908,  0.0000,  0.4152,  0.0000],
            [ 0.5735,  0.0000,  0.9044,  0.0000,  0.1732]])
    
    >>> z = torch.zeros(2, 4).scatter_(1, torch.tensor([[2], [3]]), 1.23)
    >>> z
    tensor([[ 0.0000,  0.0000,  1.2300,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  1.2300]])
    

`scatter_add_`( _dim_ , _index_ , _other_ ) → Tensor

    

将从张量`的所有值其他 `到`自 `在`索引 [指定的索引HTG11]张量以类似的方式为 `scatter_（） `。对于在`其他
`，它被添加到索引中`自 `这是由其索引指定在`[HTG27每个值]等 `为`维 ！=  暗淡 `和在`索引对应的值 `为`维 =  暗淡 `。`

For a 3-D tensor, `self`is updated as:

    
    
    self[index[i][j][k]][j][k] += other[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] += other[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] += other[i][j][k]  # if dim == 2
    

`自 `，`索引 `和`其他 `应当具有相同的维数。它也要求`index.size（d） & LT ; =  other.size（d）
[HTG19用于所有维度`d`，以及`index.size（d） & LT ; =  self.size（d） `针对所有维度`d  ！=  暗淡
`。`

此外，对于 `聚集（） `，`索引的值 `必须介于`0`和`self.size（DIM） -  1`以下，并且所有在一排值一起指定的尺寸
`暗淡 `必须是唯一的。

Note

When using the CUDA backend, this operation may induce nondeterministic
behaviour that is not easily switched off. Please see the notes on
[Reproducibility](notes/randomness.html) for background.

Parameters

    

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – the axis along which to index

  * **索引** （ _LongTensor_ ） - 元件的散射，并添加，指数可以是空的或SRC的相同的尺寸。当空，操作返回身份。

  * **其他** （ _张量_ ） - 源元件散射和添加

Example:

    
    
    >>> x = torch.rand(2, 5)
    >>> x
    tensor([[0.7404, 0.0427, 0.6480, 0.3806, 0.8328],
            [0.7953, 0.2009, 0.9154, 0.6782, 0.9620]])
    >>> torch.ones(3, 5).scatter_add_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
    tensor([[1.7404, 1.2009, 1.9154, 1.3806, 1.8328],
            [1.0000, 1.0427, 1.0000, 1.6782, 1.0000],
            [1.7953, 1.0000, 1.6480, 1.0000, 1.9620]])
    

`scatter_add`( _dim_ , _index_ , _source_ ) → Tensor

    

外的地方的 `版本torch.Tensor.scatter_add_（） `

`select`( _dim_ , _index_ ) → Tensor

    

切片的`自 `沿着给定的索引在所选择的尺寸张量。这个函数返回删除了给定尺寸的张量。

Parameters

    

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的尺寸切

  * **索引** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 索引与选择

Note

`选择（） `等价于切片。例如，`tensor.select（0， 索引） `等于`张量[指数]`和`tensor.select（2， 索引）
`等于`张量[：，：，指数]`。

`set_`( _source=None_ , _storage_offset=0_ , _size=None_ , _stride=None_ ) →
Tensor

    

设置底层存储，大小和进步。如果`源 `是一个张量，`自 `张量将共享相同的存储，并且具有相同的尺寸和步幅为`源 `。在一个张量元素的变化将反映在其他。

如果`源 `是`存放 `，该方法将底层存储，偏移量，大小，和步幅。

Parameters

    

  * **源** （ _张量_ _或_ _存放_ ） - 张量或存储使用

  * **storage_offset** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 在存储器中的偏移

  * **大小** （ _torch.Size_ _，_ _可选_ ） - 所需的大小。默认为源的大小。

  * **步幅** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选_ ） - 所需的步幅。默认为C-连续进展。

`share_memory_`()[[source]](_modules/torch/tensor.html#Tensor.share_memory_)

    

移动底层存储到共享存储器中。

这是如果底层存储已经在共享存储器和用于CUDA张量无操作。在共享存储器中张量不能被调整大小。

`short`() → Tensor

    

`self.short（） `等于`self.to（torch.int16） `。参见 `至（） `。

`sigmoid`() → Tensor

    

参见[ `torch.sigmoid（） `](torch.html#torch.sigmoid "torch.sigmoid")

`sigmoid_`() → Tensor

    

就地版本 `乙状结肠的（） `

`sign`() → Tensor

    

参见[ `torch.sign（） `](torch.html#torch.sign "torch.sign")

`sign_`() → Tensor

    

就地版本 `符号的（） `

`sin`() → Tensor

    

参见[ `torch.sin（） `](torch.html#torch.sin "torch.sin")

`sin_`() → Tensor

    

就地版本 `罪的（） `

`sinh`() → Tensor

    

参见[ `torch.sinh（） `](torch.html#torch.sinh "torch.sinh")

`sinh_`() → Tensor

    

就地版本的 `的sinh（） `

`size`() → torch.Size

    

返回`自 `张量的大小。返回的值是[ `元组 `
](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python
v3.7\)")的子类。

Example:

    
    
    >>> torch.empty(3, 4, 5).size()
    torch.Size([3, 4, 5])
    

`slogdet`( _) - > (Tensor_, _Tensor_ )

    

参见[ `torch.slogdet（） `](torch.html#torch.slogdet "torch.slogdet")

`solve`( _A_ ) → Tensor, Tensor

    

参见[ `torch.solve（） `](torch.html#torch.solve "torch.solve")

`sort`( _dim=-1_ , _descending=False) - > (Tensor_, _LongTensor_ )

    

参见[ `torch.sort（） `](torch.html#torch.sort "torch.sort")

`split`( _split_size_ , _dim=0_
)[[source]](_modules/torch/tensor.html#Tensor.split)

    

参见[ `torch.split（） `](torch.html#torch.split "torch.split")

`sparse_mask`( _input_ , _mask_ ) → Tensor

    

返回 过滤通过`指数与值的新SparseTensor从张量`输入掩码 `和值将被忽略。 `输入 `和`掩模 `必须具有相同的形状。`

Parameters

    

  * **输入** （ _张量_ ） - 的输入张量

  * **掩模** （ _SparseTensor_ ） - 一个SparseTensor我们筛选`根据它的索引输入`

Example:

    
    
    >>> nnz = 5
    >>> dims = [5, 5, 2, 2]
    >>> I = torch.cat([torch.randint(0, dims[0], size=(nnz,)),
                       torch.randint(0, dims[1], size=(nnz,))], 0).reshape(2, nnz)
    >>> V = torch.randn(nnz, dims[2], dims[3])
    >>> size = torch.Size(dims)
    >>> S = torch.sparse_coo_tensor(I, V, size).coalesce()
    >>> D = torch.randn(dims)
    >>> D.sparse_mask(S)
    tensor(indices=tensor([[0, 0, 0, 2],
                           [0, 1, 4, 3]]),
           values=tensor([[[ 1.6550,  0.2397],
                           [-0.1611, -0.0779]],
    
                          [[ 0.2326, -1.0558],
                           [ 1.4711,  1.9678]],
    
                          [[-0.5138, -0.0411],
                           [ 1.9417,  0.5158]],
    
                          [[ 0.0793,  0.0036],
                           [-0.2569, -0.1055]]]),
           size=(5, 5, 2, 2), nnz=4, layout=torch.sparse_coo)
    

`sparse_dim`() → int

    

如果`自 `是一个稀疏COO张量（即，与`torch.sparse_coo`布局），它返回稀疏的维度数目。否则，这将引发一个错误。

另请参见 `Tensor.dense_dim（） `。

`sqrt`() → Tensor

    

参见[ `torch.sqrt（） `](torch.html#torch.sqrt "torch.sqrt")

`sqrt_`() → Tensor

    

就地版本的 `SQRT（） `

`squeeze`( _dim=None_ ) → Tensor

    

参见[ `torch.squeeze（） `](torch.html#torch.squeeze "torch.squeeze")

`squeeze_`( _dim=None_ ) → Tensor

    

就地版本 `挤压的（） `

`std`( _dim=None_ , _unbiased=True_ , _keepdim=False_ ) → Tensor

    

参见[ `torch.std（） `](torch.html#torch.std "torch.std")

`stft`( _n_fft_ , _hop_length=None_ , _win_length=None_ , _window=None_ ,
_center=True_ , _pad_mode='reflect'_ , _normalized=False_ , _onesided=True_
)[[source]](_modules/torch/tensor.html#Tensor.stft)

    

参见[ `torch.stft（） `](torch.html#torch.stft "torch.stft")

Warning

此功能在0.4.1版本中更改签名。与先前的签名调用可能会导致错误或返回不正确的结果。

`storage`() → torch.Storage

    

返回底层存储。

`storage_offset`() → int

    

返回`自 `张量在底层存储的偏移在存储元件（未字节）的数目方面。

Example:

    
    
    >>> x = torch.tensor([1, 2, 3, 4, 5])
    >>> x.storage_offset()
    0
    >>> x[3:].storage_offset()
    3
    

`storage_type`() → type

    

返回底层存储的类型。

`stride`( _dim_ ) → tuple or int

    

返回`自 `张量步幅。

步幅是必要跳转到从一个元件到下一个指定维度 `暗淡 `。当没有参数在被传递返回所有步幅的元组，否则，一个整数值返回为在特定维度 `步幅暗淡 `。

Parameters

    

**暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in
Python v3.7\)") _，_ _可选_ ） - 其中需要步幅所希望的尺寸

Example:

    
    
    >>> x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> x.stride()
    (5, 1)
    >>>x.stride(0)
    5
    >>> x.stride(-1)
    1
    

`sub`( _value_ , _other_ ) → Tensor

    

减去从`自 `张量标量或张量。如果两个`值 `和`其他被指定`的`每个元素等 `被缩放通过`值 `之前被使用。

当`其他 `是一个张量，的`形状其他 `必须[ broadcastable
](notes/broadcasting.html#broadcasting-semantics)与底层张量的形状。

`sub_`( _x_ ) → Tensor

    

就地版本 `子的（） `

`sum`( _dim=None_ , _keepdim=False_ , _dtype=None_ ) → Tensor

    

参见[ `torch.sum（） `](torch.html#torch.sum "torch.sum")

`sum_to_size`( _*size_ ) → Tensor

    

总和`这里 `张量为 `大小 `。`大小 `必须broadcastable为`这里 `张量的大小。

Parameters

    

**大小** （ _INT ..._ ） - 定义输出张量的形状的整数序列。

`svd`( _some=True_ , _compute_uv=True) - > (Tensor_, _Tensor_ , _Tensor_ )

    

参见[ `torch.svd（） `](torch.html#torch.svd "torch.svd")

`symeig`( _eigenvectors=False_ , _upper=True) - > (Tensor_, _Tensor_ )

    

参见[ `torch.symeig（） `](torch.html#torch.symeig "torch.symeig")

`t`() → Tensor

    

参见[ `torch.t（） `](torch.html#torch.t "torch.t")

`t_`() → Tensor

    

就地版本的 `T（） `

`to`( _*args_ , _**kwargs_ ) → Tensor

    

执行张量D型细胞和/或设备的转换。 A [ `torch.dtype`
](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")和[ `
torch.device`](tensor_attributes.html#torch.torch.device
"torch.torch.device")从的参数推断出`self.to（*指定参数时， ** kwargs） `。

Note

如果`自 `张量已经有了正确的[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype
"torch.torch.dtype")和[ `torch.device`
](tensor_attributes.html#torch.torch.device "torch.torch.device")，然后`自被返回
`。否则，返回的张量是`自 `与复制所需的[ `torch.dtype`
](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")和[ `
torch.device`](tensor_attributes.html#torch.torch.device
"torch.torch.device")。

下面是调用`至 `方式：

`to`( _dtype_ , _non_blocking=False_ , _copy=False_ ) → Tensor

    

返回与指定`DTYPE`张量

`to`( _device=None_ , _dtype=None_ , _non_blocking=False_ , _copy=False_ ) →
Tensor

    

返回与指定 `装置 `和张量（可选）`DTYPE`。如果`DTYPE`是`无 `它被推断为`self.dtype`。当`
non_blocking`，尝试如果可能的话，以相对于异步转换到主机，例如，转换CPU张量与固定内存到CUDA张量。当`复制
`设置，即使当已经张量相匹配的所需的转化创建一个新的张量。

`to`( _other_ , _non_blocking=False_ , _copy=False_ ) → Tensor

    

返回与相同[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype
"torch.torch.dtype")和[ `torch.device`
](tensor_attributes.html#torch.torch.device "torch.torch.device")作为一个张量张量`其他
`。当`non_blocking`，尝试如果可能的话，以相对于异步转换到主机，例如，转换CPU张量与固定内存到CUDA张量。当`复制
`设置，即使当已经张量相匹配的所需的转化创建一个新的张量。

Example:

    
    
    >>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
    >>> tensor.to(torch.float64)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], dtype=torch.float64)
    
    >>> cuda0 = torch.device('cuda:0')
    >>> tensor.to(cuda0)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], device='cuda:0')
    
    >>> tensor.to(cuda0, dtype=torch.float64)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
    
    >>> other = torch.randn((), dtype=torch.float64, device=cuda0)
    >>> tensor.to(other, non_blocking=True)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
    

`to_mkldnn`() → Tensor

    

返回`torch.mkldnn`布局张的副本。

`take`( _indices_ ) → Tensor

    

参见[ `torch.take（） `](torch.html#torch.take "torch.take")

`tan`() → Tensor

    

参见[ `torch.tan（） `](torch.html#torch.tan "torch.tan")

`tan_`() → Tensor

    

就地版本的 `黄褐色（） `

`tanh`() → Tensor

    

参见[ `torch.tanh（） `](torch.html#torch.tanh "torch.tanh")

`tanh_`() → Tensor

    

就地版本 `的tanh（） `

`tolist`()

    

” tolist（） - & GT ;列表或数

返回张量为（嵌套）名单。标量，则返回一个标准的Python数目，只是（） 像 `项。张量自动移动到CPU首先，如果必要的。`

This operation is not differentiable.

例子：

    
    
    >>> a = torch.randn(2, 2)
    >>> a.tolist()
    [[0.012766935862600803, 0.5415473580360413],
     [-0.08909505605697632, 0.7729271650314331]]
    >>> a[0,0].tolist()
    0.012766935862600803
    

`topk`( _k_ , _dim=None_ , _largest=True_ , _sorted=True) - > (Tensor_,
_LongTensor_ )

    

参见[ `torch.topk（） `](torch.html#torch.topk "torch.topk")

`to_sparse`( _sparseDims_ ) → Tensor

    

返回张量的稀疏副本。 PyTorch支持[ 坐标格式 ](sparse.html#sparse-docs)稀疏张量。

Parameters

    

**sparseDims** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int
"\(in Python v3.7\)") _，_ _可选_ ） - 稀疏维数，以在新的稀疏张量包括

Example:

    
    
    >>> d = torch.tensor([[0, 0, 0], [9, 0, 10], [0, 0, 0]])
    >>> d
    tensor([[ 0,  0,  0],
            [ 9,  0, 10],
            [ 0,  0,  0]])
    >>> d.to_sparse()
    tensor(indices=tensor([[1, 1],
                           [0, 2]]),
           values=tensor([ 9, 10]),
           size=(3, 3), nnz=2, layout=torch.sparse_coo)
    >>> d.to_sparse(1)
    tensor(indices=tensor([[1]]),
           values=tensor([[ 9,  0, 10]]),
           size=(3, 3), nnz=1, layout=torch.sparse_coo)
    

`trace`() → Tensor

    

参见[ `torch.trace（） `](torch.html#torch.trace "torch.trace")

`transpose`( _dim0_ , _dim1_ ) → Tensor

    

参见[ `torch.transpose（） `](torch.html#torch.transpose "torch.transpose")

`transpose_`( _dim0_ , _dim1_ ) → Tensor

    

就地版本的 `转置（） `

`triangular_solve`( _A_ , _upper=True_ , _transpose=False_ ,
_unitriangular=False) - > (Tensor_, _Tensor_ )

    

参见[ `torch.triangular_solve（） `](torch.html#torch.triangular_solve
"torch.triangular_solve")

`tril`( _k=0_ ) → Tensor

    

参见[ `torch.tril（） `](torch.html#torch.tril "torch.tril")

`tril_`( _k=0_ ) → Tensor

    

就地版本的 `TRIL（） `

`triu`( _k=0_ ) → Tensor

    

参见[ `torch.triu（） `](torch.html#torch.triu "torch.triu")

`triu_`( _k=0_ ) → Tensor

    

就地版本 `triu的（） `

`trunc`() → Tensor

    

参见[ `torch.trunc（） `](torch.html#torch.trunc "torch.trunc")

`trunc_`() → Tensor

    

就地版本 `TRUNC的（） `

`type`( _dtype=None_ , _non_blocking=False_ , _**kwargs_ ) → str or Tensor

    

返回类型，如果 DTYPE 不设置，否则铸件此对象为指定的类型。

如果这是正确的类型已经没有执行复制操作，并返回原来的对象。

Parameters

    

  * **DTYPE** （[ _输入_ ](https://docs.python.org/3/library/functions.html#type "\(in Python v3.7\)") _或_ _串_ ） - 所需的类型

  * **non_blocking** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 如果`真 `，并且源是在固定存储器和目的地是在GPU或反之亦然，副本被相对于所述主机异步地执行。另外，参数没有任何影响。

  * **** kwargs** \- 对于相容性，可以含有键`异步 `代替`non_blocking`参数的。的`异步 `ARG被弃用。

`type_as`( _tensor_ ) → Tensor

    

返回此张投给定张的类型。

这是一个无操作，如果张量已经是正确的类型。这等同于`self.type（tensor.type（）） `

Parameters

    

**张量** （ _张量_ ） - 具有所需类型的张量

`unbind`( _dim=0_ ) → seq

    

参见[ `torch.unbind（） `](torch.html#torch.unbind "torch.unbind")

`unfold`( _dimension_ , _size_ , _step_ ) → Tensor

    

返回一个包含大小 `大小 `的所有片的张量从在尺寸`[`自 `张量HTG11]维 `。

两片之间的步骤是通过`步骤给出 `。

如果 sizedim 是维度的大小`维 `为`自 `，尺寸`[大小HTG11 ]维 `在返回的张量将是（sizedim - 大小）/步+ 1 。

大小 `大小的额外维度 `在返回的张量追加。

Parameters

    

  * **维** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 维，其中展开发生

  * **大小** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 其被展开每个切片的大小

  * **步骤** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 每个切片之间的台阶

Example:

    
    
    >>> x = torch.arange(1., 8)
    >>> x
    tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
    >>> x.unfold(0, 2, 1)
    tensor([[ 1.,  2.],
            [ 2.,  3.],
            [ 3.,  4.],
            [ 4.,  5.],
            [ 5.,  6.],
            [ 6.,  7.]])
    >>> x.unfold(0, 2, 2)
    tensor([[ 1.,  2.],
            [ 3.,  4.],
            [ 5.,  6.]])
    

`uniform_`( _from=0_ , _to=1_ ) → Tensor

    

填充`自 `张量与从连续均匀分布采样的数字：

P(x)=1to−fromP(x) = \dfrac{1}{\text{to} - \text{from}} P(x)=to−from1​

`unique`( _sorted=True_ , _return_inverse=False_ , _return_counts=False_ ,
_dim=None_ )[[source]](_modules/torch/tensor.html#Tensor.unique)

    

返回输入张量的独特元素。

参见[ `torch.unique（） `](torch.html#torch.unique "torch.unique")

`unique_consecutive`( _return_inverse=False_ , _return_counts=False_ ,
_dim=None_ )[[source]](_modules/torch/tensor.html#Tensor.unique_consecutive)

    

消除了所有的但等效从元件的每个连续组的第一个元素。

参见[ `torch.unique_consecutive（） `](torch.html#torch.unique_consecutive
"torch.unique_consecutive")

`unsqueeze`( _dim_ ) → Tensor

    

参见[ `torch.unsqueeze（） `](torch.html#torch.unsqueeze "torch.unsqueeze")

`unsqueeze_`( _dim_ ) → Tensor

    

就地版本的 `unsqueeze（） `

`values`() → Tensor

    

如果`自 `是一个稀疏COO张量（即，与`torch.sparse_coo`布局），这将返回包含的值张量的图。否则，这将引发一个错误。

另请参见 `Tensor.indices（） `。

Note

This method can only be called on a coalesced sparse tensor. See
`Tensor.coalesce()`for details.

`var`( _dim=None_ , _unbiased=True_ , _keepdim=False_ ) → Tensor

    

参见[ `torch.var（） `](torch.html#torch.var "torch.var")

`view`( _*shape_ ) → Tensor

    

返回与相同的数据`自 `张量但具有不同的`形状 `的新的张量。

返回的张量共享相同的数据，并且必须有相同数量的元素，但可以具有不同的大小。对于待观察的张量，新视图大小必须与它的原始尺寸和步幅，即兼容，每一个新的视图尺寸必须是一个原始尺寸的子空间，或仅跨跨度原始尺寸
d  ， d  \+  1  ...  ， d  \+  K  d，d + 1， \点，d + K  d  ， d  \+  1  ， ...  ， d
\+  K  满足以下的邻接样条件 ∀ i的 =  0  ， ...  ， K  \-  1  \ forall的I = 0，\点，K-1  ∀ i的 =
0  ， ...  ， ķ  \-  1

stride[i]=stride[i+1]×size[i+1]\text{stride}[i] = \text{stride}[i+1] \times
\text{size}[i+1]stride[i]=stride[i+1]×size[i+1]

否则， `连续（）需要 `被称为可观察的张量之前。参见：[ `重塑（） `](torch.html#torch.reshape
"torch.reshape")，如果形状是兼容它返回一个视图，并复制（等效于调用 `连续的（） `）否则。

Parameters

    

**定型** （ _torch.Size_ _或_ _INT ..._ ） - 所需的大小

Example:

    
    
    >>> x = torch.randn(4, 4)
    >>> x.size()
    torch.Size([4, 4])
    >>> y = x.view(16)
    >>> y.size()
    torch.Size([16])
    >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
    >>> z.size()
    torch.Size([2, 8])
    
    >>> a = torch.randn(1, 2, 3, 4)
    >>> a.size()
    torch.Size([1, 2, 3, 4])
    >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
    >>> b.size()
    torch.Size([1, 3, 2, 4])
    >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
    >>> c.size()
    torch.Size([1, 3, 2, 4])
    >>> torch.equal(b, c)
    False
    

`view_as`( _other_ ) → Tensor

    

查看这个张量为相同的大小为`其他 `。 `self.view_as（其他）​​ `等于`self.view（other.size（）） `。

请参见 `视图（） `约`视图 `的更多信息。

Parameters

    

**other** (`torch.Tensor`) – The result tensor has the same size as `other`.

`where`( _condition_ , _y_ ) → Tensor

    

`self.where（条件， y）的 `等于`torch.where（条件， 自， y）的 `。参见[ `torch.where（） `
](torch.html#torch.where "torch.where")

`zero_`() → Tensor

    

填充`自 `用零张量。

_class_`torch.``BoolTensor`

    

下面的方法是独特的 `torch.BoolTensor`。

`all`()

    

`all`() → bool

    

如果张量的所有元素都是真，假，否则返回True。

Example:

    
    
    >>> a = torch.rand(1, 2).bool()
    >>> a
    tensor([[False, True]], dtype=torch.bool)
    >>> a.all()
    tensor(False, dtype=torch.bool)
    

`all`( _dim_ , _keepdim=False_ , _out=None_ ) → Tensor

    

如果张量的每一行中的所有元素在指定维度`暗淡 `是真，假，否则返回True。

如果`keepdim`是`真 `，输出张量是相同的大小为`输入 `除了在尺寸`暗淡 `其中它是尺寸1的否则，`暗淡 `被挤出（见[ `
torch.squeeze（） `](torch.html#torch.squeeze "torch.squeeze")），导致具有比1种`输入
`更少尺寸的输出张量。

Parameters

    

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的尺寸，以减少

  * **keepdim** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 输出张量是否有`暗淡 `保留或不

  * **OUT** （ _张量_ _，_ _可选_ ） - 输出张量

Example:

    
    
    >>> a = torch.rand(4, 2).bool()
    >>> a
    tensor([[True, True],
            [True, False],
            [True, True],
            [True, True]], dtype=torch.bool)
    >>> a.all(dim=1)
    tensor([ True, False,  True,  True], dtype=torch.bool)
    >>> a.all(dim=0)
    tensor([ True, False], dtype=torch.bool)
    

`any`()

    

`any`() → bool

    

如果在任何张元素是真，假，否则返回True。

Example:

    
    
    >>> a = torch.rand(1, 2).bool()
    >>> a
    tensor([[False, True]], dtype=torch.bool)
    >>> a.any()
    tensor(True, dtype=torch.bool)
    

`any`( _dim_ , _keepdim=False_ , _out=None_ ) → Tensor

    

如果张量的每行中的任何元件在给定的尺寸`暗淡 `是真，假否则返回真。

If `keepdim`is `True`, the output tensor is of the same size as `input`
except in the dimension `dim`where it is of size 1. Otherwise, `dim`is
squeezed (see [`torch.squeeze()`](torch.html#torch.squeeze "torch.squeeze")),
resulting in the output tensor having 1 fewer dimension than `input`.

Parameters

    

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – the dimension to reduce

  * **keepdim** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether the output tensor has `dim`retained or not

  * **out** ( _Tensor_ _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4, 2) < 0
    >>> a
    tensor([[ True,  True],
            [False,  True],
            [ True,  True],
            [False, False]])
    >>> a.any(1)
    tensor([ True,  True,  True, False])
    >>> a.any(0)
    tensor([True, True])
    

[Next ![](_static/images/chevron-right-orange.svg)](tensor_attributes.html
"Tensor Attributes") [![](_static/images/chevron-right-orange.svg)
Previous](torch.html "torch")

* * *

©版权所有2019年，Torch 贡献者。