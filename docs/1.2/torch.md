# torch

Torch 包中包含用于对这些多维张量和数学运算被定义的数据结构。此外，它提供了张量的高效串行化和任意类型，以及其他有用的工具的许多工具。

它具有CUDA对应，使您可以在一个NVIDIA GPU计算能力& GT运行计算张[] = 3.0。

## 张量

`torch.``is_tensor`( _obj_ )[[source]](_modules/torch.html#is_tensor)

    

如果 OBJ 是一个PyTorch张返回True。

Parameters

    

**OBJ** （ _对象_ ） - 对象测试

`torch.``is_storage`( _obj_ )[[source]](_modules/torch.html#is_storage)

    

如果 OBJ 是一个PyTorch存储对象返回真。

Parameters

    

**obj** ( _Object_ ) – Object to test

`torch.``is_floating_point`( _input) - > (bool_)

    

返回true如果输入的`的数据类型 `是一个浮点数据类型，即，torch.float64 的`之一，`torch.float32`和`
torch.float16`。`

Parameters

    

**输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的PyTorch张量，以测试

`torch.``set_default_dtype`( _d_
)[[source]](_modules/torch.html#set_default_dtype)

    

设置默认浮点D型为`d`。这种类型的将被用作默认浮点类型为类型推断在 `torch.tensor（） `。

默认浮点D型细胞是最初`torch.float32`。

Parameters

    

**d** （[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype
"torch.torch.dtype")） - 浮点D型细胞，使默认

例：

    
    
    >>> torch.tensor([1.2, 3]).dtype           # initial default for floating point is torch.float32
    torch.float32
    >>> torch.set_default_dtype(torch.float64)
    >>> torch.tensor([1.2, 3]).dtype           # a new floating point tensor
    torch.float64
    

`torch.``get_default_dtype`() → torch.dtype

    

获得当前默认的浮点[ `torch.dtype`[HTG5。](tensor_attributes.html#torch.torch.dtype
"torch.torch.dtype")

Example:

    
    
    >>> torch.get_default_dtype()  # initial default for floating point is torch.float32
    torch.float32
    >>> torch.set_default_dtype(torch.float64)
    >>> torch.get_default_dtype()  # default is now changed to torch.float64
    torch.float64
    >>> torch.set_default_tensor_type(torch.FloatTensor)  # setting tensor type also affects this
    >>> torch.get_default_dtype()  # changed to torch.float32, the dtype for torch.FloatTensor
    torch.float32
    

`torch.``set_default_tensor_type`( _t_
)[[source]](_modules/torch.html#set_default_tensor_type)

    

设置默认`torch.Tensor`类型到浮点型张量 `T`。这种类型也将被用作默认浮点类型为类型推断在 `torch.tensor（） `
。

默认浮点张量类型最初是`torch.FloatTensor`。

Parameters

    

**T** （[ _输入_ ](https://docs.python.org/3/library/functions.html#type "\(in
Python v3.7\)") _或_ _串_ ） - 浮点张量类型或名称

Example:

    
    
    >>> torch.tensor([1.2, 3]).dtype    # initial default for floating point is torch.float32
    torch.float32
    >>> torch.set_default_tensor_type(torch.DoubleTensor)
    >>> torch.tensor([1.2, 3]).dtype    # a new floating point tensor
    torch.float64
    

`torch.``numel`( _input_ ) → int

    

返回元件在`输入 `张量的总数。

Parameters

    

**输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入张量

Example:

    
    
    >>> a = torch.randn(1, 2, 3, 4, 5)
    >>> torch.numel(a)
    120
    >>> a = torch.zeros(4,4)
    >>> torch.numel(a)
    16
    

`torch.``set_printoptions`( _precision=None_ , _threshold=None_ ,
_edgeitems=None_ , _linewidth=None_ , _profile=None_ , _sci_mode=None_
)[[source]](_modules/torch/_tensor_str.html#set_printoptions)

    

打印设置选项。项目无耻地从NumPy的拍摄

Parameters

    

  * **精度** \- 的用于浮点输出（缺省值= 4）的精度位数。

  * **阈** \- 其中触发总结而不是完整再版（默认= 1000）的数组元素的总数。

  * **edgeitems** \- 中的每个维度（缺省= 3）的开始和结束时在摘要数组项数。

  * **线宽** \- 每行的字符用于插入换行（缺省值= 80）的目的的数量。阈值的矩阵将忽略此参数。

  * **个人资料** \- 为漂亮的印刷理智的默认值。可与上述任何选项覆盖。 （的中任一项默认，短，充满）

  * **sci_mode** \- 启用（True）或禁用（假）科学记数法。如果指定无（默认值），该值是由 _Formatter定义

`torch.``set_flush_denormal`( _mode_ ) → bool

    

禁止非正规浮于CPU编号。

返回`真 `如果你的系统支持非标准冲洗数字和它成功配置刷新非标准模式。`set_flush_denormal（） `
仅支持x86架构，支持SSE3。

Parameters

    

**模式** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in
Python v3.7\)")） - 控制是否启用冲洗反规范模式或不

Example:

    
    
    >>> torch.set_flush_denormal(True)
    True
    >>> torch.tensor([1e-323], dtype=torch.float64)
    tensor([ 0.], dtype=torch.float64)
    >>> torch.set_flush_denormal(False)
    True
    >>> torch.tensor([1e-323], dtype=torch.float64)
    tensor(9.88131e-324 *
           [ 1.0000], dtype=torch.float64)
    

### 创建行动

注意

随机采样生成OPS被下列出随机采样 和包括： `torch.rand（） ``torch.rand_like（） ``torch.randn（）
``torch.randn_like（） ``torch.randint（） ``torch。 randint_like（） ``
torch.randperm（） `您也可以使用 `Torch 。空（） `与 就地随机抽样 的方法来创建[ `torch.Tensor`
](tensors.html#torch.Tensor "torch.Tensor") s的值从更广阔的范围内分布的采样。

`torch.``tensor`( _data_ , _dtype=None_ , _device=None_ ,
_requires_grad=False_ , _pin_memory=False_ ) → Tensor

    

构造具有`数据 `的张量。

警告

`torch.tensor（） `总是副本`数据 `。如果你有一个张量`数据 `，并希望避免拷贝，使用[ `
torch.Tensor.requires_grad_（） `](tensors.html#torch.Tensor.requires_grad_
"torch.Tensor.requires_grad_")或[ `torch.Tensor.detach（） `
](tensors.html#torch.Tensor.detach "torch.Tensor.detach")。如果你有一个与NumPy `
ndarray`，并希望避免拷贝，使用 `torch.as_tensor（） `[HTG35。

Warning

当数据是张量×， `torch.tensor（） `读出从不管它是通过 '数据'，和构造叶变量。因此`torch.tensor（X） `等于`
x.clone（）。分离（） `和`torch.tensor（X， requires_grad =真） `等于`
x.clone（）。分离（）。requires_grad_（真） `。使用`克隆（） `和`分离（）的当量的建议`。

Parameters

    

  * **数据** （ _array_like_ ） - 为对张量初始数据。可进行列表，元组，NumPy的`ndarray`，标量，和其他类型。

  * **DTYPE** （[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")，可选） - 返回的张量的所希望的数据类型。默认值：如果`无 `，从`数据 `推断数据类型。

  * **装置** （[ `torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device")，可选） - 返回的张量的所需的设备。默认值：如果`无 `，使用当前设备的默认张量类型（见 `torch.set_default_tensor_type（） `）。 `装置 `将成为CPU张量类型的CPU和用于CUDA张量类型当前CUDA设备。

  * **requires_grad** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果autograd应返回的记录张操作。默认值：`假 [HTG13。`

  * **pin_memory** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果设置，返回张量将在固定的内存分配。只为CPU张量工作。默认值：`假 [HTG13。`

Example:

    
    
    >>> torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
    tensor([[ 0.1000,  1.2000],
            [ 2.2000,  3.1000],
            [ 4.9000,  5.2000]])
    
    >>> torch.tensor([0, 1])  # Type inference on data
    tensor([ 0,  1])
    
    >>> torch.tensor([[0.11111, 0.222222, 0.3333333]],
                     dtype=torch.float64,
                     device=torch.device('cuda:0'))  # creates a torch.cuda.DoubleTensor
    tensor([[ 0.1111,  0.2222,  0.3333]], dtype=torch.float64, device='cuda:0')
    
    >>> torch.tensor(3.14159)  # Create a scalar (zero-dimensional tensor)
    tensor(3.1416)
    
    >>> torch.tensor([])  # Create an empty tensor (of size (0,))
    tensor([])
    

`torch.``sparse_coo_tensor`( _indices_ , _values_ , _size=None_ , _dtype=None_
, _device=None_ , _requires_grad=False_ ) → Tensor

    

构造在COO（rdinate）格式的稀疏张量与在给定的`指数 `与给定的`值
`的非零元素。稀疏张量可以未聚结的，在这种情况下，有在索引重复坐标，该索引的值是所有重复值项之和：[ torch.sparse
](https://pytorch.org/docs/stable/sparse.html)。

Parameters

    

  * 用于张量初始数据 - **指数** （ _array_like_ ）。可进行列表，元组，NumPy的`ndarray`，标量，和其他类型。将被转换为`torch.LongTensor`内部。该指数是在矩阵中的非零值的坐标，且因此应是二维，其中第一维是张量的维数，第二维是非零值的数目。

  * 用于张量的初始值 - **值** （ _array_like_ ）。可进行列表，元组，NumPy的`ndarray`，标量，和其他类型。

  * **大小** （列表，元组，或`torch.Size`，可选） - 稀疏张量的大小。如果没有提供规模将被推断为最小尺寸大到足以容纳所有非零元素。

  * **DTYPE** （[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")，可选） - 返回的张量的所希望的数据类型。默认值：如果无，从`值 `推断数据类型。

  * **装置** （[ `torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device")，可选） - 返回的张量的所需的设备。默认值：如果没有，则使用当前设备的默认张量类型（见 `torch.set_default_tensor_type（） `）。 `装置 `将成为CPU张量类型的CPU和用于CUDA张量类型当前CUDA设备。

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> i = torch.tensor([[0, 1, 1],
                          [2, 0, 2]])
    >>> v = torch.tensor([3, 4, 5], dtype=torch.float32)
    >>> torch.sparse_coo_tensor(i, v, [2, 4])
    tensor(indices=tensor([[0, 1, 1],
                           [2, 0, 2]]),
           values=tensor([3., 4., 5.]),
           size=(2, 4), nnz=3, layout=torch.sparse_coo)
    
    >>> torch.sparse_coo_tensor(i, v)  # Shape inference
    tensor(indices=tensor([[0, 1, 1],
                           [2, 0, 2]]),
           values=tensor([3., 4., 5.]),
           size=(2, 3), nnz=3, layout=torch.sparse_coo)
    
    >>> torch.sparse_coo_tensor(i, v, [2, 4],
                                dtype=torch.float64,
                                device=torch.device('cuda:0'))
    tensor(indices=tensor([[0, 1, 1],
                           [2, 0, 2]]),
           values=tensor([3., 4., 5.]),
           device='cuda:0', size=(2, 4), nnz=3, dtype=torch.float64,
           layout=torch.sparse_coo)
    
    # Create an empty sparse tensor with the following invariants:
    #   1. sparse_dim + dense_dim = len(SparseTensor.shape)
    #   2. SparseTensor._indices().shape = (sparse_dim, nnz)
    #   3. SparseTensor._values().shape = (nnz, SparseTensor.shape[sparse_dim:])
    #
    # For instance, to create an empty sparse tensor with nnz = 0, dense_dim = 0 and
    # sparse_dim = 1 (hence indices is a 2D tensor of shape = (1, 0))
    >>> S = torch.sparse_coo_tensor(torch.empty([1, 0]), [], [1])
    tensor(indices=tensor([], size=(1, 0)),
           values=tensor([], size=(0,)),
           size=(1,), nnz=0, layout=torch.sparse_coo)
    
    # and to create an empty sparse tensor with nnz = 0, dense_dim = 1 and
    # sparse_dim = 1
    >>> S = torch.sparse_coo_tensor(torch.empty([1, 0]), torch.empty([0, 2]), [1, 2])
    tensor(indices=tensor([], size=(1, 0)),
           values=tensor([], size=(0, 2)),
           size=(1, 2), nnz=0, layout=torch.sparse_coo)
    

`torch.``as_tensor`( _data_ , _dtype=None_ , _device=None_ ) → Tensor

    

将数据转换成 torch.Tensor 。如果数据已经是一个张量用相同的 DTYPE
和装置，没有副本将被执行，否则一个新的张量将是与计算图形返回保留如果数据张量具有`requires_grad =真 `。类似地，如果数据是一个`
ndarray`的相应的 DTYPE 和装置是CPU，没有复制将被执行。

Parameters

    

  * **data** ( _array_like_ ) – Initial data for the tensor. Can be a list, tuple, NumPy `ndarray`, scalar, and other types.

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, infers data type from `data`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

Example:

    
    
    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.as_tensor(a)
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([-1,  2,  3])
    
    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.as_tensor(a, device=torch.device('cuda'))
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([1,  2,  3])
    

`torch.``as_strided`( _input_ , _size_ , _stride_ , _storage_offset=0_ ) →
Tensor

    

与指定的`大小创建现有 torch.Tensor`输入 `的视图 `，`跨步 `和`storage_offset`。

Warning

创建的张量的多于一个的元件可指代单个存储器位置。其结果是，就地操作（特别是那些有量化的）可能会导致不正确的行为。如果你需要写张量，请先克隆它们。

许多PyTorch功能，它会返回一个张量的观点，在内部使用此功能来实现。这些功能，如[ `torch.Tensor.expand（） `
](tensors.html#torch.Tensor.expand "torch.Tensor.expand")，更容易阅读，因此更可取的使用。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **大小** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _或_ _整数_ ） - 输出张量的形状

  * **步幅** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _或_ _整数_ ） - 输出张量的步幅

  * **storage_offset** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 在输出张量的底层存储的偏移

Example:

    
    
    >>> x = torch.randn(3, 3)
    >>> x
    tensor([[ 0.9039,  0.6291,  1.0795],
            [ 0.1586,  2.1939, -0.4900],
            [-0.1909, -0.7503,  1.9355]])
    >>> t = torch.as_strided(x, (2, 2), (1, 2))
    >>> t
    tensor([[0.9039, 1.0795],
            [0.6291, 0.1586]])
    >>> t = torch.as_strided(x, (2, 2), (1, 2), 1)
    tensor([[0.6291, 0.1586],
            [1.0795, 2.1939]])
    

`torch.``from_numpy`( _ndarray_ ) → Tensor

    

创建[ `张量 `](tensors.html#torch.Tensor "torch.Tensor")从[ `numpy.ndarray`
](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray
"\(in NumPy v1.17\)")。

返回的张量和`ndarray`共享相同的存储器。修改对张量将反映在`ndarray`，反之亦然。返回的张量是不可调整大小。

它目前接受`ndarray`与`dtypes numpy.float64`，`numpy.float32``numpy.float16
`，`numpy.int64`，`numpy.int32`，`numpy.int16`，`numpy.int8`，`
numpy.uint8`和`numpy.bool`。

Example:

    
    
    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.from_numpy(a)
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([-1,  2,  3])
    

`torch.``zeros`( _*size_ , _out=None_ , _dtype=None_ , _layout=torch.strided_
, _device=None_ , _requires_grad=False_ ) → Tensor

    

返回填充有标量值 0 的张量，由可变参数`大小 `中定义的形状。

Parameters

    

  * **大小** （ _INT ..._ ） - 定义输出张量的形状的整数序列。可以的参数个数可变或类似的列表或元组的集合。

  * **OUT** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 输出张量

  * **DTYPE** （[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")，可选） - 返回的张量的所希望的数据类型。默认值：如果`无 `，使用全局默认设置（见 `torch.set_default_tensor_type（） `）。

  * **布局** （[ `torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout")，可选） - 返回的张量的所需布局。默认值：torch.strided ``。

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> torch.zeros(2, 3)
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])
    
    >>> torch.zeros(5)
    tensor([ 0.,  0.,  0.,  0.,  0.])
    

`torch.``zeros_like`( _input_ , _dtype=None_ , _layout=None_ , _device=None_ ,
_requires_grad=False_ ) → Tensor

    

返回填充有标量值 0 的张量，以相同的大小为`输入 `。 `torch.zeros_like（输入） `等于`
torch.zeros（input.size（）， D型细胞= input.dtype， 布局= input.layout， 设备=
input.device） `。

Warning

如为0.4，该功能不支持`OUT`关键字。作为替代方案，旧`torch.zeros_like（输入， OUT =输出） `等于`
torch.zeros（input.size （）， OUT =输出） `。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - `输入的大小 `将确定输出张量的大小

  * **DTYPE** （[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")，可选） - 所需的数据返回张量的类型。默认值：如果`无 `，默认为`输入 `的D型。

  * **布局** （[ `torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout")，可选） - 返回的张量的所需布局。默认值：如果`无 `，默认为`输入 `布局。

  * **装置** （[ `torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device")，可选） - 返回的张量的所需的设备。默认值：如果`无 `，默认为`输入 `该设备。

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> input = torch.empty(2, 3)
    >>> torch.zeros_like(input)
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])
    

`torch.``ones`( _*size_ , _out=None_ , _dtype=None_ , _layout=torch.strided_ ,
_device=None_ , _requires_grad=False_ ) → Tensor

    

返回填充有标量值 1 的张量，由可变参数`大小 `中定义的形状。

Parameters

    

  * **size** ( _int..._ ) – a sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see `torch.set_default_tensor_type()`).

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned Tensor. Default: `torch.strided`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> torch.ones(2, 3)
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])
    
    >>> torch.ones(5)
    tensor([ 1.,  1.,  1.,  1.,  1.])
    

`torch.``ones_like`( _input_ , _dtype=None_ , _layout=None_ , _device=None_ ,
_requires_grad=False_ ) → Tensor

    

返回填充有标量值 1 的张量，以相同的大小为`输入 `。 `torch.ones_like（输入） `等于`
torch.ones（input.size（）， D型细胞= input.dtype， 布局= input.layout， 设备=
input.device） `。

Warning

如为0.4，该功能不支持`OUT`关键字。作为替代方案，旧`torch.ones_like（输入， OUT =输出） `等于`
torch.ones（input.size （）， OUT =输出） `。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the size of `input`will determine size of the output tensor

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned Tensor. Default: if `None`, defaults to the dtype of `input`.

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned tensor. Default: if `None`, defaults to the layout of `input`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, defaults to the device of `input`.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> input = torch.empty(2, 3)
    >>> torch.ones_like(input)
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])
    

`torch.``arange`( _start=0_ , _end_ , _step=1_ , _out=None_ , _dtype=None_ ,
_layout=torch.strided_ , _device=None_ , _requires_grad=False_ ) → Tensor

    

返回的一个1-d张量大小 ⌈ 结束 \-  开始 步骤 ⌉ \左\ lceil \压裂{\文本{端} - \文本{开始}} {\文本{步骤}} \右\
rceil  ⌈  步骤 端 \-  开始 ⌉ 与来自间隔值`[开始， 端） `与普通差分采取`步骤 `从开始启动。

;注意，非整数`步骤 `针对`结束 `比较时受到浮点舍入误差，以避免不一致，我们建议增加一个小的ε-为`在这样的情况下结束 `。

outi+1=outi+step\text{out}_{{i+1}} = \text{out}_{i} + \text{step}
outi+1​=outi​+step

Parameters

    

  * **开始** （ _号码_ ） - 为对设定点的初始值。默认值：`0`。

  * **结束** （ _号码_ ） - 为对设定点的结束值

  * **步骤** （ _号码_ ） - 每一对相邻的点之间的间隙。默认值：`1`。

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

  * **DTYPE** （[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")，可选） - 返回的张量的所希望的数据类型。默认值：如果`无 `，使用全局默认设置（见 `torch.set_default_tensor_type（） `）。如果 D型细胞没有给出，推断从其他输入参数的数据类型。如果有任何的开始，结束或停止是浮点时， DTYPE 被推断为默认D型，请参阅 `get_default_dtype（） `。否则， DTYPE 被推断为 torch.int64 。

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned Tensor. Default: `torch.strided`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> torch.arange(5)
    tensor([ 0,  1,  2,  3,  4])
    >>> torch.arange(1, 4)
    tensor([ 1,  2,  3])
    >>> torch.arange(1, 2.5, 0.5)
    tensor([ 1.0000,  1.5000,  2.0000])
    

`torch.``range`( _start=0_ , _end_ , _step=1_ , _out=None_ , _dtype=None_ ,
_layout=torch.strided_ , _device=None_ , _requires_grad=False_ ) → Tensor

    

返回的一个1-d张量大小 ⌊ 结束 \-  开始 步骤 ⌋ \+  1  \左\ lfloor \压裂{\文本{端} - \文本{开始}}
{\文本{步骤}} \右\ rfloor + 1  ⌊ 步骤 结束 \-  开始 [HTG9 5]  ⌋ \+  1  从`值开始 `至`结束 `与步骤`
步骤 `。步骤是在张量的两个值之间的差距。

outi+1=outi+step.\text{out}_{i+1} = \text{out}_i + \text{step}.
outi+1​=outi​+step.

Warning

此功能有利于弃用`torch.arange（） `。

Parameters

    

  * **开始** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 为对设定点的初始值。默认值：`0`。

  * 对于该组点的结束值 - **端** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")）

  * **步骤** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 每一对相邻的点之间的间隙。默认值：`1`。

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see `torch.set_default_tensor_type()`). If dtype is not given, infer the data type from the other input arguments. If any of start, end, or stop are floating-point, the dtype is inferred to be the default dtype, see `get_default_dtype()`. Otherwise, the dtype is inferred to be torch.int64.

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned Tensor. Default: `torch.strided`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> torch.range(1, 4)
    tensor([ 1.,  2.,  3.,  4.])
    >>> torch.range(1, 4, 0.5)
    tensor([ 1.0000,  1.5000,  2.0000,  2.5000,  3.0000,  3.5000,  4.0000])
    

`torch.``linspace`( _start_ , _end_ , _steps=100_ , _out=None_ , _dtype=None_
, _layout=torch.strided_ , _device=None_ , _requires_grad=False_ ) → Tensor

    

返回步骤`一维张量 `之间`等距点开始 `和`结束 `。

输出张量是大小`步骤 `1-d。

Parameters

    

  * 对于该组点的初始值 - **开始** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")）

  * **end** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – the ending value for the set of points

  * **步骤** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 点数为`开始 `和`端之间采样 `。默认值：`100`。

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see `torch.set_default_tensor_type()`).

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned Tensor. Default: `torch.strided`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> torch.linspace(3, 10, steps=5)
    tensor([  3.0000,   4.7500,   6.5000,   8.2500,  10.0000])
    >>> torch.linspace(-10, 10, steps=5)
    tensor([-10.,  -5.,   0.,   5.,  10.])
    >>> torch.linspace(start=-10, end=10, steps=5)
    tensor([-10.,  -5.,   0.,   5.,  10.])
    >>> torch.linspace(start=-10, end=10, steps=1)
    tensor([-10.])
    

`torch.``logspace`( _start_ , _end_ , _steps=100_ , _base=10.0_ , _out=None_ ,
_dtype=None_ , _layout=torch.strided_ , _device=None_ , _requires_grad=False_
) → Tensor

    

返回与碱`碱对数间隔的`步骤 `点的一维张量 `之间 基 开始 {\文本{碱}} ^ {\文本{开始}}  基 开始 和 基 结束 {\文本{碱}} ^
{\文本{端}}  基 结束 。

The output tensor is 1-D of size `steps`.

Parameters

    

  * **start** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – the starting value for the set of points

  * **end** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – the ending value for the set of points

  * **steps** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – number of points to sample between `start`and `end`. Default: `100`.

  * **基** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 的对数函数的基础。默认值：`10.0`。

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see `torch.set_default_tensor_type()`).

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned Tensor. Default: `torch.strided`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> torch.logspace(start=-10, end=10, steps=5)
    tensor([ 1.0000e-10,  1.0000e-05,  1.0000e+00,  1.0000e+05,  1.0000e+10])
    >>> torch.logspace(start=0.1, end=1.0, steps=5)
    tensor([  1.2589,   2.1135,   3.5481,   5.9566,  10.0000])
    >>> torch.logspace(start=0.1, end=1.0, steps=1)
    tensor([1.2589])
    >>> torch.logspace(start=2, end=2, steps=1, base=2)
    tensor([4.0])
    

`torch.``eye`( _n_ , _m=None_ , _out=None_ , _dtype=None_ ,
_layout=torch.strided_ , _device=None_ , _requires_grad=False_ ) → Tensor

    

返回对角线上，一和零其他地方2 d张量。

Parameters

    

  * **n的** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的行数

  * **M** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 列的默认数目为`N`

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see `torch.set_default_tensor_type()`).

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned Tensor. Default: `torch.strided`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Returns

    

A 2-d张量的对角和其他地方的零那些

Return type

    

[张量](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> torch.eye(3)
    tensor([[ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.]])
    

`torch.``empty`( _*size_ , _out=None_ , _dtype=None_ , _layout=torch.strided_
, _device=None_ , _requires_grad=False_ , _pin_memory=False_ ) → Tensor

    

返回填充未初始化的数据的张量。张量的形状由可变参数`大小 `中定义。

Parameters

    

  * **size** ( _int..._ ) – a sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see `torch.set_default_tensor_type()`).

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned Tensor. Default: `torch.strided`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

  * **pin_memory** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If set, returned tensor would be allocated in the pinned memory. Works only for CPU tensors. Default: `False`.

Example:

    
    
    >>> torch.empty(2, 3)
    tensor(1.00000e-08 *
           [[ 6.3984,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000]])
    

`torch.``empty_like`( _input_ , _dtype=None_ , _layout=None_ , _device=None_ ,
_requires_grad=False_ ) → Tensor

    

返回与相同尺寸`输入 `一个未初始化的张量。 `torch.empty_like（输入） `等于`torch.empty（input.size（），
D型细胞= input.dtype， 布局= input.layout， 设备= input.device） `。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the size of `input`will determine size of the output tensor

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned Tensor. Default: if `None`, defaults to the dtype of `input`.

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned tensor. Default: if `None`, defaults to the layout of `input`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, defaults to the device of `input`.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> torch.empty((2,3), dtype=torch.int64)
    tensor([[ 9.4064e+13,  2.8000e+01,  9.3493e+13],
            [ 7.5751e+18,  7.1428e+18,  7.5955e+18]])
    

`torch.``empty_strided`( _size_ , _stride_ , _dtype=None_ , _layout=None_ ,
_device=None_ , _requires_grad=False_ , _pin_memory=False_ ) → Tensor

    

返回填充未初始化的数据的张量。的形状和张量的进展是由可变参数`大小 `和`步幅 `分别定义。 `torch.empty_strided（大小， 步幅）
`等价于.as_strided `torch.empty（大小）（大小， 步幅） `。

Warning

所创建的张量的多于一个的元件可指代单个存储器位置。其结果是，就地操作（特别是那些有量化的）可能会导致不正确的行为。如果你需要写张量，请先克隆它们。

Parameters

    

  * **大小** （ _蟒的元组：整数_ ） - 输出张量的形状

  * **步幅** （ _蟒的元组：整数_ ） - 输出张量的步幅

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see `torch.set_default_tensor_type()`).

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned Tensor. Default: `torch.strided`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

  * **pin_memory** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If set, returned tensor would be allocated in the pinned memory. Works only for CPU tensors. Default: `False`.

Example:

    
    
    >>> a = torch.empty_strided((2, 3), (1, 2))
    >>> a
    tensor([[8.9683e-44, 4.4842e-44, 5.1239e+07],
            [0.0000e+00, 0.0000e+00, 3.0705e-41]])
    >>> a.stride()
    (1, 2)
    >>> a.size()
    torch.Size([2, 3])
    

`torch.``full`( _size_ , _fill_value_ , _out=None_ , _dtype=None_ ,
_layout=torch.strided_ , _device=None_ , _requires_grad=False_ ) → Tensor

    

返回充满`fill_value`大小`大小 `的张量。

Parameters

    

  * **大小** （ _INT ..._ ） - 列表，元组，或`torch.Size`定义输出张量的形状的整数。

  * **fill_value** \- 数以填充与输出张量。

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see `torch.set_default_tensor_type()`).

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned Tensor. Default: `torch.strided`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> torch.full((2, 3), 3.141592)
    tensor([[ 3.1416,  3.1416,  3.1416],
            [ 3.1416,  3.1416,  3.1416]])
    

`torch.``full_like`( _input_ , _fill_value_ , _out=None_ , _dtype=None_ ,
_layout=torch.strided_ , _device=None_ , _requires_grad=False_ ) → Tensor

    

返回与`输入填充有`fill_value`如 `相同尺寸的张量。 `torch.full_like（输入， fill_value） `等于`
torch.full（input.size（）， fill_value， D型细胞= input.dtype， 布局= input.layout， 设备=
input.device） `。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the size of `input`will determine size of the output tensor

  * **fill_value** – the number to fill the output tensor with.

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned Tensor. Default: if `None`, defaults to the dtype of `input`.

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned tensor. Default: if `None`, defaults to the layout of `input`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, defaults to the device of `input`.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

### 索引，切片，加入，变异行动

`torch.``cat`( _tensors_ , _dim=0_ , _out=None_ ) → Tensor

    

串接SEQ 在给定的尺寸张量`给定的序列。所有张量必须具有相同的形状（除了在串接的尺寸）或为空。`

`torch.cat（） `可以被看作是对 `torch.split）的逆操作（ `和 `torch.chunk（） `。

`torch.cat（） `可以通过最好的实施例的理解。

Parameters

    

  * **张量** （ _张量_ 的序列） - 相同类型的张量的任何蟒序列。提供必须具有相同的形状，除了在猫尺寸非空张量。

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 在其上张量被级联的尺寸

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 0)
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 1)
    tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
             -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
             -0.5790,  0.1497]])
    

`torch.``chunk`( _input_ , _chunks_ , _dim=0_ ) → List of Tensors

    

拆分一个张成大块的具体数量。

最后一块会比较小，如果沿给定尺寸的大小张`暗淡 `是不是`块 `整除。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 张量来分割

  * **块** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 组块的数目返回

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 维沿着分裂张量

`torch.``gather`( _input_ , _dim_ , _index_ , _out=None_ , _sparse_grad=False_
) → Tensor

    

集值一起由暗淡中指定的轴。

对于3-d张量由指定的输出：

    
    
    out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
    

如果`输入 `是具有n维张量大小 （ × 0  ， × 1  。[HTG27 。]  ， × i的 \-  1  ， × i的 ， × i的 \+  1
， 。 。 ， × n的 \-  1  ） （X_0，X_1 ...，X_ {I-1}，X_I，X_ {I + 1}，... 。，X_ {N-1}） （ ×
0  ， × 1  。  。  。  ， × i的 \-  1  ， × i的 ， × i的 \+  1  ​​  ， 。  。  。  ， × n的 \-
1  ） 和`暗淡 =  i的 `，然后`索引 `必须为 n的 n的 n的 维张量与尺寸 （ × [HTG37 8]  0  ， × 1  ， 。  。
。  ， × i的 \-  1  ， Y  ， × i的 \+  1  ， 。  。  。  ， × n的 \-  1  ） （X_0，X_1，...，X_
{I-1}，Y，X_ {I + 1}，...，X_ {N-1}） （ × 0  ， × 1  ， 。  。  。  ， × i的 \-  1  ， Y  ，
× i的 \+  1  ， 。  。  。  ， × n的 \-  1  ） 其中 Y  ≥ 1  Y \ GEQ 1  Y  ≥ 1  和`OUT
`将具有相同的大小为`索引 [ HTG719。`

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 源张量

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的轴，沿着该索引

  * **索引** （ _LongTensor_ ） - 元素的索引来收集

  * **OUT** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 目的地张量

  * **sparse_grad** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，梯度WRT `输入 `将是一个稀疏张量。

Example:

    
    
    >>> t = torch.tensor([[1,2],[3,4]])
    >>> torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))
    tensor([[ 1,  1],
            [ 4,  3]])
    

`torch.``index_select`( _input_ , _dim_ , _index_ , _out=None_ ) → Tensor

    

使用中的条目`索引返回一个新的张量，其索引`输入 `沿着维度张量`暗淡 ``其是 LongTensor 。

返回的张量具有相同的维数与原始张量（`输入 `）的。的`暗淡 `次尺寸有大小为`的长度相同的索引 `;其它尺寸具有相同的大小与在原始张量。

Note

返回的张量确实 **不是** 使用相同的存储与原张量。如果`出 `具有与预期不同的形状，我们默默地将其更改为正确的形状，必要时重新分配基础存储空间。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的尺寸，其中我们索引

  * **索引** （ _LongTensor_ ） - 包含索引到索引1-d张量

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> x = torch.randn(3, 4)
    >>> x
    tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
            [-0.4664,  0.2647, -0.1228, -1.1068],
            [-1.1734, -0.6571,  0.7230, -0.6004]])
    >>> indices = torch.tensor([0, 2])
    >>> torch.index_select(x, 0, indices)
    tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
            [-1.1734, -0.6571,  0.7230, -0.6004]])
    >>> torch.index_select(x, 1, indices)
    tensor([[ 0.1427, -0.5414],
            [-0.4664, -0.1228],
            [-1.1734,  0.7230]])
    

`torch.``masked_select`( _input_ , _mask_ , _out=None_ ) → Tensor

    

返回一个新的1-d张量的`输入 `根据布尔掩模张量`掩模哪些索引 `其是 BoolTensor [ HTG9。

在`面具 `张量的形状和`输入 `张量不需要匹配，但是他们必须[ broadcastable
](notes/broadcasting.html#broadcasting-semantics)。

Note

返回的张量并 **不** 使用相同的存储作为原始张量

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入数据

  * **掩模** （[ _BoolTensor_ ](tensors.html#torch.BoolTensor "torch.BoolTensor")） - 包含布尔掩码索引与张力

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> x = torch.randn(3, 4)
    >>> x
    tensor([[ 0.3552, -2.3825, -0.8297,  0.3477],
            [-1.2035,  1.2252,  0.5002,  0.6248],
            [ 0.1307, -2.0608,  0.1244,  2.0139]])
    >>> mask = x.ge(0.5)
    >>> mask
    tensor([[False, False, False, False],
            [False, True, True, True],
            [False, False, False, True]])
    >>> torch.masked_select(x, mask)
    tensor([ 1.2252,  0.5002,  0.6248,  2.0139])
    

`torch.``narrow`( _input_ , _dim_ , _start_ , _length_ ) → Tensor

    

返回一个新的张量是`输入 `张量缩小版本。尺寸`暗淡 `是从`输入开始 `至`开始 +  长度 `。返回的张量和`输入 `张量共享同一基础存储。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 张量来缩小

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 沿其尺寸缩小

  * **开始** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的起始尺寸

  * **长度** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的距离到结束尺寸

Example:

    
    
    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> torch.narrow(x, 0, 0, 2)
    tensor([[ 1,  2,  3],
            [ 4,  5,  6]])
    >>> torch.narrow(x, 1, 1, 2)
    tensor([[ 2,  3],
            [ 5,  6],
            [ 8,  9]])
    

`torch.``nonzero`( _input_ , _*_ , _out=None_ , _as_tuple=False_ ) →
LongTensor or tuple of LongTensors

    

[HTG0当 `as_tuple`**是假或不特定：**

返回包含输入的`所有非零元素 `的索引的张量。结果中的每行包含一个非零元素的索引在`输入 `。其结果是字典顺序排序，最后指数变化最快的（C风格）。

如果`输入 `具有 n的的尺寸，然后将所得的指数张量`OUT`是大小为 （ Z  × n的 ） （Z \ n次） （ Z  × n的 ） ，其中 Z
Z  Z  是在`输入的非零元素的总数 `张量。

[HTG0当 `as_tuple`**是否成立：**

返回的1-d张量在`输入每个维度一个元组，一个 `，各含有的`所有非零元素的索引（该维度）输入 `。

如果`输入 `具有 n的的尺寸，然后将所得的元组包含 n的大小 Z ，其中[HTG10的张量] Z 是在`输入 `张量的非零元素的总数。

作为一种特殊的情况下，当`输入 `具有零种尺寸和非零标量值，它被视为一个元素的一维张量。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **OUT** （ _LongTensor_ _，_ _可选_ ） - 包含索引的输出张量

Returns

    

如果`as_tuple`是假的，含有索引输出张量。如果`as_tuple`为真，一个1-d张量针对每个维度，沿包含该维度的每个非零元素的索引。

Return type

    

LongTensor或LongTensor的元组

Example:

    
    
    >>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))
    tensor([[ 0],
            [ 1],
            [ 2],
            [ 4]])
    >>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
                                    [0.0, 0.4, 0.0, 0.0],
                                    [0.0, 0.0, 1.2, 0.0],
                                    [0.0, 0.0, 0.0,-0.4]]))
    tensor([[ 0,  0],
            [ 1,  1],
            [ 2,  2],
            [ 3,  3]])
    >>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]), as_tuple=True)
    (tensor([0, 1, 2, 4]),)
    >>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
                                    [0.0, 0.4, 0.0, 0.0],
                                    [0.0, 0.0, 1.2, 0.0],
                                    [0.0, 0.0, 0.0,-0.4]]), as_tuple=True)
    (tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]))
    >>> torch.nonzero(torch.tensor(5), as_tuple=True)
    (tensor([0]),)
    

`torch.``reshape`( _input_ , _shape_ ) → Tensor

    

返回与元件为`输入 `相同的数据和数字的张量，但随着特定的形状。如果可能的话，返回的张量将是`输入
`的图。否则，这将是一个副本。连续的投入和兼容的进步投入而不用拷贝进行塑形，但你不应该依赖于复制与收视行为。

参见[ `torch.Tensor.view（） `](tensors.html#torch.Tensor.view
"torch.Tensor.view")上时，它可以返回的图。

单个维度可以是-1，在这种情况下，它从剩余的尺寸和元件的在`输入 `的数量推断。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 要重塑张量

  * **形状** （ _蟒的元组：整数_ ） - 新的形状

Example:

    
    
    >>> a = torch.arange(4.)
    >>> torch.reshape(a, (2, 2))
    tensor([[ 0.,  1.],
            [ 2.,  3.]])
    >>> b = torch.tensor([[0, 1], [2, 3]])
    >>> torch.reshape(b, (-1,))
    tensor([ 0,  1,  2,  3])
    

`torch.``split`( _tensor_ , _split_size_or_sections_ , _dim=0_
)[[source]](_modules/torch/functional.html#split)

    

拆分张成块。

如果`split_size_or_sections`是整数类型，则 `张量 `将被分成相等大小的块（如果可能）
。最后一块会比较小，如果沿给定尺寸的大小张`暗淡 `是不是`split_size`整除。

如果`split_size_or_sections`是一个列表，然后 `张量 `将被分割为`LEN（根据`
split_size_or_sections`split_size_or_sections） `与`大小的块暗淡 `。

Parameters

    

  * **张量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 张量来分割。

  * **split_size_or_sections** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _）或_ _（_ [ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)") _（_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _）_ ） - 对于每个大块单个块或列表尺寸的大小

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 维沿着分裂张量。

`torch.``squeeze`( _input_ , _dim=None_ , _out=None_ ) → Tensor

    

返回与除去 大小的 1输入的`所有尺寸的张量。`

例如，如果输入是形状的： （ A  × 1  × B  × C  × 1  × d  ） （A \倍1 \倍乙\ C时代\倍1 \倍d） （ A  × 1
× B  × C  × 1  × d  ） ，则 OUT 张量将是形状的： （ A  × B  × C  × d  ） （A \倍乙\ C时代\倍d） （
A  × B  × C  × d  ） 。

当`暗淡 `给出，挤压操作仅在给定的尺寸完成。如果输入是形状的： （ A  × 1  × B  ） （A \倍1 \倍B） （ A  × 1  × B
） ，`挤压（输入， 0） `离开张量不变，但是`挤压（输入， 1） `会挤压张量的形状 （ A  × B  [H TG96]）  （A \倍B） （
A  × B  ） 。

Note

返回的张股与输入张量的存储，所以改变一个的内容会改变其他的内容。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 如果给定的，该输入将被只在这个尺寸挤

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> x = torch.zeros(2, 1, 2, 1, 2)
    >>> x.size()
    torch.Size([2, 1, 2, 1, 2])
    >>> y = torch.squeeze(x)
    >>> y.size()
    torch.Size([2, 2, 2])
    >>> y = torch.squeeze(x, 0)
    >>> y.size()
    torch.Size([2, 1, 2, 1, 2])
    >>> y = torch.squeeze(x, 1)
    >>> y.size()
    torch.Size([2, 2, 1, 2])
    

`torch.``stack`( _tensors_ , _dim=0_ , _out=None_ ) → Tensor

    

串接沿着一个新的层面张量的序列。

所有的张量需要是相同大小的。

Parameters

    

  * **张量** （ _张量的序列_ ） - 张量的序列来连接

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 要插入的尺寸。必须是0和级联张量的维数（含）之间

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

`torch.``t`( _input_ ) → Tensor

    

预计`输入 `为& LT ; = 2-d张量和调换尺寸0和1。

0-d和1-d张量返回，因为它是和2- d张量可以被看作是一个短手功能为`转置（输入， 0， 1） `。

Parameters

    

**input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input
tensor

Example:

    
    
    >>> x = torch.randn(())
    >>> x
    tensor(0.1995)
    >>> torch.t(x)
    tensor(0.1995)
    >>> x = torch.randn(3)
    >>> x
    tensor([ 2.4320, -0.4608,  0.7702])
    >>> torch.t(x)
    tensor([.2.4320,.-0.4608,..0.7702])
    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 0.4875,  0.9158, -0.5872],
            [ 0.3938, -0.6929,  0.6932]])
    >>> torch.t(x)
    tensor([[ 0.4875,  0.3938],
            [ 0.9158, -0.6929],
            [-0.5872,  0.6932]])
    

`torch.``take`( _input_ , _index_ ) → Tensor

    

返回与输入 的`在给定的索引的元素的新的张量。如同其被看作是一个1-d张量的输入张量得到治疗。结果取相同的形状指数。`

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **索引** （ _LongTensor_ ） - 索引为张量

Example:

    
    
    >>> src = torch.tensor([[4, 3, 5],
                            [6, 7, 8]])
    >>> torch.take(src, torch.tensor([0, 2, 5]))
    tensor([ 4,  5,  8])
    

`torch.``transpose`( _input_ , _dim0_ , _dim1_ ) → Tensor

    

返回一个张量是`输入 `转置版本。给定尺寸的`DIM0`和`DIM1`被交换。

将得到的`OUT`张量股它底层存储与`输入 `张量，所以改变一个的内容会改变其他的内容。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **DIM0** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 要调换第一维

  * **DIM1** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 要调换的第二维

Example:

    
    
    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 1.0028, -0.9893,  0.5809],
            [-0.1669,  0.7299,  0.4942]])
    >>> torch.transpose(x, 0, 1)
    tensor([[ 1.0028, -0.1669],
            [-0.9893,  0.7299],
            [ 0.5809,  0.4942]])
    

`torch.``unbind`( _input_ , _dim=0_ ) → seq

    

删除一个张量尺寸。

返回所有切片的元组沿着一个给定的尺寸，已经离不开它。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 张量解除绑定

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 尺寸，以除去

Example:

    
    
    >>> torch.unbind(torch.tensor([[1, 2, 3],
    >>>                            [4, 5, 6],
    >>>                            [7, 8, 9]]))
    (tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))
    

`torch.``unsqueeze`( _input_ , _dim_ , _out=None_ ) → Tensor

    

返回与在指定的位置插入一个大小的尺寸新的张量。

返回的张股与此张量相同的基础数据。

1， [ - A ``[-input.dim（范围） 内暗淡 `值HTG11] input.dim（） +  1） `可被使用。负`暗淡 `
施加将对应于 `unsqueeze（）在`暗淡 `= `暗淡 \+  input.dim（） \+  1`。`

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 索引处插入单维

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> x = torch.tensor([1, 2, 3, 4])
    >>> torch.unsqueeze(x, 0)
    tensor([[ 1,  2,  3,  4]])
    >>> torch.unsqueeze(x, 1)
    tensor([[ 1],
            [ 2],
            [ 3],
            [ 4]])
    

`torch.``where`()

    

`torch.``where`( _condition_ , _input_ , _other_ ) → Tensor

    

返回元件的从任一`输入 `或`其他 `，取决于`条件 [HTG11选择的张量]。`

该操作被定义为：

outi={inputiif conditioniotheriotherwise\text{out}_i = \begin{cases}
\text{input}_i & \text{if } \text{condition}_i \\\ \text{other}_i &
\text{otherwise} \\\ \end{cases} outi​={inputi​otheri​​if
conditioni​otherwise​

Note

张量`条件 `，`输入 `，`其他 `必须[ broadcastable
](notes/broadcasting.html#broadcasting-semantics)。

Parameters

    

  * **条件** （[ _BoolTensor_ ](tensors.html#torch.BoolTensor "torch.BoolTensor")） - 当真（非零），产率的x，否则成品率Y

  * **×** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 在索引选择的值，其中`条件 `是`真 `

  * **Y** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 在索引选择的值，其中`条件 `是`假 `

Returns

    

形状的张量等于`条件 `所广播的形状，`输入 `，`其他 `

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> x = torch.randn(3, 2)
    >>> y = torch.ones(3, 2)
    >>> x
    tensor([[-0.4620,  0.3139],
            [ 0.3898, -0.7197],
            [ 0.0478, -0.1657]])
    >>> torch.where(x > 0, x, y)
    tensor([[ 1.0000,  0.3139],
            [ 0.3898,  1.0000],
            [ 0.0478,  1.0000]])
    

`torch.``where`( _condition_ ) → tuple of LongTensor

    

`torch.where（条件） `是相同的`torch.nonzero（条件， as_tuple =真） `。

Note

另请参见 `torch.nonzero（） `。

## 发电机

_class_`torch._C.``Generator`( _device='cpu'_ ) → Generator

    

创建并返回其管理产生的伪随机数的算法的状态的生成器对象。用作许多 就地随机抽样 功能的关键字参数。

Parameters

    

**装置** （`torch.device`，可选） - 用于发电机所需的设备。

Returns

    

一个torch.Generator对象。

Return type

    

发生器

Example:

    
    
    >>> g_cpu = torch.Generator()
    >>> g_cuda = torch.Generator(device='cuda')
    

`device`

    

Generator.device - & GT ;装置

获取发电机的电流设备。

Example:

    
    
    >>> g_cpu = torch.Generator()
    >>> g_cpu.device
    device(type='cpu')
    

`get_state`() → Tensor

    

返回发电机状态作为`torch.ByteTensor`。

Returns

    

A `torch.ByteTensor ，其包含所有必要的比特到发电机还原到在特定时间点`。

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> g_cpu = torch.Generator()
    >>> g_cpu.get_state()
    

`initial_seed`() → int

    

返回用于产生随机数的初始种子。

Example:

    
    
    >>> g_cpu = torch.Generator()
    >>> g_cpu.initial_seed()
    2147483647
    

`manual_seed`( _seed_ ) → Generator

    

设置生成随机数种子。返回 torch.Generator 对象。它建议设置一个大的种子，即一个数字，具有0和1位的良好平衡。避免在种子具有许多0比特。

Parameters

    

**种子** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in
Python v3.7\)")） - 所需的种子。

Returns

    

An torch.Generator object.

Return type

    

Generator

Example:

    
    
    >>> g_cpu = torch.Generator()
    >>> g_cpu.manual_seed(2147483647)
    

`seed`() → int

    

从获取的std :: random_device或当前时间的非确定性的随机数，并使用该种子的发电机。

Example:

    
    
    >>> g_cpu = torch.Generator()
    >>> g_cpu.seed()
    1516516984916
    

`set_state`( _new_state_ ) → void

    

设置发电机的状态。

Parameters

    

**NEW_STATE** （ _torch.ByteTensor_ ） - 所需的状态。

Example:

    
    
    >>> g_cpu = torch.Generator()
    >>> g_cpu_other = torch.Generator()
    >>> g_cpu.set_state(g_cpu_other.get_state())
    

## 随机取样

`torch.``seed`()[[source]](_modules/torch/random.html#seed)

    

设置用于产生随机数，以非确定性的随机数种子。返回用于播种RNG一个64位的数。

`torch.``manual_seed`( _seed_
)[[source]](_modules/torch/random.html#manual_seed)

    

设置生成随机数种子。返回 torch.Generator 对象。

Parameters

    

**seed** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in
Python v3.7\)")) – The desired seed.

`torch.``initial_seed`()[[source]](_modules/torch/random.html#initial_seed)

    

返回初始种子用于产生随机数作为一个Python 长。

`torch.``get_rng_state`()[[source]](_modules/torch/random.html#get_rng_state)

    

返回随机数发生器状态作为 torch.ByteTensor 。

`torch.``set_rng_state`( _new_state_
)[[source]](_modules/torch/random.html#set_rng_state)

    

设置随机数生成器的状态。

Parameters

    

**NEW_STATE** （ _torch.ByteTensor_ ） - 期望状态

`torch.``default_generator`_Returns the default CPU torch.Generator_

    

`torch.``bernoulli`( _input_ , _*_ , _generator=None_ , _out=None_ ) → Tensor

    

从伯努利分布绘制二进制随机数（0或1）。

的`输入 `张量应该是包含概率被用于绘制二进制随机数的张量。因此，在`输入的所有值 `必须在范围： 0  ≤ 输入 i的 ≤ 1  0
\当量\文本{输入} _i \当量1  0  ≤  输入 i的 ≤ 1  。

的 i的 T  H  \文本{I} ^ {第}  I  T  H  输出张量的元件将以此为值 1  1  1  根据 i的 T  H  \文本{I} ^
{}第 i的 T  H  的概率值在`输入给定的 `。

outi∼Bernoulli(p=inputi)\text{out}_{i} \sim \mathrm{Bernoulli}(p =
\text{input}_{i}) outi​∼Bernoulli(p=inputi​)

返回的`OUT`张量仅具有值0或1，是相同的形状的作为`输入 `。

`OUT`可以具有积分`DTYPE`，但`输入 `必须浮点`DTYPE`。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 概率值的伯努利分布的输入张量

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
    >>> a
    tensor([[ 0.1737,  0.0950,  0.3609],
            [ 0.7148,  0.0289,  0.2676],
            [ 0.9456,  0.8937,  0.7202]])
    >>> torch.bernoulli(a)
    tensor([[ 1.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 1.,  1.,  1.]])
    
    >>> a = torch.ones(3, 3) # probability of drawing "1" is 1
    >>> torch.bernoulli(a)
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])
    >>> a = torch.zeros(3, 3) # probability of drawing "1" is 0
    >>> torch.bernoulli(a)
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])
    

`torch.``multinomial`( _input_ , _num_samples_ , _replacement=False_ ,
_out=None_ ) → LongTensor

    

返回其中每行都包含`张量num_samples`指数从位于张量输入 的`的相应行中的多项式概率分布进行采样。`

Note

的行`输入 `不需要总和为1（在这种情况下，我们使用的值作为权重），但必须是非负的，有限的，并且有一个非零和。

指数从左到右，根据当每个取样（第一样品放置在第一列）来排序。

如果`输入 `是矢量，`OUT`是大小num_samples 的`的载体。`

如果`输入 `是与的矩阵M 行，`OUT`是形状 [HTG11的矩阵]  （ M  × num_samples  ） （M \倍\文本{NUM \
_samples}） （ M  × num_samples  ） 。

如果替换为`真 `，样品绘制更换。

如果不是，他们绘制无需更换，这意味着当指数样本绘制为行，不能再为该行画出。

Note

当不需更换绘制，`num_samples`必须大于（在`输入 `非零元素的数目或最小数目的非下输入 如果它是一个矩阵）的`每行中非零元素。`

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 包含概率输入张量

  * **num_samples** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 样本的数目来绘制

  * **替换** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 是否与更换或不画

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> weights = torch.tensor([0, 10, 3, 0], dtype=torch.float) # create a tensor of weights
    >>> torch.multinomial(weights, 2)
    tensor([1, 2])
    >>> torch.multinomial(weights, 4) # ERROR!
    RuntimeError: invalid argument 2: invalid multinomial distribution (with replacement=False,
    not enough non-negative category to sample) at ../aten/src/TH/generic/THTensorRandom.cpp:320
    >>> torch.multinomial(weights, 4, replacement=True)
    tensor([ 2,  1,  1,  1])
    

`torch.``normal`()

    

`torch.``normal`( _mean_ , _std_ , _out=None_ ) → Tensor

    

返回从独立的正态分布，其平均值和标准偏差给出绘制随机数的张量。

的 `意味着 `是与每个输出元件的正常分布的平均值的张量

的 `STD`是与每个输出元件的正态分布的标准偏差的张量

的形状 `[平均HTG3] `和 `性病 `不需要匹配，但是在各张量元素的总数量需要是相同的。

Note

当形状不匹配，的 `形状意味着 `被用作形状为返回的输出张量

Parameters

    

  * **意味着** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 每个元素的装置的张量

  * **STD** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 每个元素的标准偏差的张量

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
    tensor([  1.0425,   3.5672,   2.7969,   4.2925,   4.7229,   6.2134,
              8.0505,   8.1408,   9.0563,  10.0566])
    

`torch.``normal`( _mean=0.0_ , _std_ , _out=None_ ) → Tensor

    

上述功能，但是类似的装置都被绘制的元素之间共享。

Parameters

    

  * **意味着** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 平均为所有的发行

  * **std** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor of per-element standard deviations

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> torch.normal(mean=0.5, std=torch.arange(1., 6.))
    tensor([-1.2793, -1.0732, -2.0687,  5.1177, -1.2303])
    

`torch.``normal`( _mean_ , _std=1.0_ , _out=None_ ) → Tensor

    

与上述类似的功能，但标准偏差都绘制的元素之间共享。

Parameters

    

  * **mean** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor of per-element means

  * **STD** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 所有分布的标准偏差

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> torch.normal(mean=torch.arange(1., 6.))
    tensor([ 1.1552,  2.6148,  2.6535,  5.8318,  4.2361])
    

`torch.``normal`( _mean_ , _std_ , _size_ , _*_ , _out=None_ ) → Tensor

    

类似于上面的功能，但是平均值和标准偏差都绘制的元素之间共享。将得到的张量具有由`大小 `给定的大小。

Parameters

    

  * **意味着** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 平均为所有的发行

  * **STD** （[ _浮_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 标准对于所有分布偏差

  * **大小** （ _INT ..._ ） - 定义输出张量的形状的整数序列。

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> torch.normal(2, 3, size=(1, 4))
    tensor([[-1.3987, -1.9544,  3.6048,  0.7909]])
    

`torch.``rand`( _*size_ , _out=None_ , _dtype=None_ , _layout=torch.strided_ ,
_device=None_ , _requires_grad=False_ ) → Tensor

    

返回在区间 [ 0  填充有随机数从均匀分布的张量，  1  ） [0，1） [ 0  ， 1  ）

张量的形状由可变参数`大小 `中定义。

Parameters

    

  * **size** ( _int..._ ) – a sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see `torch.set_default_tensor_type()`).

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned Tensor. Default: `torch.strided`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> torch.rand(4)
    tensor([ 0.5204,  0.2503,  0.3525,  0.5673])
    >>> torch.rand(2, 3)
    tensor([[ 0.8237,  0.5781,  0.6879],
            [ 0.3816,  0.7249,  0.0998]])
    

`torch.``rand_like`( _input_ , _dtype=None_ , _layout=None_ , _device=None_ ,
_requires_grad=False_ ) → Tensor

    

返回具有相同尺寸的张量为`输入 `填充有随机数从均匀分布在区间 [ 0  ， 1  ） [0，1） [ 0  ， 1  ） 。 `
torch.rand_like（输入） `等于`torch.rand（input.size（）， D型细胞= input.dtype， 布局=
input.layout， 设备= input.device） `。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the size of `input`will determine size of the output tensor

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned Tensor. Default: if `None`, defaults to the dtype of `input`.

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned tensor. Default: if `None`, defaults to the layout of `input`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, defaults to the device of `input`.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

`torch.``randint`( _low=0_ , _high_ , _size_ , _out=None_ , _dtype=None_ ,
_layout=torch.strided_ , _device=None_ , _requires_grad=False_ ) → Tensor

    

返回填充有随机整数的张量产生均匀之间`低 `（含）和`高 `（异）。

The shape of the tensor is defined by the variable argument `size`.

Parameters

    

  * **低** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 以从分布中抽取最低整数。默认值：0。

  * **高** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 一个以上的最大整数为从分布中抽取。

  * **大小** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")） - 元组限定所述输出张量的形状。

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see `torch.set_default_tensor_type()`).

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned Tensor. Default: `torch.strided`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> torch.randint(3, 5, (3,))
    tensor([4, 3, 4])
    
    
    >>> torch.randint(10, (2, 2))
    tensor([[0, 2],
            [5, 5]])
    
    
    >>> torch.randint(3, 10, (2, 2))
    tensor([[4, 5],
            [6, 7]])
    

`torch.``randint_like`( _input_ , _low=0_ , _high_ , _dtype=None_ ,
_layout=torch.strided_ , _device=None_ , _requires_grad=False_ ) → Tensor

    

返回与相同形状的张量作为填充有随机整数张量`输入 `产生均匀之间`低 `（含）和`高 `（异）。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the size of `input`will determine size of the output tensor

  * **low** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _,_ _optional_ ) – Lowest integer to be drawn from the distribution. Default: 0.

  * **high** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – One above the highest integer to be drawn from the distribution.

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned Tensor. Default: if `None`, defaults to the dtype of `input`.

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned tensor. Default: if `None`, defaults to the layout of `input`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, defaults to the device of `input`.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

`torch.``randn`( _*size_ , _out=None_ , _dtype=None_ , _layout=torch.strided_
, _device=None_ , _requires_grad=False_ ) → Tensor

    

返回填充的随机数从正态分布与平均值的张量 0 和方差HTG2] 1 （也称为标准正态分布）。

outi∼N(0,1)\text{out}_{i} \sim \mathcal{N}(0, 1) outi​∼N(0,1)

The shape of the tensor is defined by the variable argument `size`.

Parameters

    

  * **size** ( _int..._ ) – a sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see `torch.set_default_tensor_type()`).

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned Tensor. Default: `torch.strided`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> torch.randn(4)
    tensor([-2.1436,  0.9966,  2.3426, -0.6366])
    >>> torch.randn(2, 3)
    tensor([[ 1.5954,  2.8929, -1.0923],
            [ 1.1719, -0.4709, -0.1996]])
    

`torch.``randn_like`( _input_ , _dtype=None_ , _layout=None_ , _device=None_ ,
_requires_grad=False_ ) → Tensor

    

返回一个张量的大小相同`输入 `填充有随机数从正态分布均值为0，方差为1。`torch.randn_like（输入）`等于`
torch.randn（input.size（）， D型细胞= input.dtype， 布局= input.layout， 设备=
input.device） `。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the size of `input`will determine size of the output tensor

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned Tensor. Default: if `None`, defaults to the dtype of `input`.

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned tensor. Default: if `None`, defaults to the layout of `input`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, defaults to the device of `input`.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

`torch.``randperm`( _n_ , _out=None_ , _dtype=torch.int64_ ,
_layout=torch.strided_ , _device=None_ , _requires_grad=False_ ) → LongTensor

    

返回整数的随机置换从`0`至`n的 -  1`。

Parameters

    

  * **n的** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 上限（不包括）

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

  * **DTYPE** （[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")，可选） - 返回的张量的所希望的数据类型。默认值：`torch.int64`。

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned Tensor. Default: `torch.strided`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Example:

    
    
    >>> torch.randperm(4)
    tensor([2, 1, 0, 3])
    

### 就地随机抽样

有关于张量的定义以及几个就地随机抽样功能。点击进入是指他们的文档：

  * [ `torch.Tensor.bernoulli_（） `](tensors.html#torch.Tensor.bernoulli_ "torch.Tensor.bernoulli_") \- 就地版本 `torch.bernoulli的（） `

  * [ `torch.Tensor.cauchy_（） `](tensors.html#torch.Tensor.cauchy_ "torch.Tensor.cauchy_") \- 从柯西分布抽取的数字

  * [ `torch.Tensor.exponential_（） `](tensors.html#torch.Tensor.exponential_ "torch.Tensor.exponential_") \- 从所述指数分布抽取的数字

  * [ `torch.Tensor.geometric_（） `](tensors.html#torch.Tensor.geometric_ "torch.Tensor.geometric_") \- 从所述几何分布绘制的元素

  * [ `torch.Tensor.log_normal_（） `](tensors.html#torch.Tensor.log_normal_ "torch.Tensor.log_normal_") \- 从所述对数正态分布的样品

  * [ `torch.Tensor.normal_（） `](tensors.html#torch.Tensor.normal_ "torch.Tensor.normal_") \- 就地版本 `torch.normal的（） `

  * [ `torch.Tensor.random_（） `](tensors.html#torch.Tensor.random_ "torch.Tensor.random_") \- 从所述离散均匀分布采样的数字

  * [ `torch.Tensor.uniform_（） `](tensors.html#torch.Tensor.uniform_ "torch.Tensor.uniform_") \- 从所述连续均匀分布采样的数字

### 准随机采样

_class_`torch.quasirandom.``SobolEngine`( _dimension_ , _scramble=False_ ,
_seed=None_ )[[source]](_modules/torch/quasirandom.html#SobolEngine)

    

的 `torch.quasirandom.SobolEngine`是用于生成（加扰）的发动机Sobol序列。
Sobol序列是低差异准随机序列的一个例子。

用于Sobol序列的发动机的这种实现能够采样序列的高达1111它使用方向编号，以产生这些序列的最大尺寸，并且这些数字已经适应从[这里](http://web.maths.unsw.edu.au/~fkuo/sobol/joe-
kuo-old.1111)。

参考

  * 艺术B.欧文。扰Sobol和的Niederreiter星点。轴颈复杂性，14（4）：466-489，1998年12月。

  * I. M. Sobol。点的立方体的分布和积分的准确评估。深航。 Vychisl。垫。我在。物理学，7：784-802，1967。

Parameters

    

  * **维** （ _INT_ ） - 的序列的维度要绘制

  * **加扰** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 此设定为`真 `将产生加扰Sobol序列。扰是能够产生更好Sobol序列。默认值：`假 [HTG17。`

  * **种子** （ _INT_ _，_ _可选_ ） - 这是对加扰种子。随机数发生器的种子被设定为这一点，如果指定。默认值：`无 `

例子：

    
    
    >>> soboleng = torch.quasirandom.SobolEngine(dimension=5)
    >>> soboleng.draw(3)
    tensor([[0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            [0.7500, 0.2500, 0.7500, 0.2500, 0.7500],
            [0.2500, 0.7500, 0.2500, 0.7500, 0.2500]])
    

`draw`( _n=1_ , _out=None_ , _dtype=torch.float32_
)[[source]](_modules/torch/quasirandom.html#SobolEngine.draw)

    

函数从Sobol序列绘制的`n的 `点的序列。需要注意的是样品是依赖于先前的样本。结果的大小是 （ n的 ， d  i的 M  E  n的 S  i的 O
n的 ） （N，尺寸） （ n的 ， d  i的 M  E  n的 S  i的 O  n的 ） 。

Parameters

    

  * **n的** （ _INT_ _，_ _可选_ ） - 点的序列的长度来绘制。默认值：1

  * **OUT** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 输出张量

  * **DTYPE** （`torch.dtype`，可选） - 返回的张量的所希望的数据类型。默认值：`torch.float32`

`fast_forward`( _n_
)[[source]](_modules/torch/quasirandom.html#SobolEngine.fast_forward)

    

通过`n的 `步功能快进的`SobolEngine`的状态。这等同于在不使用样品绘图`n的 `样品。

Parameters

    

**n的** （ _INT_ ） - 的步数由快进。

`reset`()[[source]](_modules/torch/quasirandom.html#SobolEngine.reset)

    

功能重置`SobolEngine`到基本状态。

## 序列

`torch.``save`( _obj_ , _f_ , _pickle_module= <module 'pickle' from
'/opt/conda/lib/python3.6/pickle.py'>_, _pickle_protocol=2_
)[[source]](_modules/torch/serialization.html#save)

    

保存对象到磁盘文件。

参见：[ 用于保存模型推荐的方法 ](notes/serialization.html#recommend-saving-models)

Parameters

    

  * **OBJ** \- 保存对象

  * **F** \- 一个类文件对象（必须实现写和flush）或包含文件名的字符串

  * **pickle_module** \- 模块用于酸洗的元数据和对象

  * **pickle_protocol** \- 可以指定覆盖默认协议

Warning

如果您在使用Python 2， `torch.save（） `不支持`StringIO.StringIO
`作为有效的类文件对象。这是因为写方法应该返回写入的字节数; `StringIO.write（） `不执行此操作。

请使用类似[ `io.BytesIO`](https://docs.python.org/3/library/io.html#io.BytesIO
"\(in Python v3.7\)")代替。

例

    
    
    >>> # Save to file
    >>> x = torch.tensor([0, 1, 2, 3, 4])
    >>> torch.save(x, 'tensor.pt')
    >>> # Save to io.BytesIO buffer
    >>> buffer = io.BytesIO()
    >>> torch.save(x, buffer)
    

`torch.``load`( _f_ , _map_location=None_ , _pickle_module= <module 'pickle'
from '/opt/conda/lib/python3.6/pickle.py'>_, _**pickle_load_args_
)[[source]](_modules/torch/serialization.html#load)

    

加载一个对象保存 `从文件torch.save（） `。

`torch.load（） `
使用Python的在unpickle设施，但对待仓库，其背后张量，特别是。他们首先反序列化在CPU上，然后被转移到他们从已保存的设备。如果失败（例如，由于运行时系统不具有一定的设备），将引发一个例外。然而，存储器可以被动态地重新映射到一组替代使用`
map_location`参数的设备。

如果`map_location
`是一个可调用，它将被一次为每个串行化存储用两个参数称为：存储和位置。存储参数将是存储的初始反序列化，驻留在所述CPU上。每个串行化存储具有与其相关联的位置标签识别它是从保存该设备，该标签是传递给`
map_location`第二个参数。内置的位置代码是`'CPU' [HTG11用于CPU张量和`'CUDA：DEVICE_ID' `（例如`'
CUDA：2'`）为CUDA张量。 `map_location`返回值应当是`无 `或存储。如果`map_location
`返回一个存储，它将被用作最终反序列化对象，已被移动到正确的设备。否则， `torch.load（） `将回落到默认的行为，仿佛`
map_location`WASN” Ť规定。`

如果`map_location`是[ `torch.device`
](tensor_attributes.html#torch.torch.device
"torch.torch.device")对象或字符串contraining设备标记，其指示的位置所有张量应该被加载。

否则，如果`map_location`是一个字典，它将被用于重新映射出现的文件（密钥）的位置标记，以那些指定在哪里放置存储器（值）。

用户扩展可使用`torch.serialization.register_package（） `注册自己的位置的标签和标记和反串行化的方法。

Parameters

    

  * **F** \- 一个类文件对象（必须实现`读（） `：meth`readline`，：meth`tell`，并且：meth`seek `），或包含文件名的字符串

  * **map_location** \- 的函数，[ `torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device")，字符串或一个字典指定如何重新映射的存储位置

  * **pickle_module** \- 用于取储存元数据和对象模块（具有相匹配的`pickle_module`用于序列化文件）

  * **pickle_load_args** \- 可选的关键字参数传递到`pickle_module.load（） `和`pickle_module.Unpickler（） `例如，`编码= ...`。

Note

当你调用 `torch.load（） `在包含GPU张量的文件时，这些张量将被默认加载到GPU。可以调用`torch.load（..，
map_location = 'CPU'） `→`load_state_dict（） `以避免GPU RAM浪涌加载模型的检查点时。

Note

在Python 3，加载由Python 2中保存的文件时，可能会遇到`的UnicodeDecodeError： 'ASCII' 编解码器 不能 解码 字节
0X ......  [HTG15。这是由你可以使用额外的`编码 `关键字参数来指定这些对象应该如何被加载后，在Python2和Python
3字节串处理差异造成的如`编码= 'latin1的' `使用`LATIN1`编码对它们进行解码为字符串，并`编码= '字节' `使他们作为可稍后`
byte_array.decode（......） `解码字节阵列。`

Example

    
    
    >>> torch.load('tensors.pt')
    # Load all tensors onto the CPU
    >>> torch.load('tensors.pt', map_location=torch.device('cpu'))
    # Load all tensors onto the CPU, using a function
    >>> torch.load('tensors.pt', map_location=lambda storage, loc: storage)
    # Load all tensors onto GPU 1
    >>> torch.load('tensors.pt', map_location=lambda storage, loc: storage.cuda(1))
    # Map tensors from GPU 1 to GPU 0
    >>> torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})
    # Load tensor from io.BytesIO object
    >>> with open('tensor.pt', 'rb') as f:
            buffer = io.BytesIO(f.read())
    >>> torch.load(buffer)
    

## 并行

`torch.``get_num_threads`() → int

    

返回用于CPU并行操作的线程数

`torch.``set_num_threads`( _int_ )

    

设置用于CPU并行操作的线程数。警告：为确保使用的线程数目正确，set_num_threads必须在运行心切，JIT或autograd代码之前调用。

`torch.``get_num_interop_threads`() → int

    

返回用于-OP间并行CPU上的线程的数目（例如，在JIT解释器）

`torch.``set_num_interop_threads`( _int_ )

    

设置用于互操作并行（例如，在JIT解释器）上的CPU线程的数目。警告：可以在任何跨运并行工作开始（如JIT执行）之前只能被调用一次和。

## 本地禁用梯度计算

，torch.set_grad_enabled的上下文管理器`torch.no_grad（），`torch.enable_grad（） `和`（ ）
`是用于局部禁用和启用梯度计算有帮助。参见[ 本地禁用梯度计算 ](autograd.html#locally-disable-
grad)关于其使用的更多细节。这些情境经理线程局部的，所以如果你使用 发送工作到另一个线程，他们将无法正常工作：模块：`threading` 模块等。`

Examples:

    
    
    >>> x = torch.zeros(1, requires_grad=True)
    >>> with torch.no_grad():
    ...     y = x * 2
    >>> y.requires_grad
    False
    
    >>> is_train = False
    >>> with torch.set_grad_enabled(is_train):
    ...     y = x * 2
    >>> y.requires_grad
    False
    
    >>> torch.set_grad_enabled(True)  # this can also be used as a function
    >>> y = x * 2
    >>> y.requires_grad
    True
    
    >>> torch.set_grad_enabled(False)
    >>> y = x * 2
    >>> y.requires_grad
    False
    

## 数学运算

### 逐点行动

`torch.``abs`( _input_ , _out=None_ ) → Tensor

    

计算给定的`输入 `张量的逐元素的绝对值。

outi=∣inputi∣\text{out}_{i} = |\text{input}_{i}| outi​=∣inputi​∣

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> torch.abs(torch.tensor([-1, -2, 3]))
    tensor([ 1,  2,  3])
    

`torch.``acos`( _input_ , _out=None_ ) → Tensor

    

返回与输入的`的元素 `的反余弦一个新的张量。

outi=cos⁡−1(inputi)\text{out}_{i} = \cos^{-1}(\text{input}_{i})
outi​=cos−1(inputi​)

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.3348, -0.5889,  0.2005, -0.1584])
    >>> torch.acos(a)
    tensor([ 1.2294,  2.2004,  1.3690,  1.7298])
    

`torch.``add`()

    

`torch.``add`( _input_ , _other_ , _out=None_ )

    

增加了标量`其他 `到输入`输入 `中的每个元素，并返回一个新的由此而来张量。

out=input+other\text{out} = \text{input} + \text{other} out=input+other

如果`输入 `是类型FloatTensor或DoubleTensor的，`其他 `必须是一个实数，否则它应该是整数。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **其他** （ _号码_ ） - 的数量被添加到`输入 `的各要素

Keyword Arguments

    

**out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_
) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.0202,  1.0985,  1.3506, -0.6056])
    >>> torch.add(a, 20)
    tensor([ 20.0202,  21.0985,  21.3506,  19.3944])
    

`torch.``add`( _input_ , _alpha=1_ , _other_ , _out=None_ )

    

张量的每个元素`其他 `由标量`阿尔法 `相乘，并加入到该张量的每个元素`输入 `。得到的张量返回。

输入的`形状 `和`其他 `必须[ broadcastable  ](notes/broadcasting.html#broadcasting-
semantics)。

out=input+alpha×other\text{out} = \text{input} + \text{alpha} \times
\text{other} out=input+alpha×other

如果`其他 `是类型FloatTensor或DoubleTensor的，`阿尔法 `必须是一个实数，否则它应该是整数。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 第一输入张量

  * **阿尔法** （ _号码_ ） - 为`标量乘数其他 `

  * **其他** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 第二输入张量

Keyword Arguments

    

**out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_
) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([-0.9732, -0.3497,  0.6245,  0.4022])
    >>> b = torch.randn(4, 1)
    >>> b
    tensor([[ 0.3743],
            [-1.7724],
            [-0.5811],
            [-0.8017]])
    >>> torch.add(a, 10, b)
    tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
            [-18.6971, -18.0736, -17.0994, -17.3216],
            [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
            [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])
    

`torch.``addcdiv`( _input_ , _value=1_ , _tensor1_ , _tensor2_ , _out=None_ )
→ Tensor

    

通过`执行tensor1`的`逐元素除法tensor2`，由标量`值[HTG10相乘的结果] `，并将其添加到`输入 `。

outi=inputi+value×tensor1itensor2i\text{out}_i = \text{input}_i + \text{value}
\times \frac{\text{tensor1}_i}{\text{tensor2}_i}
outi​=inputi​+value×tensor2i​tensor1i​​

的形状`输入 `，`tensor1`和`tensor2`必须[ broadcastable
](notes/broadcasting.html#broadcasting-semantics)。

对于类型的输入 FloatTensor 或 DoubleTensor ，`值 `必须是一个实数，否则的整数。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 要添加的张量

  * **值** （ _号码_ _，_ _可选_ ） - 乘数 tensor1  /  tensor2  \文本{tensor1} / \文本{tensor2}  tensor1  /  tensor2 

  * **tensor1** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 分子张量

  * **tensor2** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 分母张量

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> t = torch.randn(1, 3)
    >>> t1 = torch.randn(3, 1)
    >>> t2 = torch.randn(1, 3)
    >>> torch.addcdiv(t, 0.1, t1, t2)
    tensor([[-0.2312, -3.6496,  0.1312],
            [-1.0428,  3.4292, -0.1030],
            [-0.5369, -0.9829,  0.0430]])
    

`torch.``addcmul`( _input_ , _value=1_ , _tensor1_ , _tensor2_ , _out=None_ )
→ Tensor

    

执行由`tensor2 tensor1`的`逐元素乘法 `，由标量`值[HTG10相乘的结果] `，并将其添加到`输入 `。

outi=inputi+value×tensor1i×tensor2i\text{out}_i = \text{input}_i +
\text{value} \times \text{tensor1}_i \times \text{tensor2}_i
outi​=inputi​+value×tensor1i​×tensor2i​

The shapes of `input`, `tensor1`, and `tensor2`must be
[broadcastable](notes/broadcasting.html#broadcasting-semantics).

For inputs of type FloatTensor or DoubleTensor, `value`must be a real number,
otherwise an integer.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor to be added

  * **值** （ _号码_ _，_ _可选_ ） - 乘数 T  E  n的 S  O  R  1\.  *  T  E  n的 S  O  R  2  tensor1。* tensor2  T  E  n的 S  O  R  1  。  *  T  E  n的 S  O  R  2 

  * **tensor1** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 要被乘的张量

  * **tensor2** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 要被乘的张量

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> t = torch.randn(1, 3)
    >>> t1 = torch.randn(3, 1)
    >>> t2 = torch.randn(1, 3)
    >>> torch.addcmul(t, 0.1, t1, t2)
    tensor([[-0.8635, -0.6391,  1.6174],
            [-0.7617, -0.5879,  1.7388],
            [-0.8353, -0.6249,  1.6511]])
    

`torch.``asin`( _input_ , _out=None_ ) → Tensor

    

返回与输入的`的元素 `的反正弦新张量。

outi=sin⁡−1(inputi)\text{out}_{i} = \sin^{-1}(\text{input}_{i})
outi​=sin−1(inputi​)

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([-0.5962,  1.4985, -0.4396,  1.4525])
    >>> torch.asin(a)
    tensor([-0.6387,     nan, -0.4552,     nan])
    

`torch.``atan`( _input_ , _out=None_ ) → Tensor

    

返回与输入的`的元素 `的反正切一个新的张量。

outi=tan⁡−1(inputi)\text{out}_{i} = \tan^{-1}(\text{input}_{i})
outi​=tan−1(inputi​)

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.2341,  0.2539, -0.6256, -0.6448])
    >>> torch.atan(a)
    tensor([ 0.2299,  0.2487, -0.5591, -0.5727])
    

`torch.``atan2`( _input_ , _other_ , _out=None_ ) → Tensor

    

返回与输入的`和元素 ``其他 `的反正切一个新的张量。

The shapes of `input`and `other`must be
[broadcastable](notes/broadcasting.html#broadcasting-semantics).

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the first input tensor

  * **other** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the second input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.9041,  0.0196, -0.3108, -2.4423])
    >>> torch.atan2(a, torch.randn(4))
    tensor([ 0.9833,  0.0811, -1.9743, -1.4151])
    

`torch.``ceil`( _input_ , _out=None_ ) → Tensor

    

返回与输入的`的元素 `的最小整数大于或等于每个元件的小区一个新的张量。

outi=⌈inputi⌉=⌊inputi⌋+1\text{out}_{i} = \left\lceil \text{input}_{i}
\right\rceil = \left\lfloor \text{input}_{i} \right\rfloor + 1
outi​=⌈inputi​⌉=⌊inputi​⌋+1

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([-0.6341, -1.4208, -1.0900,  0.5826])
    >>> torch.ceil(a)
    tensor([-0., -1., -1.,  1.])
    

`torch.``clamp`( _input_ , _min_ , _max_ , _out=None_ ) → Tensor

    

夹住`输入 `的所有元素为范围 [ `分钟HTG9] `， `MAX`，并返回所得到的张量：

yi={minif xi<minxiif min≤xi≤maxmaxif xi>maxy_i = \begin{cases} \text{min} &
\text{if } x_i < \text{min} \\\ x_i & \text{if } \text{min} \leq x_i \leq
\text{max} \\\ \text{max} & \text{if } x_i > \text{max} \end{cases}
yi​=⎩⎪⎨⎪⎧​minxi​max​if xi​<minif min≤xi​≤maxif xi​>max​

如果`输入 `是式 FloatTensor 或 DoubleTensor ，ARGS`分钟HTG11] `和 `MAX`
必须是实数，否则他们应该是整数。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **分钟HTG1]（ _号码_ ） - 结合的较低的范围内的被夹紧到**

  * **MAX** （ _号码_ ） - 上限的范围内的被夹紧到

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([-1.7120,  0.1734, -0.0478, -0.0922])
    >>> torch.clamp(a, min=-0.5, max=0.5)
    tensor([-0.5000,  0.1734, -0.0478, -0.0922])
    

`torch.``clamp`( _input_ , _*_ , _min_ , _out=None_ ) → Tensor

    

夹具在`输入 `为大于或等于 `分钟HTG7] `的所有元素。

如果`输入 `是式 FloatTensor 或 DoubleTensor ，`值 `应该是一个真正号，否则它应该是整数。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **值** （ _号码_ ） - 在输出中的各元素的最小值

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([-0.0299, -2.3184,  2.1593, -0.8883])
    >>> torch.clamp(a, min=0.5)
    tensor([ 0.5000,  0.5000,  2.1593,  0.5000])
    

`torch.``clamp`( _input_ , _*_ , _max_ , _out=None_ ) → Tensor

    

夹具在`输入的所有元素 `是小于或等于 `MAX`。

If `input`is of type FloatTensor or DoubleTensor, `value`should be a real
number, otherwise it should be an integer.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **值** （ _号码_ ） - 在输出中每个元素的最大值

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.7753, -0.4702, -0.4599,  1.1899])
    >>> torch.clamp(a, max=0.5)
    tensor([ 0.5000, -0.4702, -0.4599,  0.5000])
    

`torch.``cos`( _input_ , _out=None_ ) → Tensor

    

返回与输入的`的元素 `的余弦一个新的张量。

outi=cos⁡(inputi)\text{out}_{i} = \cos(\text{input}_{i}) outi​=cos(inputi​)

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
    >>> torch.cos(a)
    tensor([ 0.1395,  0.2957,  0.6553,  0.5574])
    

`torch.``cosh`( _input_ , _out=None_ ) → Tensor

    

返回与输入的`的元素 `的双曲余弦一个新的张量。

outi=cosh⁡(inputi)\text{out}_{i} = \cosh(\text{input}_{i}) outi​=cosh(inputi​)

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.1632,  1.1835, -0.6979, -0.7325])
    >>> torch.cosh(a)
    tensor([ 1.0133,  1.7860,  1.2536,  1.2805])
    

`torch.``div`()

    

`torch.``div`( _input_ , _other_ , _out=None_ ) → Tensor

    

将输入`输入 `中的每个元素与所述标量`其他 `并返回一个新产生的张量。

outi=inputiother\text{out}_i = \frac{\text{input}_i}{\text{other}}
outi​=otherinputi​​

如果`输入 `是式 FloatTensor 或 DoubleTensor ，`其他 `应该是一个真正号，否则应该是一个整数

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **其他** （ _号码_ ） - 的数量被划分为`输入 `的各要素

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(5)
    >>> a
    tensor([ 0.3810,  1.2774, -0.2972, -0.3719,  0.4637])
    >>> torch.div(a, 0.5)
    tensor([ 0.7620,  2.5548, -0.5944, -0.7439,  0.9275])
    

`torch.``div`( _input_ , _other_ , _out=None_ ) → Tensor

    

张量`输入 `的每个元素由张量`其他 `的各要素分割。得到的张量返回。输入的`形状 `和`其他 `必须[ broadcastable
](notes/broadcasting.html#broadcasting-semantics)。

outi=inputiotheri\text{out}_i = \frac{\text{input}_i}{\text{other}_i}
outi​=otheri​inputi​​

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 分子张量

  * **其他** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 分母张量

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
            [ 0.1815, -1.0111,  0.9805, -1.5923],
            [ 0.1062,  1.4581,  0.7759, -1.2344],
            [-0.1830, -0.0313,  1.1908, -1.4757]])
    >>> b = torch.randn(4)
    >>> b
    tensor([ 0.8032,  0.2930, -0.8113, -0.2308])
    >>> torch.div(a, b)
    tensor([[-0.4620, -6.6051,  0.5676,  1.2637],
            [ 0.2260, -3.4507, -1.2086,  6.8988],
            [ 0.1322,  4.9764, -0.9564,  5.3480],
            [-0.2278, -0.1068, -1.4678,  6.3936]])
    

`torch.``digamma`( _input_ , _out=None_ ) → Tensor

    

计算在输入伽马函数的对数导数。

ψ(x)=ddxln⁡(Γ(x))=Γ′(x)Γ(x)\psi(x) = \frac{d}{dx}
\ln\left(\Gamma\left(x\right)\right) = \frac{\Gamma'(x)}{\Gamma(x)}
ψ(x)=dxd​ln(Γ(x))=Γ(x)Γ′(x)​

Parameters

    

**输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 张量来计算对双伽玛函数

Example:

    
    
    >>> a = torch.tensor([1, 0.5])
    >>> torch.digamma(a)
    tensor([-0.5772, -1.9635])
    

`torch.``erf`( _input_ , _out=None_ ) → Tensor

    

计算每个元件的误差函数。误差函数定义如下：

erf(x)=2π∫0xe−t2dt\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2}
dt erf(x)=π​2​∫0x​e−t2dt

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> torch.erf(torch.tensor([0, -1., 10.]))
    tensor([ 0.0000, -0.8427,  1.0000])
    

`torch.``erfc`( _input_ , _out=None_ ) → Tensor

    

计算的`输入 `的各元素的互补误差函数。互补误差函数定义如下：

erfc(x)=1−2π∫0xe−t2dt\mathrm{erfc}(x) = 1 - \frac{2}{\sqrt{\pi}} \int_{0}^{x}
e^{-t^2} dt erfc(x)=1−π​2​∫0x​e−t2dt

Parameters

    

  * **张量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入张量

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> torch.erfc(torch.tensor([0, -1., 10.]))
    tensor([ 1.0000, 1.8427,  0.0000])
    

`torch.``erfinv`( _input_ , _out=None_ ) → Tensor

    

计算的`输入 `的各元素的逆误差函数。  1  [HTG16 - 逆误差函数的范围在 （ 中定义]， 1  ） （-1，1） （ \-  1  ， 1
）  为：

erfinv(erf(x))=x\mathrm{erfinv}(\mathrm{erf}(x)) = x erfinv(erf(x))=x

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> torch.erfinv(torch.tensor([0, 0.5, -1.]))
    tensor([ 0.0000,  0.4769,    -inf])
    

`torch.``exp`( _input_ , _out=None_ ) → Tensor

    

返回与输入张量`输入 `的元素的指数一个新的张量。

yi=exiy_{i} = e^{x_{i}} yi​=exi​

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> torch.exp(torch.tensor([0, math.log(2.)]))
    tensor([ 1.,  2.])
    

`torch.``expm1`( _input_ , _out=None_ ) → Tensor

    

返回与元件的指数减去`输入 `1新的张量。

yi=exi−1y_{i} = e^{x_{i}} - 1 yi​=exi​−1

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> torch.expm1(torch.tensor([0, math.log(2.)]))
    tensor([ 0.,  1.])
    

`torch.``floor`( _input_ , _out=None_ ) → Tensor

    

返回与输入的`的元素 `的最大整数是小于或等于每一个元件的底板中的新张量。

outi=⌊inputi⌋\text{out}_{i} = \left\lfloor \text{input}_{i} \right\rfloor
outi​=⌊inputi​⌋

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([-0.8166,  1.5308, -0.2530, -0.2091])
    >>> torch.floor(a)
    tensor([-1.,  1., -1., -1.])
    

`torch.``fmod`( _input_ , _other_ , _out=None_ ) → Tensor

    

计算事业部的元素方面的剩余部分。

该被除数和除数可以同时包含了整数和浮点数。其余具有相同的符号作为被除数`输入 `。

当`其他 `是一个张量，`输入 `和`其他 `必须的形状[ broadcastable
](notes/broadcasting.html#broadcasting-semantics)。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 被除数

  * **其他** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _或_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 除数，其可以是数字或作为被除数的相同形状的张量

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> torch.fmod(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
    tensor([-1., -0., -1.,  1.,  0.,  1.])
    >>> torch.fmod(torch.tensor([1., 2, 3, 4, 5]), 1.5)
    tensor([ 1.0000,  0.5000,  0.0000,  1.0000,  0.5000])
    

`torch.``frac`( _input_ , _out=None_ ) → Tensor

    

计算每个元件的在`输入 `的小数部分。

outi=inputi−⌊∣inputi∣⌋∗sgn⁡(inputi)\text{out}_{i} = \text{input}_{i} -
\left\lfloor |\text{input}_{i}| \right\rfloor *
\operatorname{sgn}(\text{input}_{i}) outi​=inputi​−⌊∣inputi​∣⌋∗sgn(inputi​)

Example:

    
    
    >>> torch.frac(torch.tensor([1, 2.5, -3.2]))
    tensor([ 0.0000,  0.5000, -0.2000])
    

`torch.``lerp`( _input_ , _end_ , _weight_ , _out=None_ )

    

做两张量的线性内插`开始 `（由`输入 `中给出）和`结束 [HTG11基于标量或张量`重量 `]并返回生成的`OUT`张量。`

outi=starti+weighti×(endi−starti)\text{out}_i = \text{start}_i +
\text{weight}_i \times (\text{end}_i - \text{start}_i)
outi​=starti​+weighti​×(endi​−starti​)

的`形状开始 `和`结束 `必须[ broadcastable  ](notes/broadcasting.html#broadcasting-
semantics)。如果`重量 `是一个张量，则`形状重量 `，`开始 `和`结束 `必须[ broadcastable
](notes/broadcasting.html#broadcasting-semantics)。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - ，起点的张量

  * **端** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 与该结束点的张量

  * **重量** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ _张量_ ） - 用于内插公式的权重

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> start = torch.arange(1., 5.)
    >>> end = torch.empty(4).fill_(10)
    >>> start
    tensor([ 1.,  2.,  3.,  4.])
    >>> end
    tensor([ 10.,  10.,  10.,  10.])
    >>> torch.lerp(start, end, 0.5)
    tensor([ 5.5000,  6.0000,  6.5000,  7.0000])
    >>> torch.lerp(start, end, torch.full_like(start, 0.5))
    tensor([ 5.5000,  6.0000,  6.5000,  7.0000])
    

`torch.``log`( _input_ , _out=None_ ) → Tensor

    

返回与输入的`的元素 `的自然对数新的张量。

yi=log⁡e(xi)y_{i} = \log_{e} (x_{i}) yi​=loge​(xi​)

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(5)
    >>> a
    tensor([-0.7168, -0.5471, -0.8933, -1.4428, -0.1190])
    >>> torch.log(a)
    tensor([ nan,  nan,  nan,  nan,  nan])
    

`torch.``log10`( _input_ , _out=None_ ) → Tensor

    

返回与对数新的张量，以输入的`的元素 `的基极10。

yi=log⁡10(xi)y_{i} = \log_{10} (x_{i}) yi​=log10​(xi​)

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.rand(5)
    >>> a
    tensor([ 0.5224,  0.9354,  0.7257,  0.1301,  0.2251])
    
    
    >>> torch.log10(a)
    tensor([-0.2820, -0.0290, -0.1392, -0.8857, -0.6476])
    

`torch.``log1p`( _input_ , _out=None_ ) → Tensor

    

返回与（1 + `输入 `）的自然对数新的张量。

yi=log⁡e(xi+1)y_i = \log_{e} (x_i + 1) yi​=loge​(xi​+1)

Note

此函数是更精确的比 `torch.log（） `为`输入 `的值较小

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(5)
    >>> a
    tensor([-1.0090, -0.9923,  1.0249, -0.5372,  0.2492])
    >>> torch.log1p(a)
    tensor([    nan, -4.8653,  0.7055, -0.7705,  0.2225])
    

`torch.``log2`( _input_ , _out=None_ ) → Tensor

    

返回与对数新的张量，以输入的`的元素 `的基体2。

yi=log⁡2(xi)y_{i} = \log_{2} (x_{i}) yi​=log2​(xi​)

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.rand(5)
    >>> a
    tensor([ 0.8419,  0.8003,  0.9971,  0.5287,  0.0490])
    
    
    >>> torch.log2(a)
    tensor([-0.2483, -0.3213, -0.0042, -0.9196, -4.3504])
    

`torch.``mul`()

    

`torch.``mul`( _input_ , _other_ , _out=None_ )

    

乘以所述标量`其他 `输入`输入 `中的每个元素，并返回一个新产生的张量。

outi=other×inputi\text{out}_i = \text{other} \times \text{input}_i
outi​=other×inputi​

If `input`is of type FloatTensor or DoubleTensor, `other`should be a real
number, otherwise it should be an integer

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **其他** （ _号码_ ） - 数相乘以`输入 `的各要素

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(3)
    >>> a
    tensor([ 0.2015, -0.4255,  2.6087])
    >>> torch.mul(a, 100)
    tensor([  20.1494,  -42.5491,  260.8663])
    

`torch.``mul`( _input_ , _other_ , _out=None_ )

    

张量`输入 `的每个元素是由相应的元素相乘的张量`其他 `。得到的张量返回。

The shapes of `input`and `other`must be
[broadcastable](notes/broadcasting.html#broadcasting-semantics).

outi=inputi×otheri\text{out}_i = \text{input}_i \times \text{other}_i
outi​=inputi​×otheri​

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 第一被乘数张量

  * **其他** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 第二被乘数张量

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4, 1)
    >>> a
    tensor([[ 1.1207],
            [-0.3137],
            [ 0.0700],
            [ 0.8378]])
    >>> b = torch.randn(1, 4)
    >>> b
    tensor([[ 0.5146,  0.1216, -0.5244,  2.2382]])
    >>> torch.mul(a, b)
    tensor([[ 0.5767,  0.1363, -0.5877,  2.5083],
            [-0.1614, -0.0382,  0.1645, -0.7021],
            [ 0.0360,  0.0085, -0.0367,  0.1567],
            [ 0.4312,  0.1019, -0.4394,  1.8753]])
    

`torch.``mvlgamma`( _input_ , _p_ ) → Tensor

    

与尺寸 P  [HTG11计算多元对数伽玛函数（[ [参考文献]
](https://en.wikipedia.org/wiki/Multivariate_gamma_function)） ] p  p  逐元素，给出通过

log⁡(Γp(a))=C+∑i=1plog⁡(Γ(a−i−12))\log(\Gamma_{p}(a)) = C + \displaystyle
\sum_{i=1}^{p} \log\left(\Gamma\left(a - \frac{i - 1}{2}\right)\right)
log(Γp​(a))=C+i=1∑p​log(Γ(a−2i−1​))

其中 C  =  日志 ⁡ （ π ） × p  （ p  \-  1  ） 4  C = \日志（\ PI） \倍\压裂{对 - （对1）} {4}  C
=  LO  G  （ π ） × 4  P  （ P  \-  1  ） 和 Γ （ ⋅ ） \伽玛（\ CDOT） Γ （ ⋅ ） 是Gamma函数。

如果任何元件都小于或等于 P  \-  1  2  \压裂{对 - 1} {2}  2  p  \-  1  ，然后引发错误。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 张量来计算多变量数伽玛函数

  * **P** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 维数

Example:

    
    
    >>> a = torch.empty(2, 3).uniform_(1, 2)
    >>> a
    tensor([[1.6835, 1.8474, 1.1929],
            [1.0475, 1.7162, 1.4180]])
    >>> torch.mvlgamma(a, 2)
    tensor([[0.3928, 0.4007, 0.7586],
            [1.0311, 0.3901, 0.5049]])
    

`torch.``neg`( _input_ , _out=None_ ) → Tensor

    

返回与负输入的`的元素 `的新的张量。

out=−1×input\text{out} = -1 \times \text{input} out=−1×input

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(5)
    >>> a
    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])
    >>> torch.neg(a)
    tensor([-0.0090,  0.2262,  0.0682,  0.2866, -0.3940])
    

`torch.``pow`()

    

`torch.``pow`( _input_ , _exponent_ , _out=None_ ) → Tensor

    

注意到每个元件的功率在`输入 `与`指数 `，并返回一个张量的结果。

`指数 `可以是一个单一的`浮动 `号码或张量具有相同数量的元素[的HTG10 ] 输入 。

当`指数 `是一个标量值，所施加的操作：

outi=xiexponent\text{out}_i = x_i ^ \text{exponent} outi​=xiexponent​

当`指数 `是一个张量，所施加的操作：

outi=xiexponenti\text{out}_i = x_i ^ {\text{exponent}_i} outi​=xiexponenti​​

当`指数 `是一个张量，`输入 `和`指数 `必须的形状[ broadcastable
](notes/broadcasting.html#broadcasting-semantics)。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **指数** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ _张量_ ） - 指数值

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.4331,  1.2475,  0.6834, -0.2791])
    >>> torch.pow(a, 2)
    tensor([ 0.1875,  1.5561,  0.4670,  0.0779])
    >>> exp = torch.arange(1., 5.)
    
    >>> a = torch.arange(1., 5.)
    >>> a
    tensor([ 1.,  2.,  3.,  4.])
    >>> exp
    tensor([ 1.,  2.,  3.,  4.])
    >>> torch.pow(a, exp)
    tensor([   1.,    4.,   27.,  256.])
    

`torch.``pow`( _self_ , _exponent_ , _out=None_ ) → Tensor

    

`自 `是一个标量`浮动 `值和`指数 `是一个张量。返回的张量`OUT`是相同的形状的作为`指数 `

所施加的操作是：

outi=selfexponenti\text{out}_i = \text{self} ^ {\text{exponent}_i}
outi​=selfexponenti​

Parameters

    

  * 为电源操作的标量基值 - **自** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")）

  * **指数** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 指数张量

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> exp = torch.arange(1., 5.)
    >>> base = 2
    >>> torch.pow(base, exp)
    tensor([  2.,   4.,   8.,  16.])
    

`torch.``reciprocal`( _input_ , _out=None_ ) → Tensor

    

返回一个新的张量`输入 `的元素的倒数

outi=1inputi\text{out}_{i} = \frac{1}{\text{input}_{i}} outi​=inputi​1​

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([-0.4595, -2.1219, -1.4314,  0.7298])
    >>> torch.reciprocal(a)
    tensor([-2.1763, -0.4713, -0.6986,  1.3702])
    

`torch.``remainder`( _input_ , _other_ , _out=None_ ) → Tensor

    

Computes the element-wise remainder of division.

该除数和被除数可以同时包含了整数和浮点数。其余有相同的符号与除数。

When `other`is a tensor, the shapes of `input`and `other`must be
[broadcastable](notes/broadcasting.html#broadcasting-semantics).

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the dividend

  * **其他** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _或_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 其可以是除数数或作为被除数的相同形状的张量

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> torch.remainder(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
    tensor([ 1.,  0.,  1.,  1.,  0.,  1.])
    >>> torch.remainder(torch.tensor([1., 2, 3, 4, 5]), 1.5)
    tensor([ 1.0000,  0.5000,  0.0000,  1.0000,  0.5000])
    

也可以看看

`torch.fmod（） `，它等效计算分割的逐元素其余部分C库函数`FMOD（） `。

`torch.``round`( _input_ , _out=None_ ) → Tensor

    

返回与每个`的元素的一个新的张量 `舍入到最接近的整数的输入。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.9920,  0.6077,  0.9734, -1.0362])
    >>> torch.round(a)
    tensor([ 1.,  1.,  1., -1.])
    

`torch.``rsqrt`( _input_ , _out=None_ ) → Tensor

    

返回与每个输入的`的元素 `的平方根的倒数的新张量。

outi=1inputi\text{out}_{i} = \frac{1}{\sqrt{\text{input}_{i}}}
outi​=inputi​​1​

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([-0.0370,  0.2970,  1.5420, -0.9105])
    >>> torch.rsqrt(a)
    tensor([    nan,  1.8351,  0.8053,     nan])
    

`torch.``sigmoid`( _input_ , _out=None_ ) → Tensor

    

返回与输入的`的元素 `的乙状结肠新张量。

outi=11+e−inputi\text{out}_{i} = \frac{1}{1 + e^{-\text{input}_{i}}}
outi​=1+e−inputi​1​

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.9213,  1.0887, -0.8858, -1.7683])
    >>> torch.sigmoid(a)
    tensor([ 0.7153,  0.7481,  0.2920,  0.1458])
    

`torch.``sign`( _input_ , _out=None_ ) → Tensor

    

返回与输入的`的元素 `的标志新的张量。

outi=sgn⁡(inputi)\text{out}_{i} = \operatorname{sgn}(\text{input}_{i})
outi​=sgn(inputi​)

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.tensor([0.7, -1.2, 0., 2.3])
    >>> a
    tensor([ 0.7000, -1.2000,  0.0000,  2.3000])
    >>> torch.sign(a)
    tensor([ 1., -1.,  0.,  1.])
    

`torch.``sin`( _input_ , _out=None_ ) → Tensor

    

返回与输入的`的元素 `的正弦一个新的张量。

outi=sin⁡(inputi)\text{out}_{i} = \sin(\text{input}_{i}) outi​=sin(inputi​)

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([-0.5461,  0.1347, -2.7266, -0.2746])
    >>> torch.sin(a)
    tensor([-0.5194,  0.1343, -0.4032, -0.2711])
    

`torch.``sinh`( _input_ , _out=None_ ) → Tensor

    

返回与输入的`的元素 `的双曲正弦一个新的张量。

outi=sinh⁡(inputi)\text{out}_{i} = \sinh(\text{input}_{i}) outi​=sinh(inputi​)

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.5380, -0.8632, -0.1265,  0.9399])
    >>> torch.sinh(a)
    tensor([ 0.5644, -0.9744, -0.1268,  1.0845])
    

`torch.``sqrt`( _input_ , _out=None_ ) → Tensor

    

返回与输入的`的元素 `的平方根一个新的张量。

outi=inputi\text{out}_{i} = \sqrt{\text{input}_{i}} outi​=inputi​​

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([-2.0755,  1.0226,  0.0831,  0.4806])
    >>> torch.sqrt(a)
    tensor([    nan,  1.0112,  0.2883,  0.6933])
    

`torch.``tan`( _input_ , _out=None_ ) → Tensor

    

返回与输入的`的元素 `的切线一个新的张量。

outi=tan⁡(inputi)\text{out}_{i} = \tan(\text{input}_{i}) outi​=tan(inputi​)

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([-1.2027, -1.7687,  0.4412, -1.3856])
    >>> torch.tan(a)
    tensor([-2.5930,  4.9859,  0.4722, -5.3366])
    

`torch.``tanh`( _input_ , _out=None_ ) → Tensor

    

返回与输入的`的元素 `的双曲正切一个新的张量。

outi=tanh⁡(inputi)\text{out}_{i} = \tanh(\text{input}_{i}) outi​=tanh(inputi​)

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.8986, -0.7279,  1.1745,  0.2611])
    >>> torch.tanh(a)
    tensor([ 0.7156, -0.6218,  0.8257,  0.2553])
    

`torch.``trunc`( _input_ , _out=None_ ) → Tensor

    

返回与输入的`的元素 `的截尾整数值的新张量。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([ 3.4742,  0.5466, -0.8008, -0.9079])
    >>> torch.trunc(a)
    tensor([ 3.,  0., -0., -0.])
    

### 还原行动

`torch.``argmax`()

    

`torch.``argmax`( _input_ ) → LongTensor

    

返回在`输入 `张量的所有元素的最大值的索引。

这是通过 `torch.max（） `返回的第二值。看到它的文档，这种方法的准确语义。

Parameters

    

**input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input
tensor

Example:

    
    
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [ 0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092,  0.5419, -0.2993,  0.3195]])
    >>> torch.argmax(a)
    tensor(0)
    

`torch.``argmax`( _input_ , _dim_ , _keepdim=False_ ) → LongTensor

    

返回跨尺度张量的最大值的指标。

This is the second value returned by `torch.max()`. See its documentation for
the exact semantics of this method.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的尺寸，以减少。如果`无 `，则返回扁平输入的argmax。

  * **keepdim** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 输出张量是否有`暗淡 `保留或没有。如果`暗淡=无 `忽略。

Example:

    
    
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [ 0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092,  0.5419, -0.2993,  0.3195]])
    >>> torch.argmax(a, dim=1)
    tensor([ 0,  2,  0,  1])
    

`torch.``argmin`()

    

`torch.``argmin`( _input_ ) → LongTensor

    

返回在`输入 `张量的所有元素的最小值的索引。

这是通过 `torch.min返回的第二值（） `。看到它的文档，这种方法的准确语义。

Parameters

    

**input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input
tensor

Example:

    
    
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.1139,  0.2254, -0.1381,  0.3687],
            [ 1.0100, -1.1975, -0.0102, -0.4732],
            [-0.9240,  0.1207, -0.7506, -1.0213],
            [ 1.7809, -1.2960,  0.9384,  0.1438]])
    >>> torch.argmin(a)
    tensor(13)
    

`torch.``argmin`( _input_ , _dim_ , _keepdim=False_ , _out=None_ ) →
LongTensor

    

返回跨尺度张量的最低值的索引。

This is the second value returned by `torch.min()`. See its documentation for
the exact semantics of this method.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的尺寸，以减少。如果`无 `，则返回扁平输入的argmin。

  * **keepdim** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether the output tensors have `dim`retained or not. Ignored if `dim=None`.

Example:

    
    
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.1139,  0.2254, -0.1381,  0.3687],
            [ 1.0100, -1.1975, -0.0102, -0.4732],
            [-0.9240,  0.1207, -0.7506, -1.0213],
            [ 1.7809, -1.2960,  0.9384,  0.1438]])
    >>> torch.argmin(a, dim=1)
    tensor([ 2,  1,  3,  1])
    

`torch.``cumprod`( _input_ , _dim_ , _out=None_ , _dtype=None_ ) → Tensor

    

返回输入 的`元件的累积产物在尺寸`暗淡 `。`

例如，如果`输入 `是大小为N的向量，其结果也将大小为N的向量，包含元素。

yi=x1×x2×x3×⋯×xiy_i = x_1 \times x_2\times x_3\times \dots \times x_i
yi​=x1​×x2​×x3​×⋯×xi​

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的尺寸到超过做手术

  * **DTYPE** （[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")，可选） - 返回的张量的所希望的数据类型。如果指定，输入张量浇铸到`在执行操作之前D型细胞 `。这是为了防止数据溢出型有用。默认值：无。

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(10)
    >>> a
    tensor([ 0.6001,  0.2069, -0.1919,  0.9792,  0.6727,  1.0062,  0.4126,
            -0.2129, -0.4206,  0.1968])
    >>> torch.cumprod(a, dim=0)
    tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0158, -0.0065,
             0.0014, -0.0006, -0.0001])
    
    >>> a[5] = 0.0
    >>> torch.cumprod(a, dim=0)
    tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0000, -0.0000,
             0.0000, -0.0000, -0.0000])
    

`torch.``cumsum`( _input_ , _dim_ , _out=None_ , _dtype=None_ ) → Tensor

    

返回在尺寸输入 的`元素的累积和`暗淡 `。`

For example, if `input`is a vector of size N, the result will also be a
vector of size N, with elements.

yi=x1+x2+x3+⋯+xiy_i = x_1 + x_2 + x_3 + \dots + x_i yi​=x1​+x2​+x3​+⋯+xi​

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – the dimension to do the operation over

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. If specified, the input tensor is casted to `dtype`before the operation is performed. This is useful for preventing data type overflows. Default: None.

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(10)
    >>> a
    tensor([-0.8286, -0.4890,  0.5155,  0.8443,  0.1865, -0.1752, -2.0595,
             0.1850, -1.1571, -0.4243])
    >>> torch.cumsum(a, dim=0)
    tensor([-0.8286, -1.3175, -0.8020,  0.0423,  0.2289,  0.0537, -2.0058,
            -1.8209, -2.9780, -3.4022])
    

`torch.``dist`( _input_ , _other_ , _p=2_ ) → Tensor

    

返回的p范数（`输入 `\- `其他 `）

The shapes of `input`and `other`must be
[broadcastable](notes/broadcasting.html#broadcasting-semantics).

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **其他** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的右手侧的输入张量

  * **P** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 范数被计算

Example:

    
    
    >>> x = torch.randn(4)
    >>> x
    tensor([-1.5393, -0.8675,  0.5916,  1.6321])
    >>> y = torch.randn(4)
    >>> y
    tensor([ 0.0967, -1.0511,  0.6295,  0.8360])
    >>> torch.dist(x, y, 3.5)
    tensor(1.6727)
    >>> torch.dist(x, y, 3)
    tensor(1.6973)
    >>> torch.dist(x, y, 0)
    tensor(inf)
    >>> torch.dist(x, y, 1)
    tensor(2.6537)
    

`torch.``logsumexp`( _input_ , _dim_ , _keepdim=False_ , _out=None_ )

    

返回`输入 `张量中的每一行的求和指数的日志中的给定维度`暗淡 `。计算是数值上稳定。

用于求和指数 [HTG6：J  [HTG9：J  [HTG18：J  由暗淡和给定的其他指数 i的 i的 i的 ，其结果是

> logsumexp  （ × ） i的 =  日志 ⁡ Σ [HTG29：J  实验值 ⁡ （ × i的 f]  ）
\文本{logsumexp}（x）的_ {I} = \ LOG \ sum_j \ EXP（X_ {IJ}） logsumexp  （ × ） i的
[HTG10 0] =  LO  G  [HTG122：J  Σ EXP  （ × i的 [HTG166：J  ）

如果`keepdim`是`真 `，输出张量是相同的大小为`输入 `除了在尺寸（s）实施HTG12] 暗淡 其中它是尺寸1的否则，`暗淡
`被挤出（见 `torch.squeeze（） `），导致具有1的输出张量（或`LEN（暗） `）较少的维（S）。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ _蟒的元组：整数_ ） - 的尺寸或尺寸，以减少

  * **keepdim** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 输出张量是否有`暗淡 `保留或不

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example::

    
    
    
    >>> a = torch.randn(3, 3)
    >>> torch.logsumexp(a, 1)
    tensor([ 0.8442,  1.4322,  0.8711])
    

`torch.``mean`()

    

`torch.``mean`( _input_ ) → Tensor

    

返回所有元素的在`输入 `张量的平均值。

Parameters

    

**input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input
tensor

Example:

    
    
    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.2294, -0.5481,  1.3288]])
    >>> torch.mean(a)
    tensor(0.3367)
    

`torch.``mean`( _input_ , _dim_ , _keepdim=False_ , _out=None_ ) → Tensor

    

返回在给定维度的`输入 `张量中的每一行的平均值`暗淡 `。如果`暗淡 `为维度的列表，减少过度所有的人。

If `keepdim`is `True`, the output tensor is of the same size as `input`
except in the dimension(s) `dim`where it is of size 1. Otherwise, `dim`is
squeezed (see `torch.squeeze()`), resulting in the output tensor having 1 (or
`len(dim)`) fewer dimension(s).

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_ _tuple of python:ints_ ) – the dimension or dimensions to reduce

  * **keepdim** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether the output tensor has `dim`retained or not

  * **OUT** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输出张量

Example:

    
    
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-0.3841,  0.6320,  0.4254, -0.7384],
            [-0.9644,  1.0131, -0.6549, -1.4279],
            [-0.2951, -1.3350, -0.7694,  0.5600],
            [ 1.0842, -0.9580,  0.3623,  0.2343]])
    >>> torch.mean(a, 1)
    tensor([-0.0163, -0.5085, -0.4599,  0.1807])
    >>> torch.mean(a, 1, True)
    tensor([[-0.0163],
            [-0.5085],
            [-0.4599],
            [ 0.1807]])
    

`torch.``median`()

    

`torch.``median`( _input_ ) → Tensor

    

返回所有元素的在`输入 `张量的中值。

Parameters

    

**input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input
tensor

Example:

    
    
    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 1.5219, -1.5212,  0.2202]])
    >>> torch.median(a)
    tensor(0.2202)
    

`torch.``median`( _input_ , _dim=-1_ , _keepdim=False_ , _values=None_ ,
_indices=None) - > (Tensor_, _LongTensor_ )

    

返回namedtuple `（值 索引） `其中`值 `为各行的中值的`输入 `张量在给定的尺寸`暗淡 `。和`指数
`是找到的每个中间值的索引位置。

默认情况下，`暗淡 `为`输入 `张量的最后维度。

如果`keepdim`是`真 `，输出张量是相同大小的作为`输入 `除了在尺寸`暗淡 `其中它们是尺寸1的否则，`暗淡 `被挤出（见 `
torch.squeeze（） `），导致具有比1种`输入 `更少尺寸的输出张量。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的尺寸，以减少

  * **keepdim** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 输出张量是否有`暗淡 `保留或不

  * **值** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 输出张量

  * **指数** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 输出索引张量

Example:

    
    
    >>> a = torch.randn(4, 5)
    >>> a
    tensor([[ 0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
            [ 0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
            [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
            [ 1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
    >>> torch.median(a, 1)
    torch.return_types.median(values=tensor([-0.3982,  0.2270,  0.2488,  0.4742]), indices=tensor([1, 4, 4, 3]))
    

`torch.``mode`( _input_ , _dim=-1_ , _keepdim=False_ , _values=None_ ,
_indices=None) - > (Tensor_, _LongTensor_ )

    

返回namedtuple `（值 索引） `其中`值 `是每行的模式值的`输入 `张量在给定的尺寸`暗淡 `，即，最经常出现在该行中的值，并`索引
`是找到的每个模式值的索引位置。

By default, `dim`is the last dimension of the `input`tensor.

如果`keepdim`是`真 `，输出张量是相同大小的作为`输入 `除了在尺寸`暗淡 `其中它们是尺寸1的否则，`暗淡 `被挤出（见 `
torch.squeeze（） `），导致具有比1种`输入 `更少尺寸的输出张量。

Note

此功能还没有为`torch.cuda.Tensor`中定义爱好。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – the dimension to reduce

  * **keepdim** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether the output tensors have `dim`retained or not

  * **values** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

  * **indices** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output index tensor

Example:

    
    
    >>> a = torch.randint(10, (5,))
    >>> a
    tensor([6, 5, 1, 0, 2])
    >>> b = a + (torch.randn(50, 1) * 5).long()
    >>> torch.mode(b, 0)
    torch.return_types.mode(values=tensor([6, 5, 1, 0, 2]), indices=tensor([2, 2, 2, 2, 2]))
    

`torch.``norm`( _input_ , _p='fro'_ , _dim=None_ , _keepdim=False_ ,
_out=None_ , _dtype=None_ )[[source]](_modules/torch/functional.html#norm)

    

返回给定张量的矩阵范数或向量范数。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **P** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _INF_ _，_ _-INF_ _，_ _'来回'_ _，_ _'NUC'_ _，_ _可选_ ） - 

规范的秩序。默认值：`'来回' `可以计算以下规范：

ORD

|

矩阵范

|

向量模  
  
---|---|---  
  
没有

|

弗洛比尼范数

|

2范  
  
“回回”

|

Frobenius norm

|

\-  
  
“国统会”

|

核标准

|

–  
  
其他

|

作为VEC规范时，昏暗的是无

|

总和（ABS（X）** ORD）**（1./ord）  
  
  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _蟒的2元组：整型_ _，_ _蟒2-列表：整数_ _，_ _可选_ ） - 如果它是一个int，矢量范数将被计算，如果是整数的2元组，矩阵范数将被计算。如果该值是无，当输入仅张量具有两个维度矩阵范数将被计算，当输入张量只有一个维度矢量范数将被计算。如果输入张量具有多于两个尺寸，矢量范数将被应用到最后尺寸。

  * **keepdim** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 输出张量是否有`暗淡 `保留或没有。忽略如果`暗淡 `= `无 `和`OUT`= `无 `。默认值：`假 `

  * **OUT** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 输出张量。忽略如果`暗淡 `= `无 `和`OUT`= `无 `。

  * **DTYPE** （[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")，可选） - 返回的张量的所希望的数据类型。执行操作“D型”：如果指定，输入张量浇铸到：ATTR。默认值：无。

Example:

    
    
    >>> import torch
    >>> a = torch.arange(9, dtype= torch.float) - 4
    >>> b = a.reshape((3, 3))
    >>> torch.norm(a)
    tensor(7.7460)
    >>> torch.norm(b)
    tensor(7.7460)
    >>> torch.norm(a, float('inf'))
    tensor(4.)
    >>> torch.norm(b, float('inf'))
    tensor(4.)
    >>> c = torch.tensor([[ 1, 2, 3],[-1, 1, 4]] , dtype= torch.float)
    >>> torch.norm(c, dim=0)
    tensor([1.4142, 2.2361, 5.0000])
    >>> torch.norm(c, dim=1)
    tensor([3.7417, 4.2426])
    >>> torch.norm(c, p=1, dim=1)
    tensor([6., 6.])
    >>> d = torch.arange(8, dtype= torch.float).reshape(2,2,2)
    >>> torch.norm(d, dim=(1,2))
    tensor([ 3.7417, 11.2250])
    >>> torch.norm(d[0, :, :]), torch.norm(d[1, :, :])
    (tensor(3.7417), tensor(11.2250))
    

`torch.``prod`()

    

`torch.``prod`( _input_ , _dtype=None_ ) → Tensor

    

返回在`输入 `张量的所有元素的乘积。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. If specified, the input tensor is casted to `dtype`before the operation is performed. This is useful for preventing data type overflows. Default: None.

Example:

    
    
    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[-0.8020,  0.5428, -1.5854]])
    >>> torch.prod(a)
    tensor(0.6902)
    

`torch.``prod`( _input_ , _dim_ , _keepdim=False_ , _dtype=None_ ) → Tensor

    

返回`输入 `张量的各行的产品在给定的尺寸`暗淡 `。

如果`keepdim`是`真 `，输出张量是相同的大小为`输入 `除了在尺寸`暗淡 `其中它是尺寸1的否则，`暗淡 `被挤出（见 `
torch.squeeze（） `），导致具有比1种`输入 `更少尺寸的输出张量。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – the dimension to reduce

  * **keepdim** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether the output tensor has `dim`retained or not

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. If specified, the input tensor is casted to `dtype`before the operation is performed. This is useful for preventing data type overflows. Default: None.

Example:

    
    
    >>> a = torch.randn(4, 2)
    >>> a
    tensor([[ 0.5261, -0.3837],
            [ 1.1857, -0.2498],
            [-1.1646,  0.0705],
            [ 1.1131, -1.0629]])
    >>> torch.prod(a, 1)
    tensor([-0.2018, -0.2962, -0.0821, -1.1831])
    

`torch.``std`()

    

`torch.``std`( _input_ , _unbiased=True_ ) → Tensor

    

返回在`输入 `张量的所有元素的标准偏差。

如果`无偏 `是`假 `，则标准偏差将被经由偏估计计算。否则，贝塞尔修正将被使用。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **无偏** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 是否使用无偏估计或不

Example:

    
    
    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[-0.8166, -1.3802, -0.3560]])
    >>> torch.std(a)
    tensor(0.5130)
    

`torch.``std`( _input_ , _dim_ , _keepdim=False_ , _unbiased=True_ ,
_out=None_ ) → Tensor

    

返回维度中的`输入 `张量的各行的标准偏差`暗淡 `。如果`暗淡 `为维度的列表，减少过度所有的人。

If `keepdim`is `True`, the output tensor is of the same size as `input`
except in the dimension(s) `dim`where it is of size 1. Otherwise, `dim`is
squeezed (see `torch.squeeze()`), resulting in the output tensor having 1 (or
`len(dim)`) fewer dimension(s).

If `unbiased`is `False`, then the standard-deviation will be calculated via
the biased estimator. Otherwise, Bessel’s correction will be used.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_ _tuple of python:ints_ ) – the dimension or dimensions to reduce

  * **keepdim** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether the output tensor has `dim`retained or not

  * **unbiased** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether to use the unbiased estimation or not

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.2035,  1.2959,  1.8101, -0.4644],
            [ 1.5027, -0.3270,  0.5905,  0.6538],
            [-1.5745,  1.3330, -0.5596, -0.6548],
            [ 0.1264, -0.5080,  1.6420,  0.1992]])
    >>> torch.std(a, dim=1)
    tensor([ 1.0311,  0.7477,  1.2204,  0.9087])
    

`torch.``std_mean`()

    

`torch.``std_mean`( _input_ , _unbiased=True) - > (Tensor_, _Tensor_ )

    

返回在`输入 `张量的所有元素的标准偏差和平均值。

If `unbiased`is `False`, then the standard-deviation will be calculated via
the biased estimator. Otherwise, Bessel’s correction will be used.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **unbiased** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether to use the unbiased estimation or not

Example:

    
    
    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[0.3364, 0.3591, 0.9462]])
    >>> torch.std_mean(a)
    (tensor(0.3457), tensor(0.5472))
    

`torch.``std`( _input_ , _dim_ , _keepdim=False_ , _unbiased=True) - >
(Tensor_, _Tensor_ )

    

返回维度中的`输入 `张量的各行的标准偏差和平均值`暗淡 `。如果`暗淡 `为维度的列表，减少过度所有的人。

If `keepdim`is `True`, the output tensor is of the same size as `input`
except in the dimension(s) `dim`where it is of size 1. Otherwise, `dim`is
squeezed (see `torch.squeeze()`), resulting in the output tensor having 1 (or
`len(dim)`) fewer dimension(s).

If `unbiased`is `False`, then the standard-deviation will be calculated via
the biased estimator. Otherwise, Bessel’s correction will be used.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_ _tuple of python:ints_ ) – the dimension or dimensions to reduce

  * **keepdim** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether the output tensor has `dim`retained or not

  * **unbiased** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether to use the unbiased estimation or not

Example:

    
    
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.5648, -0.5984, -1.2676, -1.4471],
            [ 0.9267,  1.0612,  1.1050, -0.6014],
            [ 0.0154,  1.9301,  0.0125, -1.0904],
            [-1.9711, -0.7748, -1.3840,  0.5067]])
    >>> torch.std_mean(a, 1)
    (tensor([0.9110, 0.8197, 1.2552, 1.0608]), tensor([-0.6871,  0.6229,  0.2169, -0.9058]))
    

`torch.``sum`()

    

`torch.``sum`( _input_ , _dtype=None_ ) → Tensor

    

返回在`输入 `张量的所有元素的总和。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. If specified, the input tensor is casted to `dtype`before the operation is performed. This is useful for preventing data type overflows. Default: None.

Example:

    
    
    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.1133, -0.9567,  0.2958]])
    >>> torch.sum(a)
    tensor(-0.5475)
    

`torch.``sum`( _input_ , _dim_ , _keepdim=False_ , _dtype=None_ ) → Tensor

    

返回在给定维度的`输入 `张量的每一行的总和`暗淡 `。如果`暗淡 `为维度的列表，减少过度所有的人。

If `keepdim`is `True`, the output tensor is of the same size as `input`
except in the dimension(s) `dim`where it is of size 1. Otherwise, `dim`is
squeezed (see `torch.squeeze()`), resulting in the output tensor having 1 (or
`len(dim)`) fewer dimension(s).

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_ _tuple of python:ints_ ) – the dimension or dimensions to reduce

  * **keepdim** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether the output tensor has `dim`retained or not

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. If specified, the input tensor is casted to `dtype`before the operation is performed. This is useful for preventing data type overflows. Default: None.

Example:

    
    
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
            [-0.2993,  0.9138,  0.9337, -1.6864],
            [ 0.1132,  0.7892, -0.1003,  0.5688],
            [ 0.3637, -0.9906, -0.4752, -1.5197]])
    >>> torch.sum(a, 1)
    tensor([-0.4598, -0.1381,  1.3708, -2.6217])
    >>> b = torch.arange(4 * 5 * 6).view(4, 5, 6)
    >>> torch.sum(b, (2, 1))
    tensor([  435.,  1335.,  2235.,  3135.])
    

`torch.``unique`( _input_ , _sorted=True_ , _return_inverse=False_ ,
_return_counts=False_ , _dim=None_
)[[source]](_modules/torch/functional.html#unique)

    

返回输入张量的独特元素。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **排序** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 是否返回作为输出之前按升序对独特的元素进行排序。

  * **return_inverse** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 是否也返回如在原有的输入元素在返回的唯一列表结束了索引。

  * **return_counts** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 是否也返回的计数为每个唯一的元件。

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 维度应用是唯一的。如果`无 `，独特的扁平输入的被返回。默认值：`无 `

Returns

    

的张量或张量的含有一个元组

>   * **输出** （ _张量_ ）：唯一的标量元件的输出列表。

>

>   * **inverse_indices** （ _张量_ ）：（可选）如果`return_inverse
`为True，将有一个附加的返回张量（相同的形状作为输入）表示用于其中在原始输入地图元素到输出索引;否则，该函数将只返回单个张量。

>

>   * **计数** （ _张量_ ）：（可选）如果`return_counts
`为True，将有一个附加的返回张量（相同的形状的输出或output.size（DIM），如果调光被指定）代表出现的每个唯一值，或张量的数目。

>

>

Return type

    

（[张量](tensors.html#torch.Tensor "torch.Tensor")，[张量](tensors.html#torch.Tensor
"torch.Tensor")（可选），[张量](tensors.html#torch.Tensor "torch.Tensor")（可选））

Example:

    
    
    >>> output = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long))
    >>> output
    tensor([ 2,  3,  1])
    
    >>> output, inverse_indices = torch.unique(
            torch.tensor([1, 3, 2, 3], dtype=torch.long), sorted=True, return_inverse=True)
    >>> output
    tensor([ 1,  2,  3])
    >>> inverse_indices
    tensor([ 0,  2,  1,  2])
    
    >>> output, inverse_indices = torch.unique(
            torch.tensor([[1, 3], [2, 3]], dtype=torch.long), sorted=True, return_inverse=True)
    >>> output
    tensor([ 1,  2,  3])
    >>> inverse_indices
    tensor([[ 0,  2],
            [ 1,  2]])
    

`torch.``unique_consecutive`( _input_ , _return_inverse=False_ ,
_return_counts=False_ , _dim=None_
)[[source]](_modules/torch/functional.html#unique_consecutive)

    

消除了所有的但等效从元件的每个连续组的第一个元素。

Note

此功能是由不同`torch.unique（） `在这个意义上，此功能仅消除连续重复的值。这个语义类似于的std ::用C独特的 ++。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **return_inverse** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – Whether to also return the indices for where elements in the original input ended up in the returned unique list.

  * **return_counts** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – Whether to also return the counts for each unique element.

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – the dimension to apply unique. If `None`, the unique of the flattened input is returned. default: `None`

Returns

    

A tensor or a tuple of tensors containing

>   * **output** ( _Tensor_ ): the output list of unique scalar elements.

>

>   * **inverse_indices** ( _Tensor_ ): (optional) if `return_inverse`is
True, there will be an additional returned tensor (same shape as input)
representing the indices for where elements in the original input map to in
the output; otherwise, this function will only return a single tensor.

>

>   * **counts** ( _Tensor_ ): (optional) if `return_counts`is True, there
will be an additional returned tensor (same shape as output or
output.size(dim), if dim was specified) representing the number of occurrences
for each unique value or tensor.

>

>

Return type

    

([Tensor](tensors.html#torch.Tensor "torch.Tensor"),
[Tensor](tensors.html#torch.Tensor "torch.Tensor") (optional),
[Tensor](tensors.html#torch.Tensor "torch.Tensor") (optional))

Example:

    
    
    >>> x = torch.tensor([1, 1, 2, 2, 3, 1, 1, 2])
    >>> output = torch.unique_consecutive(x)
    >>> output
    tensor([1, 2, 3, 1, 2])
    
    >>> output, inverse_indices = torch.unique_consecutive(x, return_inverse=True)
    >>> output
    tensor([1, 2, 3, 1, 2])
    >>> inverse_indices
    tensor([0, 0, 1, 1, 2, 3, 3, 4])
    
    >>> output, counts = torch.unique_consecutive(x, return_counts=True)
    >>> output
    tensor([1, 2, 3, 1, 2])
    >>> counts
    tensor([2, 2, 1, 2, 1])
    

`torch.``var`()

    

`torch.``var`( _input_ , _unbiased=True_ ) → Tensor

    

返回所有元素的在`输入 `张量的方差。

如果`无偏 `是`假 `，然后方差将经由偏估计计算。否则，贝塞尔修正将被使用。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **unbiased** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether to use the unbiased estimation or not

Example:

    
    
    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[-0.3425, -1.2636, -0.4864]])
    >>> torch.var(a)
    tensor(0.2455)
    

`torch.``var`( _input_ , _dim_ , _keepdim=False_ , _unbiased=True_ ,
_out=None_ ) → Tensor

    

返回`输入 `张量中的每一行的方差在给定的尺寸`暗淡 `。

If `keepdim`is `True`, the output tensor is of the same size as `input`
except in the dimension(s) `dim`where it is of size 1. Otherwise, `dim`is
squeezed (see `torch.squeeze()`), resulting in the output tensor having 1 (or
`len(dim)`) fewer dimension(s).

If `unbiased`is `False`, then the variance will be calculated via the biased
estimator. Otherwise, Bessel’s correction will be used.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_ _tuple of python:ints_ ) – the dimension or dimensions to reduce

  * **keepdim** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether the output tensor has `dim`retained or not

  * **unbiased** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether to use the unbiased estimation or not

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-0.3567,  1.7385, -1.3042,  0.7423],
            [ 1.3436, -0.1015, -0.9834, -0.8438],
            [ 0.6056,  0.1089, -0.3112, -1.4085],
            [-0.7700,  0.6074, -0.1469,  0.7777]])
    >>> torch.var(a, 1)
    tensor([ 1.7444,  1.1363,  0.7356,  0.5112])
    

`torch.``var_mean`()

    

`torch.``var_mean`( _input_ , _unbiased=True) - > (Tensor_, _Tensor_ )

    

返回所有元素的在`输入 `张量的方差和平均值。

If `unbiased`is `False`, then the variance will be calculated via the biased
estimator. Otherwise, Bessel’s correction will be used.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **unbiased** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether to use the unbiased estimation or not

Example:

    
    
    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[0.0146, 0.4258, 0.2211]])
    >>> torch.var_mean(a)
    (tensor(0.0423), tensor(0.2205))
    

`torch.``var_mean`( _input_ , _dim_ , _keepdim=False_ , _unbiased=True) - >
(Tensor_, _Tensor_ )

    

返回`输入 `张量中的每一行的方差和平均值在给定的尺寸`暗淡 `。

If `keepdim`is `True`, the output tensor is of the same size as `input`
except in the dimension(s) `dim`where it is of size 1. Otherwise, `dim`is
squeezed (see `torch.squeeze()`), resulting in the output tensor having 1 (or
`len(dim)`) fewer dimension(s).

If `unbiased`is `False`, then the variance will be calculated via the biased
estimator. Otherwise, Bessel’s correction will be used.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_ _tuple of python:ints_ ) – the dimension or dimensions to reduce

  * **keepdim** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether the output tensor has `dim`retained or not

  * **unbiased** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether to use the unbiased estimation or not

Example:

    
    
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-1.5650,  2.0415, -0.1024, -0.5790],
            [ 0.2325, -2.6145, -1.6428, -0.3537],
            [-0.2159, -1.1069,  1.2882, -1.3265],
            [-0.6706, -1.5893,  0.6827,  1.6727]])
    >>> torch.var_mean(a, 1)
    (tensor([2.3174, 1.6403, 1.4092, 2.0791]), tensor([-0.0512, -1.0946, -0.3403,  0.0239]))
    

### 比较行动

`torch.``allclose`( _input_ , _other_ , _rtol=1e-05_ , _atol=1e-08_ ,
_equal_nan=False_ ) → bool

    

此功能检查是否所有`输入 `和`其他 `满足条件：

∣input−other∣≤atol+rtol×∣other∣\lvert \text{input} - \text{other} \rvert \leq
\texttt{atol} + \texttt{rtol} \times \lvert \text{other} \rvert
∣input−other∣≤atol+rtol×∣other∣

的elementwise，对于 和输入的`所有元素`其它 `。此函数的行为类似于[ numpy.allclose
](https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html)`

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 第一伸张器，以比较

  * **其他** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 第二伸张器，以比较

  * **蒂** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 绝对公差。默认值：1E-08

  * **RTOL** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 相对公差。默认值：1E-05

  * **equal_nan** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，然后2 `的NaN`S将被相等比较。默认值：`假 `

Example:

    
    
    >>> torch.allclose(torch.tensor([10000., 1e-07]), torch.tensor([10000.1, 1e-08]))
    False
    >>> torch.allclose(torch.tensor([10000., 1e-08]), torch.tensor([10000.1, 1e-09]))
    True
    >>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]))
    False
    >>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]), equal_nan=True)
    True
    

`torch.``argsort`( _input_ , _dim=-1_ , _descending=False_ , _out=None_ ) →
LongTensor

    

返回由值升序排列沿给定尺寸的张量的指数。

这是通过 `torch.sort（） `返回的第二值。看到它的文档，这种方法的准确语义。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 的尺寸进行排序沿

  * **降序** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 控制所述排序顺序（升序或降序）

Example:

    
    
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.0785,  1.5267, -0.8521,  0.4065],
            [ 0.1598,  0.0788, -0.0745, -1.2700],
            [ 1.2208,  1.0722, -0.7064,  1.2564],
            [ 0.0669, -0.2318, -0.8229, -0.9280]])
    
    
    >>> torch.argsort(a, dim=1)
    tensor([[2, 0, 3, 1],
            [3, 2, 1, 0],
            [2, 1, 0, 3],
            [3, 2, 1, 0]])
    

`torch.``eq`( _input_ , _other_ , _out=None_ ) → Tensor

    

计算元素方面的平等

第二个参数可以是一个数字或一个张量，其形状为[ broadcastable  ](notes/broadcasting.html#broadcasting-
semantics)的第一个参数。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 张量来比较

  * **其他** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _或_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 张量或值进行比较

  * **OUT** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 输出张量。必须是 BoolTensor 

Returns

    

A `torch.BoolTensor  [HTG3含有一个True在每个位置处，其中比较结果为真`

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[True, False], [False, True]])
    

`torch.``equal`( _input_ , _other_ ) → bool

    

`真 `如果两个张量具有相同的尺寸和元素，`假 `否则。

Example:

    
    
    >>> torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2]))
    True
    

`torch.``ge`( _input_ , _other_ , _out=None_ ) → Tensor

    

计算 输入 ≥ 其他 \文本{输入} \ GEQ \文本{其它}  输入 ≥ 其他 逐元素。

The second argument can be a number or a tensor whose shape is
[broadcastable](notes/broadcasting.html#broadcasting-semantics) with the first
argument.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor to compare

  * **other** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_[ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – the tensor or value to compare

  * **OUT** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 输出张量必须是一个 BoolTensor 

Returns

    

A `torch.BoolTensor`containing a True at each location where comparison is
true

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> torch.ge(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[True, True], [False, True]])
    

`torch.``gt`( _input_ , _other_ , _out=None_ ) → Tensor

    

计算 输入 & GT ;  其他 \ {文本输入} & GT ; \ {文本其他}  输入 & GT ;  其他 逐元素。

The second argument can be a number or a tensor whose shape is
[broadcastable](notes/broadcasting.html#broadcasting-semantics) with the first
argument.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor to compare

  * **other** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_[ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – the tensor or value to compare

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor that must be a BoolTensor

Returns

    

A `torch.BoolTensor`containing a True at each location where comparison is
true

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> torch.gt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[False, True], [False, False]])
    

`torch.``isfinite`( _tensor_
)[[source]](_modules/torch/functional.html#isfinite)

    

返回与表示如果每个元素是有限或不布尔元素的新张量。

Parameters

    

**张量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 甲张量，以检查

Returns

    

`A  torch.Tensor  与 DTYPE  torch.bool  [HTG11含有一个True在每个有限元素和假的位置，否则`

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> torch.isfinite(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
    tensor([True,  False,  True,  False,  False])
    

`torch.``isinf`( _tensor_ )[[source]](_modules/torch/functional.html#isinf)

    

返回与表示如果每个元素是 +/- INF 或不布尔元素的新张量。

Parameters

    

**tensor** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – A tensor
to check

Returns

    

`A  torch.Tensor  与 DTYPE  torch.bool  [HTG11含有一个True在每个 +/- INF 元素和假的位置，否则`

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> torch.isinf(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
    tensor([False,  True,  False,  True,  False])
    

`torch.``isnan`()

    

返回与表示如果每个元素是的NaN 或不布尔元素的新张量。

Parameters

    

**输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 甲张量，以检查

Returns

    

A `torch.BoolTensor`含有在真NaN的元素每个位置。

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> torch.isnan(torch.tensor([1, float('nan'), 2]))
    tensor([False, True, False])
    

`torch.``kthvalue`( _input_ , _k_ , _dim=None_ , _keepdim=False_ , _out=None)
- > (Tensor_, _LongTensor_ )

    

返回一个namedtuple `（值 索引） `其中`值 `在`K`在给定的维度上的`输入 `张量中的每一行的第最小元素`暗淡 `。和`指数
`是找到的每个元素的索引位置。

如果`暗淡 `没有给出，则输入的最后尺寸被选择。

如果`keepdim`是`真 `，无论是`值 `和`指数 `张量是相同的大小为`输入 `，除了在尺寸`暗淡 `其中它们的尺寸1，否则`暗淡
`被挤出（见 `torch.squeeze（） `），从而导致在两个`值 `和`指数 `具有1名比`输入 `张量较少维张量。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **K** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - K为第k个最小的元素

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 的尺寸沿着找到的第k个值

  * **keepdim** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether the output tensors have `dim`retained or not

  * **OUT** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选_ ） - （张量，LongTensor）的输出元组可任选地给定的以用作输出缓冲器

Example:

    
    
    >>> x = torch.arange(1., 6.)
    >>> x
    tensor([ 1.,  2.,  3.,  4.,  5.])
    >>> torch.kthvalue(x, 4)
    torch.return_types.kthvalue(values=tensor(4.), indices=tensor(3))
    
    >>> x=torch.arange(1.,7.).resize_(2,3)
    >>> x
    tensor([[ 1.,  2.,  3.],
            [ 4.,  5.,  6.]])
    >>> torch.kthvalue(x, 2, 0, True)
    torch.return_types.kthvalue(values=tensor([[4., 5., 6.]]), indices=tensor([[1, 1, 1]]))
    

`torch.``le`( _input_ , _other_ , _out=None_ ) → Tensor

    

计算 输入 ≤ 其他 \文本{输入} \当量\文本{其它}  输入 ≤ 其他 逐元素。

The second argument can be a number or a tensor whose shape is
[broadcastable](notes/broadcasting.html#broadcasting-semantics) with the first
argument.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor to compare

  * **other** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_[ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – the tensor or value to compare

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor that must be a BoolTensor

Returns

    

A `torch.BoolTensor`containing a True at each location where comparison is
true

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> torch.le(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[True, False], [True, True]])
    

`torch.``lt`( _input_ , _other_ , _out=None_ ) → Tensor

    

计算 输入 & LT ;  其他 \ {文本输入} & LT ; \ {文本其他}  输入 & LT ;  其他 逐元素。

The second argument can be a number or a tensor whose shape is
[broadcastable](notes/broadcasting.html#broadcasting-semantics) with the first
argument.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor to compare

  * **other** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_[ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – the tensor or value to compare

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor that must be a BoolTensor

Returns

    

A  torch.BoolTensor [HTG1含有一个True在每个位置处，其中比较结果为真

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> torch.lt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[False, False], [True, False]])
    

`torch.``max`()

    

`torch.``max`( _input_ ) → Tensor

    

返回所有元素的在`输入 `张量的最大值。

Parameters

    

**input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input
tensor

Example:

    
    
    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.6763,  0.7445, -2.2369]])
    >>> torch.max(a)
    tensor(0.7445)
    

`torch.``max`( _input_ , _dim_ , _keepdim=False_ , _out=None) - > (Tensor_,
_LongTensor_ )

    

返回namedtuple `（值 索引） `其中`值 `是每行的最大值`输入 `张量在给定的尺寸`暗淡 `。和`指数
`是找到的每个最大值的索引位置（argmax）。

如果`keepdim`是`真 `，输出张量是相同大小的作为`输入 `除了在尺寸`暗淡 `其中它们是尺寸1的否则，`暗淡 `被挤出（见 `
torch.squeeze（） `），导致具有比1种`输入 `更少尺寸的输出张量。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – the dimension to reduce

  * **keepdim** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 输出张量是否有`暗淡 `保留或没有。默认值：`假 [HTG17。`

  * **OUT** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选_ ） - 两个输出张量的结果元组（最大，max_indices）

Example:

    
    
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
            [ 1.1949, -1.1127, -2.2379, -0.6702],
            [ 1.5717, -0.9207,  0.1297, -1.8768],
            [-0.6172,  1.0036, -0.6060, -0.2432]])
    >>> torch.max(a, 1)
    torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))
    

`torch.``max`( _input_ , _other_ , _out=None_ ) → Tensor

    

张量`输入 `的每个元素与张力`其他 `和逐元素最大取的对应元素进行比较。

`输入 `和`等 `不需要匹配，但是他们必须[ broadcastable [HTG10的形状]
](notes/broadcasting.html#broadcasting-semantics)。

outi=max⁡(tensori,otheri)\text{out}_i = \max(\text{tensor}_i, \text{other}_i)
outi​=max(tensori​,otheri​)

Note

当形状不匹配，则返回的输出张量的形状遵循[ 广播规则 ](notes/broadcasting.html#broadcasting-semantics)。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **other** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the second input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.2942, -0.7416,  0.2653, -0.1584])
    >>> b = torch.randn(4)
    >>> b
    tensor([ 0.8722, -1.7421, -0.4141, -0.5055])
    >>> torch.max(a, b)
    tensor([ 0.8722, -0.7416,  0.2653, -0.1584])
    

`torch.``min`()

    

`torch.``min`( _input_ ) → Tensor

    

返回所有元素的在`输入 `张量的最小值。

Parameters

    

**input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input
tensor

Example:

    
    
    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.6750,  1.0857,  1.7197]])
    >>> torch.min(a)
    tensor(0.6750)
    

`torch.``min`( _input_ , _dim_ , _keepdim=False_ , _out=None) - > (Tensor_,
_LongTensor_ )

    

返回namedtuple `（值 索引） `其中`值 `是每行的最小值`输入 `张量在给定的尺寸`暗淡 `。和`指数
`是发现（argmin）各最小值的索引位置。

如果`keepdim`是`真 `，输出张量是相同大小的作为`输入 `除了在尺寸`暗淡 `其中它们是尺寸1的否则，`暗淡 `被挤出（见 `
torch.squeeze（） `），导致具有比1种`输入 `更少尺寸的输出张量。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – the dimension to reduce

  * **keepdim** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – whether the output tensors have `dim`retained or not

  * **OUT** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选_ ） - 两个输出张量的元组（分钟，min_indices）

Example:

    
    
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-0.6248,  1.1334, -1.1899, -0.2803],
            [-1.4644, -0.2635, -0.3651,  0.6134],
            [ 0.2457,  0.0384,  1.0128,  0.7015],
            [-0.1153,  2.9849,  2.1458,  0.5788]])
    >>> torch.min(a, 1)
    torch.return_types.min(values=tensor([-1.1899, -1.4644,  0.0384, -0.1153]), indices=tensor([2, 0, 1, 0]))
    

`torch.``min`( _input_ , _other_ , _out=None_ ) → Tensor

    

张量`输入 `的每个元素与张力`其他 `和被取逐元素的最小的对应元素进行比较。得到的张量返回。

The shapes of `input`and `other`don’t need to match, but they must be
[broadcastable](notes/broadcasting.html#broadcasting-semantics).

outi=min⁡(tensori,otheri)\text{out}_i = \min(\text{tensor}_i, \text{other}_i)
outi​=min(tensori​,otheri​)

Note

When the shapes do not match, the shape of the returned output tensor follows
the [broadcasting rules](notes/broadcasting.html#broadcasting-semantics).

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **other** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the second input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.8137, -1.1740, -0.6460,  0.6308])
    >>> b = torch.randn(4)
    >>> b
    tensor([-0.1369,  0.1555,  0.4019, -0.1929])
    >>> torch.min(a, b)
    tensor([-0.1369, -1.1740, -0.6460, -0.1929])
    

`torch.``ne`( _input_ , _other_ , _out=None_ ) → Tensor

    

计算 i的 n的 P  U  T  ≠ O  T  H  E  R  输入\ NEQ其他 i的 n的 p  U  T   =  O  T  H  E  R
[HT G105]逐元素。

The second argument can be a number or a tensor whose shape is
[broadcastable](notes/broadcasting.html#broadcasting-semantics) with the first
argument.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor to compare

  * **other** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _or_[ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – the tensor or value to compare

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor that must be a BoolTensor

Returns

    

A `torch.BoolTensor`含有在真其中比较为真每个位置。

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> torch.ne(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[False, True], [True, False]])
    

`torch.``sort`( _input_ , _dim=-1_ , _descending=False_ , _out=None) - >
(Tensor_, _LongTensor_ )

    

排序`输入 `张量的沿给定的维度以升序通过值的元素。

If `dim`is not given, the last dimension of the input is chosen.

如果`降序 `是`真 `然后将元件在由值降序排列。

（值，索引）的namedtuple被返回，其中，所述值是排序的值和指数是在原输入张量的元素的索引。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _,_ _optional_ ) – the dimension to sort along

  * **descending** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – controls the sorting order (ascending or descending)

  * **OUT** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选_ ） - 的（张量的输出元组，  LongTensor ），其可任选地给定的用作输出缓冲器

Example:

    
    
    >>> x = torch.randn(3, 4)
    >>> sorted, indices = torch.sort(x)
    >>> sorted
    tensor([[-0.2162,  0.0608,  0.6719,  2.3332],
            [-0.5793,  0.0061,  0.6058,  0.9497],
            [-0.5071,  0.3343,  0.9553,  1.0960]])
    >>> indices
    tensor([[ 1,  0,  2,  3],
            [ 3,  1,  0,  2],
            [ 0,  3,  1,  2]])
    
    >>> sorted, indices = torch.sort(x, 0)
    >>> sorted
    tensor([[-0.5071, -0.2162,  0.6719, -0.5793],
            [ 0.0608,  0.0061,  0.9497,  0.3343],
            [ 0.6058,  0.9553,  1.0960,  2.3332]])
    >>> indices
    tensor([[ 2,  0,  0,  1],
            [ 0,  1,  1,  2],
            [ 1,  2,  2,  0]])
    

`torch.``topk`( _input_ , _k_ , _dim=None_ , _largest=True_ , _sorted=True_ ,
_out=None) - > (Tensor_, _LongTensor_ )

    

返回沿给定的维度上的给定的`输入 `张量的`K`最大元素。

If `dim`is not given, the last dimension of the input is chosen.

如果`大 `是`假 `然后按 K 返回最小的元素。

的甲namedtuple（值，索引）被返回，其中，所述指数是在原输入张量的元素的索引。

`排序的布尔选项 `如果`真 `，将确保返回 K 元素本身也是分类

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **K** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 在“前k”第k

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _,_ _optional_ ) – the dimension to sort along

  * **最大** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 控制是否返回最大或最小的元素

  * **排序** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 控制是否返回按排序顺序中的元素

  * **OUT** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选_ ） - （张量，LongTensor）的输出元组，可以是任选给定将被用作输出缓冲器

Example:

    
    
    >>> x = torch.arange(1., 6.)
    >>> x
    tensor([ 1.,  2.,  3.,  4.,  5.])
    >>> torch.topk(x, 3)
    torch.return_types.topk(values=tensor([5., 4., 3.]), indices=tensor([4, 3, 2]))
    

### 光谱行动

`torch.``fft`( _input_ , _signal_ndim_ , _normalized=False_ ) → Tensor

    

复杂到复杂的离散傅立叶变换

此方法计算复杂到复杂的离散傅立叶变换。忽略批次尺寸，它计算下面的表达式：

X[ω1,…,ωd]=∑n1=0N1−1⋯∑nd=0Nd−1x[n1,…,nd]e−j 2π∑i=0dωiniNi,X[\omega_1, \dots,
\omega_d] = \sum_{n_1=0}^{N_1-1} \dots \sum_{n_d=0}^{N_d-1} x[n_1, \dots, n_d]
e^{-j\ 2 \pi \sum_{i=0}^d \frac{\omega_i n_i}{N_i}},
X[ω1​,…,ωd​]=n1​=0∑N1​−1​⋯nd​=0∑Nd​−1​x[n1​,…,nd​]e−j 2π∑i=0d​Ni​ωi​ni​​,

其中 d  d  d  = `signal_ndim`是尺寸为信号数目，和 N  i的 n_i个 N  i的 是信号维度 i的 [大小HTG90 ]
i的 [HT G99]  i的 。

此方法支持一维，二维和三维复杂到复杂的变换，由`signal_ndim`表示。 `输入
`必须与尺寸2，代表复数的实和虚分量的最后一维的张量，并应具有至少[H​​TG8]  signal_ndim  \+  1
尺寸与领先的批量尺寸的任选任意数量。如果`归 `被设定为`真 `，这通过用 [HTG27除以归一化的结果]  Π i的 =  1  K  N  i的 \
SQRT {\ prod_ {I = 1}-1K-n_i个}  Π i的 =  1  K  [HT G112]  N  i的 ，使得操作者是一体的。

返回的实部和虚部的`输入 `相同的形状的联为一体张量。

此函数的逆是 `IFFT（） `。

Note

对于CUDA张量，一个LRU缓存用于CUFFT计划加快与相同配置相同的几何形状的张量重复运行FFT方法。参见[ CUFFT计划缓存
[HTG3对于如何监视和控制缓存的更多细节。](notes/cuda.html#cufft-plan-cache)

Warning

对于CPU张量，这种方法目前只适用于MKL。使用`torch.backends.mkl.is_available（） `检查是否安装MKL。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的输入张量至少`signal_ndim``+  1`尺寸

  * **signal_ndim** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 中的每个信号的维数。 `signal_ndim`只能是1，2或3个

  * **归** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 控制是否返回归一化结果。默认值：`假 `

Returns

    

将含有复合到复数傅立叶变换结果张量

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> # unbatched 2D FFT
    >>> x = torch.randn(4, 3, 2)
    >>> torch.fft(x, 2)
    tensor([[[-0.0876,  1.7835],
             [-2.0399, -2.9754],
             [ 4.4773, -5.0119]],
    
            [[-1.5716,  2.7631],
             [-3.8846,  5.2652],
             [ 0.2046, -0.7088]],
    
            [[ 1.9938, -0.5901],
             [ 6.5637,  6.4556],
             [ 2.9865,  4.9318]],
    
            [[ 7.0193,  1.1742],
             [-1.3717, -2.1084],
             [ 2.0289,  2.9357]]])
    >>> # batched 1D FFT
    >>> torch.fft(x, 1)
    tensor([[[ 1.8385,  1.2827],
             [-0.1831,  1.6593],
             [ 2.4243,  0.5367]],
    
            [[-0.9176, -1.5543],
             [-3.9943, -2.9860],
             [ 1.2838, -2.9420]],
    
            [[-0.8854, -0.6860],
             [ 2.4450,  0.0808],
             [ 1.3076, -0.5768]],
    
            [[-0.1231,  2.7411],
             [-0.3075, -1.7295],
             [-0.5384, -2.0299]]])
    >>> # arbitrary number of batch dimensions, 2D FFT
    >>> x = torch.randn(3, 3, 5, 5, 2)
    >>> y = torch.fft(x, 2)
    >>> y.shape
    torch.Size([3, 3, 5, 5, 2])
    

`torch.``ifft`( _input_ , _signal_ndim_ , _normalized=False_ ) → Tensor

    

复杂到复杂的离散傅立叶逆变换

此方法计算复杂到复杂逆离散傅立叶变换。忽略批次尺寸，它计算下面的表达式：

X[ω1,…,ωd]=1∏i=1dNi∑n1=0N1−1⋯∑nd=0Nd−1x[n1,…,nd]e j 2π∑i=0dωiniNi,X[\omega_1,
\dots, \omega_d] = \frac{1}{\prod_{i=1}^d N_i} \sum_{n_1=0}^{N_1-1} \dots
\sum_{n_d=0}^{N_d-1} x[n_1, \dots, n_d] e^{\ j\ 2 \pi \sum_{i=0}^d
\frac{\omega_i n_i}{N_i}},
X[ω1​,…,ωd​]=∏i=1d​Ni​1​n1​=0∑N1​−1​⋯nd​=0∑Nd​−1​x[n1​,…,nd​]e j
2π∑i=0d​Ni​ωi​ni​​,

where ddd = `signal_ndim`is number of dimensions for the signal, and NiN_iNi​
is the size of signal dimension iii .

的参数规格是与 `几乎相同的FFT（） `。然而，如果`归 `被设定为`真 `，这而是返回乘以 [结果HTG17]  Π i的 =  1  d  N
i的 \ SQRT {\ prod_ {I = 1} ^ d n_i个}  Π i的 =  1  d  N  i的 ，成为一个整体的操作。因此，反转一个 `
FFT（） `时，`归 `参数应该被相同地用于设置 `FFT（） `。

Returns the real and the imaginary parts together as one tensor of the same
shape of `input`.

此函数的逆是 `FFT（） `。

Note

For CUDA tensors, an LRU cache is used for cuFFT plans to speed up repeatedly
running FFT methods on tensors of same geometry with same configuration. See
[cuFFT plan cache](notes/cuda.html#cufft-plan-cache) for more details on how
to monitor and control the cache.

Warning

For CPU tensors, this method is currently only available with MKL. Use
`torch.backends.mkl.is_available()`to check if MKL is installed.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor of at least `signal_ndim``+ 1`dimensions

  * **signal_ndim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – the number of dimensions in each signal. `signal_ndim`can only be 1, 2 or 3

  * **normalized** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – controls whether to return normalized results. Default: `False`

Returns

    

将含有复合物到复杂傅立叶逆变换结果张量

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> x = torch.randn(3, 3, 2)
    >>> x
    tensor([[[ 1.2766,  1.3680],
             [-0.8337,  2.0251],
             [ 0.9465, -1.4390]],
    
            [[-0.1890,  1.6010],
             [ 1.1034, -1.9230],
             [-0.9482,  1.0775]],
    
            [[-0.7708, -0.8176],
             [-0.1843, -0.2287],
             [-1.9034, -0.2196]]])
    >>> y = torch.fft(x, 2)
    >>> torch.ifft(y, 2)  # recover x
    tensor([[[ 1.2766,  1.3680],
             [-0.8337,  2.0251],
             [ 0.9465, -1.4390]],
    
            [[-0.1890,  1.6010],
             [ 1.1034, -1.9230],
             [-0.9482,  1.0775]],
    
            [[-0.7708, -0.8176],
             [-0.1843, -0.2287],
             [-1.9034, -0.2196]]])
    

`torch.``rfft`( _input_ , _signal_ndim_ , _normalized=False_ , _onesided=True_
) → Tensor

    

真正到复杂的离散傅立叶变换

该方法可以计算真正到复杂的离散傅立叶变换。它与 `数学上是等效的fft（） `仅在输入和输出的格式的差异。

此方法支持1D，2D和3D真实到复杂的变换，由`signal_ndim`表示。 `输入 `必须与至少`
与领先的批量尺寸的任选任意数量signal_ndim`尺寸的张量。如果`归 `被设定为`真 `，这通过用 [HTG23除以归一化的结果]  Π i的
=  1  K  N  i的 \ SQRT {\ prod_ {I = 1}-1K-n_i个}  Π i的 =  1  K  N  i的
，使得操作者是单一的，其中 N  i的 n_i个 N  i的 是信号维度 i的 i的 [HTG233的大小]  i的 。

真正到复杂的傅立叶变换结果如下共轭对称：

X[ω1,…,ωd]=X∗[N1−ω1,…,Nd−ωd],X[\omega_1, \dots, \omega_d] = X^*[N_1 -
\omega_1, \dots, N_d - \omega_d], X[ω1​,…,ωd​]=X∗[N1​−ω1​,…,Nd​−ωd​],

其中索引算术计算模量的对应尺寸的尺寸， *  \ ^ *  *  为共轭算子，并 d  d  d  = `signal_ndim  [ HTG73。 `
片面 `标志控制，以避免在输出结果的冗余。如果设置为`真 `（默认）中，输出将不会被的形状 [HTG88全复数结果]（  *  ， 2  ） （*，2）
（ *  ， 2  ） ，其中 *  *  *  为输入的`形状 `，而是最后尺寸将被减半了作为尺寸 的⌊ [H TG160]  N  d  2  ⌋ +
1  \ lfloor \压裂{N_d} {2} \ rfloor + 1  ⌊ 2  N  d  ⌋ +  1  。`

此函数的逆是 `irfft（） `。

Note

For CUDA tensors, an LRU cache is used for cuFFT plans to speed up repeatedly
running FFT methods on tensors of same geometry with same configuration. See
[cuFFT plan cache](notes/cuda.html#cufft-plan-cache) for more details on how
to monitor and control the cache.

Warning

For CPU tensors, this method is currently only available with MKL. Use
`torch.backends.mkl.is_available()`to check if MKL is installed.

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的输入张量至少`signal_ndim`尺寸

  * **signal_ndim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – the number of dimensions in each signal. `signal_ndim`can only be 1, 2 or 3

  * **normalized** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – controls whether to return normalized results. Default: `False`

  * **片面** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 控制是否返回一半的结果，以避免冗余。默认值：`真 `

Returns

    

将含有实数到复数傅立叶变换结果张量

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> x = torch.randn(5, 5)
    >>> torch.rfft(x, 2).shape
    torch.Size([5, 3, 2])
    >>> torch.rfft(x, 2, onesided=False).shape
    torch.Size([5, 5, 2])
    

`torch.``irfft`( _input_ , _signal_ndim_ , _normalized=False_ ,
_onesided=True_ , _signal_sizes=None_ ) → Tensor

    

复杂到真正的离散傅立叶逆变换

此方法计算复杂到实逆离散傅立叶变换。它与数学上等效`IFFT（） `仅在输入和输出的格式的差异。

的参数规格是与几乎相同`IFFT（） `。类似于 `IFFT（） `时，如果`归 `被设定为`真 `，这通过用 乘以归一化结果Π i的 =  1
K  N  i的 \ SQRT {\ prod_ {I = 1}-1K-n_i个}  Π i的 =  1  K  [HTG1 01]  N  i的
，使得操作者是单一的，其中 N  i的 n_i个 N  i的 是信号维度 [大小HTG226 ]  i的 i的 i的 。

Note

由于共轭对称，`输入 `不需要包含完整的复频率值。大致的值的一半将是足够的，因为是当`输入 `由 `rfft给定的情况下（） `与`
rfft（信号， 片面=真） `。在这种情况下，设置此方法的为`真 `中的`片面 `参数。此外，原来的信号形状的信息有时会丢失，任意设定`
signal_sizes`是原始信号的大小（无批次尺寸，如果在成批模式）与正确恢复它形状。

因此，反转的 `rfft（） `，则归一化``和`片面 `参数应该被相同地设定为 `irfft（） `和preferrably一个`
signal_sizes`是鉴于以避免大小不匹配。参见尺寸不匹配的情况下的例子。

参见 `rfft（） `关于共轭对称的细节。

此函数的逆是 `rfft（） `。

Warning

一般来说，输入这个功能应该包含以下的共轭对称性值。需要注意的是片面 即使`是`真 `，常为对称性上仍然需要一些部分。当该要求不被满足，的 `
行为irfft（） `是未定义的。由于[ `torch.autograd.gradcheck（） `
](autograd.html#torch.autograd.gradcheck
"torch.autograd.gradcheck")估计数值雅可比与点扰动， `irfft（） `几乎肯定会失败的检查。`

Note

For CUDA tensors, an LRU cache is used for cuFFT plans to speed up repeatedly
running FFT methods on tensors of same geometry with same configuration. See
[cuFFT plan cache](notes/cuda.html#cufft-plan-cache) for more details on how
to monitor and control the cache.

Warning

For CPU tensors, this method is currently only available with MKL. Use
`torch.backends.mkl.is_available()`to check if MKL is installed.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor of at least `signal_ndim``+ 1`dimensions

  * **signal_ndim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – the number of dimensions in each signal. `signal_ndim`can only be 1, 2 or 3

  * **normalized** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – controls whether to return normalized results. Default: `False`

  * **片面** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 控制是否`输入 `被减半了避免冗余，例如，通过 `rfft（） `。默认值：`真 `

  * **signal_sizes** （列表或`torch.Size`，可选） - 原始信号（无批次尺寸）的尺寸。默认值：`无 `

Returns

    

将含有复合物到实傅立叶逆变换结果张量

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> x = torch.randn(4, 4)
    >>> torch.rfft(x, 2, onesided=True).shape
    torch.Size([4, 3, 2])
    >>>
    >>> # notice that with onesided=True, output size does not determine the original signal size
    >>> x = torch.randn(4, 5)
    
    >>> torch.rfft(x, 2, onesided=True).shape
    torch.Size([4, 3, 2])
    >>>
    >>> # now we use the original shape to recover x
    >>> x
    tensor([[-0.8992,  0.6117, -1.6091, -0.4155, -0.8346],
            [-2.1596, -0.0853,  0.7232,  0.1941, -0.0789],
            [-2.0329,  1.1031,  0.6869, -0.5042,  0.9895],
            [-0.1884,  0.2858, -1.5831,  0.9917, -0.8356]])
    >>> y = torch.rfft(x, 2, onesided=True)
    >>> torch.irfft(y, 2, onesided=True, signal_sizes=x.shape)  # recover x
    tensor([[-0.8992,  0.6117, -1.6091, -0.4155, -0.8346],
            [-2.1596, -0.0853,  0.7232,  0.1941, -0.0789],
            [-2.0329,  1.1031,  0.6869, -0.5042,  0.9895],
            [-0.1884,  0.2858, -1.5831,  0.9917, -0.8356]])
    

`torch.``stft`( _input_ , _n_fft_ , _hop_length=None_ , _win_length=None_ ,
_window=None_ , _center=True_ , _pad_mode='reflect'_ , _normalized=False_ ,
_onesided=True_ )[[source]](_modules/torch/functional.html#stft)

    

短时傅立叶变换（STFT）。

忽略可选批次尺寸，此方法计算下列表达式：

X[m,ω]=∑k=0win_length-1window[k] input[m×hop_length+k]
exp⁡(−j2π⋅ωkwin_length),X[m, \omega] = \sum_{k = 0}^{\text{win\\_length-1}}%
\text{window}[k]\ \text{input}[m \times \text{hop\\_length} + k]\ %
\exp\left(- j \frac{2 \pi \cdot \omega k}{\text{win\\_length}}\right),
X[m,ω]=k=0∑win_length-1​window[k] input[m×hop_length+k]
exp(−jwin_length2π⋅ωk​),

其中 M  M  M  在滑动窗口的索引，和 ω \的ω ω 是频率 0  ≤ ω & LT ;  N_FFT  0 \当量\欧米加& LT ; \文本{N
\ _fft}  0  ≤ ω & LT ;  [ H T G94]  N_FFT  。当`片面 `为默认值`真 `

  * `输入 `必须是1-d的时间序列或2- d批次时间序列。

  * 如果`hop_length`是`无 `（默认），它被视为等于`地板（N_FFT  /  4） `。

  * 如果`win_length`是`无 `（默认），它被视为等于`N_FFT`。

  * `窗口 `可以是大小`win_length`，例如1-d张量，由 `torch。 hann_window（） `。如果`窗口 `是`无 `（默认），它被视为好像具有 1  1  1  无处不在的窗口。如果 win_length  & LT ;  N_FFT  \ {文本赢得\ _length} & LT ; \文本{N \ _fft}  win_length  & LT ;  N_FFT  ，`窗口 `将在两侧长度被填充`N_FFT`之前被施加。

  * 如果`中心 `是`真 `（默认），`输入 `将在两个填充侧，使得所述 T  T  T  个帧在时间 [HTG40中心]  T  × hop_length  吨\倍\文本{一跳\ _length}  T  × hop_length  。否则， T  T  T  个帧开始于时间 T  × hop_length  吨\倍\文本{一跳\ _length}  T  × hop_length  。

  * `pad_mode`确定在`输入 `中使用的填补方法，当`中心 `是`真 `。参见[ `torch.nn.functional.pad（） `](nn.functional.html#torch.nn.functional.pad "torch.nn.functional.pad")所有可用的选项。默认值是`“反映” `。

  * 如果`片面 `是`真 `（默认）中，仅值 ω \的ω ω 在 [ 0  ， 1  ， 2  ， ...  ， ⌊  N_FFT  2  ⌋ \+  1  ]  \左[0，1，2，\点，\左\ lfloor \压裂{\文本{N \ _fft}} {2} \右\ rfloor + 1 \右]  [ 0  1  ， 2  ， ...  ， ⌊ 2  N_FFT  ⌋ \+  1  被返回因为真正的到复杂的傅立叶变换满足共轭对称的，即， X  [  M  ， ω =  X  [ M  ， N_FFT  \-  ω *  X [米，\ω= X [米，\文本{N \ _fft} - \ω-^ *  X  [ M  ， ω  =  X  [ ​​ M  [HTG27 2]， N_FFT  \-  ω *  。

  * 如果`归 `是`真 `（默认设定为`假 `），该函数返回归一化的STFT的结果，即，乘以 （ 帧_  ） \-  0.5  （\文本{帧\ _length}）^ { - 0.5}  （ 帧_  ） \-  0  。  5  。

返回的实部和虚部一起作为大小 （ *  ×[之一张量HTG11]  N  × T  × 2  ） （* \次数N \时间T \倍2） （ *  × N  ×
T  × 2  ） ，其中 *  *  [H TG96]  *  是`可选的批量大小输入 `， N  N  N  是其中应用STFT的频率的数量， T  T
T  是使用的帧的总数量，并且每对在最后一维表示复数作为实部和虚部。

Warning

此功能在0.4.1版本中更改签名。与先前的签名调用可能会导致错误或返回不正确的结果。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **N_FFT** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 傅立叶变换大小

  * **hop_length** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 相邻滑动窗帧之间的距离。默认值：`无 `（视为等于`地板（N_FFT  /  4） `）

  * **win_length** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 窗框和STFT滤波器的尺寸。默认值：`无 `（视为等于`N_FFT`）

  * **窗口** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 可选的窗口函数。默认值：`无 `（视作的窗口中的所有 1  1  1  S）

  * **中心** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 是否垫`输入 `在两侧，使得 T  T  T  个帧在时间居中 T  × hop_length  吨\倍\文本{一跳\ _length}  T  × hop_length  。默认值：`真 `

  * **pad_mode** （ _串_ _，_ _可选_ ） - 控制所使用的填补方法，当`中心 `是`真 `。默认值：`“反映” `

  * **归** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 控制是否返回的归一化STFT结果默认：`假 `

  * **片面** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 控制是否返回一半的结果，以避免冗余默认值：`真 `

Returns

    

如上所述包含与形状STFT结果张量

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

`torch.``bartlett_window`( _window_length_ , _periodic=True_ , _dtype=None_ ,
_layout=torch.strided_ , _device=None_ , _requires_grad=False_ ) → Tensor

    

巴特利特窗函数。

w[n]=1−∣2nN−1−1∣={2nN−1if 0≤n≤N−122−2nN−1if N−12<n<N,w[n] = 1 - \left|
\frac{2n}{N-1} - 1 \right| = \begin{cases} \frac{2n}{N - 1} & \text{if } 0
\leq n \leq \frac{N - 1}{2} \\\ 2 - \frac{2n}{N - 1} & \text{if } \frac{N -
1}{2} < n < N \\\ \end{cases}, w[n]=1−∣∣∣∣​N−12n​−1∣∣∣∣​={N−12n​2−N−12n​​if
0≤n≤2N−1​if 2N−1​<n<N​,

其中 N  N  N  是全窗口大小。

输入`window_length`是正整数控制返回窗口大小。 `周期性
`标志确定所返回的窗口是否剪掉从对称窗口中的最后重复的值，并准备用作周期性窗口中包含 `[HTG10功能] torch.stft（） `。因此，如果`
周期性 `为真，则 N  N  N  在上述式中事实上 window_length  \+  1  \文本{窗口\ _length} + 1
window_length  \+  1  。另外，我们始终有`torch.bartlett_window（L， 周期性=真） `等于`
torch.bartlett_window（L  +  1， 周期性=假）[： - 1]） `。

Note

如果`window_length`=  1  = 1  =  1  ，返回的窗口包含一个值1。

Parameters

    

  * **window_length** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 返回的窗口的大小

  * **周期性** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果为True，返回到被用作周期函数的窗口。如果为False，返回一个对称窗口。

  * **DTYPE** （[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")，可选） - 返回的张量的所希望的数据类型。默认值：如果`无 `，使用全局默认设置（见 `torch.set_default_tensor_type（） `）。只有浮点类型的支持。

  * **布局** （[ `torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout")，可选） - 返回的窗口张量的所需布局。只有`torch.strided`（密集布局）被支撑。

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Returns

    

的A 1-d张量大小 （ window_length  ， ）  （\文本{窗口\ _length}，） （ window_length  ， ） 包含窗口

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

`torch.``blackman_window`( _window_length_ , _periodic=True_ , _dtype=None_ ,
_layout=torch.strided_ , _device=None_ , _requires_grad=False_ ) → Tensor

    

布莱克曼窗函数。

w[n]=0.42−0.5cos⁡(2πnN−1)+0.08cos⁡(4πnN−1)w[n] = 0.42 - 0.5 \cos \left(
\frac{2 \pi n}{N - 1} \right) + 0.08 \cos \left( \frac{4 \pi n}{N - 1} \right)
w[n]=0.42−0.5cos(N−12πn​)+0.08cos(N−14πn​)

where NNN is the full window size.

输入`window_length`是正整数控制返回窗口大小。 `周期性
`标志确定所返回的窗口是否剪掉从对称窗口中的最后重复的值，并准备用作周期性窗口中包含 `[HTG10功能] torch.stft（） `。因此，如果`
周期性 `为真，则 N  N  N  在上述式中事实上 window_length  \+  1  \文本{窗口\ _length} + 1
window_length  \+  1  。另外，我们始终有`torch.blackman_window（L， 周期性=真） `等于`
torch.blackman_window（L  +  1， 周期性=假）[： - 1]） `。

Note

If `window_length`=1=1=1 , the returned window contains a single value 1.

Parameters

    

  * **window_length** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – the size of returned window

  * **periodic** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If True, returns a window to be used as periodic function. If False, return a symmetric window.

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see `torch.set_default_tensor_type()`). Only floating point types are supported.

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned window tensor. Only `torch.strided`(dense layout) is supported.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Returns

    

A 1-D tensor of size (window_length,)(\text{window\\_length},)(window_length,)
containing the window

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

`torch.``hamming_window`( _window_length_ , _periodic=True_ , _alpha=0.54_ ,
_beta=0.46_ , _dtype=None_ , _layout=torch.strided_ , _device=None_ ,
_requires_grad=False_ ) → Tensor

    

海明窗函数。

w[n]=α−β cos⁡(2πnN−1),w[n] = \alpha - \beta\ \cos \left( \frac{2 \pi n}{N - 1}
\right), w[n]=α−β cos(N−12πn​),

where NNN is the full window size.

输入`window_length`是正整数控制返回窗口大小。 `周期性
`标志确定所返回的窗口是否剪掉从对称窗口中的最后重复的值，并准备用作周期性窗口中包含 `[HTG10功能] torch.stft（） `。因此，如果`
周期性 `为真，则 N  N  N  在上述式中事实上 window_length  \+  1  \文本{窗口\ _length} + 1
window_length  \+  1  。另外，我们始终有`torch.hamming_window（L， 周期性=真） `等于`
torch.hamming_window（L  +  1， 周期性=假）[： - 1]） `。

Note

If `window_length`=1=1=1 , the returned window contains a single value 1.

Note

这是 `的一般化版本torch.hann_window（） `。

Parameters

    

  * **window_length** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – the size of returned window

  * **periodic** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If True, returns a window to be used as periodic function. If False, return a symmetric window.

  * **阿尔法** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 系数 α \阿尔法 α  在上面的等式

  * **的β** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 系数 β \的β β  在上面的等式

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see `torch.set_default_tensor_type()`). Only floating point types are supported.

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned window tensor. Only `torch.strided`(dense layout) is supported.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Returns

    

A 1-D tensor of size (window_length,)(\text{window\\_length},)(window_length,)
containing the window

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

`torch.``hann_window`( _window_length_ , _periodic=True_ , _dtype=None_ ,
_layout=torch.strided_ , _device=None_ , _requires_grad=False_ ) → Tensor

    

Hann窗函数。

w[n]=12 [1−cos⁡(2πnN−1)]=sin⁡2(πnN−1),w[n] = \frac{1}{2}\ \left[1 - \cos
\left( \frac{2 \pi n}{N - 1} \right)\right] = \sin^2 \left( \frac{\pi n}{N -
1} \right), w[n]=21​ [1−cos(N−12πn​)]=sin2(N−1πn​),

where NNN is the full window size.

输入`window_length`是正整数控制返回窗口大小。 `周期性
`标志确定所返回的窗口是否剪掉从对称窗口中的最后重复的值，并准备用作周期性窗口中包含 `[HTG10功能] torch.stft（） `。因此，如果`
周期性 `为真，则 N  N  N  在上述式中事实上 window_length  \+  1  \文本{窗口\ _length} + 1
window_length  \+  1  。另外，我们始终有`torch.hann_window（L， 周期性=真） `等于`
torch.hann_window（L  +  1， 周期性=假）[： - 1]） `。

Note

If `window_length`=1=1=1 , the returned window contains a single value 1.

Parameters

    

  * **window_length** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – the size of returned window

  * **periodic** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If True, returns a window to be used as periodic function. If False, return a symmetric window.

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, uses a global default (see `torch.set_default_tensor_type()`). Only floating point types are supported.

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – the desired layout of returned window tensor. Only `torch.strided`(dense layout) is supported.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **requires_grad** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If autograd should record operations on the returned tensor. Default: `False`.

Returns

    

A 1-D tensor of size (window_length,)(\text{window\\_length},)(window_length,)
containing the window

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

### 其他操作

`torch.``bincount`( _input_ , _weights=None_ , _minlength=0_ ) → Tensor

    

计数每个值的频率在非负整数的数组。

（尺寸1）段的数目会比`最大值较大的一个输入 `除非`输入 `是空的，在这种情况下结果是大小为0的张量如果`中指定
`时MINLENGTH，箱柜的数目至少为`MINLENGTH`如果`输入 `是空的，那么结果是大小`填充 MINLENGTH `用零的张量。如果`
n的 `在位置值`i的 `，`OUT [N]  + =  权重[I]`如果`的权重被别的指定 ``OUT [N]  + =  1`。

Note

当使用CUDA后端，该操作可以诱导非确定性的行为是不容易断开。请参阅[ 重复性 ](notes/randomness.html)为背景的音符。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 1-d INT张量

  * **权重** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 可选的，重量为输入张量的每个值。应该是相同的大小作为输入张量。

  * **MINLENGTH** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 仓的可选的，最小数量。应为非负。

Returns

    

形状`大小的张量（[最大值（输入） +  1]） `如果`输入 `非空，否则`尺寸（0） `

Return type

    

输出（[张量](tensors.html#torch.Tensor "torch.Tensor")）

Example:

    
    
    >>> input = torch.randint(0, 8, (5,), dtype=torch.int64)
    >>> weights = torch.linspace(0, 1, steps=5)
    >>> input, weights
    (tensor([4, 3, 6, 3, 4]),
     tensor([ 0.0000,  0.2500,  0.5000,  0.7500,  1.0000])
    
    >>> torch.bincount(input)
    tensor([0, 0, 0, 2, 2, 0, 1])
    
    >>> input.bincount(weights)
    tensor([0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.5000])
    

`torch.``broadcast_tensors`( _*tensors_ ) → List of
Tensors[[source]](_modules/torch/functional.html#broadcast_tensors)

    

根据[ 广播语义 ](notes/broadcasting.html#broadcasting-semantics)广播给定的张量。

Parameters

    

***张量** \- 任何数量的相同类型的张量的

Warning

广播的张量的多于一个的元件可指代单个存储器位置。其结果是，就地操作（特别是那些有量化的）可能会导致不正确的行为。如果你需要写张量，请先克隆它们。

Example:

    
    
    >>> x = torch.arange(3).view(1, 3)
    >>> y = torch.arange(2).view(2, 1)
    >>> a, b = torch.broadcast_tensors(x, y)
    >>> a.size()
    torch.Size([2, 3])
    >>> a
    tensor([[0, 1, 2],
            [0, 1, 2]])
    

`torch.``cartesian_prod`( _*tensors_
)[[source]](_modules/torch/functional.html#cartesian_prod)

    

做张量的定序列的笛卡尔乘积。该行为类似于Python的 itertools.product [HTG1。

Parameters

    

***张量** \- 任何数量的1维张量。

Returns

    

A tensor equivalent to converting all the input tensors into lists,

    

做 itertools.product 在这些名单，最后结果列表转换成张量。

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> a = [1, 2, 3]
    >>> b = [4, 5]
    >>> list(itertools.product(a, b))
    [(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]
    >>> tensor_a = torch.tensor(a)
    >>> tensor_b = torch.tensor(b)
    >>> torch.cartesian_prod(tensor_a, tensor_b)
    tensor([[1, 4],
            [1, 5],
            [2, 4],
            [2, 5],
            [3, 4],
            [3, 5]])
    

`torch.``combinations`( _input_ , _r=2_ , _with_replacement=False_ ) → seq

    

长度 R  R  [HTG14的计算组合]  R  [HTG23给定的张量。该行为类似于Python的 itertools.combinations 当
with_replacement 设置为假和 itertools.combinations_with_replacement 当
with_replacement 设置为真[HTG35。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 1D向量。

  * **R** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 元素的数目相结合

  * **with_replacement** （ _布尔_ _，_ _可选_ ） - 是否允许在组合的重复

Returns

    

张量相当于将所有的输入张量成列表，执行 itertools.combinations 或
itertools.combinations_with_replacement 在这些名单，最后结果列表转换成张量。

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> a = [1, 2, 3]
    >>> list(itertools.combinations(a, r=2))
    [(1, 2), (1, 3), (2, 3)]
    >>> list(itertools.combinations(a, r=3))
    [(1, 2, 3)]
    >>> list(itertools.combinations_with_replacement(a, r=2))
    [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    >>> tensor_a = torch.tensor(a)
    >>> torch.combinations(tensor_a)
    tensor([[1, 2],
            [1, 3],
            [2, 3]])
    >>> torch.combinations(tensor_a, r=3)
    tensor([[1, 2, 3]])
    >>> torch.combinations(tensor_a, with_replacement=True)
    tensor([[1, 1],
            [1, 2],
            [1, 3],
            [2, 2],
            [2, 3],
            [3, 3]])
    

`torch.``cross`( _input_ , _other_ , _dim=-1_ , _out=None_ ) → Tensor

    

返回向量的叉积的尺寸`暗淡 `的`输入 `和`其他 `。

`输入 `和`其他 `必须具有相同的尺寸，并且它们的`暗淡 [HTG11的大小]维应该是3。`

如果`暗淡 `没有给出，则默认为与尺寸3找到的第一个维度。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **other** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the second input tensor

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 的尺寸取跨产品英寸

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(4, 3)
    >>> a
    tensor([[-0.3956,  1.1455,  1.6895],
            [-0.5849,  1.3672,  0.3599],
            [-1.1626,  0.7180, -0.0521],
            [-0.1339,  0.9902, -2.0225]])
    >>> b = torch.randn(4, 3)
    >>> b
    tensor([[-0.0257, -1.4725, -1.2251],
            [-1.1479, -0.7005, -1.9757],
            [-1.3904,  0.3726, -1.1836],
            [-0.9688, -0.7153,  0.2159]])
    >>> torch.cross(a, b, dim=1)
    tensor([[ 1.0844, -0.5281,  0.6120],
            [-2.4490, -1.5687,  1.9792],
            [-0.8304, -1.3037,  0.5650],
            [-1.2329,  1.9883,  1.0551]])
    >>> torch.cross(a, b)
    tensor([[ 1.0844, -0.5281,  0.6120],
            [-2.4490, -1.5687,  1.9792],
            [-0.8304, -1.3037,  0.5650],
            [-1.2329,  1.9883,  1.0551]])
    

`torch.``diag`( _input_ , _diagonal=0_ , _out=None_ ) → Tensor

    

  * 如果`输入 `是矢量（1-d张量），然后返回一个2-d平方张量与`输入 `为一体的元件对角线。

  * 如果`输入 `是矩阵（2- d张量），则返回1-d张量与输入的``的对角元素。

的参数 `对角线 `控制以考虑其对角：

  * 如果 `对角线 `= 0，它是主对角线。

  * 如果 `对角线 `& GT ; 0，它上面的主对角线。

  * 如果 `对角线 `& LT ; 0，它是下面的主对角线。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **对角线** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 对角线考虑

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

See also

`torch.diagonal（） `总是返回对角线其输入。

`torch.diagflat（） `始终构成与由输入指定的对角元素的张量。

Examples:

获取方阵，其中输入向量为对角：

    
    
    >>> a = torch.randn(3)
    >>> a
    tensor([ 0.5950,-0.0872, 2.3298])
    >>> torch.diag(a)
    tensor([[ 0.5950, 0.0000, 0.0000],
            [ 0.0000,-0.0872, 0.0000],
            [ 0.0000, 0.0000, 2.3298]])
    >>> torch.diag(a, 1)
    tensor([[ 0.0000, 0.5950, 0.0000, 0.0000],
            [ 0.0000, 0.0000,-0.0872, 0.0000],
            [ 0.0000, 0.0000, 0.0000, 2.3298],
            [ 0.0000, 0.0000, 0.0000, 0.0000]])
    

获取给定矩阵的第k个对角线：

    
    
    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[-0.4264, 0.0255,-0.1064],
            [ 0.8795,-0.2429, 0.1374],
            [ 0.1029,-0.6482,-1.6300]])
    >>> torch.diag(a, 0)
    tensor([-0.4264,-0.2429,-1.6300])
    >>> torch.diag(a, 1)
    tensor([ 0.0255, 0.1374])
    

`torch.``diag_embed`( _input_ , _offset=0_ , _dim1=-2_ , _dim2=-1_ ) → Tensor

    

创建一个张量，其一定的2D平面的对角线（由`指定DIM1`和`DIM2`）由`输入填充
`。为了便于创建批处理对角矩阵，用返回的张量的最后两个维度形成二维平面默认选中。

的参数`偏移HTG2] `控制以考虑其对角：

  * 如果offset = 0 `，它是主对角线。`

  * 如果`偏移HTG2] `& GT ; 0，它上面的主对角线。

  * 如果`偏移HTG2] `& LT ; 0，它是下面的主对角线。

新矩阵的大小将被计算为使最后输入尺寸大小的指定对角线。注意，`偏移 `除 0  0  0  ，的顺序`DIM1`和`DIM2
`事项。交换它们是等效于改变的`偏移HTG38] `符号。

施加 `torch.diagonal（） `该函数的输出与相同参数产生相同的输入的矩阵。然而， `torch.diagonal（） `
具有不同的默认尺寸，因此这些需要被明确指定。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入张量。必须至少为1维的。

  * **偏移HTG1]（[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 考虑哪些对角线。默认值：0（主对角线）。**

  * **DIM1** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 相对于第一维度，其采取对角线。默认值：-2。

  * **DIM2** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 相对于第二尺寸，其采取对角线。缺省值：-1。

Example:

    
    
    >>> a = torch.randn(2, 3)
    >>> torch.diag_embed(a)
    tensor([[[ 1.5410,  0.0000,  0.0000],
             [ 0.0000, -0.2934,  0.0000],
             [ 0.0000,  0.0000, -2.1788]],
    
            [[ 0.5684,  0.0000,  0.0000],
             [ 0.0000, -1.0845,  0.0000],
             [ 0.0000,  0.0000, -1.3986]]])
    
    >>> torch.diag_embed(a, offset=1, dim1=0, dim2=2)
    tensor([[[ 0.0000,  1.5410,  0.0000,  0.0000],
             [ 0.0000,  0.5684,  0.0000,  0.0000]],
    
            [[ 0.0000,  0.0000, -0.2934,  0.0000],
             [ 0.0000,  0.0000, -1.0845,  0.0000]],
    
            [[ 0.0000,  0.0000,  0.0000, -2.1788],
             [ 0.0000,  0.0000,  0.0000, -1.3986]],
    
            [[ 0.0000,  0.0000,  0.0000,  0.0000],
             [ 0.0000,  0.0000,  0.0000,  0.0000]]])
    

`torch.``diagflat`( _input_ , _offset=0_ ) → Tensor

    

  * If `input`is a vector (1-D tensor), then returns a 2-D square tensor with the elements of `input`as the diagonal.

  * 如果`输入 `是与多于一个的维度的张量，然后返回2-d张量的对角元素等于扁平`输入 `。

The argument `offset`controls which diagonal to consider:

  * If `offset`= 0, it is the main diagonal.

  * If `offset`> 0, it is above the main diagonal.

  * If `offset`< 0, it is below the main diagonal.

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **偏移HTG1]（[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 对角线来考虑。默认值：0（主对角线）。**

Examples:

    
    
    >>> a = torch.randn(3)
    >>> a
    tensor([-0.2956, -0.9068,  0.1695])
    >>> torch.diagflat(a)
    tensor([[-0.2956,  0.0000,  0.0000],
            [ 0.0000, -0.9068,  0.0000],
            [ 0.0000,  0.0000,  0.1695]])
    >>> torch.diagflat(a, 1)
    tensor([[ 0.0000, -0.2956,  0.0000,  0.0000],
            [ 0.0000,  0.0000, -0.9068,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.1695],
            [ 0.0000,  0.0000,  0.0000,  0.0000]])
    
    >>> a = torch.randn(2, 2)
    >>> a
    tensor([[ 0.2094, -0.3018],
            [-0.1516,  1.9342]])
    >>> torch.diagflat(a)
    tensor([[ 0.2094,  0.0000,  0.0000,  0.0000],
            [ 0.0000, -0.3018,  0.0000,  0.0000],
            [ 0.0000,  0.0000, -0.1516,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  1.9342]])
    

`torch.``diagonal`( _input_ , _offset=0_ , _dim1=0_ , _dim2=1_ ) → Tensor

    

返回`输入 `与它的对角元素的局部视图相对于`DIM1`和`DIM2`作为附加在所述形状的端部的尺寸。

The argument `offset`controls which diagonal to consider:

  * If `offset`= 0, it is the main diagonal.

  * If `offset`> 0, it is above the main diagonal.

  * If `offset`< 0, it is below the main diagonal.

`torch.diag_embed施加（） `该函数的输出与相同的参数产生与所述输入的对角项的对角矩阵。然而， `torch.diag_embed（）
`具有不同的默认尺寸，因此这些需要被明确指定。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入张量。必须至少2维的。

  * **offset** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _,_ _optional_ ) – which diagonal to consider. Default: 0 (main diagonal).

  * **DIM1** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 相对于第一维度，其采取对角线。默认值：0。

  * **DIM2** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 相对于第二尺寸，其采取对角线。默认值：1。

Note

采取分批对角线，传入DIM1 = -2，DIM2 = -1。

Examples:

    
    
    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[-1.0854,  1.1431, -0.1752],
            [ 0.8536, -0.0905,  0.0360],
            [ 0.6927, -0.3735, -0.4945]])
    
    
    >>> torch.diagonal(a, 0)
    tensor([-1.0854, -0.0905, -0.4945])
    
    
    >>> torch.diagonal(a, 1)
    tensor([ 1.1431,  0.0360])
    
    
    >>> x = torch.randn(2, 5, 4, 2)
    >>> torch.diagonal(x, offset=-1, dim1=1, dim2=2)
    tensor([[[-1.2631,  0.3755, -1.5977, -1.8172],
             [-1.1065,  1.0401, -0.2235, -0.7938]],
    
            [[-1.7325, -0.3081,  0.6166,  0.2335],
             [ 1.0500,  0.7336, -0.3836, -1.1015]]])
    

`torch.``einsum`( _equation_ , _*operands_ ) →
Tensor[[source]](_modules/torch/functional.html#einsum)

    

这个功能提供计算多线性表达式的方式使用爱因斯坦求和约定（即乘积的和）。

Parameters

    

  * **方程** （ _串_ ） - 方程式中的小写字母（索引）来给出将与操作数和结果的每个维度相关联。左侧列出了操作数的尺寸，用逗号分开。应该有每张量维度中的一个索引字母。之后的右手边如下 - & GT ; ，并给出了指数的输出。如果 \- [ - ] GT ; 和右侧被省略，它含蓄地定义为左侧恰好出现一次所有指数的按字母顺序排序列表。在输出不apprearing该指数的操作数项相乘后求和。如果指数出现几次同样的操作，对角线取。椭圆 ... 代表尺寸的固定数目。如果右侧推断，省略号尺寸在输出的开始。

  * **操作数** （ _张量_ 的列表中） - 的操作数来计算的爱因斯坦总和。

Examples:

    
    
    >>> x = torch.randn(5)
    >>> y = torch.randn(4)
    >>> torch.einsum('i,j->ij', x, y)  # outer product
    tensor([[-0.0570, -0.0286, -0.0231,  0.0197],
            [ 1.2616,  0.6335,  0.5113, -0.4351],
            [ 1.4452,  0.7257,  0.5857, -0.4984],
            [-0.4647, -0.2333, -0.1883,  0.1603],
            [-1.1130, -0.5588, -0.4510,  0.3838]])
    
    
    >>> A = torch.randn(3,5,4)
    >>> l = torch.randn(2,5)
    >>> r = torch.randn(2,4)
    >>> torch.einsum('bn,anm,bm->ba', l, A, r) # compare torch.nn.functional.bilinear
    tensor([[-0.3430, -5.2405,  0.4494],
            [ 0.3311,  5.5201, -3.0356]])
    
    
    >>> As = torch.randn(3,2,5)
    >>> Bs = torch.randn(3,5,4)
    >>> torch.einsum('bij,bjk->bik', As, Bs) # batch matrix multiplication
    tensor([[[-1.0564, -1.5904,  3.2023,  3.1271],
             [-1.6706, -0.8097, -0.8025, -2.1183]],
    
            [[ 4.2239,  0.3107, -0.5756, -0.2354],
             [-1.4558, -0.3460,  1.5087, -0.8530]],
    
            [[ 2.8153,  1.8787, -4.3839, -1.2112],
             [ 0.3728, -2.1131,  0.0921,  0.8305]]])
    
    >>> A = torch.randn(3, 3)
    >>> torch.einsum('ii->i', A) # diagonal
    tensor([-0.7825,  0.8291, -0.1936])
    
    >>> A = torch.randn(4, 3, 3)
    >>> torch.einsum('...ii->...i', A) # batch diagonal
    tensor([[-1.0864,  0.7292,  0.0569],
            [-0.9725, -1.0270,  0.6493],
            [ 0.5832, -1.1716, -1.5084],
            [ 0.4041, -1.1690,  0.8570]])
    
    >>> A = torch.randn(2, 3, 4, 5)
    >>> torch.einsum('...ij->...ji', A).shape # batch permute
    torch.Size([2, 3, 5, 4])
    

`torch.``flatten`( _input_ , _start_dim=0_ , _end_dim=-1_ ) → Tensor

    

变平的张量DIMS的连续范围。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **start_dim** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 第一暗淡变平

  * **end_dim** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 最后暗淡变平

Example:

    
    
    >>> t = torch.tensor([[[1, 2],
                           [3, 4]],
                          [[5, 6],
                           [7, 8]]])
    >>> torch.flatten(t)
    tensor([1, 2, 3, 4, 5, 6, 7, 8])
    >>> torch.flatten(t, start_dim=1)
    tensor([[1, 2, 3, 4],
            [5, 6, 7, 8]])
    

`torch.``flip`( _input_ , _dims_ ) → Tensor

    

反向沿DIMS定轴线正d张量的顺序。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **变暗** （ _列表_ _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")） - 轴翻转上

Example:

    
    
    >>> x = torch.arange(8).view(2, 2, 2)
    >>> x
    tensor([[[ 0,  1],
             [ 2,  3]],
    
            [[ 4,  5],
             [ 6,  7]]])
    >>> torch.flip(x, [0, 1])
    tensor([[[ 6,  7],
             [ 4,  5]],
    
            [[ 2,  3],
             [ 0,  1]]])
    

`torch.``rot90`( _input_ , _k_ , _dims_ ) → Tensor

    

在通过DIMS轴规定的平面内旋转90度后的正d张量。旋转方向是从第一向第二轴如果k & GT ; 0，并且从第二向第一对于k & LT ; 0。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **K** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 次数旋转

  * **变暗** （ _列表_ _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")） - 轴旋转

Example:

    
    
    >>> x = torch.arange(4).view(2, 2)
    >>> x
    tensor([[0, 1],
            [2, 3]])
    >>> torch.rot90(x, 1, [0, 1])
    tensor([[1, 3],
            [0, 2]])
    
    >>> x = torch.arange(8).view(2, 2, 2)
    >>> x
    tensor([[[0, 1],
             [2, 3]],
    
            [[4, 5],
             [6, 7]]])
    >>> torch.rot90(x, 1, [1, 2])
    tensor([[[1, 3],
             [0, 2]],
    
            [[5, 7],
             [4, 6]]])
    

`torch.``histc`( _input_ , _bins=100_ , _min=0_ , _max=0_ , _out=None_ ) →
Tensor

    

计算张量的柱状图。

这些元件之间 `分成相等的宽度仓分钟HTG3] `和 `MAX`。如果 `分钟HTG15] `和 `MAX`
均为零，最小和的最大值的数据被使用。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **仓** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 直方图区间的数

  * **分钟HTG1]（[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 该范围的下端（含）**

  * **MAX** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 该范围的上端（含）

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Returns

    

直方图表示为张量

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> torch.histc(torch.tensor([1., 2, 1]), bins=4, min=0, max=3)
    tensor([ 0.,  2.,  1.,  0.])
    

`torch.``meshgrid`( _*tensors_ , _**kwargs_
)[[source]](_modules/torch/functional.html#meshgrid)

    

取 N  N  N  张量，其中的每一个可以是标量或1维向量，并创建 N  N  N  N维网格，其中，所述 i的 i的 i的 第网格由扩大 i的 i的
[HTG85定义]  i的[H TG93]  第在由其他输入来定义的尺寸输入。

> Args:

>  
>

> 张量（张量的列表）：标量或1维的张量的列表。标量将被视为的张量大小 （ 1  ， ） （1） （ 1  ， ） 自动

>

> Returns:

>  
>

> SEQ（张量的序列）：如果输入具有 K  K  K  大小的张量 （ N  1  ， ） ， （ N  2  ， ） ...  ， （ N  K  ，
） （N_1，），（N_2，），\ ldots，（N_k，） （ N  1  ） ， （ N  2  ， ） ， ...  ， （ N  K  ， ）
则输出也将具有 K  K  K  张量，其中所有的张量的大小 （ N  1  ， N  2  ， ...  ， N  K  ​​） （N_1，N_2，\
ldots，N_k） （ N  1  ， N  2  ， ...  ， N  K  [HT G382]  ） 。

>

> Example:

>  
>  
>     >>> x = torch.tensor([1, 2, 3])

>     >>> y = torch.tensor([4, 5, 6])

>     >>> grid_x, grid_y = torch.meshgrid(x, y)

>     >>> grid_x

>     tensor([[1, 1, 1],

>             [2, 2, 2],

>             [3, 3, 3]])

>     >>> grid_y

>     tensor([[4, 5, 6],

>             [4, 5, 6],

>             [4, 5, 6]])

>  

`torch.``renorm`( _input_ , _p_ , _dim_ , _maxnorm_ , _out=None_ ) → Tensor

    

返回的张量，其中输入的`每个子张量 `沿着维度`暗淡 `进行归一化，使得 P  \- 子张量的范数低于值`maxnorm`

Note

如果行的范数大于 maxnorm 下，该行是不变

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **P** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 功率为范数计算

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的尺寸切过，以获得子张量

  * **maxnorm** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 最大范保持每个子张量下

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> x = torch.ones(3, 3)
    >>> x[1].fill_(2)
    tensor([ 2.,  2.,  2.])
    >>> x[2].fill_(3)
    tensor([ 3.,  3.,  3.])
    >>> x
    tensor([[ 1.,  1.,  1.],
            [ 2.,  2.,  2.],
            [ 3.,  3.,  3.]])
    >>> torch.renorm(x, 1, 0, 5)
    tensor([[ 1.0000,  1.0000,  1.0000],
            [ 1.6667,  1.6667,  1.6667],
            [ 1.6667,  1.6667,  1.6667]])
    

`torch.``repeat_interleave`()

    

`torch.``repeat_interleave`( _input_ , _repeats_ , _dim=None_ ) → Tensor

    

重复张量元素。

Warning

这是从`torch.repeat（） `不同但类似于 numpy.repeat 。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入张量

  * **重复** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _或_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 重复用于每个元件的数目。重复广播到符合给定的轴的形状。

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 沿其以重复的值的尺寸。缺省情况下，使用压平输入阵列，并返回一个平坦的输出阵列。

Returns

    

Repeated tensor which has the same shape as input, except along the

    

定轴。

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> x = torch.tensor([1, 2, 3])
    >>> x.repeat_interleave(2)
    tensor([1, 1, 2, 2, 3, 3])
    >>> y = torch.tensor([[1, 2], [3, 4]])
    >>> torch.repeat_interleave(y, 2)
    tensor([1, 1, 2, 2, 3, 3, 4, 4])
    >>> torch.repeat_interleave(y, 3, dim=1)
    tensor([[1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4]])
    >>> torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0)
    tensor([[1, 2],
            [3, 4],
            [3, 4]])
    

`torch.``repeat_interleave`( _repeats_ ) → Tensor

    

如果重复是张量（[N1，N2，N3，...]），那么输出将是张量（[0，0，...，1，1， ...，2,2，...，...] 其中 0 出现） N1 ，
1 出现倍 N 2 次， 2 出现 N3 时间等

`torch.``roll`( _input_ , _shifts_ , _dims=None_ ) → Tensor

    

一起滚动给定的尺寸（S）的张量。在第一位置被移动超过最后一个位置元素被重新引入。如果没有指定一个尺寸，张量将轧制前的被平坦化，然后恢复到原来的形状。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **移位** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ _蟒的元组：整数_ ） - 的地方数，通过该元件张量移位。如果移位是一个元组，DIMS必须是相同大小的元组，并且每个维度将由相应的值被卷起

  * **变暗** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ _蟒的元组：整数_ ） - 轴沿其滚动

Example:

    
    
    >>> x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
    >>> x
    tensor([[1, 2],
            [3, 4],
            [5, 6],
            [7, 8]])
    >>> torch.roll(x, 1, 0)
    tensor([[7, 8],
            [1, 2],
            [3, 4],
            [5, 6]])
    >>> torch.roll(x, -1, 0)
    tensor([[3, 4],
            [5, 6],
            [7, 8],
            [1, 2]])
    >>> torch.roll(x, shifts=(2, 1), dims=(0, 1))
    tensor([[6, 5],
            [8, 7],
            [2, 1],
            [4, 3]])
    

`torch.``tensordot`( _a_ , _b_ , _dims=2_
)[[source]](_modules/torch/functional.html#tensordot)

    

返回的收缩和b在多个维度。

`tensordot`实现了一个广义矩阵乘积。

Parameters

    

  * **一** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 左张量收缩

  * **B** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 右张量收缩

  * **变暗** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ _蟒两个列表的元组：整数_ ） - 维数收缩或为`一 `和分别`b`尺寸的显式列表

当与一个整数参数称为`变暗 `=  d  d  d  和数量的尺寸`一 `和`b`是 M  M  M  和 n的 n的 n的 ，分别，它计算

ri0,...,im−d,id,...,in=∑k0,...,kd−1ai0,...,im−d,k0,...,kd−1×bk0,...,kd−1,id,...,in.r_{i_0,...,i_{m-d},
i_d,...,i_n} = \sum_{k_0,...,k_{d-1}} a_{i_0,...,i_{m-d},k_0,...,k_{d-1}}
\times b_{k_0,...,k_{d-1}, i_d,...,i_n}.
ri0​,...,im−d​,id​,...,in​​=k0​,...,kd−1​∑​ai0​,...,im−d​,k0​,...,kd−1​​×bk0​,...,kd−1​,id​,...,in​​.

当与`所谓变暗 `列表的形式，给定尺寸将取代过去的 承包 d  d  d  的`一 `和第一 d  d  d  的 b  b  b
。在这些维度的尺寸必须匹配，但是 `tensordot`将处理广播尺寸。

Examples:

    
    
    >>> a = torch.arange(60.).reshape(3, 4, 5)
    >>> b = torch.arange(24.).reshape(4, 3, 2)
    >>> torch.tensordot(a, b, dims=([1, 0], [0, 1]))
    tensor([[4400., 4730.],
            [4532., 4874.],
            [4664., 5018.],
            [4796., 5162.],
            [4928., 5306.]])
    
    >>> a = torch.randn(3, 4, 5, device='cuda')
    >>> b = torch.randn(4, 5, 6, device='cuda')
    >>> c = torch.tensordot(a, b, dims=2).cpu()
    tensor([[ 8.3504, -2.5436,  6.2922,  2.7556, -1.0732,  3.2741],
            [ 3.3161,  0.0704,  5.0187, -0.4079, -4.3126,  4.8744],
            [ 0.8223,  3.9445,  3.2168, -0.2400,  3.4117,  1.7780]])
    

`torch.``trace`( _input_ ) → Tensor

    

返回对角线输入2-d的矩阵的元素的总和。

Example:

    
    
    >>> x = torch.arange(1., 10.).view(3, 3)
    >>> x
    tensor([[ 1.,  2.,  3.],
            [ 4.,  5.,  6.],
            [ 7.,  8.,  9.]])
    >>> torch.trace(x)
    tensor(15.)
    

`torch.``tril`( _input_ , _diagonal=0_ , _out=None_ ) → Tensor

    

返回矩阵（2- d张量）或分批矩阵`输入的下三角部分 `，结果张量`OUT 的其它元件`被设置为0。

矩阵的下三角部分定义为上和下面的对角线的元素。

的参数 `对角线 `控制考虑哪些对角线。如果 `对角线 `=
0，上面和下面的主对角线的所有元素被保留。正值包括正上方的主对角线许多对角线，并且类似地负值排除正下方的主对角线许多对角线。主对角线是该组的索引 { （
i的 i的 ） }  \ lbrace（I，i）的\ rbrace  { （ i的 ， i的 ） }  为 i的 ∈ [ 0  ， 分钟HTG79] ⁡ {
d  1  ， d  2  }  \-  1  I \在[0，\分钟\ {D_ {1 }，D_ {2} \\} - 1]  i的 ∈ [ 0  ，
分钟HTG137]  { d  1  ， d  2  [H TG204]}  \-  1  其中 d  1  ， d  2  D_ {1}，D_ {2}
d  1  ​​  ， d  2  是矩阵的维数。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **diagonal** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _,_ _optional_ ) – the diagonal to consider

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[-1.0813, -0.8619,  0.7105],
            [ 0.0935,  0.1380,  2.2112],
            [-0.3409, -0.9828,  0.0289]])
    >>> torch.tril(a)
    tensor([[-1.0813,  0.0000,  0.0000],
            [ 0.0935,  0.1380,  0.0000],
            [-0.3409, -0.9828,  0.0289]])
    
    >>> b = torch.randn(4, 6)
    >>> b
    tensor([[ 1.2219,  0.5653, -0.2521, -0.2345,  1.2544,  0.3461],
            [ 0.4785, -0.4477,  0.6049,  0.6368,  0.8775,  0.7145],
            [ 1.1502,  3.2716, -1.1243, -0.5413,  0.3615,  0.6864],
            [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0978]])
    >>> torch.tril(b, diagonal=1)
    tensor([[ 1.2219,  0.5653,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.4785, -0.4477,  0.6049,  0.0000,  0.0000,  0.0000],
            [ 1.1502,  3.2716, -1.1243, -0.5413,  0.0000,  0.0000],
            [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0000]])
    >>> torch.tril(b, diagonal=-1)
    tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.4785,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 1.1502,  3.2716,  0.0000,  0.0000,  0.0000,  0.0000],
            [-0.0614, -0.7344, -1.3164,  0.0000,  0.0000,  0.0000]])
    

`torch.``tril_indices`( _row_ , _col_ , _offset=0_ , _dtype=torch.long_ ,
_device='cpu'_ , _layout=torch.strided_ ) → Tensor

    

返回`行的下三角部分的索引 `-by- `COL
`矩阵在一个2-N张量，其中第一行包含所有索引的列坐标和第二行包含列坐标。指数是基于行，然后列排序。

The lower triangular part of the matrix is defined as the elements on and
below the diagonal.

的参数`偏移HTG2] `控制考虑哪些对角线。如果`偏移HTG6] `=
0，上面和下面的主对角线的所有元素被保留。正值包括正上方的主对角线许多对角线，并且类似地负值排除正下方的主对角线许多对角线。主对角线是该组的索引 { （
i的 i的 ） }  \ lbrace（I，i）的\ rbrace  { （ i的 ， i的 ） }  为 i的 ∈ [ 0  ， 分钟HTG75] ⁡ {
d  1  ， d  2  }  \-  1  I \在[0，\分钟\ {D_ {1 }，{D_ 2} \\} - 1]  i的 ∈ [ 0  ， 分钟 {
d  1  ， d  2  }  \-  1  其中 d  1  ， d  2  D_ {1}，D_ {2}  d  1  ​​  ， d  2
是矩阵的维数。

注：在 'CUDA' 运行时，行* COL必须低于 2  59  2 ^ {59}  2  5  9  可以防止计算期间的溢出。

Parameters

    

  * 在2 d矩阵中的行数 - **行** （`INT`）。

  * **COL** （`INT`） - 在2-d矩阵的列数。

  * **偏移HTG1]（`INT`） - 对角线从所述主对角线的偏移。默认值：如果没有提供，0。**

  * **DTYPE** （[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")，可选） - 返回的张量的所希望的数据类型。默认值：如果`无 `，`torch.long`。

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **布局** （[ `torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout")，可选） - 目前仅支持`torch.strided`。

Example::

    
    
    
    >>> a = torch.tril_indices(3, 3)
    >>> a
    tensor([[0, 1, 1, 2, 2, 2],
            [0, 0, 1, 0, 1, 2]])
    
    
    
    >>> a = torch.tril_indices(4, 3, -1)
    >>> a
    tensor([[1, 2, 2, 3, 3, 3],
            [0, 0, 1, 0, 1, 2]])
    
    
    
    >>> a = torch.tril_indices(4, 3, 1)
    >>> a
    tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            [0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2]])
    

`torch.``triu`( _input_ , _diagonal=0_ , _out=None_ ) → Tensor

    

返回一个矩阵（2- d张量）或分批矩阵`输入的上三角部分 `，结果张量`OUT 的其它元件`被设置为0。

矩阵的上三角部分定义为上和上面的对角线的元素。

的参数 `对角线 `控制考虑哪些对角线。如果 `对角线 `=
0，上面和下面的主对角线的所有元素被保留。正值排除同样多的对角线上方的主对角线，并且类似地负值包括正下方的主对角线许多对角线。主对角线是该组的索引 { （
i的 i的 ） }  \ lbrace（I，i）的\ rbrace  { （ i的 ， i的 ） }  为 i的 ∈ [ 0  ， 分钟HTG79] ⁡ {
d  1  ， d  2  }  \-  1  I \在[0，\分钟\ {D_ {1 }，D_ {2} \\} - 1]  i的 ∈ [ 0  ，
分钟HTG137]  { d  1  ， d  2  [H TG204]}  \-  1  其中 d  1  ， d  2  D_ {1}，D_ {2}
d  1  ​​  ， d  2  是矩阵的维数。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **diagonal** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _,_ _optional_ ) – the diagonal to consider

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[ 0.2309,  0.5207,  2.0049],
            [ 0.2072, -1.0680,  0.6602],
            [ 0.3480, -0.5211, -0.4573]])
    >>> torch.triu(a)
    tensor([[ 0.2309,  0.5207,  2.0049],
            [ 0.0000, -1.0680,  0.6602],
            [ 0.0000,  0.0000, -0.4573]])
    >>> torch.triu(a, diagonal=1)
    tensor([[ 0.0000,  0.5207,  2.0049],
            [ 0.0000,  0.0000,  0.6602],
            [ 0.0000,  0.0000,  0.0000]])
    >>> torch.triu(a, diagonal=-1)
    tensor([[ 0.2309,  0.5207,  2.0049],
            [ 0.2072, -1.0680,  0.6602],
            [ 0.0000, -0.5211, -0.4573]])
    
    >>> b = torch.randn(4, 6)
    >>> b
    tensor([[ 0.5876, -0.0794, -1.8373,  0.6654,  0.2604,  1.5235],
            [-0.2447,  0.9556, -1.2919,  1.3378, -0.1768, -1.0857],
            [ 0.4333,  0.3146,  0.6576, -1.0432,  0.9348, -0.4410],
            [-0.9888,  1.0679, -1.3337, -1.6556,  0.4798,  0.2830]])
    >>> torch.triu(b, diagonal=1)
    tensor([[ 0.0000, -0.0794, -1.8373,  0.6654,  0.2604,  1.5235],
            [ 0.0000,  0.0000, -1.2919,  1.3378, -0.1768, -1.0857],
            [ 0.0000,  0.0000,  0.0000, -1.0432,  0.9348, -0.4410],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.4798,  0.2830]])
    >>> torch.triu(b, diagonal=-1)
    tensor([[ 0.5876, -0.0794, -1.8373,  0.6654,  0.2604,  1.5235],
            [-0.2447,  0.9556, -1.2919,  1.3378, -0.1768, -1.0857],
            [ 0.0000,  0.3146,  0.6576, -1.0432,  0.9348, -0.4410],
            [ 0.0000,  0.0000, -1.3337, -1.6556,  0.4798,  0.2830]])
    

`torch.``triu_indices`( _row_ , _col_ , _offset=0_ , _dtype=torch.long_ ,
_device='cpu'_ , _layout=torch.strided_ ) → Tensor

    

通过`返回`行 `的上三角部分的索引COL`矩阵在一个2-N张量，其中，所述第一行包含所有索引的列坐标和第二行包含列坐标。指数是基于行，然后列排序。

The upper triangular part of the matrix is defined as the elements on and
above the diagonal.

的参数`偏移HTG2] `控制考虑哪些对角线。如果`偏移HTG6] `=
0，上和上方的主对角线的所有元素被保留。正值排除同样多的对角线上方的主对角线，并且类似地负值包括正下方的主对角线许多对角线。主对角线是该组的索引 { （
i的 i的 ） }  \ lbrace（I，i）的\ rbrace  { （ i的 ， i的 ） }  为 i的 ∈ [ 0  ， 分钟HTG75] ⁡ {
d  1  ， d  2  }  \-  1  I \在[0，\分钟\ {D_ {1 }，{D_ 2} \\} - 1]  i的 ∈ [ 0  ， 分钟 {
d  1  ， d  2  }  \-  1  其中 d  1  ， d  2  D_ {1}，D_ {2}  d  1  ​​  ， d  2
是矩阵的维数。

NOTE: when running on ‘cuda’, row * col must be less than 2592^{59}259 to
prevent overflow during calculation.

Parameters

    

  * **row** (`int`) – number of rows in the 2-D matrix.

  * **col** (`int`) – number of columns in the 2-D matrix.

  * **offset** (`int`) – diagonal offset from the main diagonal. Default: if not provided, 0.

  * **dtype** ([`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), optional) – the desired data type of returned tensor. Default: if `None`, `torch.long`.

  * **device** ([`torch.device`](tensor_attributes.html#torch.torch.device "torch.torch.device"), optional) – the desired device of returned tensor. Default: if `None`, uses the current device for the default tensor type (see `torch.set_default_tensor_type()`). `device`will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

  * **layout** ([`torch.layout`](tensor_attributes.html#torch.torch.layout "torch.torch.layout"), optional) – currently only support `torch.strided`.

Example::

    
    
    
    >>> a = torch.triu_indices(3, 3)
    >>> a
    tensor([[0, 0, 0, 1, 1, 2],
            [0, 1, 2, 1, 2, 2]])
    
    
    
    >>> a = torch.triu_indices(4, 3, -1)
    >>> a
    tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3],
            [0, 1, 2, 0, 1, 2, 1, 2, 2]])
    
    
    
    >>> a = torch.triu_indices(4, 3, 1)
    >>> a
    tensor([[0, 0, 1],
            [1, 2, 2]])
    

### BLAS和LAPACK操作

`torch.``addbmm`( _beta=1_ , _input_ , _alpha=1_ , _batch1_ , _batch2_ ,
_out=None_ ) → Tensor

    

对存储在`BATCH1`和`batch2`，具有降低的附加步骤（所有的矩阵乘法得到积累矩阵的批次矩阵矩阵积沿着第一维度）。 `输入
`加到最终结果。

`BATCH1`和`batch2`必须各自含有相同数量的矩阵的3-d张量。

如果`BATCH1`是 （ B  × n的 × M  ） （b \ n次\乘以m） （ b  × n的 × M  ） 张量，`batch2`是
（ b  × M  × p  ） （b \倍米\倍p） [HT G100]  （ B  × M  × p  ） 张量，`输入 `必须[
broadcastable  ](notes/broadcasting.html#broadcasting-semantics)与 （ n的 × p  ）
（N \倍p） （ n的 × p  ） 几十或并`OUT`将是 （ n的 × p  ） （N \倍p） （ n的 × p  ） 张量。

out=β input+α (∑i=0b−1batch1i@batch2i)out = \beta\ \text{input} + \alpha\
(\sum_{i=0}^{b-1} \text{batch1}_i \mathbin{@} \text{batch2}_i) out=β input+α
(i=0∑b−1​batch1i​@batch2i​)

对于类型的输入 FloatTensor 或 DoubleTensor ，自变量`的β `和`阿尔法 `必须实数，否则他们应该是整数。

Parameters

    

  * **的β** （ _号码_ _，_ _可选_ ） - 乘数`输入 `（ β \的β β ）

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 要添加矩阵

  * **阿尔法** （ _号码_ _，_ _可选_ ） - 乘数 BATCH1 @ batch2 （ α \阿尔法 α ）

  * **BATCH1** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 第一批矩阵的相乘

  * **batch2** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 第二批矩阵的相乘

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> M = torch.randn(3, 5)
    >>> batch1 = torch.randn(10, 3, 4)
    >>> batch2 = torch.randn(10, 4, 5)
    >>> torch.addbmm(M, batch1, batch2)
    tensor([[  6.6311,   0.0503,   6.9768, -12.0362,  -2.1653],
            [ -4.8185,  -1.4255,  -6.6760,   8.9453,   2.5743],
            [ -3.8202,   4.3691,   1.0943,  -1.1109,   5.4730]])
    

`torch.``addmm`( _beta=1_ , _input_ , _alpha=1_ , _mat1_ , _mat2_ , _out=None_
) → Tensor

    

执行矩阵`MAT1`和`MAT2`的矩阵乘法。矩阵`输入 `加到最终结果。

如果`MAT1`是 （ n的 × M  ） （N \乘以m） （ n的 × M  ） 张量，`MAT2`是 （ M  × p  ） （M
\倍p） （ M  × p  ） 张量，然后`输入 `必须[ broadcastable
](notes/broadcasting.html#broadcasting-semantics)与 （ n的 × p  ） （N \倍p ） （ n的 ×
p  ） 张量和`OUT`将是 （ n的 ×  p  ） （N \倍p） （ n的 × P  ） 张量。

`阿尔法 `和`的β `的之间`MAT1`和上的矩阵矢量乘积的缩放因子`MAT2`和分别添加的矩阵`输入 `。

out=β input+α (mat1i@mat2i)\text{out} = \beta\ \text{input} + \alpha\
(\text{mat1}_i \mathbin{@} \text{mat2}_i) out=β input+α (mat1i​@mat2i​)

对于类型的输入 FloatTensor 或 DoubleTensor ，自变量`的β `和`阿尔法 `必须实数，否则他们应该是整数。

Parameters

    

  * **beta** ( _Number_ _,_ _optional_ ) – multiplier for `input`(β\betaβ )

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – matrix to be added

  * **阿尔法** （ _号码_ _，_ _可选_ ） - 乘数 M  一 T  1  @  M  一 T  2  MAT1 @ MAT2  M  一 T  1  @  M  一 吨 2  （ α \阿尔法 α ）

  * **MAT1** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 要被相乘的第一矩阵

  * **MAT2** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 第二矩阵相乘

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> M = torch.randn(2, 3)
    >>> mat1 = torch.randn(2, 3)
    >>> mat2 = torch.randn(3, 3)
    >>> torch.addmm(M, mat1, mat2)
    tensor([[-4.8716,  1.4671, -1.3746],
            [ 0.7573, -3.9555, -2.8681]])
    

`torch.``addmv`( _beta=1_ , _input_ , _alpha=1_ , _mat_ , _vec_ , _out=None_ )
→ Tensor

    

执行矩阵`垫 `和的矩阵矢量乘积矢量`VEC`。矢量`输入 `加到最终结果。

如果`垫 `是 （ n的 × M  ） （N \乘以m） （ n的 × M  ） 张量，`VEC`是大小[1-d张量HTG56] M ，然后`输入
`必须[ broadcastable  ](notes/broadcasting.html#broadcasting-semantics)与大小的1-d张量
n的和`OUT`将大小 n的 1- d张量。

`阿尔法 `和`的β `是比例上的矩阵矢量乘积因子之间`垫 `和`VEC`和分别添加的张量`输入 `。

out=β input+α (mat@vec)\text{out} = \beta\ \text{input} + \alpha\ (\text{mat}
\mathbin{@} \text{vec}) out=β input+α (mat@vec)

对于类型的输入 FloatTensor 或 DoubleTensor ，自变量`的β `和`阿尔法 `必须实数，否则他们应该是整数

Parameters

    

  * **beta** ( _Number_ _,_ _optional_ ) – multiplier for `input`(β\betaβ )

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 要添加矢量

  * **阿尔法** （ _号码_ _，_ _可选_ ） - 乘数 M  一 T  @  [HTG22】v  E  C  垫@ VEC  M  一 T  @  [HTG46】v  E  C  （ α \阿尔法 α ）

  * **垫** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 矩阵相乘

  * **VEC** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 向量相乘

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> M = torch.randn(2)
    >>> mat = torch.randn(2, 3)
    >>> vec = torch.randn(3)
    >>> torch.addmv(M, mat, vec)
    tensor([-0.3768, -5.5565])
    

`torch.``addr`( _beta=1_ , _input_ , _alpha=1_ , _vec1_ , _vec2_ , _out=None_
) → Tensor

    

执行的矢量`VEC1`和`VEC2`外产物，并将其添加到矩阵`输入 `。

可选值`的β `和`阿尔法 `是在间`外积缩放因子VEC1`和`VEC2`和分别添加的矩阵`输入 `。

out=β input+α (vec1⊗vec2)\text{out} = \beta\ \text{input} + \alpha\
(\text{vec1} \otimes \text{vec2}) out=β input+α (vec1⊗vec2)

如果`VEC1`是大小 n的的向量和`VEC2`是大小 M [向量HTG11]，然后`输入 `必须[ broadcastable
](notes/broadcasting.html#broadcasting-semantics)用的大小 [基质HTG24]  （ n的 × M  ）
（N \倍米） （ n的 ×  M  ） 和`OUT`将大小的矩阵 （ n的 × M  ） （N \乘以m）[HTG9 0]  （ n的 × M  ）
。

For inputs of type FloatTensor or DoubleTensor, arguments `beta`and `alpha`
must be real numbers, otherwise they should be integers

Parameters

    

  * **beta** ( _Number_ _,_ _optional_ ) – multiplier for `input`(β\betaβ )

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – matrix to be added

  * **阿尔法** （ _号码_ _，_ _可选_ ） - 乘数 VEC1  ⊗ VEC2  \文本{VEC1} \ otimes \文本{VEC2}  VEC1  ⊗ VEC2  （ α \阿尔法 α ）

  * **VEC1** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 外积的第一向量

  * **VEC2** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 外积的第二矢量

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> vec1 = torch.arange(1., 4.)
    >>> vec2 = torch.arange(1., 3.)
    >>> M = torch.zeros(3, 2)
    >>> torch.addr(M, vec1, vec2)
    tensor([[ 1.,  2.],
            [ 2.,  4.],
            [ 3.,  6.]])
    

`torch.``baddbmm`( _beta=1_ , _input_ , _alpha=1_ , _batch1_ , _batch2_ ,
_out=None_ ) → Tensor

    

执行矩阵的批次矩阵矩阵产物`BATCH1`和`batch2`。 `输入 `加到最终结果。

`BATCH1`和`batch2`必须各自含有相同数量的矩阵的3-d张量。

如果`BATCH1`是 （ B  × n的 × M  ） （b \ n次\乘以m） （ b  × n的 × M  ） 张量，`batch2`是
（ b  × M  × p  ） （b \倍米\倍p） [HT G100]  （ B  × M  × p  ） 张量，然后`输入 `必须[
broadcastable  ](notes/broadcasting.html#broadcasting-semantics)与 （ b  × n的 ×
p  ） （b \ n次\倍p） （ b  × n的 × P  ） 张量和`OUT`将是 （ b  × n的 × p  ） （b \ n次\倍p） （
b  × n的 × p  ​​） 张量。既`阿尔法 `和`的β `意味着相同于 `torch.addbmm所使用的比例因子（ ） `。

outi=β inputi+α (batch1i@batch2i)\text{out}_i = \beta\ \text{input}_i +
\alpha\ (\text{batch1}_i \mathbin{@} \text{batch2}_i) outi​=β inputi​+α
(batch1i​@batch2i​)

For inputs of type FloatTensor or DoubleTensor, arguments `beta`and `alpha`
must be real numbers, otherwise they should be integers.

Parameters

    

  * **beta** ( _Number_ _,_ _optional_ ) – multiplier for `input`(β\betaβ )

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tensor to be added

  * **阿尔法** （ _号码_ _，_ _可选_ ） - 乘数 BATCH1  @  batch2  \文本{BATCH1} \ mathbin {@} \文本{batch2}  BATCH1  @  batch2  （ α \阿尔法 α ）

  * **batch1** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the first batch of matrices to be multiplied

  * **batch2** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the second batch of matrices to be multiplied

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> M = torch.randn(10, 3, 5)
    >>> batch1 = torch.randn(10, 3, 4)
    >>> batch2 = torch.randn(10, 4, 5)
    >>> torch.baddbmm(M, batch1, batch2).size()
    torch.Size([10, 3, 5])
    

`torch.``bmm`( _input_ , _mat2_ , _out=None_ ) → Tensor

    

对存储在`输入 `和`MAT2`矩阵的批次矩阵矩阵乘积。

`输入 `和`MAT2`必须各自含有相同数量的矩阵的3-d张量。

如果`输入 `是 （ B  × n的 × M  ） （b \ n次\乘以m） （ b  × n的 × M  ） 张量，`MAT2`是 （ b  ×
M  × p  ） （b \倍米\倍p） [HTG10 0]  （ B  × M  × p  ） 张量，`OUT`将是 （ b  × n的 × p
） （b \ n次\倍p） （ b  × n的 ×  [HTG19 3]  P  ） 张量。

outi=inputi@mat2i\text{out}_i = \text{input}_i \mathbin{@} \text{mat2}_i
outi​=inputi​@mat2i​

Note

此功能不播[ [HTG3。用于广播基质的产品，见](notes/broadcasting.html#broadcasting-semantics) `
torch.matmul（） `。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 矩阵的第一批要被乘

  * **MAT2** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 第二批矩阵的相乘

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> input = torch.randn(10, 3, 4)
    >>> mat2 = torch.randn(10, 4, 5)
    >>> res = torch.bmm(input, mat2)
    >>> res.size()
    torch.Size([10, 3, 5])
    

`torch.``bitwise_not`( _input_ , _out=None_ ) → Tensor

    

计算给定输入张量的位NOT。输入必须是整数或布尔类型。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example

    
    
    >>> torch.bitwise_not(torch.tensor([-1, -2, 3], dtype=torch.int8))
    tensor([ 0,  1, -4], dtype=torch.int8)
    

`torch.``chain_matmul`( _*matrices_
)[[source]](_modules/torch/functional.html#chain_matmul)

    

返回的矩阵积的 N  N  N  2-d张量。此产物用其选择其中招致算术操作方面的成本最低的（[ [CLRS]
](https://mitpress.mit.edu/books/introduction-algorithms-third-
edition)）的顺序进行矩阵链顺序算法有效地计算。注意，由于这是一个函数来计算的产物， N  N  N  需要为大于或等于2
;如果等于2，则一个简单的矩阵矩阵产品退还。如果 N  N  N  为1，那么这是一个无操作 - 原始矩阵返回原样。

Parameters

    

**矩阵** （ _张量..._ ） - 2以上2-d张量，其产物是待确定的序列。

Returns

    

如果 i的 T  H  I ^ {个}  i的 T  H  张量是尺寸 p  i的 × p  i的 \+  1  P_ {I} \倍P_ {I + 1}
p  i的 × p  i的 \+  1  ，则产品将是尺寸 p的 1  × p  N  \+  1  P_ {1} \ TI MES P_ {N + 1}
P  1  × p  N  \+  1  ​​  。

Return type

    

[Tensor](tensors.html#torch.Tensor "torch.Tensor")

Example:

    
    
    >>> a = torch.randn(3, 4)
    >>> b = torch.randn(4, 5)
    >>> c = torch.randn(5, 6)
    >>> d = torch.randn(6, 7)
    >>> torch.chain_matmul(a, b, c, d)
    tensor([[ -2.3375,  -3.9790,  -4.1119,  -6.6577,   9.5609, -11.5095,  -3.2614],
            [ 21.4038,   3.3378,  -8.4982,  -5.2457, -10.2561,  -2.4684,   2.7163],
            [ -0.9647,  -5.8917,  -2.3213,  -5.2284,  12.8615, -12.2816,  -2.5095]])
    

`torch.``cholesky`( _input_ , _upper=False_ , _out=None_ ) → Tensor

    

计算对称正定矩阵 A  A  [的Cholesky分解HTG12]  A  或对称正定矩阵的批次。

如果`上 `是`真 `，返回的矩阵`U`是上三角，和分解的形式为：

A=UTUA = U^TUA=UTU

如果`上 `是`假 `，返回的矩阵`L`是下三角，和分解的形式为：

A=LLTA = LL^TA=LLT

如果`上 `是`真 `和 A  A  A  是分批对称正定矩阵，则返回的张量将组成每个单独的矩阵的上三角的Cholesky因素的。类似地，当`上
`是`假 `，返回的张量将组成的每一个单独的矩阵的下三角的Cholesky因素。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入张量 A  A  A  大小的 （ *  ， n的 ， n的 ） （*，N，N） （ *  ， n的 ， n的 ） 其中 * 是零个或多个选自由对称正定的批次尺寸矩阵。

  * **上** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 标志，指示是否以返回上或下三角矩阵。默认值：`假 `

  * **OUT** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 输出矩阵

Example:

    
    
    >>> a = torch.randn(3, 3)
    >>> a = torch.mm(a, a.t()) # make symmetric positive-definite
    >>> l = torch.cholesky(a)
    >>> a
    tensor([[ 2.4112, -0.7486,  1.4551],
            [-0.7486,  1.3544,  0.1294],
            [ 1.4551,  0.1294,  1.6724]])
    >>> l
    tensor([[ 1.5528,  0.0000,  0.0000],
            [-0.4821,  1.0592,  0.0000],
            [ 0.9371,  0.5487,  0.7023]])
    >>> torch.mm(l, l.t())
    tensor([[ 2.4112, -0.7486,  1.4551],
            [-0.7486,  1.3544,  0.1294],
            [ 1.4551,  0.1294,  1.6724]])
    >>> a = torch.randn(3, 2, 2)
    >>> a = torch.matmul(a, a.transpose(-1, -2)) + 1e-03 # make symmetric positive-definite
    >>> l = torch.cholesky(a)
    >>> z = torch.matmul(l, l.transpose(-1, -2))
    >>> torch.max(torch.abs(z - a)) # Max non-zero
    tensor(2.3842e-07)
    

`torch.``cholesky_inverse`( _input_ , _upper=False_ , _out=None_ ) → Tensor

    

计算对称正定矩阵 A  的倒数A  A  使用其的Cholesky因数 U  U  U  ：返回矩阵`INV`。逆使用LAPACK例程`
dpotri计算 `和`spotri`（和相应的MAGMA例程）。

如果`上 `是`假 `， U  U  U  是下三角使得返回的张量是

inv=(uuT)−1inv = (uu^{T})^{-1} inv=(uuT)−1

如果`上 `是`真 `或没有提供， U  U  U  是上三角使得返回的张量是

inv=(uTu)−1inv = (u^T u)^{-1} inv=(uTu)−1

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入2-d张量 U  U  U  ，一个上或下三角的Cholesky因数

  * **上** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 是否返回低级（默认）或上三角矩阵

  * **OUT** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 输出张量为 INV 

Example:

    
    
    >>> a = torch.randn(3, 3)
    >>> a = torch.mm(a, a.t()) + 1e-05 * torch.eye(3) # make symmetric positive definite
    >>> u = torch.cholesky(a)
    >>> a
    tensor([[  0.9935,  -0.6353,   1.5806],
            [ -0.6353,   0.8769,  -1.7183],
            [  1.5806,  -1.7183,  10.6618]])
    >>> torch.cholesky_inverse(u)
    tensor([[ 1.9314,  1.2251, -0.0889],
            [ 1.2251,  2.4439,  0.2122],
            [-0.0889,  0.2122,  0.1412]])
    >>> a.inverse()
    tensor([[ 1.9314,  1.2251, -0.0889],
            [ 1.2251,  2.4439,  0.2122],
            [-0.0889,  0.2122,  0.1412]])
    

`torch.``cholesky_solve`( _input_ , _input2_ , _upper=False_ , _out=None_ ) →
Tensor

    

解决方程与半正定矩阵的线性系统被倒置给予其的Cholesky因数矩阵 U  U  U  。

如果`上 `是`假 `， U  U  U  是和和 C 被返回下三角使得：

c=(uuT)−1bc = (u u^T)^{-1} b c=(uuT)−1b

如果`上 `是`真 `或没有提供， U  U  U  是上三角和 C 被返回，使得：

c=(uTu)−1bc = (u^T u)^{-1} b c=(uTu)−1b

torch.cholesky_solve（B，U）可以在2D输入 B，U 或是2D矩阵的批输入。如果输入是批次，然后返回成批输出 C

Note

的`OUT`关键字仅支持2D矩阵输入，即， B，U 必须2D矩阵。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入矩阵 B  b  b  大小的 （ *  ， M  ， K  ） （*，M，K） （ *  ， M  ， K  ） ，其中 *  *  *  [H TG102]  是零点或多个批次的尺寸

  * **输入2** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入矩阵 U  U  U  大小的 （ *  ， M  ， M  ） （*，M，M） （ *  ， M  ， M  ） ，其中 *  *  *  为上限或下三角的Cholesky因数组成多个批处理尺寸的零

  * **上** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 是否考虑的Cholesky因数作为下或上三角矩阵。默认值：`假 [HTG13。`

  * **OUT** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 输出张量为 C 

Example:

    
    
    >>> a = torch.randn(3, 3)
    >>> a = torch.mm(a, a.t()) # make symmetric positive definite
    >>> u = torch.cholesky(a)
    >>> a
    tensor([[ 0.7747, -1.9549,  1.3086],
            [-1.9549,  6.7546, -5.4114],
            [ 1.3086, -5.4114,  4.8733]])
    >>> b = torch.randn(3, 2)
    >>> b
    tensor([[-0.6355,  0.9891],
            [ 0.1974,  1.4706],
            [-0.4115, -0.6225]])
    >>> torch.cholesky_solve(b, u)
    tensor([[ -8.1625,  19.6097],
            [ -5.8398,  14.2387],
            [ -4.3771,  10.4173]])
    >>> torch.mm(a.inverse(), b)
    tensor([[ -8.1626,  19.6097],
            [ -5.8398,  14.2387],
            [ -4.3771,  10.4173]])
    

`torch.``dot`( _input_ , _tensor_ ) → Tensor

    

计算两个张量的点积（内积）。

Note

此功能不播[ [HTG3。](notes/broadcasting.html#broadcasting-semantics)

Example:

    
    
    >>> torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))
    tensor(7)
    

`torch.``eig`( _input_ , _eigenvectors=False_ , _out=None) - > (Tensor_,
_Tensor_ )

    

计算实方阵的特征值和特征向量。

Note

因为特征向量可能是复杂的，向后通仅对支持`torch.symeig（） `

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的形状 方阵（  n的 × n的 ） （N \ n次） （ n的 × n的 ） 的量，特征向量将被计算

  * **本征向量** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - `真 `来计算两个特征向量;否则，只有特征值将计算

  * **OUT** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选_ ） - 输出张量

Returns

    

将含有namedtuple（本征值，本征矢量）

>   * **本征值** （ _张量_ ）：形状 （ n的 × 2  ） （N \次2） （ n的 × 2  ） 。每一行是输入
，其中，所述第一元件是实部和所述第二元件是虚数部分的`的本征值。特征值不一定有序。`

>

>   * **本征向量** （ _张量_ ）：如果`本征向量=假 `，它是一个空的张量。否则，该张量的形状 （ n的 × n的 ） （N \ n次） （
n的 × n的 ） 可以被用于计算归一化的（单元长度）如下对应的特征值的特征向量。如果相应的本征值[J] 是一个实数，柱本征向量[:,
j]的是对应于本征值[j]的本征向量。如果相应的本征值[J] 和本征值[J + 1] 形成的复共轭对，则真正的本征向量可被计算为 真实特征矢量 [
[HTG76：J  =  E  i的 克 E  n的 [HTG92】v  E  C  T  O  R  S  [ ： ， [ HTG112：J  \+
i的 × E  i的 G  E  n的 [HTG132】v  E  C  T  问题o  R  S  [ ： ， [HTG152：J  \+  1  \
{文本特征向量真} [j]的本征向量= [：，j]的+ I \倍特征向量[：，J + 1]  真实特征矢量 [ [HTG176：J  =  E  i的 克
E  n的 [HTG200】v  E  C  T  O  R  S  [ ： ， [HTG226：J  \+  i的 × E  i的 克 E  n的
[HTG262】v  E  C  T  ​​ O  R  S  [ ： ， [HTG288：J  \+  1  ， 真实特征矢量 [ [HTG318：J
\+  1  =  E  i的 克 E  n的 [HTG338】V  E  C  T  O  R  S  [ ： ， [HTG358：J  \-  i的 ×
E  i的 G  E  n的 [HTG378】v  E  C  T  O  R  S  [ ： ， [HTG398：J  \+  1  \
{文本特征向量真} [J + 1] =特征向量[:, j]的 - I \倍特征向量[:, J + 1]  [HTG4 16] 真实特征矢量 [
[HTG422：J  \+  1  =  E  i的 克 E  n的 [HTG458】v  E  C  T  O  R  S  [ ： ，
[HTG484：J  \-  i的 × E  i的 克 E  n的 [HTG520】V  E  C  T  O  R  S  [ ： ， [HTG546：J
\+  1  。

>

>

Return type

    

（[张量](tensors.html#torch.Tensor "torch.Tensor")，[张量](tensors.html#torch.Tensor
"torch.Tensor")）

`torch.``gels`( _input_ , _A_ , _out=None_
)[[source]](_modules/torch/functional.html#gels)

    

计算的解最小二乘和最小范数问题为满秩矩阵 A  A  A  大小的 （ M  × n的 ） （M \ n次） （ M  × n的 ） 和一个矩阵 B  B
B  大小的 （ M  × K  ） （M \倍K） （ M  × K  ） 。

有关 `torch.gels（）更多信息 `，请检查 `torch.lstsq（） `。

Warning

`torch.gels（） `以有利于 `torch.lstsq的（废弃） `，将在未来的版本中删除。请使用 `torch.lstsq（） `
代替。

`torch.``geqrf`( _input_ , _out=None) - > (Tensor_, _Tensor_ )

    

这是直接调用LAPACK一个低级别的功能。此函数返回如[ LAPACK文档定义geqrf ](https://software.intel.com/en-
us/node/521004)一个namedtuple（一，tau蛋白）。

您通常需要使用 `torch.qr（） `代替。

计算`输入的QR分解 `，但没有构建 Q  Q  Q  和 R  R  R  作为明确的分离矩阵。

相反，该直接调用底层LAPACK函数 geqrf产生“基本反射”的序列。

参见[HTG0对于geqrf 为进一步的细节LAPACK文档。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入矩阵

  * **OUT** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选_ ） - 的输出元组（张量，张量）

`torch.``ger`( _input_ , _vec2_ , _out=None_ ) → Tensor

    

的`输入 `和`VEC2`外积。如果`输入 `是大小为向量 n的 N  n的 和`VEC2`是大小为向量 M  M  M  ，然后`OUT
`必须的大小 （ n的 × [HTG80一个矩阵] M ） （N \乘以m） （ n的 × M  ） 。

Note

This function does not [broadcast](notes/broadcasting.html#broadcasting-
semantics).

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 1-d输入矢量

  * **VEC2** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 1-d输入矢量

  * **OUT** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 可选的输出矩阵

Example:

    
    
    >>> v1 = torch.arange(1., 5.)
    >>> v2 = torch.arange(1., 4.)
    >>> torch.ger(v1, v2)
    tensor([[  1.,   2.,   3.],
            [  2.,   4.,   6.],
            [  3.,   6.,   9.],
            [  4.,   8.,  12.]])
    

`torch.``inverse`( _input_ , _out=None_ ) → Tensor

    

取正方形矩阵`输入 `的倒数。 `输入 `可以是2D方形张量，在这种情况下，这函数将返回一个个体逆组成张量的批次。

Note

不管原始进展的，返回的张量将被转置，即具有如步幅input.contiguous（）。转置（-2，-1）.stride（）

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 大小 的输入张量（  *  ， n的 ， n的 ） （*，N，N） （ *  n的 ， n的 ） 其中 * 是零点或多个批次的尺寸

  * **OUT** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 可选输出张量

Example:

    
    
    >>> x = torch.rand(4, 4)
    >>> y = torch.inverse(x)
    >>> z = torch.mm(x, y)
    >>> z
    tensor([[ 1.0000, -0.0000, -0.0000,  0.0000],
            [ 0.0000,  1.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  1.0000,  0.0000],
            [ 0.0000, -0.0000, -0.0000,  1.0000]])
    >>> torch.max(torch.abs(z - torch.eye(4))) # Max non-zero
    tensor(1.1921e-07)
    >>> # Batched inverse example
    >>> x = torch.randn(2, 3, 4, 4)
    >>> y = torch.inverse(x)
    >>> z = torch.matmul(x, y)
    >>> torch.max(torch.abs(z - torch.eye(4).expand_as(x))) # Max non-zero
    tensor(1.9073e-06)
    

`torch.``det`( _input_ ) → Tensor

    

计算方阵或方阵的批次的决定因素。

Note

向后通过 `DET（） `在内部使用时`输入 `不可逆SVD的结果。在这种情况下，双向后通过 `DET（） `将在当`输入
`没有不稳定不同奇异值。参见 `对于细节SVD（） `。

Parameters

    

**输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 大小的输入张量（ *
中，n，n）其中 * 是零米或多个批次的尺寸。

Example:

    
    
    >>> A = torch.randn(3, 3)
    >>> torch.det(A)
    tensor(3.7641)
    
    >>> A = torch.randn(3, 2, 2)
    >>> A
    tensor([[[ 0.9254, -0.6213],
             [-0.5787,  1.6843]],
    
            [[ 0.3242, -0.9665],
             [ 0.4539, -0.0887]],
    
            [[ 1.1336, -0.4025],
             [-0.7089,  0.9032]]])
    >>> A.det()
    tensor([1.1990, 0.4099, 0.7386])
    

`torch.``logdet`( _input_ ) → Tensor

    

计算方阵或方阵的批次的日志决定因素。

Note

结果是`-INF`如果`输入 `具有零日志行列式，并且是`楠 `如果`输入 `具有负的决定因素。

Note

向后通过 `logdet（） `在内部使用时`输入 `不可逆SVD的结果。在这种情况下，双向后通过 `logdet（） `将在当`输入
`没有不稳定不同奇异值。参见 `对于细节SVD（） `。

Parameters

    

**输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 大小的输入张量（ *
中，n，n）其中 * 是零米或多个批次的尺寸。

Example:

    
    
    >>> A = torch.randn(3, 3)
    >>> torch.det(A)
    tensor(0.2611)
    >>> torch.logdet(A)
    tensor(-1.3430)
    >>> A
    tensor([[[ 0.9254, -0.6213],
             [-0.5787,  1.6843]],
    
            [[ 0.3242, -0.9665],
             [ 0.4539, -0.0887]],
    
            [[ 1.1336, -0.4025],
             [-0.7089,  0.9032]]])
    >>> A.det()
    tensor([1.1990, 0.4099, 0.7386])
    >>> A.det().log()
    tensor([ 0.1815, -0.8917, -0.3031])
    

`torch.``slogdet`( _input) - > (Tensor_, _Tensor_ )

    

计算的符号和记录一个方阵或方阵的批次的行列式（S）的绝对值。

Note

如果`输入 `具有零行列式，这将返回`（0， -INF） `。

Note

向后通过 `slogdet（） `在内部使用时`输入 `不可逆SVD的结果。在这种情况下，双向后通过 `slogdet（） `将在当`输入
`没有不稳定不同奇异值。参见 `对于细节SVD（） `。

Parameters

    

**输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 大小的输入张量（ *
中，n，n）其中 * 是零米或多个批次的尺寸。

Returns

    

将含有行列式的符号namedtuple（标志，logabsdet），并且绝对行列式的log值。

Example:

    
    
    >>> A = torch.randn(3, 3)
    >>> A
    tensor([[ 0.0032, -0.2239, -1.1219],
            [-0.6690,  0.1161,  0.4053],
            [-1.6218, -0.9273, -0.0082]])
    >>> torch.det(A)
    tensor(-0.7576)
    >>> torch.logdet(A)
    tensor(nan)
    >>> torch.slogdet(A)
    torch.return_types.slogdet(sign=tensor(-1.), logabsdet=tensor(-0.2776))
    

`torch.``lstsq`( _input_ , _A_ , _out=None_ ) → Tensor

    

Computes the solution to the least squares and least norm problems for a full
rank matrix AAA of size (m×n)(m \times n)(m×n) and a matrix BBB of size
(m×k)(m \times k)(m×k) .

如果 M  ≥ n的 米\ GEQÑ  M  ≥ n的 ， `lstsq（） `解决了最小平方问题：

min⁡X∥AX−B∥2.\begin{array}{ll} \min_X & \|AX-B\|_2.
\end{array}minX​​∥AX−B∥2​.​

如果 M  & LT ;  n的 M & LT ; N  M  & LT ;  n的 ， `lstsq（） `解决了至少范数问题：

min⁡X∥X∥2subject toAX=B.\begin{array}{ll} \min_X & \|X\|_2 & \text{subject to}
& AX = B. \end{array}minX​​∥X∥2​​subject to​AX=B.​

返回张量 X  X  X  具有形状 （ MAX  ⁡ （ M  ， n的 ）  × K  ） （\ MAX（M，N）\倍K） （ MAX  （ M  ，
n的 ） × K  ） 。第一 n的 n的 n的 的行 X  X  X  包含溶液。如果 M  ≥ n的 米\ GEQÑ  M  ≥ n的
，正方形的用于每一列中的溶液中的剩余之和由平方和给定在剩余的 M  元素 -  n的 米 - N  M  \-  n的 该列的行。

Note

的情况下，当 M  & LT ;  n的 M & LT ; N  M  & LT ;  n的 不支持在GPU上。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 矩阵 B  B  B 

  * **A** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的 M  M  M  通过 n的 n的 n的 矩阵 A  A  A 

  * **OUT** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选_ ） - 可选的目的地张量

Returns

    

将含有namedtuple（溶液，QR）：

>   * **溶液** （ _张量_ ）：最小二乘解

>

>   * **QR** （ _张量_ ）：将QR分解的细节

>

>

Return type

    

([Tensor](tensors.html#torch.Tensor "torch.Tensor"),
[Tensor](tensors.html#torch.Tensor "torch.Tensor"))

Note

返回的矩阵总是会换位不论输入矩阵的进步。也就是说，他们将有步幅（1，M）而非（M，1）。

Example:

    
    
    >>> A = torch.tensor([[1., 1, 1],
                          [2, 3, 4],
                          [3, 5, 2],
                          [4, 2, 5],
                          [5, 4, 3]])
    >>> B = torch.tensor([[-10., -3],
                          [ 12, 14],
                          [ 14, 12],
                          [ 16, 16],
                          [ 18, 16]])
    >>> X, _ = torch.lstsq(B, A)
    >>> X
    tensor([[  2.0000,   1.0000],
            [  1.0000,   1.0000],
            [  1.0000,   2.0000],
            [ 10.9635,   4.8501],
            [  8.9332,   5.2418]])
    

`torch.``lu`( _A_ , _pivot=True_ , _get_infos=False_ , _out=None_
)[[source]](_modules/torch/functional.html#lu)

    

计算方阵或方阵`A`的批次的LU分解。返回包含LU分解和`A`枢转的元组。如果`枢 `设置为`真 `旋转完成。

Note

由该函数返回的枢轴是1-索引。如果`枢轴 `是`假 `，则返回的枢转是填充有适当大小的零的张量。

Note

LU分解与`枢 `= `假 `不适用于CPU，并试图这样做会引发错误。然而，LU分解与`枢轴 `= `假 `可用于CUDA。

Note

此功能不检查分解是否成功，如果`get_infos`是`真 `由于分解的状态出现在返回的元组的第三个元素。

Parameters

    

  * **A** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 张量对因子大小的 （ *  ， M  ， M  ） （*，M，M） （ *  M  ， M  ）

  * **枢轴** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 枢转控制是否已完成。默认值：`真 `

  * **get_infos** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果设定为`真 `，返回一个信息IntTensor。默认值：`假 `

  * **OUT** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选_ ） - 可选的输出元组。如果`get_infos`是`真 `，然后在所述元组的元素是张量，IntTensor，和IntTensor。如果`get_infos`是`假 `，然后在所述元组的元素是张量，IntTensor。默认值：`无 `

Returns

    

张量的含有A元组

>   * **因式分解** （ _张量_ ）：的大小 因式分解（ *  ， M  ， M  ） （*，M，M ） （ *  ， M  ， M  ）

>

>   * **枢轴** （ _IntTensor_ ）：大小 的枢轴（ *  ， M  ） （*，M） （ *  ， M  ）

>

>   * **的相关信息** （ _IntTensor_ ， _可选_ ）：如果`get_infos`是`真 `，这是大小 （ *  ）的张量
（*） （ *  ）  其中非零值指示因式分解对矩阵或每个minibatch是否成功或失败

>

>

Return type

    

（[张量](tensors.html#torch.Tensor "torch.Tensor")，IntTensor，IntTensor（可选））

Example:

    
    
    >>> A = torch.randn(2, 3, 3)
    >>> A_LU, pivots = torch.lu(A)
    >>> A_LU
    tensor([[[ 1.3506,  2.5558, -0.0816],
             [ 0.1684,  1.1551,  0.1940],
             [ 0.1193,  0.6189, -0.5497]],
    
            [[ 0.4526,  1.2526, -0.3285],
             [-0.7988,  0.7175, -0.9701],
             [ 0.2634, -0.9255, -0.3459]]])
    >>> pivots
    tensor([[ 3,  3,  3],
            [ 3,  3,  3]], dtype=torch.int32)
    >>> A_LU, pivots, info = torch.lu(A, get_infos=True)
    >>> if info.nonzero().size(0) == 0:
    ...   print('LU factorization succeeded for all samples!')
    LU factorization succeeded for all samples!
    

`torch.``lu_solve`( _input_ , _LU_data_ , _LU_pivots_ , _out=None_ ) → Tensor

    

返回LU求解线性系统 的A  × =  b  Ax = b的 A  × =  b  [HTG43使用来自 `torch.lu（） `
A的部分枢转LU分解。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的大小 的RHS张量（  b  ， M  ， K  ） （b，M，K） （ b  M  ， K  ）

  * **LU_data** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 从 `A的枢转LU分解torch.lu（） `大小的 （ b  ， M  ， M  ） （b，M，M） （ b  ， M  ， M  ）

  * **LU_pivots** （ _IntTensor_ ） - 的LU分解的从 `枢轴torch.lu（） `的大小 （ b  ， M  ） （b，M） （ b  ， M  ）

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the optional output tensor

Example:

    
    
    >>> A = torch.randn(2, 3, 3)
    >>> b = torch.randn(2, 3, 1)
    >>> A_LU = torch.lu(A)
    >>> x = torch.lu_solve(b, *A_LU)
    >>> torch.norm(torch.bmm(A, x) - b)
    tensor(1.00000e-07 *
           2.8312)
    

`torch.``lu_unpack`( _LU_data_ , _LU_pivots_ , _unpack_data=True_ ,
_unpack_pivots=True_ )[[source]](_modules/torch/functional.html#lu_unpack)

    

从张量的LU分解解包的数据，并枢转。

返回张量的元组为`（在 枢转时， 中的 L  张量， 的 U  张量） `。

Parameters

    

  * **LU_data** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 打包LU分解数据

  * **LU_pivots** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 填充LU分解枢转

  * **unpack_data** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 标志，指示如果数据应被解压缩

  * **unpack_pivots** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 标志，指示如果枢轴应解压

Example:

    
    
    >>> A = torch.randn(2, 3, 3)
    >>> A_LU, pivots = A.lu()
    >>> P, A_L, A_U = torch.lu_unpack(A_LU, pivots)
    >>>
    >>> # can recover A from factorization
    >>> A_ = torch.bmm(P, torch.bmm(A_L, A_U))
    

`torch.``matmul`( _input_ , _other_ , _out=None_ ) → Tensor

    

2张量的矩阵产品。

该行为取决于张量的维数如下：

  * 如果两个张量是1维的，点积（标量）被返回。

  * 如果两个参数是2维的，则返回矩阵，矩阵产品。

  * 如果第一个参数是1维的，并且第二个参数是2维的，一个1被预置到其尺寸为矩阵乘法的目的。的矩阵乘法后，将预谋尺寸被去除。

  * 如果第一个参数是2维的，并且第二个参数是1维的，则返回矩阵矢量乘积。

  * 如果两个参数是至少一维和至少一个参数是N维（N & GT其中; 2），则返回一个批处理矩阵乘法。如果第一个参数是1维的，一个1被预置到其尺寸为成批矩阵乘法的目的，并且之后被去除。如果第二个参数是1维的，1被附加到其尺寸为成批矩阵的多个目的和之后被去除。非矩阵（即批）尺寸[ 广播 ](notes/broadcasting.html#broadcasting-semantics)（并因此必须是broadcastable）。例如，如果`输入 `是 （ [HTG16：J × 1  × n的 × M  ） （j \倍1 \ n次\乘以m） （ [ HTG44：J  × 1  × n的 × M  ） 张量和`其他 `是 （ K  ×[H TG103]  M  × P  ） （K \倍米\倍P） （ K  × M  × p  ） 张量，`OUT`将是 （ [HTG168：J × K  × n的 × p  ） （j \乘K \ n次\倍p） （ [HTG196：J  × K  × n的 × p  ） 张量。

Note

该函数的1维的点积版本不支持`OUT`参数。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 要被相乘的第一张量

  * **其他** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 要被相乘的第二张量

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> # vector x vector
    >>> tensor1 = torch.randn(3)
    >>> tensor2 = torch.randn(3)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([])
    >>> # matrix x vector
    >>> tensor1 = torch.randn(3, 4)
    >>> tensor2 = torch.randn(4)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([3])
    >>> # batched matrix x broadcasted vector
    >>> tensor1 = torch.randn(10, 3, 4)
    >>> tensor2 = torch.randn(4)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([10, 3])
    >>> # batched matrix x batched matrix
    >>> tensor1 = torch.randn(10, 3, 4)
    >>> tensor2 = torch.randn(10, 4, 5)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([10, 3, 5])
    >>> # batched matrix x broadcasted matrix
    >>> tensor1 = torch.randn(10, 3, 4)
    >>> tensor2 = torch.randn(4, 5)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([10, 3, 5])
    

`torch.``matrix_power`( _input_ , _n_ ) → Tensor

    

返回幂`n的 `为方阵矩阵。对于批量矩阵，每个单独的矩阵被升高到功率`n的 `。

如果`n的 `是否定的，则矩阵（如果可逆）的逆被升高到功率`n的 `。对于间歇矩阵，成批的逆（如果可逆）提高到电源`n的 `。如果`n的
`为0，则单位矩阵被返回。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  * **n的** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 功率，以提高基质中以

Example:

    
    
    >>> a = torch.randn(2, 2, 2)
    >>> a
    tensor([[[-1.9975, -1.9610],
             [ 0.9592, -2.3364]],
    
            [[-1.2534, -1.3429],
             [ 0.4153, -1.4664]]])
    >>> torch.matrix_power(a, 3)
    tensor([[[  3.9392, -23.9916],
             [ 11.7357,  -0.2070]],
    
            [[  0.2468,  -6.7168],
             [  2.0774,  -0.8187]]])
    

`torch.``matrix_rank`( _input_ , _tol=None_ , _bool symmetric=False_ ) →
Tensor

    

返回一个2-d张量的数值等级。计算矩阵的秩的方法是使用SVD默认情况下完成的。如果`对称 `是`真 `，然后`输入
`被假设为是对称的，并且排名的计算是通过获取特征值来完成。

`TOL`是低于该奇异值（或本征值时`对称 `是`真阈值`）都被认为是0。如果`TOL`没有指定，`TOL`被设定为`S.max（）
*  最大值（S.size（）） *  EPS`其中 S 为奇异值（或本征值时`对称 `是`真 `）和`EPS`是用于`输入
`的数据类型的ε值。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入2-d张量

  * **TOL** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 公差值。默认值：`无 `

  * **对称** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 指示是否`输入 `是对称的。默认值：`假 `

Example:

    
    
    >>> a = torch.eye(10)
    >>> torch.matrix_rank(a)
    tensor(10)
    >>> b = torch.eye(10)
    >>> b[0, 0] = 0
    >>> torch.matrix_rank(b)
    tensor(9)
    

`torch.``mm`( _input_ , _mat2_ , _out=None_ ) → Tensor

    

执行矩阵`输入 `和`MAT2`的矩阵乘法。

如果`输入 `是 （ n的 × M  ） （N \乘以m） （ n的 × M  ） 张量，`MAT2`是 （ M  × p  ） （M \倍p） （
M  × p  ） 张量，`OUT`将是 （ N  × p  ） （N \倍p） （ n的 × p  ） 张量。

Note

This function does not [broadcast](notes/broadcasting.html#broadcasting-
semantics). For broadcasting matrix products, see `torch.matmul()`.

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 要被相乘的第一矩阵

  * **mat2** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the second matrix to be multiplied

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> mat1 = torch.randn(2, 3)
    >>> mat2 = torch.randn(3, 3)
    >>> torch.mm(mat1, mat2)
    tensor([[ 0.4851,  0.5037, -0.3633],
            [-0.0760, -3.6705,  2.4784]])
    

`torch.``mv`( _input_ , _vec_ , _out=None_ ) → Tensor

    

执行矩阵`输入 `和的矩阵矢量乘积矢量`VEC`。

如果`输入 `是 （ n的 × M  ） （N \乘以m） （ n的 × M  ） 张量，`VEC`是大小[1-d张量HTG56]  M  M
[大小的HTG72]  M  ，`OUT`将1-d  n的 n的 n的 。

Note

This function does not [broadcast](notes/broadcasting.html#broadcasting-
semantics).

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 要被相乘的矩阵

  * **vec** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – vector to be multiplied

  * **out** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – the output tensor

Example:

    
    
    >>> mat = torch.randn(2, 3)
    >>> vec = torch.randn(3)
    >>> torch.mv(mat, vec)
    tensor([ 1.0404, -0.6361])
    

`torch.``orgqr`( _input_ , _input2_ ) → Tensor

    

计算正交矩阵 Q 一个QR分解的，从（输入，输入2）元组通过返回 `torch.geqrf（） `。

这直接调用底层的LAPACK函数 orgqr [HTG1。参见[HTG2对于orgqr 为进一步的细节LAPACK文档。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的一从 `torch.geqrf（） `。

  * **输入2** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的 tau蛋白从 `torch.geqrf（） `。

`torch.``ormqr`( _input_ , _input2_ , _input3_ , _left=True_ ,
_transpose=False_ ) → Tensor

    

乘法垫通过 `[形成的QR分解的正交 Q 矩阵HTG10（由`输入3`给出） ] torch.geqrf（） `由（A，tau蛋白）表示（由（`
输入 `，[HTG20给出] 输入2  ））。

这直接调用底层的LAPACK函数 ormqr [HTG1。参见[ LAPACK文档ormqr
](https://software.intel.com/en-us/mkl-developer-reference-c-ormqr)为进一步的细节。

Parameters

    

  * **input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the a from `torch.geqrf()`.

  * **input2** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – the tau from `torch.geqrf()`.

  * **输入3** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 要相乘的矩阵。

`torch.``pinverse`( _input_ , _rcond=1e-15_ ) → Tensor

    

计算2D张量的伪逆（也被称为Moore-Penrose逆）。请看[ Moore-
Penrose逆](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)更多细节

Note

该方法是使用奇异值分解来实现。

Note

伪逆不一定是矩阵[ [1]
](https://epubs.siam.org/doi/10.1137/0117004)的元素的连续函数。因此，衍生物并不总是存在的，并且存在对于恒定秩只[
[2]
](https://www.jstor.org/stable/2156365)。但是，这种方法是backprop，由于能够通过使用SVD结果的执行，可能是不稳定的。双落后也将是不稳定的，由于SVD的使用内部。参见
`SVD（） `的更多细节。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的输入的2D张量尺寸 M  × n的 米\ n次 M  × n的

  * **rcond** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 甲浮点值来确定用于小奇异值截止。默认值：1E-15

Returns

    

的`输入尺寸的`n的 ×伪逆 M  n的\乘以m  n的 × M

Example:

    
    
    >>> input = torch.randn(3, 5)
    >>> input
    tensor([[ 0.5495,  0.0979, -1.4092, -0.1128,  0.4132],
            [-1.1143, -0.3662,  0.3042,  1.6374, -0.9294],
            [-0.3269, -0.5745, -0.0382, -0.5922, -0.6759]])
    >>> torch.pinverse(input)
    tensor([[ 0.0600, -0.1933, -0.2090],
            [-0.0903, -0.0817, -0.4752],
            [-0.7124, -0.1631, -0.2272],
            [ 0.1356,  0.3933, -0.5023],
            [-0.0308, -0.1725, -0.5216]])
    

`torch.``qr`( _input_ , _some=True_ , _out=None) - > (Tensor_, _Tensor_ )

    

计算一个矩阵的QR分解或分批矩阵`输入 `的，并返回张量的namedtuple（Q，R），使得 输入 =  Q  R  \文本{输入} = QR  输入
=  Q  R  与 Q  Q  Q  为正交矩阵的正交矩阵或分批和 R  R  R  为上三角矩阵或批量上三角矩阵。

如果`一些 `是`真 `，则该函数将返回薄（减小）QR分解。否则，如果`一些 `是`假 `，该函数返回完整的QR分解。

Note

如果`输入的元素的量值 `是大精度可能会丢失

Note

虽然它应该总是给你一个有效的分解，它可能不会给你跨平台的同一个 - 这将取决于你的LAPACK实现。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 大小 的输入张量（  *  ， M  ， n的 ） （*，M，N） （ *  M  ， n的 ） 其中 * 是零个或多个由尺寸 M 矩阵[批量尺寸HTG68]×  n的 米\ n次 M  × n的 。

  * **一些** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 设置为`真 [ HTG13用于减少QR分解和`[HTG15用于完整QR分解假 `。`

  * **OUT** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选_ ） - 的 Q 和元组 [R 张量满足`输入 =  torch.matmul（Q， R） `。  Q 和的尺寸 R 是 （ *  ， M  ， K  ） （*，M，K） （ *  ， M  ， K  ） 和 （ *  ， K  ， n的 ） （*，K，N） （ *  ， K  ， n的 ） 分别，其中 K  =  分钟HTG143] ⁡ （ M  ， n的 ） K = \分钟（M，N） K  =  分钟HTG179] （ M  ， n的 ） 如果`一些： `是`真 `和 K  =  M  K = M  K  =  M  否则。

Example:

    
    
    >>> a = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
    >>> q, r = torch.qr(a)
    >>> q
    tensor([[-0.8571,  0.3943,  0.3314],
            [-0.4286, -0.9029, -0.0343],
            [ 0.2857, -0.1714,  0.9429]])
    >>> r
    tensor([[ -14.0000,  -21.0000,   14.0000],
            [   0.0000, -175.0000,   70.0000],
            [   0.0000,    0.0000,  -35.0000]])
    >>> torch.mm(q, r).round()
    tensor([[  12.,  -51.,    4.],
            [   6.,  167.,  -68.],
            [  -4.,   24.,  -41.]])
    >>> torch.mm(q.t(), q).round()
    tensor([[ 1.,  0.,  0.],
            [ 0.,  1., -0.],
            [ 0., -0.,  1.]])
    >>> a = torch.randn(3, 4, 5)
    >>> q, r = torch.qr(a, some=False)
    >>> torch.allclose(torch.matmul(q, r), a)
    True
    >>> torch.allclose(torch.matmul(q.transpose(-2, -1), q), torch.eye(5))
    True
    

`torch.``solve`( _input_ , _A_ , _out=None) - > (Tensor_, _Tensor_ )

    

该函数返回溶液到线性方程组由 A  X  = [所表示的系统HTG11]  B  AX = B  A  X  =  B
和A的LU分解，为了作为namedtuple 溶液，LU 。

LU 包含 L 和 U 因素的 A  LU分解。

torch.solve（B，A）可以在那些2D矩阵的批次的2D输入端 B，A 或输入。如果输入是批次，然后返回成批输出溶液，LU 。

Note

不管原始进展的，则返回的矩阵溶液和 LU 将被转置，即具有如步幅B.contiguous（）。转置（-1，-2）。步幅（）和
A.contiguous（）。转置（-1，-2）.stride（）分别。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入矩阵 B  B  B  大小的 （ *  ， M  ， K  ） （*，M，K） （ *  ， M  ， K  ） ，其中 *  *  *  是零米或多个批次的尺寸。

  * **A** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入尺寸 的方矩阵（  *  ， M  ， M  ） （*，M，M） （ *  M  ， M  ） ，其中 *  *  *  是零米或多个批次的尺寸。

  * **OUT** （ _（_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _）_ _，_ _可选_ ） - 可选的输出元组。

Example:

    
    
    >>> A = torch.tensor([[6.80, -2.11,  5.66,  5.97,  8.23],
                          [-6.05, -3.30,  5.36, -4.44,  1.08],
                          [-0.45,  2.58, -2.70,  0.27,  9.04],
                          [8.32,  2.71,  4.35,  -7.17,  2.14],
                          [-9.67, -5.14, -7.26,  6.08, -6.87]]).t()
    >>> B = torch.tensor([[4.02,  6.19, -8.22, -7.57, -3.03],
                          [-1.56,  4.00, -8.67,  1.75,  2.86],
                          [9.81, -4.09, -4.57, -8.61,  8.99]]).t()
    >>> X, LU = torch.solve(B, A)
    >>> torch.dist(B, torch.mm(A, X))
    tensor(1.00000e-06 *
           7.0977)
    
    >>> # Batched solver example
    >>> A = torch.randn(2, 3, 1, 4, 4)
    >>> B = torch.randn(2, 3, 1, 4, 6)
    >>> X, LU = torch.solve(B, A)
    >>> torch.dist(B, A.matmul(X))
    tensor(1.00000e-06 *
       3.6386)
    

`torch.``svd`( _input_ , _some=True_ , _compute_uv=True_ , _out=None) - >
(Tensor_, _Tensor_ , _Tensor_ )

    

该函数返回一个namedtuple `（U， S， V） `这是一个输入实矩阵或批次的奇异值分解实数矩阵`输入 `，使得 i的 n的 p  U  T
=  U  × d  i的 一 克 （ S  ） × [HTG51】V  T  输入= U \倍DIAG（S）\倍于V ^ T  i的 n的 p  U  T
=  U  × d  i的 一 克 （ S  ） × [HTG123】V  T  。

如果`一些 `是`真 `（默认），则该方法返回降低奇异值分解，即，如果[HTG8最后两个维度] 输入 是`M`和`n的 `，则返回的 U
和[HTG22】V 矩阵将仅包含 M  i的 n的 （ n的 ， M  ） 分钟（N，M） M  i的 n的 （ n的 ， M  ） 正交列。

如果`compute_uv`是`假 `，返回 U 和[HTG10】V 矩阵将是零的形状 （ 米矩阵 × M  ） （M \乘以m） （ M  × M
） 和 （ n的 ×  n的 ） （N \ n次） （ n的 × n的 ） 分别。 `一些 `将在这里被忽略了。

Note

SVD的CPU上的实现使用LAPACK例程 gesdd （分而治之算法）代替 gesvd [HTG3用于速度。类似地，在GPU上的SVD使用MAGMA例程
gesdd 为好。

Note

不管原始进展的，则返回的矩阵 U 将被转置，即具有步幅`U.contiguous（）。转置（-2， -1）。步幅（） `

Note

格外小心需要通过 U 和[HTG2】V 输出时向后服用。这样的操作真的只有稳定时`输入 `与所有不同的奇异值满秩。否则，`的NaN
`可以作为梯度未正确定义出现。此外，请注意双向后通常将通过 U 和做附加的倒V型即使原始向后只在 S 。

Note

当`一些 `= `假 `，在`梯度U [...， ： 分钟（米， N）：]`和`[HTG19】V [...， ： 分钟（米， N）：]
`将在向后忽略那些载体可以是子空间的任意碱。

Note

当`compute_uv`= `假 `，向后不能因为 U 进行并[HTG10】V 从直传是所必需的向后操作。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 大小 的输入张量（  *  ， M  ， n的 ） （*，M，N） （ *  M  ， n的 ） 其中 * 是零个或多个选自由M ×的 批次尺寸 n的 米\ n次 M  × n的 矩阵。

  * **一些** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 控制的返回形状U 和[HTG12】V 

  * **compute_uv** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 选项是否计算 U 和[ HTG12】V 或不

  * **OUT** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选_ ） - 张量的输出元组

Example:

    
    
    >>> a = torch.randn(5, 3)
    >>> a
    tensor([[ 0.2364, -0.7752,  0.6372],
            [ 1.7201,  0.7394, -0.0504],
            [-0.3371, -1.0584,  0.5296],
            [ 0.3550, -0.4022,  1.5569],
            [ 0.2445, -0.0158,  1.1414]])
    >>> u, s, v = torch.svd(a)
    >>> u
    tensor([[ 0.4027,  0.0287,  0.5434],
            [-0.1946,  0.8833,  0.3679],
            [ 0.4296, -0.2890,  0.5261],
            [ 0.6604,  0.2717, -0.2618],
            [ 0.4234,  0.2481, -0.4733]])
    >>> s
    tensor([2.3289, 2.0315, 0.7806])
    >>> v
    tensor([[-0.0199,  0.8766,  0.4809],
            [-0.5080,  0.4054, -0.7600],
            [ 0.8611,  0.2594, -0.4373]])
    >>> torch.dist(a, torch.mm(torch.mm(u, torch.diag(s)), v.t()))
    tensor(8.6531e-07)
    >>> a_big = torch.randn(7, 5, 3)
    >>> u, s, v = torch.svd(a_big)
    >>> torch.dist(a_big, torch.matmul(torch.matmul(u, torch.diag_embed(s)), v.transpose(-2, -1)))
    tensor(2.6503e-06)
    

`torch.``symeig`( _input_ , _eigenvectors=False_ , _upper=True_ , _out=None) -
> (Tensor_, _Tensor_ )

    

该函数返回一个实对称矩阵`输入 `或间歇实对称矩阵，由namedtuple（本征值，本征矢量）表示的特征向量。

此函数计算所有特征值（和载体）的`输入 `，使得 输入 =  [HTG14】V  DIAG  （ E  ） [HTG25】V  T  \文本{输入} =
V \文本{DIAG}（E）V ^ T  输入 =  [HTG54】V  DIAG  （ E  ） [HTG67 】V  T  。

布尔参数`本征向量 `定义了本征向量和本征值仅或特征值的计算。

如果是`假 `，只有特征值计算。如果是`真 `，二者特征向量被计算。

由于输入矩阵`输入 `应该是对称的，仅上三角形部分默认使用。

如果`上 `是`假 `，则使用下三角部分。

Note

不管原始进展的，则返回的矩阵[HTG0】V 将被转置，即，步幅 V.contiguous（）。转置（-1，-2）.stride（）。

Note

格外小心，需要通过输出落后时采取。这样的操作实在是唯一的稳定，当所有特征值是不同的。否则，`的NaN`可以作为梯度未正确定义出现。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 大小 的输入张量（  *  ， n的 ， n的 ） （*，N，N） （ *  n的 ， n的 ） 其中 * 是零米或多个由对称矩阵的批次的尺寸。

  * **本征向量** （ _布尔_ _，_ _可选_ ） - 控制特征向量是否必须计算

  * **上** （ _布尔_ _，_ _可选_ ） - 控制是否考虑上三角或下三角区

  * **out** ([ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _,_ _optional_ ) – the output tuple of (Tensor, Tensor)

Returns

    

A namedtuple (eigenvalues, eigenvectors) containing

>   * **本征值** （ _张量_ ）：形状 （ *  ， M  ） （*，M） （ *  ， M  ） 。按升序排列的特征值。

>

>   * **本征向量** （ _张量_ ）：形状 （ *  ， M  ， M  ） （*，M，M） （ *  ， M  ， M  ） 。如果`
的特征向量=假 `，它是用零填充的张量。否则，该张量包含`输入 `的正交的特征向量。

>

>

Return type

    

([Tensor](tensors.html#torch.Tensor "torch.Tensor"),
[Tensor](tensors.html#torch.Tensor "torch.Tensor"))

Examples:

    
    
    >>> a = torch.randn(5, 5)
    >>> a = a + a.t()  # To make a symmetric
    >>> a
    tensor([[-5.7827,  4.4559, -0.2344, -1.7123, -1.8330],
            [ 4.4559,  1.4250, -2.8636, -3.2100, -0.1798],
            [-0.2344, -2.8636,  1.7112, -5.5785,  7.1988],
            [-1.7123, -3.2100, -5.5785, -2.6227,  3.1036],
            [-1.8330, -0.1798,  7.1988,  3.1036, -5.1453]])
    >>> e, v = torch.symeig(a, eigenvectors=True)
    >>> e
    tensor([-13.7012,  -7.7497,  -2.3163,   5.2477,   8.1050])
    >>> v
    tensor([[ 0.1643,  0.9034, -0.0291,  0.3508,  0.1817],
            [-0.2417, -0.3071, -0.5081,  0.6534,  0.4026],
            [-0.5176,  0.1223, -0.0220,  0.3295, -0.7798],
            [-0.4850,  0.2695, -0.5773, -0.5840,  0.1337],
            [ 0.6415, -0.0447, -0.6381, -0.0193, -0.4230]])
    >>> a_big = torch.randn(5, 2, 2)
    >>> a_big = a_big + a_big.transpose(-2, -1)  # To make a_big symmetric
    >>> e, v = a_big.symeig(eigenvectors=True)
    >>> torch.allclose(torch.matmul(v, torch.matmul(e.diag_embed(), v.transpose(-2, -1))), a_big)
    True
    

`torch.``trapz`()

    

`torch.``trapz`( _y_ , _x_ , _*_ , _dim=-1_ ) → Tensor

    

估计 ∫ Y  d  ×  \ INT Y \，DX  ∫ Y  d  × 沿暗淡使用梯形规则。

Parameters

    

  * **Y** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 该函数的值，以整合

  * **×** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 在该函数 Y 被采样的点。如果×不是升序排列，间隔在其上降低到估计的积分（即，公约 [负贡献HTG16] ∫ 一 b  F  =  \-  ∫ b  一 F  \ int_a ^ BF = - \ int_b ^ AF  ∫ 一 b  F  =  [HT G98]  \-  ∫ b  一 F  之后）。

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 沿其尺寸集成。默认情况下，使用最后一个维度。

Returns

    

具有相同形状的输入，除了与一种张量暗淡除去。返回的张量的每个元素代表所估计的积分 ∫ Y  d  × \ INT Y \，DX  ∫ Y  d  ×
沿着暗淡。

Example:

    
    
    >>> y = torch.randn((2, 3))
    >>> y
    tensor([[-2.1156,  0.6857, -0.2700],
            [-1.2145,  0.5540,  2.0431]])
    >>> x = torch.tensor([[1, 3, 4], [1, 2, 3]])
    >>> torch.trapz(y, x)
    tensor([-1.2220,  0.9683])
    

`torch.``trapz`( _y_ , _*_ , _dx=1_ , _dim=-1_ ) → Tensor

    

如上述，但采样点被均匀地以 DX 的距离间隔开。

Parameters

    

  * **y** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – The values of the function to integrate

  * **DX** （[ _浮_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 的距离，其中 Y 被采样点之间。

  * **dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – The dimension along which to integrate. By default, use the last dimension.

Returns

    

A Tensor with the same shape as the input, except with dim removed. Each
element of the returned tensor represents the estimated integral ∫y dx\int
y\,dx∫ydx along dim.

`torch.``triangular_solve`( _input_ , _A_ , _upper=True_ , _transpose=False_ ,
_unitriangular=False) - > (Tensor_, _Tensor_ )

    

解决了方程系统具有三角形系数矩阵 A  A  A  和多个右手侧 b  b  b  。

特别是，解决了 A  X  =  B  AX = b  A  X  =  b  并假定 A  A  A  是上三角与缺省关键字参数。

torch.triangular_solve（B，A）可以在2D输入 B，A 或是2D矩阵的批输入。如果输入是批次，然后返回成批输出 X

Note

的`OUT`关键字仅支持2D矩阵输入，即， B，A 必须2D矩阵。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 大小 [HTG12的多个右手侧]（  *  ， M  ， K  ） （*，M，K） （ *  ， M  ， K  ） 其中 *  *  *  是更批量尺寸的零（ b  b  B  ）

  * **A** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 大小 [的输入三角形系数矩阵HTG12 ]（  *  ， M  ， M  ） （*，M，M） （ *  ， M  ， M  ） 其中 *  *  *  是零点或多个批次的尺寸

  * **上** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 是否求解方程的上三角系统（默认）或方程的下三角系统。默认值：`真 [HTG13。`

  * **转置** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 是否 A  A  A  应当被发送到求解器之前被调换。默认值：`假 [HTG37。`

  * **unitriangular** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 是否 A  A  A  是单位三角形。如果为True，的 A  的对角元素A  A  被假设为1，而不是从引用 A  A  A  。默认值：`假 [HTG85。`

Returns

    

甲namedtuple （溶液，cloned_coefficient）其中 cloned_coefficient 是 克隆A  A  A  和溶液为溶液 X
X  X  至 A  X  =  b  AX = b  A  X  =  b  （或任何方程系统，根据所述关键字参数的变体）。

Examples:

    
    
    >>> A = torch.randn(2, 2).triu()
    >>> A
    tensor([[ 1.1527, -1.0753],
            [ 0.0000,  0.7986]])
    >>> b = torch.randn(2, 3)
    >>> b
    tensor([[-0.0210,  2.3513, -1.5492],
            [ 1.5429,  0.7403, -1.0243]])
    >>> torch.triangular_solve(b, A)
    torch.return_types.triangular_solve(
    solution=tensor([[ 1.7841,  2.9046, -2.5405],
            [ 1.9320,  0.9270, -1.2826]]),
    cloned_coefficient=tensor([[ 1.1527, -1.0753],
            [ 0.0000,  0.7986]]))
    

## 公用事业

`torch.``compiled_with_cxx11_abi`()[[source]](_modules/torch.html#compiled_with_cxx11_abi)

    

返回PyTorch是否与_GLIBCXX_USE_CXX11_ABI = 1建成

[Next ![](_static/images/chevron-right-orange.svg)](tensors.html
"torch.Tensor") [![](_static/images/chevron-right-orange.svg)
Previous](community/persons_of_interest.html "PyTorch Governance | Persons of
Interest")

* * *

©版权所有2019年，Torch 贡献者。
