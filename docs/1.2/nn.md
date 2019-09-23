# torch.nn

## 参数

_class_`torch.nn.``Parameter`[[source]](_modules/torch/nn/parameter.html#Parameter)

    

有种张量将被认为是模块参数。

参数是[ `张量 `](tensors.html#torch.Tensor "torch.Tensor")亚类，具有 `模块 `
[HTG11使用时，是具有一个非常特殊的属性] S - 当他们指定为模块属性，它们会自动添加到其参数列表，并会出现如在 `参数（） `
迭代器。指定一个张量并没有这样的效果。这是因为人们可能会想缓存一些临时的状态，就像RNN的最后一个隐藏状态，在模型中。如果没有这样的类为 `参数 `
，这些临时工将获得注册过。

Parameters

    

  * **数据** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 参数张量。

  * **requires_grad** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果参数需要梯度。看到从向后 [ 不包括子图的更多细节。默认值：真](notes/autograd.html#excluding-subgraphs)

## 容器

### 模块

_class_`torch.nn.``Module`[[source]](_modules/torch/nn/modules/module.html#Module)

    

基类的所有神经网络模块。

你的车型也应该继承这个类。

模块也可以包含其他的模块，允许其嵌套在一个树结构。您可以分配的子模块的常规属性：

    
    
    import torch.nn as nn
    import torch.nn.functional as F
    
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)
    
        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))
    

以这种方式分配的子模块将被注册，并有自己的参数转换过，当你调用 `以（） `等。

`add_module`( _name_ , _module_
)[[source]](_modules/torch/nn/modules/module.html#Module.add_module)

    

添加一个子模块，当前模块。

该模块可以作为使用给定名称的属性进行访问。

Parameters

    

  * **名** （ _串_ ） - 子模块的名称。子模块可以从该模块使用给定的名称来访问

  * **模块** （ _模块_ ） - 子模块被添加到该模块。

`apply`( _fn_ )[[source]](_modules/torch/nn/modules/module.html#Module.apply)

    

适用`FN`递归地对每个子模块以及自（如由`。儿童（） `返回）。典型用途包括初始化一个模型的参数（也见炬-NN-INIT ）。

Parameters

    

**FN** （ `模块 `\- & GT ;无） - 函数被应用到每个子模块

Returns

    

自

Return type

    

模块

例：

    
    
    >>> def init_weights(m):
    >>>     print(m)
    >>>     if type(m) == nn.Linear:
    >>>         m.weight.data.fill_(1.0)
    >>>         print(m.weight)
    >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    >>> net.apply(init_weights)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[ 1.,  1.],
            [ 1.,  1.]])
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[ 1.,  1.],
            [ 1.,  1.]])
    Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )
    Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )
    

`buffers`( _recurse=True_
)[[source]](_modules/torch/nn/modules/module.html#Module.buffers)

    

返回在模块缓冲区的迭代器。

Parameters

    

**递归** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in
Python v3.7\)")） - 如果为True，则产生该模块和所有子模块的缓冲器。否则，仅产生是该模块的直接成员的缓冲区。

Yields

    

_torch.Tensor_ \- 模块缓冲器

Example:

    
    
    >>> for buf in model.buffers():
    >>>     print(type(buf.data), buf.size())
    <class 'torch.FloatTensor'> (20L,)
    <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)
    

`children`()[[source]](_modules/torch/nn/modules/module.html#Module.children)

    

返回在即时儿童模块的迭代器。

Yields

    

_模块_ \- 一个子模块

`cpu`()[[source]](_modules/torch/nn/modules/module.html#Module.cpu)

    

移动所有模型参数和缓冲区的CPU。

Returns

    

self

Return type

    

Module

`cuda`( _device=None_
)[[source]](_modules/torch/nn/modules/module.html#Module.cuda)

    

移动所有模型参数和缓冲区的GPU。

这也使得相关的参数和缓冲区不同的对象。所以应该构建优化模块是否将生活在GPU同时进行优化之前被调用。

Parameters

    

**装置** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in
Python v3.7\)") _，_ _可选_ ） - 如果指定，所有的参数将被复制到该设备

Returns

    

self

Return type

    

Module

`double`()[[source]](_modules/torch/nn/modules/module.html#Module.double)

    

施放所有浮点参数和缓冲液以`双 `数据类型。

Returns

    

self

Return type

    

Module

`dump_patches`_= False_

    

这允许更好BC支持`load_state_dict（） `。在 `state_dict（） `，版本号将被保存为在返回的状态字典的属性
_metadata ，因此酸洗。  _metadata 是与后面状态字典的命名约定键的字典。参见`_load_from_state_dict
`如何在加载使用该信息。

如果添加了新的参数/缓冲器/从模块中取出，这个数字将被碰撞，以及模块的 _load_from_state_dict
方法可以比较的版本号，并做适当的修改，如果状态字典是从改变之前。

`eval`()[[source]](_modules/torch/nn/modules/module.html#Module.eval)

    

设置在评估模式下的模块。

这只有在某些模块没有任何影响。见特定模块的单证在培训/评估模式，如果他们受到影响，例如他们的行为的细节 `降 `，`BatchNorm`等

这相当于与 `self.train（假） `。

Returns

    

self

Return type

    

Module

`extra_repr`()[[source]](_modules/torch/nn/modules/module.html#Module.extra_repr)

    

设置模块的额外代表性

要打印定制额外的信息，你应该在你自己的模块重新实现此方法。既单行和多行字符串是可接受的。

`float`()[[source]](_modules/torch/nn/modules/module.html#Module.float)

    

施放所有浮点参数和缓冲区浮动数据类型。

Returns

    

self

Return type

    

Module

`forward`( _*input_
)[[source]](_modules/torch/nn/modules/module.html#Module.forward)

    

定义在每个呼叫进行计算。

应该由所有子类覆盖。

注意

虽然必须在这个函数中定义的直传食谱，应该叫 `模块 `实例之后，而不是这个，因为前者需要运行的护理注册挂钩，而后者默默地忽略它们。

`half`()[[source]](_modules/torch/nn/modules/module.html#Module.half)

    

施放所有浮点参数和缓冲液以`一半 `数据类型。

Returns

    

self

Return type

    

Module

`load_state_dict`( _state_dict_ , _strict=True_
)[[source]](_modules/torch/nn/modules/module.html#Module.load_state_dict)

    

份参数和缓冲液从 `state_dict`到这个模块及其后代。如果`严格 `是`真 `，则 `键state_dict`
必须完全符合本模块的 `state_dict（） `函数的返回键。

Parameters

    

  * **state_dict** （[ _DICT_ ](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.7\)")） - 包含参数和持久性缓冲区的字典。

  * **严格** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 是否严格执行，在键`state_dict`匹配由该模块的 `state_dict（） `函数返回的键。默认值：`真 `

Returns

    

  * **missing_keys** 是STR的包含丢失密钥的列表

  * **unexpected_keys** 是STR的含有意想不到的键的列表

Return type

    

`NamedTuple`与`missing_keys`和`unexpected_keys`字段

`modules`()[[source]](_modules/torch/nn/modules/module.html#Module.modules)

    

返回在网络中的所有模块的迭代器。

Yields

    

网络中的模块 - _模块_

Note

重复模块只返回一次。在以下示例中，`L`将只返回一次。

Example:

    
    
    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.modules()):
            print(idx, '->', m)
    
    0 -> Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )
    1 -> Linear(in_features=2, out_features=2, bias=True)
    

`named_buffers`( _prefix=''_ , _recurse=True_
)[[source]](_modules/torch/nn/modules/module.html#Module.named_buffers)

    

返回在模块缓冲区的迭代器，产生缓冲中的两个名字以及缓冲区本身。

Parameters

    

  * **前缀** （[ _STR_ ](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)")） - 前缀时，预先准备所有缓冲器的名字。

  * **recurse** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")) – if True, then yields buffers of this module and all submodules. Otherwise, yields only buffers that are direct members of this module.

Yields

    

_（字符串，torch.Tensor）_ \- 元组包含名称和缓冲

Example:

    
    
    >>> for name, buf in self.named_buffers():
    >>>    if name in ['running_var']:
    >>>        print(buf.size())
    

`named_children`()[[source]](_modules/torch/nn/modules/module.html#Module.named_children)

    

返回在即时儿童模块的迭代器，产生模块的两个名字，以及模块本身。

Yields

    

_（字符串，模块）_ \- 含有名称和子模块的Tuple

Example:

    
    
    >>> for name, module in model.named_children():
    >>>     if name in ['conv4', 'conv5']:
    >>>         print(module)
    

`named_modules`( _memo=None_ , _prefix=''_
)[[source]](_modules/torch/nn/modules/module.html#Module.named_modules)

    

返回网络，在所有模块的迭代器，产生模块的两个名字，以及模块本身。

Yields

    

_（字符串，模块）_ \- 名称和模块的元组

Note

Duplicate modules are returned only once. In the following example, `l`will
be returned only once.

Example:

    
    
    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.named_modules()):
            print(idx, '->', m)
    
    0 -> ('', Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    ))
    1 -> ('0', Linear(in_features=2, out_features=2, bias=True))
    

`named_parameters`( _prefix=''_ , _recurse=True_
)[[source]](_modules/torch/nn/modules/module.html#Module.named_parameters)

    

返回在模块参数的迭代器，产生参数的两个名称以及参数本身。

Parameters

    

  * **前缀** （[ _海峡_ ](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)")） - 前缀预先考虑到所有的参数名称。

  * **递归** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 如果为True，则产生该模块和所有子模块的参数。否则，仅产生是该模块的直接成员参数。

Yields

    

_（字符串，参数）_ \- 包含元组的名称和参数

Example:

    
    
    >>> for name, param in self.named_parameters():
    >>>    if name in ['bias']:
    >>>        print(param.size())
    

`parameters`( _recurse=True_
)[[source]](_modules/torch/nn/modules/module.html#Module.parameters)

    

返回在模块参数的迭代器。

这通常是通过给优化。

Parameters

    

**recurse** ([ _bool_](https://docs.python.org/3/library/functions.html#bool
"\(in Python v3.7\)")) – if True, then yields parameters of this module and
all submodules. Otherwise, yields only parameters that are direct members of
this module.

Yields

    

_参数_ \- 模块参数

Example:

    
    
    >>> for param in model.parameters():
    >>>     print(type(param.data), param.size())
    <class 'torch.FloatTensor'> (20L,)
    <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)
    

`register_backward_hook`( _hook_
)[[source]](_modules/torch/nn/modules/module.html#Module.register_backward_hook)

    

寄存器模块上的向后钩。

钩将被称为每次相对于梯度以模块输入被计算。钩子应该具有以下特征：

    
    
    hook(module, grad_input, grad_output) -> Tensor or None
    

的`grad_input`和`grad_output
`可以是元组如果模块具有多个输入或输出。钩不应修改其参数，但它可以任选地返回一个新的梯度相对于输入，将代替`grad_input`在随后的计算中使用。

Returns

    

可以使用的一个手柄通过调用`handle.remove（）以去除所添加的钩 `

Return type

    

`torch.utils.hooks.RemovableHandle`

警告

当前的实现不会对复杂的 `模块 `执行许多操作所呈现的行为。在一些失败的情况下，`grad_input`和`grad_output
`将只包含对的输入和输出的一个子集的梯度​​。对于这样的 `模块 `，则应该使用[ `torch.Tensor.register_hook（） `
](tensors.html#torch.Tensor.register_hook
"torch.Tensor.register_hook")直接在一个特定的输入或输出，以获得所需的梯度。

`register_buffer`( _name_ , _tensor_
)[[source]](_modules/torch/nn/modules/module.html#Module.register_buffer)

    

添加一个持久缓冲区到模块。

这通常是用于注册不应被认为是一个模型参数的缓冲器。例如，BatchNorm的`running_mean`不是参数，但是持久状态的一部分。

缓冲区可以为使用给定的名称属性来访问。

Parameters

    

  * **名称** （ _串_ ） - 缓冲的名称。缓冲器可以从该模块使用给定的名称来访问

  * **张量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 缓冲液进行注册。

Example:

    
    
    >>> self.register_buffer('running_mean', torch.zeros(num_features))
    

`register_forward_hook`( _hook_
)[[source]](_modules/torch/nn/modules/module.html#Module.register_forward_hook)

    

寄存器模块上的前钩。

钩将被称为每次之后 `向前（） `已经计算的输出。它应该具有以下特征：

    
    
    hook(module, input, output) -> None or modified output
    

钩可以修改的输出。它可以修改输入就地，但它不会对转发的影响，因为这是后 `向前称为（） `被调用。

Returns

    

a handle that can be used to remove the added hook by calling
`handle.remove()`

Return type

    

`torch.utils.hooks.RemovableHandle`

`register_forward_pre_hook`( _hook_
)[[source]](_modules/torch/nn/modules/module.html#Module.register_forward_pre_hook)

    

寄存器模块上的前预挂钩。

钩将每次调用之前 `向前（） `被调用。它应该具有以下特征：

    
    
    hook(module, input) -> None or modified input
    

钩可以修改输入。用户可以返回的元组或在钩的单个修饰的值。最后，我们将值插入到一个元组如果返回一个值（除非该值已经是一个元组）。

Returns

    

a handle that can be used to remove the added hook by calling
`handle.remove()`

Return type

    

`torch.utils.hooks.RemovableHandle`

`register_parameter`( _name_ , _param_
)[[source]](_modules/torch/nn/modules/module.html#Module.register_parameter)

    

添加一个参数到模块。

该参数可以作为使用定名称的属性来访问。

Parameters

    

  * **名称** （ _串_ ） - 参数的名称。该参数可以从该模块使用给定的名称来访问

  * **PARAM** （ _参数_ ） - 参数被添加到该模块。

`requires_grad_`( _requires_grad=True_
)[[source]](_modules/torch/nn/modules/module.html#Module.requires_grad_)

    

改变，如果autograd应在此模块中的参数记录等操作。

此方法设置的参数`requires_grad`就地属性。

此方法是用于微调或单独训练的模型的部分（例如，GAN培训）冷冻所述模块的一部分有帮助的。

Parameters

    

**requires_grad** （[ _布尔_
](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")）
- autograd是否应该在该模块中的参数记录操作。默认值：`真 [HTG9。`

Returns

    

self

Return type

    

Module

`state_dict`( _destination=None_ , _prefix=''_ , _keep_vars=False_
)[[source]](_modules/torch/nn/modules/module.html#Module.state_dict)

    

返回包含模块的整体状态的字典。

这两个参数和持久性缓冲液（例如运行平均值）也包括在内。键对应的参数和缓冲区名字。

Returns

    

包含模块的整个状态的字典

Return type

    

[字典](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python
v3.7\)")

Example:

    
    
    >>> module.state_dict().keys()
    ['bias', 'weight']
    

`to`( _*args_ , _**kwargs_
)[[source]](_modules/torch/nn/modules/module.html#Module.to)

    

移动和/或注塑参数和缓冲区。

这可以被称为

`to`( _device=None_ , _dtype=None_ , _non_blocking=False_
)[[source]](_modules/torch/nn/modules/module.html#Module.to)

    

`to`( _dtype_ , _non_blocking=False_
)[[source]](_modules/torch/nn/modules/module.html#Module.to)

    

`to`( _tensor_ , _non_blocking=False_
)[[source]](_modules/torch/nn/modules/module.html#Module.to)

    

它的签名是类似于[ `torch.Tensor.to（） `](tensors.html#torch.Tensor.to
"torch.Tensor.to")，但仅接受浮点所需的`DTYPE`秒。此外，这种方法将只投浮点参数和缓冲液以`DTYPE
`（如有）。积分参数和缓冲液将被移至`装置 `，如果给定，但与dtypes不变。当`non_blocking
`被设定，它会尝试转换/如果可能异步相对于移动到主机，例如，移动CPU张量与固定内存到CUDA设备。

请参见下面的例子。

Note

此方法会修改就地模块。

Parameters

    

  * **装置** （`torch.device`） - 的参数和缓冲器在该模块中的所期望的设备

  * 此模块中的浮点参数和缓冲液的所希望的浮点类型 - **DTYPE** （`torch.dtype`）

  * **张量** （[ _torch.Tensor_ ](tensors.html#torch.Tensor "torch.Tensor")） - 张量，其D型细胞和装置所需的D型细胞和装置此模块中的所有参数和缓冲器

Returns

    

self

Return type

    

Module

Example:

    
    
    >>> linear = nn.Linear(2, 2)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]])
    >>> linear.to(torch.double)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]], dtype=torch.float64)
    >>> gpu1 = torch.device("cuda:1")
    >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
    >>> cpu = torch.device("cpu")
    >>> linear.to(cpu)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16)
    

`train`( _mode=True_
)[[source]](_modules/torch/nn/modules/module.html#Module.train)

    

设置在训练模式下的模块。

This has any effect only on certain modules. See documentations of particular
modules for details of their behaviors in training/evaluation mode, if they
are affected, e.g. `Dropout`, `BatchNorm`, etc.

Parameters

    

**模式** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in
Python v3.7\)")） - 是否设定训练模式（`真 `）或评估模式（`假 `）。默认值：`真 [HTG17。`

Returns

    

self

Return type

    

Module

`type`( _dst_type_
)[[source]](_modules/torch/nn/modules/module.html#Module.type)

    

施放的所有参数和缓冲液以`dst_type`。

Parameters

    

**dst_type** （[ _输入_ ](https://docs.python.org/3/library/functions.html#type
"\(in Python v3.7\)") _或_ _串_ ） - 所需的类型

Returns

    

self

Return type

    

Module

`zero_grad`()[[source]](_modules/torch/nn/modules/module.html#Module.zero_grad)

    

将所有模型参数为零的梯度。

### 序贯

_class_`torch.nn.``Sequential`( _*args_
)[[source]](_modules/torch/nn/modules/container.html#Sequential)

    

顺序容器。模块将被添加到它在它们在构造函数中传递的顺序。可替代地，模块的有序字典也可以通过。

为了便于理解，这里是一个小例子：

    
    
    # Example of using Sequential
    model = nn.Sequential(
              nn.Conv2d(1,20,5),
              nn.ReLU(),
              nn.Conv2d(20,64,5),
              nn.ReLU()
            )
    
    # Example of using Sequential with OrderedDict
    model = nn.Sequential(OrderedDict([
              ('conv1', nn.Conv2d(1,20,5)),
              ('relu1', nn.ReLU()),
              ('conv2', nn.Conv2d(20,64,5)),
              ('relu2', nn.ReLU())
            ]))
    

###  ModuleList 

_class_`torch.nn.``ModuleList`( _modules=None_
)[[source]](_modules/torch/nn/modules/container.html#ModuleList)

    

持有列表子模块。

`ModuleList`可以被索引像一个普通的Python列表，但它包含的模块正确注册，并且将所有 `[HTG8可见]模块 `的方法。

Parameters

    

**模块** （ _可迭代_ _，_ _可选_ ） - 模块的一个可迭代添加

Example:

    
    
    class MyModule(nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
    
        def forward(self, x):
            # ModuleList can act as an iterable, or be indexed using ints
            for i, l in enumerate(self.linears):
                x = self.linears[i // 2](x) + l(x)
            return x
    

`append`( _module_
)[[source]](_modules/torch/nn/modules/container.html#ModuleList.append)

    

追加给定的模块到列表的末尾。

Parameters

    

**模块** （ _nn.Module_ ） - 模块追加

`extend`( _modules_
)[[source]](_modules/torch/nn/modules/container.html#ModuleList.extend)

    

从追加一个Python模块可迭代到列表的末尾。

Parameters

    

**模块** （ _可迭代_ ） - 模块的可迭代追加

`insert`( _index_ , _module_
)[[source]](_modules/torch/nn/modules/container.html#ModuleList.insert)

    

列表中的给定索引前插入一个特定的模块。

Parameters

    

  * **索引** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 要插入的索引。

  * **模块** （ _nn.Module_ ） - 模块插入

###  ModuleDict 

_class_`torch.nn.``ModuleDict`( _modules=None_
)[[source]](_modules/torch/nn/modules/container.html#ModuleDict)

    

持有字典子模块。

`ModuleDict`可以被索引像一个普通的Python字典，但它包含的模块正确注册，并且将所有 `[HTG8可见]模块 `的方法。

`ModuleDict`是尊重了一个 **有序** 字典

  * 插入的顺序，并

  * 在 `更新（） `的顺序进行合并`OrderedDict`或另一个 `ModuleDict`（将参数 `更新（） `）。

需要注意的是 `更新（） `与其他无序映射类型（例如，Python的平原`快译通 `）不保留合并后的映射的顺序。

Parameters

    

**模块** （ _可迭代_ _，_ _可选_ ） - （字符串：模块）的映射（字典）或键 - 值对的一个可迭代型的（字符串，模块）

Example:

    
    
    class MyModule(nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.choices = nn.ModuleDict({
                    'conv': nn.Conv2d(10, 10, 3),
                    'pool': nn.MaxPool2d(3)
            })
            self.activations = nn.ModuleDict([
                    ['lrelu', nn.LeakyReLU()],
                    ['prelu', nn.PReLU()]
            ])
    
        def forward(self, x, choice, act):
            x = self.choices[choice](x)
            x = self.activations[act](x)
            return x
    

`clear`()[[source]](_modules/torch/nn/modules/container.html#ModuleDict.clear)

    

取下ModuleDict的所有项目。

`items`()[[source]](_modules/torch/nn/modules/container.html#ModuleDict.items)

    

返回ModuleDict键/值对的迭代。

`keys`()[[source]](_modules/torch/nn/modules/container.html#ModuleDict.keys)

    

返回ModuleDict键的迭代。

`pop`( _key_
)[[source]](_modules/torch/nn/modules/container.html#ModuleDict.pop)

    

从ModuleDict删除键和返回它的模块。

Parameters

    

**键** （ _串_ ） - 键从ModuleDict弹出

`update`( _modules_
)[[source]](_modules/torch/nn/modules/container.html#ModuleDict.update)

    

更新 `ModuleDict`与来自映射键值对或可迭代，覆盖现有的密钥。

Note

如果`模块 `是`OrderedDict`，A`ModuleDict`或键 - 值对的一个可迭代，它在新的元素的顺序被保留。

Parameters

    

**模块** （ _可迭代_ ） - 映射（字典）从字符串 `模块 `，或密钥的迭代值对类型（字符串， `模块 `）

`values`()[[source]](_modules/torch/nn/modules/container.html#ModuleDict.values)

    

返回ModuleDict值的迭代。

### 参数列表

_class_`torch.nn.``ParameterList`( _parameters=None_
)[[source]](_modules/torch/nn/modules/container.html#ParameterList)

    

持有列表参数。

`参数列表 `可以被索引像一个普通的Python列表，但参数它所包含正确注册，并且将所有 `[HTG8可见]模块 `的方法。

Parameters

    

**参数** （ _可迭代_ _，_ _可选_ ） - 的 `参数可迭代 `添加

Example:

    
    
    class MyModule(nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])
    
        def forward(self, x):
            # ParameterList can act as an iterable, or be indexed using ints
            for i, p in enumerate(self.params):
                x = self.params[i // 2].mm(x) + p.mm(x)
            return x
    

`append`( _parameter_
)[[source]](_modules/torch/nn/modules/container.html#ParameterList.append)

    

附加在列表的最后一个给定的参数。

Parameters

    

**参数** （ _nn.Parameter_ ） - 参数到附加

`extend`( _parameters_
)[[source]](_modules/torch/nn/modules/container.html#ParameterList.extend)

    

从追加一个Python参数迭代到列表的末尾。

Parameters

    

**参数** （ _可迭代_ ） - 的参数可迭代到追加

###  ParameterDict 

_class_`torch.nn.``ParameterDict`( _parameters=None_
)[[source]](_modules/torch/nn/modules/container.html#ParameterDict)

    

拥有一本字典的参数。

ParameterDict能够被索引像一个普通的Python字典，但参数它所包含正确注册，并且将所有模块的方法可见。

`ParameterDict`是尊重了一个 **有序** 字典

  * the order of insertion, and

  * 在 `更新（） `的顺序进行合并`OrderedDict`或另一个 `ParameterDict`（将参数 `更新（） `）。

需要注意的是 `更新（） `与其他无序映射类型（例如，Python的平原`快译通 `）不保留合并后的映射的顺序。

Parameters

    

**参数** （ _可迭代_ _，_ _可选_ ） - （串的映射（字典）： `参数 `）或类型的键 - 值对（串的迭代， `参数 `）

Example:

    
    
    class MyModule(nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.params = nn.ParameterDict({
                    'left': nn.Parameter(torch.randn(5, 10)),
                    'right': nn.Parameter(torch.randn(5, 10))
            })
    
        def forward(self, x, choice):
            x = self.params[choice].mm(x)
            return x
    

`clear`()[[source]](_modules/torch/nn/modules/container.html#ParameterDict.clear)

    

取下ParameterDict的所有项目。

`items`()[[source]](_modules/torch/nn/modules/container.html#ParameterDict.items)

    

返回ParameterDict键/值对的迭代。

`keys`()[[source]](_modules/torch/nn/modules/container.html#ParameterDict.keys)

    

返回ParameterDict键的迭代。

`pop`( _key_
)[[source]](_modules/torch/nn/modules/container.html#ParameterDict.pop)

    

从ParameterDict删除键和返回它的参数。

Parameters

    

**键** （ _串_ ） - 键从ParameterDict弹出

`update`( _parameters_
)[[source]](_modules/torch/nn/modules/container.html#ParameterDict.update)

    

更新 `ParameterDict`与来自映射键值对或可迭代，覆盖现有的密钥。

Note

如果`参数 `是`OrderedDict`，A`ParameterDict`或键 - 值对的一个可迭代，它在新的元素的顺序被保留。

Parameters

    

**参数** （ _可迭代_ ） - 一个从串映射（字典）为 `参数 `，或密钥的迭代值对类型（字符串， `参数 `）

`values`()[[source]](_modules/torch/nn/modules/container.html#ParameterDict.values)

    

返回ParameterDict值的迭代。

## 卷积层

###  Conv1d 

_class_`torch.nn.``Conv1d`( _in_channels_ , _out_channels_ , _kernel_size_ ,
_stride=1_ , _padding=0_ , _dilation=1_ , _groups=1_ , _bias=True_ ,
_padding_mode='zeros'_
)[[source]](_modules/torch/nn/modules/conv.html#Conv1d)

    

施加1D卷积在几个输入平面组成的输入信号。

在最简单的情况下，所述层的与输入大小 的输出值（ N  C  在 ， L  ） （N，C _ {\文本{在}}，L） （ N  ， C  在 ， L  ）
和输出 （ N  ， C  OUT  ， L  OUT  ）  （N，C _ {\文本{出}}，L _ {\文本{出}}） （ N  ， C  OUT  ，
大号 OUT  ） 可以精确地描述为：

out(Ni,Coutj)=bias(Coutj)+∑k=0Cin−1weight(Coutj,k)⋆input(Ni,k)\text{out}(N_i,
C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{in} - 1}
\text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
out(Ni​,Coutj​​)=bias(Coutj​​)+k=0∑Cin​−1​weight(Coutj​​,k)⋆input(Ni​,k)

其中 ⋆ \星 ⋆ 是有效的[互相关](https://en.wikipedia.org/wiki/Cross-correlation)运算符， N  N
N  是一个批量大小， C  C  C  表示的数的信道， L  L  L  是人信号序列的ength。

  * `步幅 `控制用于交叉相关，单个数字或一个元素的元组的步幅。

  * `填充 `控制隐含零填补处理的双方的量为`填充 `点数。

  * `扩张 `控制内核点之间的间隔;也被称为劈窗算法。这是很难形容，但这种[链接](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)有什么`扩张 `做一个很好的可视化。

  * `基团 `控制输入和输出之间的连接。 `in_channels`和`out_channels`必须都是由`基团 `整除。例如，

>     * 在基团= 1，所有的输入被卷积以所有输出。

>

>     * 在组= 2，操作变得等效于具有由一侧上的两个CONV层侧，每个看到一半的输入通道，和产生一半的输出通道，并且两个随后连接在一起。

>

>     * 在基团= `in_channels`中，每个输入信道进行卷积以它自己的一套过滤器，大小的 ⌊ O  U  T  _  C  H  一
n的 n的 E  L  S  i的 n的 _  C  H  一 n的 n的 E  L  S  ⌋ \左\ lfloor \压裂{出\ _channels}
{在\ _channels} \右\ rfloor  ⌊ i的 n的 _  C  H  一 n的 n的 E  L  S  O  U  T  _  C  H
一 n的 n的 E  L  S  ⌋ 。

Note

根据你的内核的大小，几个（最后）输入的列可能会丢失，因为它是一个有效的[互相关](https://en.wikipedia.org/wiki/Cross-
correlation)，而不是一个完整的[互相关](https://en.wikipedia.org/wiki/Cross-correlation)
。它是由用户添加适当的填充。

Note

当基团== in_channels 和 out_channels == K * in_channels ，其中 K
是一个正整数，该操作也被称为在文献中作为深度方向卷积。

换句话说，为的大小 （ N  ，输入 C  i的 n的 ， L  i的 n的 ） （N，C_ {IN}，L_ {在}） （ N  ， C  i的 n的 ，
L  一世  n的 ） ，具有深度方向乘法器深度方向卷积 K ，可通过参数[构造HTG130]  （ C  在 =  C  i的 n的 ， C  OUT
=  C  i的 n的 × K  ， 。  。  。  ， 基团 =  C  i的 n的 ） （C_ \文本{IN} = C_ {}中，C_ \文本{出}
= C_ {在} \倍K，.. 。，\文本{基} = C_ {在}） （ C  在 =  C  ​​ i的 n的 ， C  OUT  =  C  i的 n的
× K  ， 。  。  。  ， 基团 =  C  I  n的 ） 。

Note

在使用CUDA后端与CuDNN当某些情况下，这种操作者可以选择不确定性的算法来提高性能。如果这是不可取的，你可以尝试通过设置`
torch.backends.cudnn.deterministic  =  真[使操作确定性（可能以性能为代价） HTG6] `。请参阅[ 重复性
](notes/randomness.html)为背景的音符。

Parameters

    

  * 在输入图像中通道数 - **in_channels** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")）

  * **out_channels** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 由卷积产生通道数

  * **kernel_size** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")） - 的卷积内核的大小

  * **步幅** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选的_ ） - 卷积的步幅。默认值：1

  * **填充** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选的_ ） - 补零加到输入的两侧。默认值：0

  * **padding_mode** （ _串_ _，_ _可选_ ） - 零

  * **扩张** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选的_ ） - 内核元件之间的间隔。默认值：1

  * **基团** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 从输入信道到输出通道阻塞的连接的数目。默认值：1

  * **偏压** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，增加了一个可学习偏压到输出端。默认值：`真 `

Shape:

    

  * 输入： （ N  ， C  i的 n的 ， L  i的 n的 ） （N，C_ {IN}，L_ {在}） （ N  ， C  i的 n的 ， L  i的 n的 ）

  * 输出： （ N  ， C  O  U  T  ， L  O  U  T  ） （N，C_ {出}，L_ {出}） （ N  ， C  O  U  T  ， L  [HTG1 05]  O  U  T  ） 其中

Lout=⌊Lin+2×padding−dilation×(kernel_size−1)−1stride+1⌋L_{out} =
\left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation} \times
(\text{kernel\\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor
Lout​=⌊strideLin​+2×padding−dilation×(kernel_size−1)−1​+1⌋

Variables

    

  * **〜Conv1d.weight** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的形状 [该模块的可学习权重HTG11] （ out_channels  ， in_channels  基团 ， kernel_size  ） （\文本{出\ _channels}，\压裂{\文本{在\ _channels}} {\文本{基}}，\文本{内核\ _size}） （ out_channels  ， 基团 in_channels  [H TG86]  ， kernel_size  ） 。这些权重的值是从 取样U  （ \-  K  ， K  ） \ mathcal【U}（ - \ SQRT {K}，\ SQRT {K}） U  （ \-  K  ， K  [H TG206]  ） 其中 K  =  1  C  在 *  kernel_size  K = \压裂{1} {C_ \文本{IN} * \文本{内核\ _size}}  ​​  K  =  ç  在 *  kernel_size  1 

  * **〜Conv1d.bias** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 形状（out_channels）的模块的可学习偏差。如果`偏压 `是`真 `，然后这些权重的值是从 取样 U  （ \-  K  ， K  ） \ mathcal【U}（ - \ SQRT {K}，\ SQRT {K}） U  （ \-  K  K  ） 其中 K  =  1  C  在 *  kernel_size  K = \压裂{1} {C_ \文本{IN} * \文本{内核\ _size}}  K  =  C  在 *  kernel_size  1 

例子：

    
    
    >>> m = nn.Conv1d(16, 33, 3, stride=2)
    >>> input = torch.randn(20, 16, 50)
    >>> output = m(input)
    

###  Conv2d 

_class_`torch.nn.``Conv2d`( _in_channels_ , _out_channels_ , _kernel_size_ ,
_stride=1_ , _padding=0_ , _dilation=1_ , _groups=1_ , _bias=True_ ,
_padding_mode='zeros'_
)[[source]](_modules/torch/nn/modules/conv.html#Conv2d)

    

施加二维卷积在几个输入平面组成的输入信号。

在最简单的情况下，所述层的与输入大小 的输出值（ N  C  在 ， H  ， W  ） （N，C _ {\文本{在}}，H，W） （ N  ， C  在
， H  ， W  ） 和输出 （ N  ， C  OUT  ， H  OUT  ， W  OUT  ） （N，C _ {\文本{出}}，H _
{\文本{出}}，W _ {\文本{出}} ） （ N  ， C  OUT  ， H  OUT  ， W  OUT  ） 可以精确地描述为：

out(Ni,Coutj)=bias(Coutj)+∑k=0Cin−1weight(Coutj,k)⋆input(Ni,k)\text{out}(N_i,
C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k =
0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star
\text{input}(N_i, k)
out(Ni​,Coutj​​)=bias(Coutj​​)+k=0∑Cin​−1​weight(Coutj​​,k)⋆input(Ni​,k)

其中 ⋆ \星 ⋆ 是有效的2D [互相关](https://en.wikipedia.org/wiki/Cross-correlation)运算符， N
N  N  是一个批量大小， C  C  C  表示的数信道， H  H  H  是以像素为单位输入平面的高度，并 W  W  W  是以像素为单位的宽度。

  * `步幅 `控制用于交叉相关，单个数字或一个元组的步幅。

  * `填充 `控制隐含零填补处理的双方的量为`填充 `点数为每个维度。

  * `dilation`controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) has a nice visualization of what `dilation`does.

  * `groups`controls the connections between inputs and outputs. `in_channels`and `out_channels`must both be divisible by `groups`. For example,

>     * At groups=1, all inputs are convolved to all outputs.

>

>     * At groups=2, the operation becomes equivalent to having two conv
layers side by side, each seeing half the input channels, and producing half
the output channels, and both subsequently concatenated.

>

>     * 在基团= `in_channels`中，每个输入信道进行卷积以它自己的一套过滤器，大小： ⌊ O  U  T  _  C  H  一
n的 n的 E  L  S  i的 n的 _  C  H  一 n的 n的 E  L  S  ⌋ \左\ lfloor \压裂{出\ _channels}
{在\ _channels} \右\ rfloor  ⌊ [HTG9 1]  i的 n的 _  C  H  一 N  n的 E  L  S  O  U  T
_  C  H  一 n的 n的 E  L  S  ⌋ 。

参数`kernel_size`，`步幅 `，`填充 `，`扩张 `可以是：

>   * 单一`INT`\- 在这种情况下相同的值被用于高度和宽度尺寸

>

>   * 一个`元组 `的两个整数 - 在这种情况下，第一个 INT用来为高度尺寸，并且所述第二 INT 为宽度尺寸

>

>

Note

根据你的内核的大小，几个（最后）输入的列可能会丢失，因为它是一个有效的[互相关](https://en.wikipedia.org/wiki/Cross-
correlation)，而不是一个完整的[互相关](https://en.wikipedia.org/wiki/Cross-correlation)
。它是由用户添加适当的填充。

Note

When groups == in_channels and out_channels == K * in_channels, where K is a
positive integer, this operation is also termed in literature as depthwise
convolution.

换句话说，为的大小 （ N  ，输入 C  i的 n的 ， H  i的 n的 ， W  i的 n的 ） （N，C_ {IN}，H_ {IN}，W_ {在}）
（ N  ， ç  i的 n的 ， H  i的 n的 ， W  i的 n的 ） ，具有深度方向乘法器深度方向卷积 K ，可由参数 （ i的 n的 _
[H构建TG191]  C  H  一 n的 n的 E  L  S  =  C  i的 n的 ， O  U  T  _  C  H  一个 n的 n的 E
L  S  =  C  i的 n的 × K  ， 。  。  。  ​​， 克 R  O  U  P  S  =  C  i的 n的 ） （在\
_channels = C_ {}中，出\ _channels = C_ {在} \倍K，...，组= C_ {在}） （ i的 n的 _  C  H  一
n的 n的 E  L  S  =  C  i的 n的 O  U  T  _  C  H  一 n的 n的 E  L  S  =  C  i的 n的 × K
， 。  。  。  ， 克 R  O  U  P  S  =  C  i的 n的 ） 。

Note

In some circumstances when using the CUDA backend with CuDNN, this operator
may select a nondeterministic algorithm to increase performance. If this is
undesirable, you can try to make the operation deterministic (potentially at a
performance cost) by setting `torch.backends.cudnn.deterministic = True`.
Please see the notes on [Reproducibility](notes/randomness.html) for
background.

Parameters

    

  * **in_channels** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Number of channels in the input image

  * **out_channels** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Number of channels produced by the convolution

  * **kernel_size** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")) – Size of the convolving kernel

  * **stride** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _,_ _optional_ ) – Stride of the convolution. Default: 1

  * **填充** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选的_ ） - 补零加到输入的两侧。默认值：0

  * **padding_mode** ( _string_ _,_ _optional_ ) – zeros

  * **扩张** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选的_ ） - 内核元件之间的间隔。默认值：1

  * **基团** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 从输入信道到输出通道阻塞的连接的数目。默认值：1

  * **bias** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If `True`, adds a learnable bias to the output. Default: `True`

Shape:

    

  * 输入： （ N  ， C  i的 n的 ， H  i的 n的 ， W  i的 n的 ） （N，C_ {IN}，H_ {IN}，W_ {在}） （ N  ， C  i的 n的 ， H  i的 n的 ， W  i的 N  ）

  * 输出： （ N  ， C  O  U  T  ， H  O  U  T  ， W  O  U  T  ） （N，C_ {出}，H_ {出} ，W_ {出}） （ N  ， C  O  U  T  [HT G104]  ， H  O  U  T  ， W  O  U  T  ） 其中

Hout=⌊Hin+2×padding[0]−dilation[0]×(kernel_size[0]−1)−1stride[0]+1⌋H_{out} =
\left\lfloor\frac{H_{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
\times (\text{kernel\\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
Hout​=⌊stride[0]Hin​+2×padding[0]−dilation[0]×(kernel_size[0]−1)−1​+1⌋

Wout=⌊Win+2×padding[1]−dilation[1]×(kernel_size[1]−1)−1stride[1]+1⌋W_{out} =
\left\lfloor\frac{W_{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
\times (\text{kernel\\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
Wout​=⌊stride[1]Win​+2×padding[1]−dilation[1]×(kernel_size[1]−1)−1​+1⌋

Variables

    

  * **〜Conv2d.weight** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的形状 [该模块的可学习权重HTG11] （ out_channels  ， in_channels  基团 ， （\文本{出\ _channels}，\压裂{\文本{在\ _channels}} {\文本{基}}， （ out_channels  ， 基团 in_channels  [HTG8 9]  ， kernel_size [0]  ， kernel_size [1]  ） \文本{内核\ _size [0]}，\文本{内核\ _size [1]}） kernel_size [0]  ， kernel_size [1]  ） 。这些权重的值是从 取样U  （ \-  K  ， K  ） \ mathcal【U}（ - \ SQRT {K}，\ SQRT {K}） U  （ \-  K  ， K  [H TG236]  ） 其中 ​​  K  =  1  C  在 *  Π i的 =  0  1  kernel_size  [ i的 K = \压裂{1} {C_ \文本{IN} * \ prod_ {I = 0} ^ {1} \文本{内核\ _size} [I]}  K  =  C  在 *  Π i的 =  0  1  kernel_size  [ i的 1 

  * **〜Conv2d.bias** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 形状（out_channels）的模块的可学习偏差。如果`偏压 `是`真 `，然后这些权重的值是从 取样 U  （ \-  K  ， K  ） \ mathcal【U}（ - \ SQRT {K}，\ SQRT {K}） U  （ \-  K  K  ） 其中 K  =  1  C  在 *  Π i的 =  0  1  kernel_size  [ i的 K = \压裂{1} {C_ \文本{IN} * \ prod_ {I = 0} ^ {1} \文本{内核\ _size} [I]}  K  =  C  在 *  Π i的 =  0  ​​  1  kernel_size  [ i的 1 

Examples:

    
    
    >>> # With square kernels and equal stride
    >>> m = nn.Conv2d(16, 33, 3, stride=2)
    >>> # non-square kernels and unequal stride and with padding
    >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
    >>> # non-square kernels and unequal stride and with padding and dilation
    >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
    >>> input = torch.randn(20, 16, 50, 100)
    >>> output = m(input)
    

###  Conv3d 

_class_`torch.nn.``Conv3d`( _in_channels_ , _out_channels_ , _kernel_size_ ,
_stride=1_ , _padding=0_ , _dilation=1_ , _groups=1_ , _bias=True_ ,
_padding_mode='zeros'_
)[[source]](_modules/torch/nn/modules/conv.html#Conv3d)

    

施加三维卷积在几个输入平面组成的输入信号。

在最简单的情况下，所述层的与输入大小 的输出值（ N  C  i的 n的 ， d  ， H  ， W  ） （N，C_ {IN}，d ，H，W） （ N
， C  i的 n的 ， d  ， H  [H T G99]  W  ） 和输出 （ N  ， C  O  U  T  ， d  O  U  T  ， H
O  U  吨 ， W  O  U  T  ） （N，C_ {出}，D_ {出}，H_ {出}，W_ {出}） （ N [HTG1 91] ， C  O
U  T  ， d  O  U  T  ， ​​  H  O  U  T  ， W  O  U  T  ） 可以精确地描述为：

out(Ni,Coutj)=bias(Coutj)+∑k=0Cin−1weight(Coutj,k)⋆input(Ni,k)out(N_i,
C_{out_j}) = bias(C_{out_j}) + \sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k)
\star input(N_i, k)
out(Ni​,Coutj​​)=bias(Coutj​​)+k=0∑Cin​−1​weight(Coutj​​,k)⋆input(Ni​,k)

其中 ⋆ \星 ⋆ 是有效的3D [互相关](https://en.wikipedia.org/wiki/Cross-correlation)操作者

  * `步幅 `控制用于互相关的步幅。

  * `padding`controls the amount of implicit zero-paddings on both sides for `padding`number of points for each dimension.

  * `扩张 `控制内核点之间的间隔;也被称为劈窗算法。这是很难形容，但这种[链接](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)有什么`扩张 `做一个很好的可视化。

  * `groups`controls the connections between inputs and outputs. `in_channels`and `out_channels`must both be divisible by `groups`. For example,

>     * At groups=1, all inputs are convolved to all outputs.

>

>     * At groups=2, the operation becomes equivalent to having two conv
layers side by side, each seeing half the input channels, and producing half
the output channels, and both subsequently concatenated.

>

>     * 在基团= `in_channels`中，每个输入信道进行卷积以它自己的一套过滤器，大小的 ⌊ O  U  T  _  C  H  一
n的 n的 E  L  S  i的 n的 _  C  H  一 n的 n的 E  L  S  ⌋ \左\ lfloor \压裂{出\ _channels}
{在\ _channels} \右\ rfloor  ⌊ i的 n的 _  C  H  一 n的 n的 E  L  S  O  U  T  _  C  H
一 n的 n的 E  L  S  ⌋ 。

The parameters `kernel_size`, `stride`, `padding`, `dilation`can either be:

>   * 单一`INT`\- 在这种情况下相同的值被用于深度，高度和宽度尺寸

>

>   * 一个`元组 `三个整数 - 在这种情况下，第一个 INT用于深度尺寸，所述第二 INT 为高度维度和第三 INT 为宽度尺寸

>

>

Note

Depending of the size of your kernel, several (of the last) columns of the
input might be lost, because it is a valid [cross-
correlation](https://en.wikipedia.org/wiki/Cross-correlation), and not a full
[cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation). It is up
to the user to add proper padding.

Note

When groups == in_channels and out_channels == K * in_channels, where K is a
positive integer, this operation is also termed in literature as depthwise
convolution.

换句话说，为的大小 （ N  ，输入 C  i的 n的 ， d  i的 n的 ， H  i的 n的 ， W  i的 n的 ） （N，C_ {IN}，D_
{IN}，H_ {IN}，W_ {在}） （ N  ， C  i的 n的 ， d  i的 n的 ， H  i的 N  ， W  i的[H TG199]
n的 ） ，具有深度方向乘法器深度方向卷积 K ，可以通过参构造 （ i的 n的 _  C  H  一 n的 n的 E  L  S  =  C  i的 n的
， 问题o  U  T  _  C  H  一 n的 n的 E  L  S  =  C  i的 n的 × K  ， 。  。  。  ， 克 R  O  U
P  S  =  C  i的 n的 ） （在\ _channels = C_ {}中，出\ _channels = C_ {在} \倍K，...，组= C_
{在}） （ i的 n的 _  C  H  一 n的 n的 E  L  S  =  C  i的 n的 O  U  T  _  C  H  一 n的 n的 E
L  S  =  C  i的 n的 × K  ， 。  。  。  ， 克 R  O  U  P  S  =  C  i的 n的 ） 。

Note

In some circumstances when using the CUDA backend with CuDNN, this operator
may select a nondeterministic algorithm to increase performance. If this is
undesirable, you can try to make the operation deterministic (potentially at a
performance cost) by setting `torch.backends.cudnn.deterministic = True`.
Please see the notes on [Reproducibility](notes/randomness.html) for
background.

Parameters

    

  * **in_channels** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Number of channels in the input image

  * **out_channels** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Number of channels produced by the convolution

  * **kernel_size** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")) – Size of the convolving kernel

  * **stride** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _,_ _optional_ ) – Stride of the convolution. Default: 1

  * **填充** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选的_ ） - 补零加到输入的所有三个侧面。默认值：0

  * **padding_mode** ( _string_ _,_ _optional_ ) – zeros

  * **dilation** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _,_ _optional_ ) – Spacing between kernel elements. Default: 1

  * **groups** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _,_ _optional_ ) – Number of blocked connections from input channels to output channels. Default: 1

  * **bias** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If `True`, adds a learnable bias to the output. Default: `True`

Shape:

    

  * 输入： （ N  ， C  i的 n的 ， d  i的 n的 ， H  i的 n的 ， W  i的 n的 ） （N，C_ {IN}，D_ {IN}，H_ {IN}，W_ {在}） （ N  ， C  i的 n的 ， d  i的 n的 ， H  i的 n的 ， W  i的 n的 ）

  * 输出： （ N  ， C  O  U  T  ， d  O  U  T  ， H  O  U  T  ， W  O  U  T  ） （N，C_ {出}，D_ {出}，H_ {出}，W_ {出}） （ N  C  O  U  吨[H TG103]  ， d  O  U  T  ， H  O  U  T  ， [H TG201] W  O  U  T  ） 其中

Dout=⌊Din+2×padding[0]−dilation[0]×(kernel_size[0]−1)−1stride[0]+1⌋D_{out} =
\left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
\times (\text{kernel\\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
Dout​=⌊stride[0]Din​+2×padding[0]−dilation[0]×(kernel_size[0]−1)−1​+1⌋

Hout=⌊Hin+2×padding[1]−dilation[1]×(kernel_size[1]−1)−1stride[1]+1⌋H_{out} =
\left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
\times (\text{kernel\\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
Hout​=⌊stride[1]Hin​+2×padding[1]−dilation[1]×(kernel_size[1]−1)−1​+1⌋

Wout=⌊Win+2×padding[2]−dilation[2]×(kernel_size[2]−1)−1stride[2]+1⌋W_{out} =
\left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2]
\times (\text{kernel\\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor
Wout​=⌊stride[2]Win​+2×padding[2]−dilation[2]×(kernel_size[2]−1)−1​+1⌋

Variables

    

  * **〜Conv3d.weight** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的形状 [该模块的可学习权重HTG11] （ out_channels  ， in_channels  基团 ， （\文本{出\ _channels}，\压裂{\文本{在\ _channels}} {\文本{基}}， （ out_channels  ， 基团 in_channels  [HTG8 9]  ， kernel_size [0]  ， kernel_size [1]  ， kernel_size [2]  ） \文本{内核\ _size [0]}，\文本{内核\ _size [1]}，\文本{内核\ _size [2 ]}） kernel_size [0]  ， kernel_size [1]  ， kernel_size [2]  ） 。这些权重的值是从 取样U  （ \-  K  ， K  ） \ mathcal【U}（ - \ SQRT {K}，\ SQRT {K}） U  （ \-  K  ， K  [H TG248]  ​​  ） 其中 K  =  1  C  在 *  Π i的 =  0  2  kernel_size  [ i的 K = \压裂{1} {C_ \文本{IN} * \ prod_ {I = 0} ^ {2} \文本{内核\ _size} [I]}  K  =  C  在 *  Π i的 =  0  2  kernel_size  [ i的 1 

  * **〜Conv3d.bias** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 形状（out_channels）的模块的可学习偏差。如果`偏压 `是`真 `，然后这些权重的值是从 取样 U  （ \-  K  ， K  ） \ mathcal【U}（ - \ SQRT {K}，\ SQRT {K}） U  （ \-  K  K  ） 其中 K  =  1  C  在 *  Π i的 =  0  2  kernel_size  [ i的 K = \压裂{1} {C_ \文本{IN} * \ prod_ {I = 0} ^ {2} \文本{内核\ _size} [I]}  K  =  C  在 *  Π i的 =  0  ​​  2  kernel_size  [ i的 1 

Examples:

    
    
    >>> # With square kernels and equal stride
    >>> m = nn.Conv3d(16, 33, 3, stride=2)
    >>> # non-square kernels and unequal stride and with padding
    >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
    >>> input = torch.randn(20, 16, 10, 50, 100)
    >>> output = m(input)
    

###  ConvTranspose1d 

_class_`torch.nn.``ConvTranspose1d`( _in_channels_ , _out_channels_ ,
_kernel_size_ , _stride=1_ , _padding=0_ , _output_padding=0_ , _groups=1_ ,
_bias=True_ , _dilation=1_ , _padding_mode='zeros'_
)[[source]](_modules/torch/nn/modules/conv.html#ConvTranspose1d)

    

应用在多个输入平面构成的输入图像的1D换位卷积运算。

此模块可以被视为Conv1d的梯度相对于它的输入。它也被称为一分级-跨距卷积或去卷积（尽管它不是一个实际的去卷积运算）。

  * `stride`controls the stride for the cross-correlation.

  * `填充 `控制隐含零补白的两侧为`的量扩张 *  （kernel_size  -  1） -  填充 `数量的点。请参阅下面注释详情。

  * `output_padding`控制添加到输出形状的一侧上的额外尺寸。请参阅下面注释详情。

  * `dilation`controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) has a nice visualization of what `dilation`does.

  * `groups`controls the connections between inputs and outputs. `in_channels`and `out_channels`must both be divisible by `groups`. For example,

>     * At groups=1, all inputs are convolved to all outputs.

>

>     * At groups=2, the operation becomes equivalent to having two conv
layers side by side, each seeing half the input channels, and producing half
the output channels, and both subsequently concatenated.

>

>     * 在基团= `in_channels`中，每个输入信道进行卷积以它自己的一套滤波器（大小[的 HTG10]⌊  O  U  T  _
C  H  一 n的 n的 E  L  S  i的 n的 _  C  H  一 n的 n的 E  L  S  ⌋ \左\ lfloor \压裂{出\
_channels} {在\ _channels} \右\ rfloor  ⌊ i的 n的 _  C  H  一 n的 n的 E  L  S  O  U
T  _  C  H  一 n的 n的 E  L  S  ⌋ ）。

Note

Depending of the size of your kernel, several (of the last) columns of the
input might be lost, because it is a valid [cross-
correlation](https://en.wikipedia.org/wiki/Cross-correlation), and not a full
[cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation). It is up
to the user to add proper padding.

Note

的`填充 `参数有效地增加了`扩张 *  （kernel_size  -  1） -  填充 `零填充的量与输入的两个尺寸。此被设置成使得当一个 `
Conv1d`和a`ConvTranspose1d`与初始化相同的参数，它们是在考虑到输入和输出形状彼此的逆。然而，当`步幅 & GT
;  1`， `Conv1d`映射多个输入的形状，以相同的输出的形状。 `output_padding
`提供一种通过有效地增加在一侧上所计算出的输出的形状来解决此模糊性。需要注意的是`output_padding
`仅用于查找输出的形状，但实际上并没有增加零填充输出。

Note

In some circumstances when using the CUDA backend with CuDNN, this operator
may select a nondeterministic algorithm to increase performance. If this is
undesirable, you can try to make the operation deterministic (potentially at a
performance cost) by setting `torch.backends.cudnn.deterministic = True`.
Please see the notes on [Reproducibility](notes/randomness.html) for
background.

Parameters

    

  * **in_channels** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Number of channels in the input image

  * **out_channels** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Number of channels produced by the convolution

  * **kernel_size** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")) – Size of the convolving kernel

  * **stride** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _,_ _optional_ ) – Stride of the convolution. Default: 1

  * **填充** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选的_ ） - `扩张 *  （kernel_size  -  1） -  填充 `零填充将被添加到输入的两侧。默认值：0

  * **output_padding** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选的_ ） - 添加到输出形状的一侧的其他尺寸。默认值：0

  * **groups** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _,_ _optional_ ) – Number of blocked connections from input channels to output channels. Default: 1

  * **bias** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If `True`, adds a learnable bias to the output. Default: `True`

  * **dilation** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _,_ _optional_ ) – Spacing between kernel elements. Default: 1

Shape:

    

  * Input: (N,Cin,Lin)(N, C_{in}, L_{in})(N,Cin​,Lin​)

  * Output: (N,Cout,Lout)(N, C_{out}, L_{out})(N,Cout​,Lout​) where

Lout=(Lin−1)×stride−2×padding+dilation×(kernel_size−1)+output_padding+1L_{out}
= (L_{in} - 1) \times \text{stride} - 2 \times \text{padding} +
\text{dilation} \times (\text{kernel\\_size} - 1) + \text{output\\_padding} +
1 Lout​=(Lin​−1)×stride−2×padding+dilation×(kernel_size−1)+output_padding+1

Variables

    

  * **〜ConvTranspose1d.weight** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的形状 [该模块的可学习权重HTG11] （ in_channels  ， out_channels  基团 ， （\文本{在\ _channels}，\压裂{\文本{出\ _channels}} {\文本{基}}， （ in_channels  ， 基团 out_channels  [HTG8 8]  ， kernel_size  ） \文本{内核\ _size}） kernel_size  ） 。这些权重的值是从 取样U  （ \-  K  ， K  ） \ mathcal【U}（ - \ SQRT {K}，\ SQRT {K}） U  （ \-  K  ， K  [H TG224]  ） 其中 K  =  1  C ​​ 在 *  kernel_size  K = \压裂{1} {C_ \文本{IN} * \文本{内核\ _size}}  K  =  ç  在 *  kernel_size  1 

  * **〜ConvTranspose1d.bias** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 形状（out_channels）的模块的可学习偏差。如果`偏压 `是`真 `，然后这些权重的值是从 取样 U  （ \-  K  ， K  ） \ mathcal【U}（ - \ SQRT {K}，\ SQRT {K}） U  （ \-  K  K  ） 其中 K  =  1  C  在 *  kernel_size  K = \压裂{1} {C_ \文本{IN} * \文本{内核\ _size}}  K  =  C  在 *  kernel_size  1 

###  ConvTranspose2d 

_class_`torch.nn.``ConvTranspose2d`( _in_channels_ , _out_channels_ ,
_kernel_size_ , _stride=1_ , _padding=0_ , _output_padding=0_ , _groups=1_ ,
_bias=True_ , _dilation=1_ , _padding_mode='zeros'_
)[[source]](_modules/torch/nn/modules/conv.html#ConvTranspose2d)

    

应用在多个输入平面构成的输入图像的2D转卷积运算。

此模块可以被视为Conv2d的梯度相对于它的输入。它也被称为一分级-跨距卷积或去卷积（尽管它不是一个实际的去卷积运算）。

  * `stride`controls the stride for the cross-correlation.

  * `padding`controls the amount of implicit zero-paddings on both sides for `dilation * (kernel_size - 1) - padding`number of points. See note below for details.

  * `output_padding`controls the additional size added to one side of the output shape. See note below for details.

  * `dilation`controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) has a nice visualization of what `dilation`does.

  * `groups`controls the connections between inputs and outputs. `in_channels`and `out_channels`must both be divisible by `groups`. For example,

>     * At groups=1, all inputs are convolved to all outputs.

>

>     * At groups=2, the operation becomes equivalent to having two conv
layers side by side, each seeing half the input channels, and producing half
the output channels, and both subsequently concatenated.

>

>     * At groups= `in_channels`, each input channel is convolved with its own
set of filters (of size
⌊out_channelsin_channels⌋\left\lfloor\frac{out\\_channels}{in\\_channels}\right\rfloor⌊in_channelsout_channels​⌋
).

参数`kernel_size`，`步幅 `，`填充 `，`output_padding`可以是：

>   * 单一`INT`\- 在这种情况下相同的值被用于高度和宽度尺寸

>

>   * a `tuple`of two ints – in which case, the first int is used for the
height dimension, and the second int for the width dimension

>

>

Note

Depending of the size of your kernel, several (of the last) columns of the
input might be lost, because it is a valid [cross-
correlation](https://en.wikipedia.org/wiki/Cross-correlation), and not a full
[cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation). It is up
to the user to add proper padding.

Note

的`填充 `参数有效地增加了`扩张 *  （kernel_size  -  1） -  填充 `零填充的量与输入的两个尺寸。此被设置成使得当一个 `
Conv2d`和a`ConvTranspose2d`与初始化相同的参数，它们是在考虑到输入和输出形状彼此的逆。然而，当`步幅 & GT
;  1`， `Conv2d`映射多个输入的形状，以相同的输出的形状。 `output_padding
`提供一种通过有效地增加在一侧上所计算出的输出的形状来解决此模糊性。需要注意的是`output_padding
`仅用于查找输出的形状，但实际上并没有增加零填充输出。

Note

In some circumstances when using the CUDA backend with CuDNN, this operator
may select a nondeterministic algorithm to increase performance. If this is
undesirable, you can try to make the operation deterministic (potentially at a
performance cost) by setting `torch.backends.cudnn.deterministic = True`.
Please see the notes on [Reproducibility](notes/randomness.html) for
background.

Parameters

    

  * **in_channels** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Number of channels in the input image

  * **out_channels** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Number of channels produced by the convolution

  * **kernel_size** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")) – Size of the convolving kernel

  * **stride** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _,_ _optional_ ) – Stride of the convolution. Default: 1

  * **填充** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选的_ ） - `扩张 *  （kernel_size  -  1） -  填充 `零填充将被添加到输入中的每个维度的两侧。默认值：0

  * **output_padding** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选的_ ） - 加入到每个维度中的一个侧的输出形状的其他尺寸。默认值：0

  * **groups** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _,_ _optional_ ) – Number of blocked connections from input channels to output channels. Default: 1

  * **bias** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If `True`, adds a learnable bias to the output. Default: `True`

  * **dilation** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _,_ _optional_ ) – Spacing between kernel elements. Default: 1

Shape:

    

  * Input: (N,Cin,Hin,Win)(N, C_{in}, H_{in}, W_{in})(N,Cin​,Hin​,Win​)

  * Output: (N,Cout,Hout,Wout)(N, C_{out}, H_{out}, W_{out})(N,Cout​,Hout​,Wout​) where

Hout=(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1H_{out}
= (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] +
\text{dilation}[0] \times (\text{kernel\\_size}[0] - 1) +
\text{output\\_padding}[0] + 1
Hout​=(Hin​−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1

Wout=(Win−1)×stride[1]−2×padding[1]+dilation[1]×(kernel_size[1]−1)+output_padding[1]+1W_{out}
= (W_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] +
\text{dilation}[1] \times (\text{kernel\\_size}[1] - 1) +
\text{output\\_padding}[1] + 1
Wout​=(Win​−1)×stride[1]−2×padding[1]+dilation[1]×(kernel_size[1]−1)+output_padding[1]+1

Variables

    

  * **〜ConvTranspose2d.weight** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的形状 [该模块的可学习权重HTG11] （ in_channels  ， out_channels  基团 ， （\文本{在\ _channels}，\压裂{\文本{出\ _channels}} {\文本{基}}， （ in_channels  ， 基团 out_channels  [HTG8 8]  ， kernel_size [0]  ， kernel_size [1]  ） \文本{内核\ _size [0]}，\文本{内核\ _size [1]}） kernel_size [0]  ， kernel_size [1]  ） 。这些权重的值是从 取样U  （ \-  K  ， K  ） \ mathcal【U}（ - \ SQRT {K}，\ SQRT {K}） U  （ \-  K  ， K  [H TG236]  ） 其中 ​​  K  =  1  C  在 *  Π i的 =  0  1  kernel_size  [ i的 K = \压裂{1} {C_ \文本{IN} * \ prod_ {I = 0} ^ {1} \文本{内核\ _size} [I]}  K  =  C  在 *  Π i的 =  0  1  kernel_size  [ i的 1 

  * **〜ConvTranspose2d.bias** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 形状（out_channels）的模块的可学习偏压如果`偏压 `是`真 `，然后这些权重的值从采样 U  （ \-  K  ， K  ） \ mathcal【U}（ - \ SQRT {K}，\ SQRT {K}） U  （ \-  K  ， K  ） 其中 K  =  1  C  在 *  Π i的 =  0  1  kernel_size  [ i的 ķ = \压裂{1} {C_ \文本{IN} * \ prod_ {I = 0} ^ {1} \文本{内核\ _size} [I]}  K  =  C  在 *  Π i的 =  0  ​​  1  kernel_size  [ i的 1 

Examples:

    
    
    >>> # With square kernels and equal stride
    >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
    >>> # non-square kernels and unequal stride and with padding
    >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
    >>> input = torch.randn(20, 16, 50, 100)
    >>> output = m(input)
    >>> # exact output size can be also specified as an argument
    >>> input = torch.randn(1, 16, 12, 12)
    >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
    >>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
    >>> h = downsample(input)
    >>> h.size()
    torch.Size([1, 16, 6, 6])
    >>> output = upsample(h, output_size=input.size())
    >>> output.size()
    torch.Size([1, 16, 12, 12])
    

###  ConvTranspose3d 

_class_`torch.nn.``ConvTranspose3d`( _in_channels_ , _out_channels_ ,
_kernel_size_ , _stride=1_ , _padding=0_ , _output_padding=0_ , _groups=1_ ,
_bias=True_ , _dilation=1_ , _padding_mode='zeros'_
)[[source]](_modules/torch/nn/modules/conv.html#ConvTranspose3d)

    

应用在多个输入平面构成的输入图像的3D换位卷积运算。转置卷积运算符乘以在从所有输入特征平面输出每个输入值逐元素由一个可学习的内核，和求和。

此模块可以被视为Conv3d的梯度相对于它的输入。它也被称为一分级-跨距卷积或去卷积（尽管它不是一个实际的去卷积运算）。

  * `stride`controls the stride for the cross-correlation.

  * `padding`controls the amount of implicit zero-paddings on both sides for `dilation * (kernel_size - 1) - padding`number of points. See note below for details.

  * `output_padding`controls the additional size added to one side of the output shape. See note below for details.

  * `dilation`controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) has a nice visualization of what `dilation`does.

  * `groups`controls the connections between inputs and outputs. `in_channels`and `out_channels`must both be divisible by `groups`. For example,

>     * At groups=1, all inputs are convolved to all outputs.

>

>     * At groups=2, the operation becomes equivalent to having two conv
layers side by side, each seeing half the input channels, and producing half
the output channels, and both subsequently concatenated.

>

>     * At groups= `in_channels`, each input channel is convolved with its own
set of filters (of size
⌊out_channelsin_channels⌋\left\lfloor\frac{out\\_channels}{in\\_channels}\right\rfloor⌊in_channelsout_channels​⌋
).

The parameters `kernel_size`, `stride`, `padding`, `output_padding`can either
be:

>   * 单一`INT`\- 在这种情况下相同的值被用于深度，高度和宽度尺寸

>

>   * a `tuple`of three ints – in which case, the first int is used for the
depth dimension, the second int for the height dimension and the third int for
the width dimension

>

>

Note

Depending of the size of your kernel, several (of the last) columns of the
input might be lost, because it is a valid [cross-
correlation](https://en.wikipedia.org/wiki/Cross-correlation), and not a full
[cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation). It is up
to the user to add proper padding.

Note

的`填充 `参数有效地增加了`扩张 *  （kernel_size  -  1） -  填充 `零填充的量与输入的两个尺寸。此被设置成使得当一个 `
Conv3d`和a`ConvTranspose3d`与初始化相同的参数，它们是在考虑到输入和输出形状彼此的逆。然而，当`步幅 & GT
;  1`， `Conv3d`映射多个输入的形状，以相同的输出的形状。 `output_padding
`提供一种通过有效地增加在一侧上所计算出的输出的形状来解决此模糊性。需要注意的是`output_padding
`仅用于查找输出的形状，但实际上并没有增加零填充输出。

Note

In some circumstances when using the CUDA backend with CuDNN, this operator
may select a nondeterministic algorithm to increase performance. If this is
undesirable, you can try to make the operation deterministic (potentially at a
performance cost) by setting `torch.backends.cudnn.deterministic = True`.
Please see the notes on [Reproducibility](notes/randomness.html) for
background.

Parameters

    

  * **in_channels** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Number of channels in the input image

  * **out_channels** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Number of channels produced by the convolution

  * **kernel_size** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")) – Size of the convolving kernel

  * **stride** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _,_ _optional_ ) – Stride of the convolution. Default: 1

  * **padding** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _,_ _optional_ ) – `dilation * (kernel_size - 1) - padding`zero-padding will be added to both sides of each dimension in the input. Default: 0

  * **output_padding** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _,_ _optional_ ) – Additional size added to one side of each dimension in the output shape. Default: 0

  * **groups** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _,_ _optional_ ) – Number of blocked connections from input channels to output channels. Default: 1

  * **bias** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If `True`, adds a learnable bias to the output. Default: `True`

  * **dilation** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _,_ _optional_ ) – Spacing between kernel elements. Default: 1

Shape:

    

  * Input: (N,Cin,Din,Hin,Win)(N, C_{in}, D_{in}, H_{in}, W_{in})(N,Cin​,Din​,Hin​,Win​)

  * Output: (N,Cout,Dout,Hout,Wout)(N, C_{out}, D_{out}, H_{out}, W_{out})(N,Cout​,Dout​,Hout​,Wout​) where

Dout=(Din−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1D_{out}
= (D_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] +
\text{dilation}[0] \times (\text{kernel\\_size}[0] - 1) +
\text{output\\_padding}[0] + 1
Dout​=(Din​−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1

Hout=(Hin−1)×stride[1]−2×padding[1]+dilation[1]×(kernel_size[1]−1)+output_padding[1]+1H_{out}
= (H_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] +
\text{dilation}[1] \times (\text{kernel\\_size}[1] - 1) +
\text{output\\_padding}[1] + 1
Hout​=(Hin​−1)×stride[1]−2×padding[1]+dilation[1]×(kernel_size[1]−1)+output_padding[1]+1

Wout=(Win−1)×stride[2]−2×padding[2]+dilation[2]×(kernel_size[2]−1)+output_padding[2]+1W_{out}
= (W_{in} - 1) \times \text{stride}[2] - 2 \times \text{padding}[2] +
\text{dilation}[2] \times (\text{kernel\\_size}[2] - 1) +
\text{output\\_padding}[2] + 1
Wout​=(Win​−1)×stride[2]−2×padding[2]+dilation[2]×(kernel_size[2]−1)+output_padding[2]+1

Variables

    

  * **〜ConvTranspose3d.weight** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的形状 [该模块的可学习权重HTG11] （ in_channels  ， out_channels  基团 ， （\文本{在\ _channels}，\压裂{\文本{出\ _channels}} {\文本{基}}， （ in_channels  ， 基团 out_channels  [HTG8 8]  ， kernel_size [0]  ， kernel_size [1]  ， kernel_size [2]  ） \文本{内核\ _size [0]}，\文本{内核\ _size [1]}，\文本{内核\ _size [2]}） kernel_size [0]  kernel_size [1]  ， kernel_size [2]  ） 。这些权重的值是从 取样U  （ \-  K  ， K  ） \ mathcal【U}（ - \ SQRT {K}，\ SQRT {K}） U  （ \-  K  ， K  [H TG248]  ​​  ） 其中 K  =  1  C  在 *  Π i的 =  0  2  kernel_size  [ i的 K = \压裂{1} {C_ \文本{IN} * \ prod_ {I = 0} ^ {2} \文本{内核\ _size} [I]}  K  =  C  在 *  Π i的 =  0  2  kernel_size  [ i的 1 

  * **〜ConvTranspose3d.bias** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 形状（out_channels）的模块的可学习偏压如果`偏压 `是`真 `，然后这些权重的值从采样 U  （ \-  K  ， K  ） \ mathcal【U}（ - \ SQRT {K}，\ SQRT {K}） U  （ \-  K  ， K  ） 其中 K  =  1  C  在 *  Π i的 =  0  2  kernel_size  [ i的 ķ = \压裂{1} {C_ \文本{IN} * \ prod_ {I = 0} ^ {2} \文本{内核\ _size} [I]}  K  =  C  在 *  Π i的 =  0  ​​  2  kernel_size  [ i的 1 

Examples:

    
    
    >>> # With square kernels and equal stride
    >>> m = nn.ConvTranspose3d(16, 33, 3, stride=2)
    >>> # non-square kernels and unequal stride and with padding
    >>> m = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
    >>> input = torch.randn(20, 16, 10, 50, 100)
    >>> output = m(input)
    

### 展开

_class_`torch.nn.``Unfold`( _kernel_size_ , _dilation=1_ , _padding=0_ ,
_stride=1_ )[[source]](_modules/torch/nn/modules/fold.html#Unfold)

    

提取物成批输入张量滑动局部块。

考虑一个成批`输入 `的张量形状 （ N  ， C  ， *  ） （N，C，*） （ N  ， C  ， *  ） ，其中 N  N  N
是批处理尺寸， ç  C  C  是信道尺寸，并 *  *  *  表示任意的空间尺寸。该操作变平的`输入 `的空间尺寸中的每个滑动`
kernel_size`尺度的块划分成的列（即，最后的尺寸） 3-d `输出 `形状的张量 （ N  ， C  × Π （ kernel_size
） ， L  ） （N，C \倍\ PROD（\文本{内核\ _size}），L） （ N  ， C  × Π （[H TG203]
kernel_size  ） ， L  ） ，其中 C  × Π （ kernel_size  ） ç\倍\ PROD（\文本{内核\ _size}） C
× Π （ kernel_size  ​​） 是值的每个块内的总数目（一个块具有 Π （ kernel_size  ） \ PROD（\文本{内核\
_size}） Π （ kernel_size  ） 的空间位置每一个包含 C  C  C  -channeled矢量），和 L  L  L
是这样的块的总数：

L=∏d⌊spatial_size[d]+2×padding[d]−dilation[d]×(kernel_size[d]−1)−1stride[d]+1⌋,L
= \prod_d \left\lfloor\frac{\text{spatial\\_size}[d] + 2 \times
\text{padding}[d] % \- \text{dilation}[d] \times (\text{kernel\\_size}[d] - 1)
- 1}{\text{stride}[d]} + 1\right\rfloor,
L=d∏​⌊stride[d]spatial_size[d]+2×padding[d]−dilation[d]×(kernel_size[d]−1)−1​+1⌋,

其中 spatial_size  \文本{空间\ _size}  spatial_size  是由`[HTG27空间尺寸形成]输入 `（ *  *  *
段），和 d  d  d  是所有空间维度。

因此，索引`输出 `在最后一维（列维度）给出特定块内的所有值。

的`填充 `，`步幅 `和`扩张 `参数指定的滑动块如何检索。

  * `步幅 `控制用于所述滑动块的步幅。

  * `填充 `控制隐含零填补处理的双方的量为`填充 `点数重塑之前每个维度。

  * `dilation`controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) has a nice visualization of what `dilation`does.

Parameters

    

  * **kernel_size** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")） - 滑动块的大小

  * **步幅** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选的_ ） - 在输入空间维度的滑动块的步幅。默认值：1

  * **填充** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选的_ ） - 隐式零填充到上输入的两侧添加。默认值：0

  * **扩张** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _，_ _可选的_ ） - 一个控制邻域内的元素的步幅的参数。默认值：1

  * 如果`kernel_size`，`扩张 `，`填充 `或`步幅 `是一个int或长度为1的元组，它们的值将在所有空间维度上复制。

  * 对于两个输入空间维度的情况下，该操作有时被称为`im2col`。

Note

`折 `通过从含有所有块中的所有值求和来计算在所得到的大张量的每个组合的值。`展开 `
通过从大张量提​​取复制在局部块中的值。所以，如果块重叠，它们不是彼此的逆。

Warning

目前，只有4-d的输入张量（成批图像样张量）的支持。

Shape:

    

  * 输入： （ N  ， C  ， *  ） （N，C，*） （ N  ， C  ， *  ）

  * 输出： （ N  ， C  × Π （ kernel_size  ） ， L  ） （N，C \倍\ PROD（\文本{内核\ _size}），L） （ N  ， C  × Π （ kernel_size  ）  ， L  ） 如上所述

Examples:

    
    
    >>> unfold = nn.Unfold(kernel_size=(2, 3))
    >>> input = torch.randn(2, 5, 3, 4)
    >>> output = unfold(input)
    >>> # each patch contains 30 values (2x3=6 vectors, each of 5 channels)
    >>> # 4 blocks (2x3 kernels) in total in the 3x4 input
    >>> output.size()
    torch.Size([2, 30, 4])
    
    >>> # Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)
    >>> inp = torch.randn(1, 3, 10, 12)
    >>> w = torch.randn(2, 3, 4, 5)
    >>> inp_unf = torch.nn.functional.unfold(inp, (4, 5))
    >>> out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
    >>> out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
    >>> # or equivalently (and avoiding a copy),
    >>> # out = out_unf.view(1, 2, 7, 8)
    >>> (torch.nn.functional.conv2d(inp, w) - out).abs().max()
    tensor(1.9073e-06)
    

### 折叠

_class_`torch.nn.``Fold`( _output_size_ , _kernel_size_ , _dilation=1_ ,
_padding=0_ , _stride=1_
)[[source]](_modules/torch/nn/modules/fold.html#Fold)

    

结合滑动局部块到大量含有张量的阵列。

考虑包含滑动局部块，例如分批`输入 `张量，图像的补丁，的形状 （ N  ， C  × Π （ kernel_size  ） ， L  ） （N，C
\倍\ PROD（\文本{内核\ _size}），L） （ N  C  × Π （ kernel_size  ） ， L  ） ，其中 N  N  N
是批次尺寸， C  × Π （ kernel_size  ） ç\倍\ PROD（\文本{内核\ _size}） C  × Π （ kernel_size
） 是值的块内（数的那种块具有 Π （ 柯rnel_size  ） \ PROD（\文本{内核\ _size}） Π （ kernel_size  ）
的空间位置每一个包含 C  C  C  -channeled矢量），和 L  L  L  是块的总数。 （这是完全一样的说明书中的 `展开 `
的输出的形状。）此操作这些局部块结合到大`输出 `的张量形状 （ N  ​​， C  ， output_size  [ 0  ， output_size
[ 1  ， ...  ） （N，C，\文本{输出\ _size} [0]，\文本{输出\ _size} [1]，\点） （ N  ， C  ，
output_size  [ 0  ， output_size  [ 1  ， ...  ） 由重叠值求和。类似于 `展开 `，参数必须满足

L=∏d⌊output_size[d]+2×padding[d]−dilation[d]×(kernel_size[d]−1)−1stride[d]+1⌋,L
= \prod_d \left\lfloor\frac{\text{output\\_size}[d] + 2 \times
\text{padding}[d] % \- \text{dilation}[d] \times (\text{kernel\\_size}[d] - 1)
- 1}{\text{stride}[d]} + 1\right\rfloor,
L=d∏​⌊stride[d]output_size[d]+2×padding[d]−dilation[d]×(kernel_size[d]−1)−1​+1⌋,

其中 d  d  d  是所有空间维度。

  * `output_size`描述的滑动局部块的大含有张量的空间形状。它当多个输入形状地图到相同数量的滑动块，例如，具有`来解决多义性是有用的步幅 & GT ;  0`。

The `padding`, `stride`and `dilation`arguments specify how the sliding
blocks are retrieved.

  * `stride`controls the stride for the sliding blocks.

  * `padding`controls the amount of implicit zero-paddings on both sides for `padding`number of points for each dimension before reshaping.

  * `dilation`controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) has a nice visualization of what `dilation`does.

Parameters

    

  * **output_size** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")） - 的空间尺寸的形状该输出（即，`output.sizes（）[2：]`）

  * **kernel_size** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")) – the size of the sliding blocks

  * **步幅** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")） - 滑动块的步幅输入的空间尺寸。默认值：1

  * **padding** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _,_ _optional_ ) – implicit zero padding to be added on both sides of input. Default: 0

  * **dilation** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)") _,_ _optional_ ) – a parameter that controls the stride of elements within the neighborhood. Default: 1

  * 如果`output_size`，`kernel_size`，`扩张 `，`填充 `或`步幅 `是int或然后它们的值将在所有空间维度上复制长度为1的元组。

  * 对于两个输出空间维度的情况下，该操作有时被称为`col2im`。

Note

`Fold`calculates each combined value in the resulting large tensor by summing
all values from all containing blocks. `Unfold`extracts the values in the
local blocks by copying from the large tensor. So, if the blocks overlap, they
are not inverses of each other.

Warning

目前，只有4-d输出张量（成批图像样张量）的支持。

Shape:

    

  * 输入： （ N  ， C  × Π （ kernel_size  ） ， L  ） （N，C \倍\ PROD（\文本{内核\ _size}），L） （ N  ， C  × Π （ kernel_size  ）  ， L  ）

  * 输出： （ N  ， C  ， output_size  [ 0  ， output_size  [ 1  ， ...  ） （N，C，\文本{输出\ _size} [0]，\文本{输出\ _size} [1]，\点） （ N  ， C  ， output_size  [ 0  ， output_size  [ 1  ， ... [HTG9 5]  ） 如上所述

Examples:

    
    
    >>> fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 2))
    >>> input = torch.randn(1, 3 * 2 * 2, 12)
    >>> output = fold(input)
    >>> output.size()
    torch.Size([1, 3, 4, 5])
    

## 池层

###  MaxPool1d 

_class_`torch.nn.``MaxPool1d`( _kernel_size_ , _stride=None_ , _padding=0_ ,
_dilation=1_ , _return_indices=False_ , _ceil_mode=False_
)[[source]](_modules/torch/nn/modules/pooling.html#MaxPool1d)

    

应用于一维的最大汇集了多个输入飞机组成的输入信号。

在最简单的情况下，所述层的与输入大小 的输出值（ N  C  ， L  ） （N，C，L） （ N  ， C  ， L  ） 和输出 （ N  ， C  ，
L  O  U  T  ） （ N，C，L_ {出}） （ N  C  ， L  O  U  T  ） 可以精确地描述为：

out(Ni,Cj,k)=max⁡m=0,…,kernel_size−1input(Ni,Cj,stride×k+m)out(N_i, C_j, k) =
\max_{m=0, \ldots, \text{kernel\\_size} - 1} input(N_i, C_j, stride \times k +
m) out(Ni​,Cj​,k)=m=0,…,kernel_size−1max​input(Ni​,Cj​,stride×k+m)

如果`填充 `是非零，则输入是隐式地在两侧上用零填充为`填充 `点数。 `扩张
`控制内核点之间的间隔。这是很难形容，但这种[链接](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)有什么`
扩张 `做一个很好的可视化。

Parameters

    

  * **kernel_size** \- 窗口的大小，以采取最大过

  * **步幅** \- 窗口的步幅。默认值为`kernel_size`

  * **填充** \- 隐含零填充到在两侧被添加

  * **扩张** \- 一个控制元件的步幅在窗口的参数

  * **return_indices** \- 如果`真 `，将返回最大指数中，产出一起。有用的 `torch.nn.MaxUnpool1d`以后

  * **ceil_mode** \- 真时，将使用小区而非地板来计算输出形状

Shape:

    

  * 输入： （ N  ， C  ， L  i的 n的 ） （N，C，L_ {在}） （ N  ， C  ， L  i的 n的 ）

  * 输出： （ N  ， C  ， L  O  U  T  ） （N，C，L_ {出}） （ N  ， C  ， L  O  U  T  ）  ，其中

Lout=⌊Lin+2×padding−dilation×(kernel_size−1)−1stride+1⌋L_{out} = \left\lfloor
\frac{L_{in} + 2 \times \text{padding} - \text{dilation} \times
(\text{kernel\\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor
Lout​=⌊strideLin​+2×padding−dilation×(kernel_size−1)−1​+1⌋

Examples:

    
    
    >>> # pool of size=3, stride=2
    >>> m = nn.MaxPool1d(3, stride=2)
    >>> input = torch.randn(20, 16, 50)
    >>> output = m(input)
    

###  MaxPool2d 

_class_`torch.nn.``MaxPool2d`( _kernel_size_ , _stride=None_ , _padding=0_ ,
_dilation=1_ , _return_indices=False_ , _ceil_mode=False_
)[[source]](_modules/torch/nn/modules/pooling.html#MaxPool2d)

    

施加最大的2D汇集在几个输入平面组成的输入信号。

在最简单的情况下，所述层的与输入大小 的输出值（ N  C  ， H  ， W  ） （N，C，H，W） （ N  ， C  ， H  ， W  ） ，输出
（ N  ， C  ， H  O  U  T  ， W  O  U  T  ） （N，C，H_ {出}，W_ {出}） （ N  ， ç  ， H  O
U  T  ， W  O  U  T  [H TG194]  ） 和`kernel_size`（ K  H  ， K  W  ） （KH，千瓦） （
K  H  ， K  W  ） 可以精确地描述为：

out(Ni,Cj,h,w)=max⁡m=0,…,kH−1max⁡n=0,…,kW−1input(Ni,Cj,stride[0]×h+m,stride[1]×w+n)\begin{aligned}
out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1}
\\\ & \text{input}(N_i, C_j, \text{stride[0]} \times h + m, \text{stride[1]}
\times w + n) \end{aligned}
out(Ni​,Cj​,h,w)=​m=0,…,kH−1max​n=0,…,kW−1max​input(Ni​,Cj​,stride[0]×h+m,stride[1]×w+n)​

If `padding`is non-zero, then the input is implicitly zero-padded on both
sides for `padding`number of points. `dilation`controls the spacing between
the kernel points. It is harder to describe, but this
[link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) has
a nice visualization of what `dilation`does.

The parameters `kernel_size`, `stride`, `padding`, `dilation`can either be:

>   * a single `int`– in which case the same value is used for the height and
width dimension

>

>   * a `tuple`of two ints – in which case, the first int is used for the
height dimension, and the second int for the width dimension

>

>

Parameters

    

  * **kernel_size** – the size of the window to take a max over

  * **stride** – the stride of the window. Default value is `kernel_size`

  * **padding** – implicit zero padding to be added on both sides

  * **dilation** – a parameter that controls the stride of elements in the window

  * **return_indices** \- 如果`真 `，将返回最大指数中，产出一起。有用的 `torch.nn.MaxUnpool2d`以后

  * **ceil_mode** – when True, will use ceil instead of floor to compute the output shape

Shape:

    

  * 输入： （ N  ， C  ， H  i的 n的 ， W  i的 n的 ） （N，C，H_ {IN} ，W_ {在}） （ N  ， C  ， H  i的 n的 ， W  i的 n的 ）

  * 输出： （ N  ， C  ， H  O  U  T  ， W  O  U  T  ） （N，C，H_ {出}，W_ {出}） （ N  ， C  ， H  O  U  T  ， W  O  U  T  ） ，其中

Hout=⌊Hin+2∗padding[0]−dilation[0]×(kernel_size[0]−1)−1stride[0]+1⌋H_{out} =
\left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]} \times
(\text{kernel\\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor
Hout​=⌊stride[0]Hin​+2∗padding[0]−dilation[0]×(kernel_size[0]−1)−1​+1⌋

Wout=⌊Win+2∗padding[1]−dilation[1]×(kernel_size[1]−1)−1stride[1]+1⌋W_{out} =
\left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]} \times
(\text{kernel\\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor
Wout​=⌊stride[1]Win​+2∗padding[1]−dilation[1]×(kernel_size[1]−1)−1​+1⌋

Examples:

    
    
    >>> # pool of square window of size=3, stride=2
    >>> m = nn.MaxPool2d(3, stride=2)
    >>> # pool of non-square window
    >>> m = nn.MaxPool2d((3, 2), stride=(2, 1))
    >>> input = torch.randn(20, 16, 50, 32)
    >>> output = m(input)
    

###  MaxPool3d 

_class_`torch.nn.``MaxPool3d`( _kernel_size_ , _stride=None_ , _padding=0_ ,
_dilation=1_ , _return_indices=False_ , _ceil_mode=False_
)[[source]](_modules/torch/nn/modules/pooling.html#MaxPool3d)

    

应用了3D最大汇集了多个输入飞机组成的输入信号。

在最简单的情况下，所述层的与输入大小 的输出值（ N  C  ， d  ， H  ， W  ） （N，C，d，H，W） （ N  ， C  ， d  ， H
， W  ） ，输出 （ N  C  ， d  O  U  T  [HT G98]  ， H  O  U  T  ， W  O  U  T  ）
（N，C，D_ {出}，H_ {出}，W_ {出}） （ N  ， C  ， d  O  U  T  ， H  O  U  T  ， W  O  U  T
） 和`​​  kernel_size`（ K  d  ķ  H  ， K  W  ） （KD，KH，千瓦） （ K  d  K  H  ， K  W
） 可以精确地描述为：

out(Ni,Cj,d,h,w)=max⁡k=0,…,kD−1max⁡m=0,…,kH−1max⁡n=0,…,kW−1input(Ni,Cj,stride[0]×d+k,stride[1]×h+m,stride[2]×w+n)\begin{aligned}
\text{out}(N_i, C_j, d, h, w) ={} & \max_{k=0, \ldots, kD-1} \max_{m=0,
\ldots, kH-1} \max_{n=0, \ldots, kW-1} \\\ & \text{input}(N_i, C_j,
\text{stride[0]} \times d + k, \text{stride[1]} \times h + m, \text{stride[2]}
\times w + n) \end{aligned}
out(Ni​,Cj​,d,h,w)=​k=0,…,kD−1max​m=0,…,kH−1max​n=0,…,kW−1max​input(Ni​,Cj​,stride[0]×d+k,stride[1]×h+m,stride[2]×w+n)​

If `padding`is non-zero, then the input is implicitly zero-padded on both
sides for `padding`number of points. `dilation`controls the spacing between
the kernel points. It is harder to describe, but this
[link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) has
a nice visualization of what `dilation`does.

The parameters `kernel_size`, `stride`, `padding`, `dilation`can either be:

>   * a single `int`– in which case the same value is used for the depth,
height and width dimension

>

>   * a `tuple`of three ints – in which case, the first int is used for the
depth dimension, the second int for the height dimension and the third int for
the width dimension

>

>

Parameters

    

  * **kernel_size** – the size of the window to take a max over

  * **stride** – the stride of the window. Default value is `kernel_size`

  * **填充** \- 隐含零填充在所有三个侧被添加

  * **dilation** – a parameter that controls the stride of elements in the window

  * **return_indices** \- 如果`真 `，将返回最大指数中，产出一起。有用的 `torch.nn.MaxUnpool3d`以后

  * **ceil_mode** – when True, will use ceil instead of floor to compute the output shape

Shape:

    

  * 输入： （ N  ， C  ， d  i的 n的 ， H  i的 n的 ， W  i的 n的 ） （N，C，D_ {IN}，H_ {IN}，W_ {IN} ） （ N  ， C  ， d  i的 n的 ， H  i的 n的 ， W  i的 n的 ）

  * 输出： （ N  ， C  ， d  O  U  T  ， H  O  U  T  ， W  O  U  T  ） （N，C，D_ {出}，H_ {出}，W_ {出}） （  N  ， C  ， d  O  U  T  ， H  O  U  T  ， W  O  U  T  ） ，其中

Dout=⌊Din+2×padding[0]−dilation[0]×(kernel_size[0]−1)−1stride[0]+1⌋D_{out} =
\left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
\times (\text{kernel\\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
Dout​=⌊stride[0]Din​+2×padding[0]−dilation[0]×(kernel_size[0]−1)−1​+1⌋

Hout=⌊Hin+2×padding[1]−dilation[1]×(kernel_size[1]−1)−1stride[1]+1⌋H_{out} =
\left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
\times (\text{kernel\\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
Hout​=⌊stride[1]Hin​+2×padding[1]−dilation[1]×(kernel_size[1]−1)−1​+1⌋

Wout=⌊Win+2×padding[2]−dilation[2]×(kernel_size[2]−1)−1stride[2]+1⌋W_{out} =
\left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2]
\times (\text{kernel\\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor
Wout​=⌊stride[2]Win​+2×padding[2]−dilation[2]×(kernel_size[2]−1)−1​+1⌋

Examples:

    
    
    >>> # pool of square window of size=3, stride=2
    >>> m = nn.MaxPool3d(3, stride=2)
    >>> # pool of non-square window
    >>> m = nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2))
    >>> input = torch.randn(20, 16, 50,44, 31)
    >>> output = m(input)
    

###  MaxUnpool1d 

_class_`torch.nn.``MaxUnpool1d`( _kernel_size_ , _stride=None_ , _padding=0_
)[[source]](_modules/torch/nn/modules/pooling.html#MaxUnpool1d)

    

计算的 `MaxPool1d`的局部逆。

`MaxPool1d`不是完全可逆的，因为非极大值都将丢失。

`MaxUnpool1d`取入作为输入 `输出MaxPool1d`包括的索引极大值，并计算其中所有非极大值都设置为零的局部逆。

Note

`MaxPool1d`
可以映射多个输入大小，以相同的输出大小。因此，反演过程可以得到明确。为了适应这种情况，可以提供所需的输出尺寸为前向呼叫的附加自变量`output_size
`。看到输入和下面的实施例。

Parameters

    

  * **kernel_size** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")） - 最大池窗口的大小。

  * **步幅** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")） - 最大池窗口的步幅。它的默认设置为`kernel_size`。

  * 这是添加到输入填充 - **填充** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")）

Inputs:

    

  * 输入：将输入张量反转

  * 指数：由给出了索引`MaxPool1d`

  * output_size （可选）：目标输出大小

Shape:

    

  * 输入： （ N  ， C  ， H  i的 n的 ） （N，C，H_ {在}） （ N  ， C  ， H  i的 n的 ）

  * 输出： （ N  ， C  ， H  O  U  T  ） （N，C，H_ {出}） （ N  ， C  ， H  O  U  T  ）  ，其中

Hout=(Hin−1)×stride[0]−2×padding[0]+kernel_size[0]H_{out} = (H_{in} - 1)
\times \text{stride}[0] - 2 \times \text{padding}[0] + \text{kernel\\_size}[0]
Hout​=(Hin​−1)×stride[0]−2×padding[0]+kernel_size[0]

或由`output_size`在呼叫操作员给定的

Example:

    
    
    >>> pool = nn.MaxPool1d(2, stride=2, return_indices=True)
    >>> unpool = nn.MaxUnpool1d(2, stride=2)
    >>> input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8]]])
    >>> output, indices = pool(input)
    >>> unpool(output, indices)
    tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.]]])
    
    >>> # Example showcasing the use of output_size
    >>> input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8, 9]]])
    >>> output, indices = pool(input)
    >>> unpool(output, indices, output_size=input.size())
    tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.,  0.]]])
    
    >>> unpool(output, indices)
    tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.]]])
    

###  MaxUnpool2d 

_class_`torch.nn.``MaxUnpool2d`( _kernel_size_ , _stride=None_ , _padding=0_
)[[source]](_modules/torch/nn/modules/pooling.html#MaxUnpool2d)

    

计算的 `MaxPool2d`的局部逆。

`MaxPool2d`不是完全可逆的，因为非极大值都将丢失。

`MaxUnpool2d`取入作为输入 `输出MaxPool2d`包括的索引极大值，并计算其中所有非极大值都设置为零的局部逆。

Note

`MaxPool2d`
可以映射多个输入大小，以相同的输出大小。因此，反演过程可以得到明确。为了适应这种情况，可以提供所需的输出尺寸为前向呼叫的附加自变量`output_size
`。看到输入和下面的实施例。

Parameters

    

  * **kernel_size** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")) – Size of the max pooling window.

  * **stride** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")) – Stride of the max pooling window. It is set to `kernel_size`by default.

  * **padding** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")) – Padding that was added to the input

Inputs:

    

  * input: the input Tensor to invert

  * 指数：由给出了索引`MaxPool2d`

  * output_size (optional): the targeted output size

Shape:

    

  * Input: (N,C,Hin,Win)(N, C, H_{in}, W_{in})(N,C,Hin​,Win​)

  * Output: (N,C,Hout,Wout)(N, C, H_{out}, W_{out})(N,C,Hout​,Wout​) , where

Hout=(Hin−1)×stride[0]−2×padding[0]+kernel_size[0]H_{out} = (H_{in} - 1)
\times \text{stride[0]} - 2 \times \text{padding[0]} + \text{kernel\\_size[0]}
Hout​=(Hin​−1)×stride[0]−2×padding[0]+kernel_size[0]

Wout=(Win−1)×stride[1]−2×padding[1]+kernel_size[1]W_{out} = (W_{in} - 1)
\times \text{stride[1]} - 2 \times \text{padding[1]} + \text{kernel\\_size[1]}
Wout​=(Win​−1)×stride[1]−2×padding[1]+kernel_size[1]

or as given by `output_size`in the call operator

Example:

    
    
    >>> pool = nn.MaxPool2d(2, stride=2, return_indices=True)
    >>> unpool = nn.MaxUnpool2d(2, stride=2)
    >>> input = torch.tensor([[[[ 1.,  2,  3,  4],
                                [ 5,  6,  7,  8],
                                [ 9, 10, 11, 12],
                                [13, 14, 15, 16]]]])
    >>> output, indices = pool(input)
    >>> unpool(output, indices)
    tensor([[[[  0.,   0.,   0.,   0.],
              [  0.,   6.,   0.,   8.],
              [  0.,   0.,   0.,   0.],
              [  0.,  14.,   0.,  16.]]]])
    
    >>> # specify a different output size than input size
    >>> unpool(output, indices, output_size=torch.Size([1, 1, 5, 5]))
    tensor([[[[  0.,   0.,   0.,   0.,   0.],
              [  6.,   0.,   8.,   0.,   0.],
              [  0.,   0.,   0.,  14.,   0.],
              [ 16.,   0.,   0.,   0.,   0.],
              [  0.,   0.,   0.,   0.,   0.]]]])
    

###  MaxUnpool3d 

_class_`torch.nn.``MaxUnpool3d`( _kernel_size_ , _stride=None_ , _padding=0_
)[[source]](_modules/torch/nn/modules/pooling.html#MaxUnpool3d)

    

计算的 `MaxPool3d`的局部逆。

`MaxPool3d`不是完全可逆的，因为非极大值都将丢失。`MaxUnpool3d`取入作为输入 `输出MaxPool3d`
包括的索引极大值，并计算其中所有非极大值都设置为零的局部逆。

Note

`MaxPool3d`
可以映射多个输入大小，以相同的输出大小。因此，反演过程可以得到明确。为了适应这种情况，可以提供所需的输出尺寸为前向呼叫的附加自变量`output_size
`。请参阅下面的输入部分。

Parameters

    

  * **kernel_size** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")) – Size of the max pooling window.

  * **stride** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")) – Stride of the max pooling window. It is set to `kernel_size`by default.

  * **padding** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_[ _tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")) – Padding that was added to the input

Inputs:

    

  * input: the input Tensor to invert

  * 指数：由给出了索引`MaxPool3d`

  * output_size (optional): the targeted output size

Shape:

    

  * Input: (N,C,Din,Hin,Win)(N, C, D_{in}, H_{in}, W_{in})(N,C,Din​,Hin​,Win​)

  * Output: (N,C,Dout,Hout,Wout)(N, C, D_{out}, H_{out}, W_{out})(N,C,Dout​,Hout​,Wout​) , where

Dout=(Din−1)×stride[0]−2×padding[0]+kernel_size[0]D_{out} = (D_{in} - 1)
\times \text{stride[0]} - 2 \times \text{padding[0]} + \text{kernel\\_size[0]}
Dout​=(Din​−1)×stride[0]−2×padding[0]+kernel_size[0]

Hout=(Hin−1)×stride[1]−2×padding[1]+kernel_size[1]H_{out} = (H_{in} - 1)
\times \text{stride[1]} - 2 \times \text{padding[1]} + \text{kernel\\_size[1]}
Hout​=(Hin​−1)×stride[1]−2×padding[1]+kernel_size[1]

Wout=(Win−1)×stride[2]−2×padding[2]+kernel_size[2]W_{out} = (W_{in} - 1)
\times \text{stride[2]} - 2 \times \text{padding[2]} + \text{kernel\\_size[2]}
Wout​=(Win​−1)×stride[2]−2×padding[2]+kernel_size[2]

or as given by `output_size`in the call operator

Example:

    
    
    >>> # pool of square window of size=3, stride=2
    >>> pool = nn.MaxPool3d(3, stride=2, return_indices=True)
    >>> unpool = nn.MaxUnpool3d(3, stride=2)
    >>> output, indices = pool(torch.randn(20, 16, 51, 33, 15))
    >>> unpooled_output = unpool(output, indices)
    >>> unpooled_output.size()
    torch.Size([20, 16, 51, 33, 15])
    

###  AvgPool1d 

_class_`torch.nn.``AvgPool1d`( _kernel_size_ , _stride=None_ , _padding=0_ ,
_ceil_mode=False_ , _count_include_pad=True_
)[[source]](_modules/torch/nn/modules/pooling.html#AvgPool1d)

    

适用在几个输入平面组成的输入信号的平均1D池。

在最简单的情况下，所述层的与输入大小 的输出值（ N  C  ， L  ） （N，C，L） （ N  ， C  ， L  ） ，输出 （ N  ， C  ，
L  O  U  T  ） （ N，C，L_ {出}） （ N  C  ， L  O  U  T  ） 和`kernel_size`K  K  K
可以精确地描述为：

out(Ni,Cj,l)=1k∑m=0k−1input(Ni,Cj,stride×l+m)\text{out}(N_i, C_j, l) =
\frac{1}{k} \sum_{m=0}^{k-1} \text{input}(N_i, C_j, \text{stride} \times l +
m)out(Ni​,Cj​,l)=k1​m=0∑k−1​input(Ni​,Cj​,stride×l+m)

如果`填充 `是非零，则输入是隐式地在两侧上用零填充为`填充 `点数。

参数`kernel_size`，`步幅 `，`填充 `可各自为一个`INT`或一个元素的元组。

Parameters

    

  * **kernel_size** \- 窗口的大小

  * **stride** – the stride of the window. Default value is `kernel_size`

  * **padding** – implicit zero padding to be added on both sides

  * **ceil_mode** – when True, will use ceil instead of floor to compute the output shape

  * **count_include_pad** \- 真时，将包括在平均计算补零

Shape:

    

  * Input: (N,C,Lin)(N, C, L_{in})(N,C,Lin​)

  * Output: (N,C,Lout)(N, C, L_{out})(N,C,Lout​) , where

Lout=⌊Lin+2×padding−kernel_sizestride+1⌋L_{out} = \left\lfloor \frac{L_{in} +
2 \times \text{padding} - \text{kernel\\_size}}{\text{stride}} +
1\right\rfloor Lout​=⌊strideLin​+2×padding−kernel_size​+1⌋

Examples:

    
    
    >>> # pool with window of size=3, stride=2
    >>> m = nn.AvgPool1d(3, stride=2)
    >>> m(torch.tensor([[[1.,2,3,4,5,6,7]]]))
    tensor([[[ 2.,  4.,  6.]]])
    

###  AvgPool2d 

_class_`torch.nn.``AvgPool2d`( _kernel_size_ , _stride=None_ , _padding=0_ ,
_ceil_mode=False_ , _count_include_pad=True_ , _divisor_override=None_
)[[source]](_modules/torch/nn/modules/pooling.html#AvgPool2d)

    

适用在几个输入平面组成的输入信号的2D平均池。

In the simplest case, the output value of the layer with input size
(N,C,H,W)(N, C, H, W)(N,C,H,W) , output (N,C,Hout,Wout)(N, C, H_{out},
W_{out})(N,C,Hout​,Wout​) and `kernel_size`(kH,kW)(kH, kW)(kH,kW) can be
precisely described as:

out(Ni,Cj,h,w)=1kH∗kW∑m=0kH−1∑n=0kW−1input(Ni,Cj,stride[0]×h+m,stride[1]×w+n)out(N_i,
C_j, h, w) = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1} input(N_i,
C_j, stride[0] \times h + m, stride[1] \times w +
n)out(Ni​,Cj​,h,w)=kH∗kW1​m=0∑kH−1​n=0∑kW−1​input(Ni​,Cj​,stride[0]×h+m,stride[1]×w+n)

If `padding`is non-zero, then the input is implicitly zero-padded on both
sides for `padding`number of points.

参数`kernel_size`，`步幅 `，`填充 `可以是：

>   * a single `int`– in which case the same value is used for the height and
width dimension

>

>   * a `tuple`of two ints – in which case, the first int is used for the
height dimension, and the second int for the width dimension

>

>

Parameters

    

  * **kernel_size** – the size of the window

  * **stride** – the stride of the window. Default value is `kernel_size`

  * **padding** – implicit zero padding to be added on both sides

  * **ceil_mode** – when True, will use ceil instead of floor to compute the output shape

  * **count_include_pad** – when True, will include the zero-padding in the averaging calculation

  * **divisor_override** \- 如果指定的话，它将被用作除数，否则ATTR： kernel_size 将用于

Shape:

    

  * Input: (N,C,Hin,Win)(N, C, H_{in}, W_{in})(N,C,Hin​,Win​)

  * Output: (N,C,Hout,Wout)(N, C, H_{out}, W_{out})(N,C,Hout​,Wout​) , where

Hout=⌊Hin+2×padding[0]−kernel_size[0]stride[0]+1⌋H_{out} =
\left\lfloor\frac{H_{in} + 2 \times \text{padding}[0] -
\text{kernel\\_size}[0]}{\text{stride}[0]} + 1\right\rfloor
Hout​=⌊stride[0]Hin​+2×padding[0]−kernel_size[0]​+1⌋

Wout=⌊Win+2×padding[1]−kernel_size[1]stride[1]+1⌋W_{out} =
\left\lfloor\frac{W_{in} + 2 \times \text{padding}[1] -
\text{kernel\\_size}[1]}{\text{stride}[1]} + 1\right\rfloor
Wout​=⌊stride[1]Win​+2×padding[1]−kernel_size[1]​+1⌋

Examples:

    
    
    >>> # pool of square window of size=3, stride=2
    >>> m = nn.AvgPool2d(3, stride=2)
    >>> # pool of non-square window
    >>> m = nn.AvgPool2d((3, 2), stride=(2, 1))
    >>> input = torch.randn(20, 16, 50, 32)
    >>> output = m(input)
    

###  AvgPool3d 

_class_`torch.nn.``AvgPool3d`( _kernel_size_ , _stride=None_ , _padding=0_ ,
_ceil_mode=False_ , _count_include_pad=True_ , _divisor_override=None_
)[[source]](_modules/torch/nn/modules/pooling.html#AvgPool3d)

    

适用在几个输入平面组成的输入信号的平均三维池。

In the simplest case, the output value of the layer with input size
(N,C,D,H,W)(N, C, D, H, W)(N,C,D,H,W) , output (N,C,Dout,Hout,Wout)(N, C,
D_{out}, H_{out}, W_{out})(N,C,Dout​,Hout​,Wout​) and `kernel_size`
(kD,kH,kW)(kD, kH, kW)(kD,kH,kW) can be precisely described as:

out(Ni,Cj,d,h,w)=∑k=0kD−1∑m=0kH−1∑n=0kW−1input(Ni,Cj,stride[0]×d+k,stride[1]×h+m,stride[2]×w+n)kD×kH×kW\begin{aligned}
\text{out}(N_i, C_j, d, h, w) ={} & \sum_{k=0}^{kD-1} \sum_{m=0}^{kH-1}
\sum_{n=0}^{kW-1} \\\ & \frac{\text{input}(N_i, C_j, \text{stride}[0] \times d
+ k, \text{stride}[1] \times h + m, \text{stride}[2] \times w + n)} {kD \times
kH \times kW} \end{aligned}
out(Ni​,Cj​,d,h,w)=​k=0∑kD−1​m=0∑kH−1​n=0∑kW−1​kD×kH×kWinput(Ni​,Cj​,stride[0]×d+k,stride[1]×h+m,stride[2]×w+n)​​

如果`填充 `是非零，则输入是隐式地在所有三个侧面零填充为`填充 `数量的点。

参数`kernel_size`，`步幅 `可以是：

>   * a single `int`– in which case the same value is used for the depth,
height and width dimension

>

>   * a `tuple`of three ints – in which case, the first int is used for the
depth dimension, the second int for the height dimension and the third int for
the width dimension

>

>

Parameters

    

  * **kernel_size** – the size of the window

  * **stride** – the stride of the window. Default value is `kernel_size`

  * **padding** – implicit zero padding to be added on all three sides

  * **ceil_mode** – when True, will use ceil instead of floor to compute the output shape

  * **count_include_pad** – when True, will include the zero-padding in the averaging calculation

  * **divisor_override** – if specified, it will be used as divisor, otherwise attr:kernel_size will be used

Shape:

    

  * Input: (N,C,Din,Hin,Win)(N, C, D_{in}, H_{in}, W_{in})(N,C,Din​,Hin​,Win​)

  * Output: (N,C,Dout,Hout,Wout)(N, C, D_{out}, H_{out}, W_{out})(N,C,Dout​,Hout​,Wout​) , where

Dout=⌊Din+2×padding[0]−kernel_size[0]stride[0]+1⌋D_{out} =
\left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] -
\text{kernel\\_size}[0]}{\text{stride}[0]} + 1\right\rfloor
Dout​=⌊stride[0]Din​+2×padding[0]−kernel_size[0]​+1⌋

Hout=⌊Hin+2×padding[1]−kernel_size[1]stride[1]+1⌋H_{out} =
\left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] -
\text{kernel\\_size}[1]}{\text{stride}[1]} + 1\right\rfloor
Hout​=⌊stride[1]Hin​+2×padding[1]−kernel_size[1]​+1⌋

Wout=⌊Win+2×padding[2]−kernel_size[2]stride[2]+1⌋W_{out} =
\left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] -
\text{kernel\\_size}[2]}{\text{stride}[2]} + 1\right\rfloor
Wout​=⌊stride[2]Win​+2×padding[2]−kernel_size[2]​+1⌋

Examples:

    
    
    >>> # pool of square window of size=3, stride=2
    >>> m = nn.AvgPool3d(3, stride=2)
    >>> # pool of non-square window
    >>> m = nn.AvgPool3d((3, 2, 2), stride=(2, 1, 2))
    >>> input = torch.randn(20, 16, 50,44, 31)
    >>> output = m(input)
    

###  FractionalMaxPool2d 

_class_`torch.nn.``FractionalMaxPool2d`( _kernel_size_ , _output_size=None_ ,
_output_ratio=None_ , _return_indices=False_ , __random_samples=None_
)[[source]](_modules/torch/nn/modules/pooling.html#FractionalMaxPool2d)

    

适用在几个输入平面组成的输入信号的2D分数最大池。

分数MaxPooling中详细纸张[分数MaxPooling ](http://arxiv.org/abs/1412.6071)通过格雷厄姆描述

最大-池操作在施加 K  H  × K  W  的kH \倍千瓦 K  H  × K  W
区域通过由目标输出尺寸决定的随机步长。的输出特征的数量等于输入平面的数量。

Parameters

    

  * **kernel_size** \- 窗口的大小，以采取最大过来。可以是单一的数k（对于k X k的平方内核）或元组（KH，千瓦）

  * **output_size** \- 形式哦X OW 的图像的目标输出大小。可以是一个元组（OH，OW）或正方形图像的单个数字喔哦X哦

  * **output_ratio** \- 如果一个人希望有一个输出大小为输入大小的比率，这个选项可以给出。这必须是在范围内的数或元组（0，1）

  * **return_indices** \- 如果`真 `，将返回指数中，产出一起。有用传递给`nn.MaxUnpool2d（） `。默认值：`假 `

例子

    
    
    >>> # pool of square window of size=3, and target output size 13x12
    >>> m = nn.FractionalMaxPool2d(3, output_size=(13, 12))
    >>> # pool of square window and target output size being half of input image size
    >>> m = nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))
    >>> input = torch.randn(20, 16, 50, 32)
    >>> output = m(input)
    

###  LPPool1d 

_class_`torch.nn.``LPPool1d`( _norm_type_ , _kernel_size_ , _stride=None_ ,
_ceil_mode=False_
)[[source]](_modules/torch/nn/modules/pooling.html#LPPool1d)

    

适用在几个输入平面组成的输入信号的功率1D平均池。

在每个窗口中，计算出的函数是：

f(X)=∑x∈Xxppf(X) = \sqrt[p]{\sum_{x \in X} x^{p}} f(X)=p​x∈X∑​xp​

  * 在P =  ∞ \ infty  ∞ ，可以得到最大池

  * 在p = 1时，可以得到萨姆池（其正比于平均池）

Note

如果总和至p的功率为零时，该函数的梯度没有定义。此实现会在这种情况下，设置渐变至零。

Parameters

    

  * **kernel_size** \- 单个int，窗口的大小

  * **步幅** \- 一个单一的在，窗口的步幅。默认值为`kernel_size`

  * **ceil_mode** – when True, will use ceil instead of floor to compute the output shape

Shape:

    

  * Input: (N,C,Lin)(N, C, L_{in})(N,C,Lin​)

  * Output: (N,C,Lout)(N, C, L_{out})(N,C,Lout​) , where

Lout=⌊Lin+2×padding−kernel_sizestride+1⌋L_{out} = \left\lfloor\frac{L_{in} + 2
\times \text{padding} - \text{kernel\\_size}}{\text{stride}} + 1\right\rfloor
Lout​=⌊strideLin​+2×padding−kernel_size​+1⌋

Examples::

    
    
    
    >>> # power-2 pool of window of length 3, with stride 2.
    >>> m = nn.LPPool1d(2, 3, stride=2)
    >>> input = torch.randn(20, 16, 50)
    >>> output = m(input)
    

###  LPPool2d 

_class_`torch.nn.``LPPool2d`( _norm_type_ , _kernel_size_ , _stride=None_ ,
_ceil_mode=False_
)[[source]](_modules/torch/nn/modules/pooling.html#LPPool2d)

    

适用在几个输入平面组成的输入信号的2D功率平均池。

On each window, the function computed is:

f(X)=∑x∈Xxppf(X) = \sqrt[p]{\sum_{x \in X} x^{p}} f(X)=p​x∈X∑​xp​

  * At p = ∞\infty∞ , one gets Max Pooling

  * 在p = 1时，可以得到萨姆池（其正比于平均池）

The parameters `kernel_size`, `stride`can either be:

>   * a single `int`– in which case the same value is used for the height and
width dimension

>

>   * a `tuple`of two ints – in which case, the first int is used for the
height dimension, and the second int for the width dimension

>

>

Note

If the sum to the power of p is zero, the gradient of this function is not
defined. This implementation will set the gradient to zero in this case.

Parameters

    

  * **kernel_size** – the size of the window

  * **stride** – the stride of the window. Default value is `kernel_size`

  * **ceil_mode** – when True, will use ceil instead of floor to compute the output shape

Shape:

    

  * Input: (N,C,Hin,Win)(N, C, H_{in}, W_{in})(N,C,Hin​,Win​)

  * Output: (N,C,Hout,Wout)(N, C, H_{out}, W_{out})(N,C,Hout​,Wout​) , where

Hout=⌊Hin+2×padding[0]−dilation[0]×(kernel_size[0]−1)−1stride[0]+1⌋H_{out} =
\left\lfloor\frac{H_{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
\times (\text{kernel\\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
Hout​=⌊stride[0]Hin​+2×padding[0]−dilation[0]×(kernel_size[0]−1)−1​+1⌋

Wout=⌊Win+2×padding[1]−dilation[1]×(kernel_size[1]−1)−1stride[1]+1⌋W_{out} =
\left\lfloor\frac{W_{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
\times (\text{kernel\\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
Wout​=⌊stride[1]Win​+2×padding[1]−dilation[1]×(kernel_size[1]−1)−1​+1⌋

Examples:

    
    
    >>> # power-2 pool of square window of size=3, stride=2
    >>> m = nn.LPPool2d(2, 3, stride=2)
    >>> # pool of non-square window of power 1.2
    >>> m = nn.LPPool2d(1.2, (3, 2), stride=(2, 1))
    >>> input = torch.randn(20, 16, 50, 32)
    >>> output = m(input)
    

###  AdaptiveMaxPool1d 

_class_`torch.nn.``AdaptiveMaxPool1d`( _output_size_ , _return_indices=False_
)[[source]](_modules/torch/nn/modules/pooling.html#AdaptiveMaxPool1d)

    

适用在几个输入平面组成的输入信号的1D自适应最大池。

输出尺寸是H，任何输入的大小。的输出特征的数量等于输入平面的数量。

Parameters

    

  * **output_size** \- 目标输出口径H

  * **return_indices** \- 如果`真 `，将返回指数中，产出一起。有用传递给nn.MaxUnpool1d。默认值：`假 `

Examples

    
    
    >>> # target output size of 5
    >>> m = nn.AdaptiveMaxPool1d(5)
    >>> input = torch.randn(1, 64, 8)
    >>> output = m(input)
    

###  AdaptiveMaxPool2d 

_class_`torch.nn.``AdaptiveMaxPool2d`( _output_size_ , _return_indices=False_
)[[source]](_modules/torch/nn/modules/pooling.html#AdaptiveMaxPool2d)

    

适用在几个输入平面组成的输入信号的2D自适应最大池。

输出是尺寸高×宽的，对于任何输入大小。的输出特征的数量等于输入平面的数量。

Parameters

    

  * **output_size** \- 的形式高x W的图像的目标输出大小可以是一个元组（H，W）或对于方形图像高x H.单个H H和W可以是`INT`或`无 `这意味着大小将是相同的，输入的。

  * **return_indices** \- 如果`真 `，将返回指数中，产出一起。有用传递给nn.MaxUnpool2d。默认值：`假 `

Examples

    
    
    >>> # target output size of 5x7
    >>> m = nn.AdaptiveMaxPool2d((5,7))
    >>> input = torch.randn(1, 64, 8, 9)
    >>> output = m(input)
    >>> # target output size of 7x7 (square)
    >>> m = nn.AdaptiveMaxPool2d(7)
    >>> input = torch.randn(1, 64, 10, 9)
    >>> output = m(input)
    >>> # target output size of 10x7
    >>> m = nn.AdaptiveMaxPool2d((None, 7))
    >>> input = torch.randn(1, 64, 10, 9)
    >>> output = m(input)
    

###  AdaptiveMaxPool3d 

_class_`torch.nn.``AdaptiveMaxPool3d`( _output_size_ , _return_indices=False_
)[[source]](_modules/torch/nn/modules/pooling.html#AdaptiveMaxPool3d)

    

适用在几个输入平面组成的输入信号的3D自适应最大池。

输出是尺寸d x高x W的，对于任何输入大小。的输出特征的数量等于输入平面的数量。

Parameters

    

  * **output_size** \- 形式d×高×W的图像的目标输出大小可以是一个元组（d，H，W）或多维数据集d X d X D. d单个d， H和W可以是`INT`或`无 `这意味着大小将是相同的，输入的。

  * **return_indices** \- 如果`真 `，将返回指数中，产出一起。有用传递给nn.MaxUnpool3d。默认值：`假 `

Examples

    
    
    >>> # target output size of 5x7x9
    >>> m = nn.AdaptiveMaxPool3d((5,7,9))
    >>> input = torch.randn(1, 64, 8, 9, 10)
    >>> output = m(input)
    >>> # target output size of 7x7x7 (cube)
    >>> m = nn.AdaptiveMaxPool3d(7)
    >>> input = torch.randn(1, 64, 10, 9, 8)
    >>> output = m(input)
    >>> # target output size of 7x9x8
    >>> m = nn.AdaptiveMaxPool3d((7, None, None))
    >>> input = torch.randn(1, 64, 10, 9, 8)
    >>> output = m(input)
    

###  AdaptiveAvgPool1d 

_class_`torch.nn.``AdaptiveAvgPool1d`( _output_size_
)[[source]](_modules/torch/nn/modules/pooling.html#AdaptiveAvgPool1d)

    

适用在几个输入平面组成的输入信号的1D自适应平均池。

The output size is H, for any input size. The number of output features is
equal to the number of input planes.

Parameters

    

**output_size** – the target output size H

Examples

    
    
    >>> # target output size of 5
    >>> m = nn.AdaptiveAvgPool1d(5)
    >>> input = torch.randn(1, 64, 8)
    >>> output = m(input)
    

###  AdaptiveAvgPool2d 

_class_`torch.nn.``AdaptiveAvgPool2d`( _output_size_
)[[source]](_modules/torch/nn/modules/pooling.html#AdaptiveAvgPool2d)

    

适用在几个输入平面组成的输入信号的2D自适应平均池。

The output is of size H x W, for any input size. The number of output features
is equal to the number of input planes.

Parameters

    

**output_size** – the target output size of the image of the form H x W. Can
be a tuple (H, W) or a single H for a square image H x H. H and W can be
either a `int`, or `None`which means the size will be the same as that of the
input.

Examples

    
    
    >>> # target output size of 5x7
    >>> m = nn.AdaptiveAvgPool2d((5,7))
    >>> input = torch.randn(1, 64, 8, 9)
    >>> output = m(input)
    >>> # target output size of 7x7 (square)
    >>> m = nn.AdaptiveAvgPool2d(7)
    >>> input = torch.randn(1, 64, 10, 9)
    >>> output = m(input)
    >>> # target output size of 10x7
    >>> m = nn.AdaptiveMaxPool2d((None, 7))
    >>> input = torch.randn(1, 64, 10, 9)
    >>> output = m(input)
    

###  AdaptiveAvgPool3d 

_class_`torch.nn.``AdaptiveAvgPool3d`( _output_size_
)[[source]](_modules/torch/nn/modules/pooling.html#AdaptiveAvgPool3d)

    

适用在几个输入平面组成的输入信号的3D自适应平均池。

The output is of size D x H x W, for any input size. The number of output
features is equal to the number of input planes.

Parameters

    

**output_size** \- 形式d×高×W的目标输出大小可以是用于立方体d X d X D. d，H和元组（d，H，W）或单数d W可以是无论是`
INT`或`无 `这意味着大小将是相同的，输入的。

Examples

    
    
    >>> # target output size of 5x7x9
    >>> m = nn.AdaptiveAvgPool3d((5,7,9))
    >>> input = torch.randn(1, 64, 8, 9, 10)
    >>> output = m(input)
    >>> # target output size of 7x7x7 (cube)
    >>> m = nn.AdaptiveAvgPool3d(7)
    >>> input = torch.randn(1, 64, 10, 9, 8)
    >>> output = m(input)
    >>> # target output size of 7x9x8
    >>> m = nn.AdaptiveMaxPool3d((7, None, None))
    >>> input = torch.randn(1, 64, 10, 9, 8)
    >>> output = m(input)
    

## 填充层

###  ReflectionPad1d 

_class_`torch.nn.``ReflectionPad1d`( _padding_
)[[source]](_modules/torch/nn/modules/padding.html#ReflectionPad1d)

    

焊盘使用输入边界的反射输入张量。

对于 N 维填充，用[ `torch.nn.functional.pad（） `
](nn.functional.html#torch.nn.functional.pad "torch.nn.functional.pad")。

Parameters

    

**填充** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in
Python v3.7\)") _，_ [ _元组_
](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")）
- 填充的大小。如果是 INT ，使用在所有边界相同的填充。如果2- 元组，使用（ padding_left  \ {文本填充\ _Left}
padding_left  ， padding_right  \文本{填充\ _right}  padding_right  ）

Shape:

    

  * 输入： （ N  ， C  ， W  i的 n的 ） （N，C，W_ {在}） （ N  ， C  ， W  i的 n的 ）

  * 输出： （ N  ， C  ， W  O  U  T  ） （N，C，W_ {出}） （ N  ， C  ， W  O  U  T  ）  其中

W  O  U  T  =  W  i的 n的 \+  padding_left  \+  padding_right  W_ {出} = W_ {在} +
\文本{填充\ _Left} + \文本{填充\ _right}  W  O  U  T  =  W  [HT G100]  i的 n的 \+
padding_left  \+  padding_right

Examples:

    
    
    >>> m = nn.ReflectionPad1d(2)
    >>> input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
    >>> input
    tensor([[[0., 1., 2., 3.],
             [4., 5., 6., 7.]]])
    >>> m(input)
    tensor([[[2., 1., 0., 1., 2., 3., 2., 1.],
             [6., 5., 4., 5., 6., 7., 6., 5.]]])
    >>> # using different paddings for different sides
    >>> m = nn.ReflectionPad1d((3, 1))
    >>> m(input)
    tensor([[[3., 2., 1., 0., 1., 2., 3., 2.],
             [7., 6., 5., 4., 5., 6., 7., 6.]]])
    

###  ReflectionPad2d 

_class_`torch.nn.``ReflectionPad2d`( _padding_
)[[source]](_modules/torch/nn/modules/padding.html#ReflectionPad2d)

    

Pads the input tensor using the reflection of the input boundary.

For N-dimensional padding, use
[`torch.nn.functional.pad()`](nn.functional.html#torch.nn.functional.pad
"torch.nn.functional.pad").

Parameters

    

**填充** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in
Python v3.7\)") _，_ [ _元组_
](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")）
- 填充的大小。如果是 INT ，使用在所有边界相同的填充。如果4- 元组，使用（ padding_left  \ {文本填充\ _Left}
padding_left  ， padding_right  \文本{填充\ _right}  padding_right  ， padding_top
\文本{填充\ _top}  padding_top  ， padding_bottom  \ {文本paddi纳克\ _bottom}
padding_bottom  ）

Shape:

    

  * Input: (N,C,Hin,Win)(N, C, H_{in}, W_{in})(N,C,Hin​,Win​)

  * 输出： （ N  ， C  ， H  O  U  T  ， W  O  U  T  ） （N，C，H_ {出}，W_ {出}） （ N  ， C  ， H  O  U  T  ， W  O  U  T  ） 其中

H  O  U  T  =  H  i的 n的 \+  padding_top  \+  padding_bottom  H_ {出} = H_ {在} +
\文本{填充\ _top} + \文本{填充\ _bottom}  H  O  U  T  =  H  [HT G100]  i的 n的 \+
padding_top  \+  padding_bottom

Wout=Win+padding_left+padding_rightW_{out} = W_{in} + \text{padding\\_left} +
\text{padding\\_right}Wout​=Win​+padding_left+padding_right

Examples:

    
    
    >>> m = nn.ReflectionPad2d(2)
    >>> input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
    >>> input
    tensor([[[[0., 1., 2.],
              [3., 4., 5.],
              [6., 7., 8.]]]])
    >>> m(input)
    tensor([[[[8., 7., 6., 7., 8., 7., 6.],
              [5., 4., 3., 4., 5., 4., 3.],
              [2., 1., 0., 1., 2., 1., 0.],
              [5., 4., 3., 4., 5., 4., 3.],
              [8., 7., 6., 7., 8., 7., 6.],
              [5., 4., 3., 4., 5., 4., 3.],
              [2., 1., 0., 1., 2., 1., 0.]]]])
    >>> # using different paddings for different sides
    >>> m = nn.ReflectionPad2d((1, 1, 2, 0))
    >>> m(input)
    tensor([[[[7., 6., 7., 8., 7.],
              [4., 3., 4., 5., 4.],
              [1., 0., 1., 2., 1.],
              [4., 3., 4., 5., 4.],
              [7., 6., 7., 8., 7.]]]])
    

###  ReplicationPad1d 

_class_`torch.nn.``ReplicationPad1d`( _padding_
)[[source]](_modules/torch/nn/modules/padding.html#ReplicationPad1d)

    

垫使用输入边界的复制输入张量。

For N-dimensional padding, use
[`torch.nn.functional.pad()`](nn.functional.html#torch.nn.functional.pad
"torch.nn.functional.pad").

Parameters

    

**padding** ([ _int_](https://docs.python.org/3/library/functions.html#int
"\(in Python v3.7\)") _,_[
_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python
v3.7\)")) – the size of the padding. If is int, uses the same padding in all
boundaries. If a 2-tuple, uses (padding_left\text{padding\\_left}padding_left
, padding_right\text{padding\\_right}padding_right )

Shape:

    

  * Input: (N,C,Win)(N, C, W_{in})(N,C,Win​)

  * Output: (N,C,Wout)(N, C, W_{out})(N,C,Wout​) where

Wout=Win+padding_left+padding_rightW_{out} = W_{in} + \text{padding\\_left} +
\text{padding\\_right}Wout​=Win​+padding_left+padding_right

Examples:

    
    
    >>> m = nn.ReplicationPad1d(2)
    >>> input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
    >>> input
    tensor([[[0., 1., 2., 3.],
             [4., 5., 6., 7.]]])
    >>> m(input)
    tensor([[[0., 0., 0., 1., 2., 3., 3., 3.],
             [4., 4., 4., 5., 6., 7., 7., 7.]]])
    >>> # using different paddings for different sides
    >>> m = nn.ReplicationPad1d((3, 1))
    >>> m(input)
    tensor([[[0., 0., 0., 0., 1., 2., 3., 3.],
             [4., 4., 4., 4., 5., 6., 7., 7.]]])
    

###  ReplicationPad2d 

_class_`torch.nn.``ReplicationPad2d`( _padding_
)[[source]](_modules/torch/nn/modules/padding.html#ReplicationPad2d)

    

Pads the input tensor using replication of the input boundary.

For N-dimensional padding, use
[`torch.nn.functional.pad()`](nn.functional.html#torch.nn.functional.pad
"torch.nn.functional.pad").

Parameters

    

**padding** ([ _int_](https://docs.python.org/3/library/functions.html#int
"\(in Python v3.7\)") _,_[
_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python
v3.7\)")) – the size of the padding. If is int, uses the same padding in all
boundaries. If a 4-tuple, uses (padding_left\text{padding\\_left}padding_left
, padding_right\text{padding\\_right}padding_right ,
padding_top\text{padding\\_top}padding_top ,
padding_bottom\text{padding\\_bottom}padding_bottom )

Shape:

    

  * Input: (N,C,Hin,Win)(N, C, H_{in}, W_{in})(N,C,Hin​,Win​)

  * Output: (N,C,Hout,Wout)(N, C, H_{out}, W_{out})(N,C,Hout​,Wout​) where

Hout=Hin+padding_top+padding_bottomH_{out} = H_{in} + \text{padding\\_top} +
\text{padding\\_bottom}Hout​=Hin​+padding_top+padding_bottom

Wout=Win+padding_left+padding_rightW_{out} = W_{in} + \text{padding\\_left} +
\text{padding\\_right}Wout​=Win​+padding_left+padding_right

Examples:

    
    
    >>> m = nn.ReplicationPad2d(2)
    >>> input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
    >>> input
    tensor([[[[0., 1., 2.],
              [3., 4., 5.],
              [6., 7., 8.]]]])
    >>> m(input)
    tensor([[[[0., 0., 0., 1., 2., 2., 2.],
              [0., 0., 0., 1., 2., 2., 2.],
              [0., 0., 0., 1., 2., 2., 2.],
              [3., 3., 3., 4., 5., 5., 5.],
              [6., 6., 6., 7., 8., 8., 8.],
              [6., 6., 6., 7., 8., 8., 8.],
              [6., 6., 6., 7., 8., 8., 8.]]]])
    >>> # using different paddings for different sides
    >>> m = nn.ReplicationPad2d((1, 1, 2, 0))
    >>> m(input)
    tensor([[[[0., 0., 1., 2., 2.],
              [0., 0., 1., 2., 2.],
              [0., 0., 1., 2., 2.],
              [3., 3., 4., 5., 5.],
              [6., 6., 7., 8., 8.]]]])
    

###  ReplicationPad3d 

_class_`torch.nn.``ReplicationPad3d`( _padding_
)[[source]](_modules/torch/nn/modules/padding.html#ReplicationPad3d)

    

Pads the input tensor using replication of the input boundary.

For N-dimensional padding, use
[`torch.nn.functional.pad()`](nn.functional.html#torch.nn.functional.pad
"torch.nn.functional.pad").

Parameters

    

**填充** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in
Python v3.7\)") _，_ [ _元组_
](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")）
- 填充的大小。如果是 INT ，使用在所有边界相同的填充。如果6- 元组，使用（ padding_left  \ {文本填充\ _Left}
padding_left  ， padding_right  \文本{填充\ _right}  padding_right  ， padding_top
\文本{填充\ _top}  padding_top  ， padding_bottom  \ {文本paddi纳克\ _bottom}
padding_bottom  ， padding_front  \文本{填充\ _front}  padding_front  ，
padding_back  \文本{填充\ _back}  padding_back  ）

Shape:

    

  * Input: (N,C,Din,Hin,Win)(N, C, D_{in}, H_{in}, W_{in})(N,C,Din​,Hin​,Win​)

  * 输出： （ N  ， C  ， d  O  U  T  ， H  O  U  T  ， W  O  U  T  ） （N，C，D_ {出}，H_ {出}，W_ {出}） （  N  ， C  ， d  O  U  T  ， H  O  U  T  ， W  O  U  T  ） 其中

d  O  U  T  =  d  i的 n的 \+  padding_front  \+  padding_back  D_ {出} = D_ {在} +
\文本{填充\ _front} + \文本{填充\ _back}  d  O  U  T  =  d  [HT G100]  i的 n的 \+
padding_front  \+  padding_back

Hout=Hin+padding_top+padding_bottomH_{out} = H_{in} + \text{padding\\_top} +
\text{padding\\_bottom}Hout​=Hin​+padding_top+padding_bottom

Wout=Win+padding_left+padding_rightW_{out} = W_{in} + \text{padding\\_left} +
\text{padding\\_right}Wout​=Win​+padding_left+padding_right

Examples:

    
    
    >>> m = nn.ReplicationPad3d(3)
    >>> input = torch.randn(16, 3, 8, 320, 480)
    >>> output = m(input)
    >>> # using different paddings for different sides
    >>> m = nn.ReplicationPad3d((3, 3, 6, 6, 1, 1))
    >>> output = m(input)
    

###  ZeroPad2d 

_class_`torch.nn.``ZeroPad2d`( _padding_
)[[source]](_modules/torch/nn/modules/padding.html#ZeroPad2d)

    

零垫输入张量边界。

For N-dimensional padding, use
[`torch.nn.functional.pad()`](nn.functional.html#torch.nn.functional.pad
"torch.nn.functional.pad").

Parameters

    

**padding** ([ _int_](https://docs.python.org/3/library/functions.html#int
"\(in Python v3.7\)") _,_[
_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python
v3.7\)")) – the size of the padding. If is int, uses the same padding in all
boundaries. If a 4-tuple, uses (padding_left\text{padding\\_left}padding_left
, padding_right\text{padding\\_right}padding_right ,
padding_top\text{padding\\_top}padding_top ,
padding_bottom\text{padding\\_bottom}padding_bottom )

Shape:

    

  * Input: (N,C,Hin,Win)(N, C, H_{in}, W_{in})(N,C,Hin​,Win​)

  * Output: (N,C,Hout,Wout)(N, C, H_{out}, W_{out})(N,C,Hout​,Wout​) where

Hout=Hin+padding_top+padding_bottomH_{out} = H_{in} + \text{padding\\_top} +
\text{padding\\_bottom}Hout​=Hin​+padding_top+padding_bottom

Wout=Win+padding_left+padding_rightW_{out} = W_{in} + \text{padding\\_left} +
\text{padding\\_right}Wout​=Win​+padding_left+padding_right

Examples:

    
    
    >>> m = nn.ZeroPad2d(2)
    >>> input = torch.randn(1, 1, 3, 3)
    >>> input
    tensor([[[[-0.1678, -0.4418,  1.9466],
              [ 0.9604, -0.4219, -0.5241],
              [-0.9162, -0.5436, -0.6446]]]])
    >>> m(input)
    tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
              [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
              [ 0.0000,  0.0000, -0.1678, -0.4418,  1.9466,  0.0000,  0.0000],
              [ 0.0000,  0.0000,  0.9604, -0.4219, -0.5241,  0.0000,  0.0000],
              [ 0.0000,  0.0000, -0.9162, -0.5436, -0.6446,  0.0000,  0.0000],
              [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
              [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]]])
    >>> # using different paddings for different sides
    >>> m = nn.ZeroPad2d((1, 1, 2, 0))
    >>> m(input)
    tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
              [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
              [ 0.0000, -0.1678, -0.4418,  1.9466,  0.0000],
              [ 0.0000,  0.9604, -0.4219, -0.5241,  0.0000],
              [ 0.0000, -0.9162, -0.5436, -0.6446,  0.0000]]]])
    

###  ConstantPad1d 

_class_`torch.nn.``ConstantPad1d`( _padding_ , _value_
)[[source]](_modules/torch/nn/modules/padding.html#ConstantPad1d)

    

具有恒定值垫输入张量的界限。

For N-dimensional padding, use
[`torch.nn.functional.pad()`](nn.functional.html#torch.nn.functional.pad
"torch.nn.functional.pad").

Parameters

    

**填充** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in
Python v3.7\)") _，_ [ _元组_
](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")）
- 填充的大小。如果是 INT ，使用在两个边界相同的填充。如果2- 元组，使用（ padding_left  \ {文本填充\ _Left}
padding_left  ， padding_right  \文本{填充\ _right}  padding_right  ）

Shape:

    

  * Input: (N,C,Win)(N, C, W_{in})(N,C,Win​)

  * Output: (N,C,Wout)(N, C, W_{out})(N,C,Wout​) where

Wout=Win+padding_left+padding_rightW_{out} = W_{in} + \text{padding\\_left} +
\text{padding\\_right}Wout​=Win​+padding_left+padding_right

Examples:

    
    
    >>> m = nn.ConstantPad1d(2, 3.5)
    >>> input = torch.randn(1, 2, 4)
    >>> input
    tensor([[[-1.0491, -0.7152, -0.0749,  0.8530],
             [-1.3287,  1.8966,  0.1466, -0.2771]]])
    >>> m(input)
    tensor([[[ 3.5000,  3.5000, -1.0491, -0.7152, -0.0749,  0.8530,  3.5000,
               3.5000],
             [ 3.5000,  3.5000, -1.3287,  1.8966,  0.1466, -0.2771,  3.5000,
               3.5000]]])
    >>> m = nn.ConstantPad1d(2, 3.5)
    >>> input = torch.randn(1, 2, 3)
    >>> input
    tensor([[[ 1.6616,  1.4523, -1.1255],
             [-3.6372,  0.1182, -1.8652]]])
    >>> m(input)
    tensor([[[ 3.5000,  3.5000,  1.6616,  1.4523, -1.1255,  3.5000,  3.5000],
             [ 3.5000,  3.5000, -3.6372,  0.1182, -1.8652,  3.5000,  3.5000]]])
    >>> # using different paddings for different sides
    >>> m = nn.ConstantPad1d((3, 1), 3.5)
    >>> m(input)
    tensor([[[ 3.5000,  3.5000,  3.5000,  1.6616,  1.4523, -1.1255,  3.5000],
             [ 3.5000,  3.5000,  3.5000, -3.6372,  0.1182, -1.8652,  3.5000]]])
    

###  ConstantPad2d 

_class_`torch.nn.``ConstantPad2d`( _padding_ , _value_
)[[source]](_modules/torch/nn/modules/padding.html#ConstantPad2d)

    

Pads the input tensor boundaries with a constant value.

For N-dimensional padding, use
[`torch.nn.functional.pad()`](nn.functional.html#torch.nn.functional.pad
"torch.nn.functional.pad").

Parameters

    

**padding** ([ _int_](https://docs.python.org/3/library/functions.html#int
"\(in Python v3.7\)") _,_[
_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python
v3.7\)")) – the size of the padding. If is int, uses the same padding in all
boundaries. If a 4-tuple, uses (padding_left\text{padding\\_left}padding_left
, padding_right\text{padding\\_right}padding_right ,
padding_top\text{padding\\_top}padding_top ,
padding_bottom\text{padding\\_bottom}padding_bottom )

Shape:

    

  * Input: (N,C,Hin,Win)(N, C, H_{in}, W_{in})(N,C,Hin​,Win​)

  * Output: (N,C,Hout,Wout)(N, C, H_{out}, W_{out})(N,C,Hout​,Wout​) where

Hout=Hin+padding_top+padding_bottomH_{out} = H_{in} + \text{padding\\_top} +
\text{padding\\_bottom}Hout​=Hin​+padding_top+padding_bottom

Wout=Win+padding_left+padding_rightW_{out} = W_{in} + \text{padding\\_left} +
\text{padding\\_right}Wout​=Win​+padding_left+padding_right

Examples:

    
    
    >>> m = nn.ConstantPad2d(2, 3.5)
    >>> input = torch.randn(1, 2, 2)
    >>> input
    tensor([[[ 1.6585,  0.4320],
             [-0.8701, -0.4649]]])
    >>> m(input)
    tensor([[[ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
             [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
             [ 3.5000,  3.5000,  1.6585,  0.4320,  3.5000,  3.5000],
             [ 3.5000,  3.5000, -0.8701, -0.4649,  3.5000,  3.5000],
             [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
             [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000]]])
    >>> # using different paddings for different sides
    >>> m = nn.ConstantPad2d((3, 0, 2, 1), 3.5)
    >>> m(input)
    tensor([[[ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
             [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
             [ 3.5000,  3.5000,  3.5000,  1.6585,  0.4320],
             [ 3.5000,  3.5000,  3.5000, -0.8701, -0.4649],
             [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000]]])
    

###  ConstantPad3d 

_class_`torch.nn.``ConstantPad3d`( _padding_ , _value_
)[[source]](_modules/torch/nn/modules/padding.html#ConstantPad3d)

    

Pads the input tensor boundaries with a constant value.

For N-dimensional padding, use
[`torch.nn.functional.pad()`](nn.functional.html#torch.nn.functional.pad
"torch.nn.functional.pad").

Parameters

    

**padding** ([ _int_](https://docs.python.org/3/library/functions.html#int
"\(in Python v3.7\)") _,_[
_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python
v3.7\)")) – the size of the padding. If is int, uses the same padding in all
boundaries. If a 6-tuple, uses (padding_left\text{padding\\_left}padding_left
, padding_right\text{padding\\_right}padding_right ,
padding_top\text{padding\\_top}padding_top ,
padding_bottom\text{padding\\_bottom}padding_bottom ,
padding_front\text{padding\\_front}padding_front ,
padding_back\text{padding\\_back}padding_back )

Shape:

    

  * Input: (N,C,Din,Hin,Win)(N, C, D_{in}, H_{in}, W_{in})(N,C,Din​,Hin​,Win​)

  * Output: (N,C,Dout,Hout,Wout)(N, C, D_{out}, H_{out}, W_{out})(N,C,Dout​,Hout​,Wout​) where

Dout=Din+padding_front+padding_backD_{out} = D_{in} + \text{padding\\_front} +
\text{padding\\_back}Dout​=Din​+padding_front+padding_back

Hout=Hin+padding_top+padding_bottomH_{out} = H_{in} + \text{padding\\_top} +
\text{padding\\_bottom}Hout​=Hin​+padding_top+padding_bottom

Wout=Win+padding_left+padding_rightW_{out} = W_{in} + \text{padding\\_left} +
\text{padding\\_right}Wout​=Win​+padding_left+padding_right

Examples:

    
    
    >>> m = nn.ConstantPad3d(3, 3.5)
    >>> input = torch.randn(16, 3, 10, 20, 30)
    >>> output = m(input)
    >>> # using different paddings for different sides
    >>> m = nn.ConstantPad3d((3, 3, 6, 6, 0, 1), 3.5)
    >>> output = m(input)
    

## 非线性激活（加权和，非线性）

###  ELU 

_class_`torch.nn.``ELU`( _alpha=1.0_ , _inplace=False_
)[[source]](_modules/torch/nn/modules/activation.html#ELU)

    

适用逐元素的功能：

ELU(x)=max⁡(0,x)+min⁡(0,α∗(exp⁡(x)−1))\text{ELU}(x) = \max(0,x) + \min(0,
\alpha * (\exp(x) - 1)) ELU(x)=max(0,x)+min(0,α∗(exp(x)−1))

Parameters

    

  * **阿尔法** \- 的 α \阿尔法 α 为ELU制剂值。默认值：1.0

  * **就地** \- 可以任选地执行操作就地。默认值：`假 `

Shape:

    

  * 输入： （ N  ， *  ） （N，*） （ N  ， *  ） 其中 * 的装置，任意数量的附加维度的

  * 输出： （ N  ， *  ） （N，*） （ N  ， *  ） ，相同形状的输入

![_images/ELU.png](_images/ELU.png)

Examples:

    
    
    >>> m = nn.ELU()
    >>> input = torch.randn(2)
    >>> output = m(input)
    

###  Hardshrink 

_class_`torch.nn.``Hardshrink`( _lambd=0.5_
)[[source]](_modules/torch/nn/modules/activation.html#Hardshrink)

    

适用硬收缩功能元素方面：

HardShrink(x)={x, if x>λx, if x<−λ0, otherwise \text{HardShrink}(x) =
\begin{cases} x, & \text{ if } x > \lambda \\\ x, & \text{ if } x < -\lambda
\\\ 0, & \text{ otherwise } \end{cases} HardShrink(x)=⎩⎪⎨⎪⎧​x,x,0,​ if x>λ if
x<−λ otherwise ​

Parameters

    

**lambd** \- 的 λ \拉姆达 λ 为Hardshrink制剂值。默认值：0.5

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where * means, any number of additional dimensions

  * Output: (N,∗)(N, *)(N,∗) , same shape as the input

![_images/Hardshrink.png](_images/Hardshrink.png)

Examples:

    
    
    >>> m = nn.Hardshrink()
    >>> input = torch.randn(2)
    >>> output = m(input)
    

###  Hardtanh 

_class_`torch.nn.``Hardtanh`( _min_val=-1.0_ , _max_val=1.0_ , _inplace=False_
, _min_value=None_ , _max_value=None_
)[[source]](_modules/torch/nn/modules/activation.html#Hardtanh)

    

应用HardTanh功能逐元素

HardTanh定义为：

HardTanh(x)={1 if x>1−1 if x<−1x otherwise \text{HardTanh}(x) = \begin{cases}
1 & \text{ if } x > 1 \\\ -1 & \text{ if } x < -1 \\\ x & \text{ otherwise }
\\\ \end{cases} HardTanh(x)=⎩⎪⎨⎪⎧​1−1x​ if x>1 if x<−1 otherwise ​

线性区域 [ 的范围内 -  1  ， 1  [1,1]  [ \-  1  ， 1  ]  可以用被调整`MIN_VAL`和`MAX_VAL
`。

Parameters

    

  * **MIN_VAL** \- 线性区域范围的最小值。默认值：-1

  * **MAX_VAL** \- 线性区域范围的最大值。默认值：1

  * **inplace** – can optionally do the operation in-place. Default: `False`

关键字参数`MIN_VALUE`和`MAX_VALUE`已经被弃用，取而代之的`MIN_VAL`和`MAX_VAL`。

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where * means, any number of additional dimensions

  * Output: (N,∗)(N, *)(N,∗) , same shape as the input

![_images/Hardtanh.png](_images/Hardtanh.png)

Examples:

    
    
    >>> m = nn.Hardtanh(-2, 2)
    >>> input = torch.randn(2)
    >>> output = m(input)
    

###  LeakyReLU 

_class_`torch.nn.``LeakyReLU`( _negative_slope=0.01_ , _inplace=False_
)[[source]](_modules/torch/nn/modules/activation.html#LeakyReLU)

    

Applies the element-wise function:

LeakyReLU(x)=max⁡(0,x)+negative_slope∗min⁡(0,x)\text{LeakyReLU}(x) = \max(0,
x) + \text{negative\\_slope} * \min(0, x)
LeakyReLU(x)=max(0,x)+negative_slope∗min(0,x)

要么

LeakyRELU(x)={x, if x≥0negative_slope×x, otherwise \text{LeakyRELU}(x) =
\begin{cases} x, & \text{ if } x \geq 0 \\\ \text{negative\\_slope} \times x,
& \text{ otherwise } \end{cases} LeakyRELU(x)={x,negative_slope×x,​ if x≥0
otherwise ​

Parameters

    

  * **negative_slope** \- 控制负斜率的角度。默认值：1E-2

  * **inplace** – can optionally do the operation in-place. Default: `False`

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where * means, any number of additional dimensions

  * Output: (N,∗)(N, *)(N,∗) , same shape as the input

![_images/LeakyReLU.png](_images/LeakyReLU.png)

Examples:

    
    
    >>> m = nn.LeakyReLU(0.1)
    >>> input = torch.randn(2)
    >>> output = m(input)
    

###  LogSigmoid 

_class_`torch.nn.``LogSigmoid`[[source]](_modules/torch/nn/modules/activation.html#LogSigmoid)

    

Applies the element-wise function:

LogSigmoid(x)=log⁡(11+exp⁡(−x))\text{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1
+ \exp(-x)}\right) LogSigmoid(x)=log(1+exp(−x)1​)

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where * means, any number of additional dimensions

  * Output: (N,∗)(N, *)(N,∗) , same shape as the input

![_images/LogSigmoid.png](_images/LogSigmoid.png)

Examples:

    
    
    >>> m = nn.LogSigmoid()
    >>> input = torch.randn(2)
    >>> output = m(input)
    

###  MultiheadAttention 

_class_`torch.nn.``MultiheadAttention`( _embed_dim_ , _num_heads_ ,
_dropout=0.0_ , _bias=True_ , _add_bias_kv=False_ , _add_zero_attn=False_ ,
_kdim=None_ , _vdim=None_
)[[source]](_modules/torch/nn/modules/activation.html#MultiheadAttention)

    

允许模型共同出席，从不同的表示子空间的信息。见参考文献：注意是所有你需要

MultiHead(Q,K,V)=Concat(head1,…,headh)WOwhereheadi=Attention(QWiQ,KWiK,VWiV)\text{MultiHead}(Q,
K, V) = \text{Concat}(head_1,\dots,head_h)W^O \text{where} head_i =
\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
MultiHead(Q,K,V)=Concat(head1​,…,headh​)WOwhereheadi​=Attention(QWiQ​,KWiK​,VWiV​)

Parameters

    

  * **embed_dim** \- 模型的总尺寸。

  * **num_heads** \- 平行注意头。

  * **滤除** \- 关于attn_output_weights一个漏失层。默认值：0.0。

  * **偏压** \- 加偏压作为模块参数。默认值：true。

  * **add_bias_kv** \- 在昏暗= 0添加偏置的键和值的序列。

  * **add_zero_attn** \- 在昏暗= 1添加新的批次零到的键和值的序列。

  * **kdim** \- 的关键特征总数。默认值：无。

  * **VDIM** \- 的关键特征总数。默认值：无。

  * [HTG0注意 - 如果kdim和VDIM都没有，它们将被设置为embed_dim这样

  * **键，值具有相同数目的特征。** （ _查询_ _，_ ） - 

Examples:

    
    
    >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
    >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    

`forward`( _query_ , _key_ , _value_ , _key_padding_mask=None_ ,
_need_weights=True_ , _attn_mask=None_
)[[source]](_modules/torch/nn/modules/activation.html#MultiheadAttention.forward)

    

Parameters

    

  * **键，值** （ _查询_ _，[5 HTG） - 映射的查询和一组键 - 值对到输出。请参阅“注意是所有你需要”更多的细节。_

  * **key_padding_mask** \- 如果提供的话，在键配置的填充元件将被受瞩目忽略。这是一个二进制掩码。当值为True，关注层上的相应值将充满-INF。

  * **need_weights** \- 输出attn_output_weights。

  * **attn_mask** \- 掩模，防止注意某些位置。这是一种添加剂掩模（即，值将被添加到关注层）。

Shape:

    

  * 输入：

  * 查询： （ L  ， N  ， E  ） （L，N，E） （ L  ， N  ， E  ） 其中，L是所述靶序列的长度，N是批量大小，E是嵌入维数。

  * 键： （ S  ， N  ， E  ） （S，N，E） （ S  ， N  ， E  ） ，其中S是源序列长度，N是批量大小，E是嵌入维数。

  * 值： （ S  ， N  ， E  ） （S，N，E） （ S  ， N  ， E  ） 其中，S是源序列长度，N是批量大小，E是嵌入维数。

  * key_padding_mask： （ N  ， S  ） （N，S） （ N  ， S  ） ，ByteTensor，其中N是批量大小，S是源序列长度。

  * attn_mask： （ L  ， S  ） （L，S） （ L  ， S  ） 其中，L是所述靶序列的长度，S是源序列长度。

  * 输出：

  * attn_output： （ L  ， N  ， E  ） （L，N，E） （ L  ， N  ， E  ） 其中，L是所述靶序列的长度，N是批量大小，E是嵌入维数。

  * attn_output_weights： （ N  ， L  ， S  ） （N，L，S） （ N  ， L  ， S  ） 其中，N是批量大小，L是所述靶序列的长度，S是源序列长度。

###  PReLU 

_class_`torch.nn.``PReLU`( _num_parameters=1_ , _init=0.25_
)[[source]](_modules/torch/nn/modules/activation.html#PReLU)

    

Applies the element-wise function:

PReLU(x)=max⁡(0,x)+a∗min⁡(0,x)\text{PReLU}(x) = \max(0,x) + a * \min(0,x)
PReLU(x)=max(0,x)+a∗min(0,x)

or

PReLU(x)={x, if x≥0ax, otherwise \text{PReLU}(x) = \begin{cases} x, & \text{
if } x \geq 0 \\\ ax, & \text{ otherwise } \end{cases} PReLU(x)={x,ax,​ if x≥0
otherwise ​

此处 一 一 一 是一个可以学习的参数。当不带参数调用， nn.PReLU（）使用单个参数 一 A  一 所有输入通道。如果调用
nn.PReLU（nChannels），单独的 一 一 一 用于每个输入通道。

Note

权衰减不应该被用来当学习 一 一 A  获得良好的性能。

Note

昏暗的通道输入的第二暗淡。当输入具有变暗& LT ; 2，那么就没有信道暗淡和通道= 1的数。

Parameters

    

  * **num_parameters** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的 数目的 一 一 学习。虽然采用int作为输入，仅存在两个值是合法的：1，或通道中的输入的数目。默认值：1

  * **INIT** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 的 一[初始值HTG13]  一 一 [ HTG29。默认值：0.25

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where * means, any number of additional dimensions

  * Output: (N,∗)(N, *)(N,∗) , same shape as the input

Variables

    

**〜PReLU.weight** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） -
形状的可学习权重（`num_parameters`）。

![_images/PReLU.png](_images/PReLU.png)

Examples:

    
    
    >>> m = nn.PReLU()
    >>> input = torch.randn(2)
    >>> output = m(input)
    

###  RELU 

_class_`torch.nn.``ReLU`( _inplace=False_
)[[source]](_modules/torch/nn/modules/activation.html#ReLU)

    

施加整流的线性单元函数逐元素：

RELU  （ × ） =  MAX  ⁡ （ 0  ， × ） \文本{RELU}（X）= \ MAX（0，x）的 RELU  （ × ） =  MAX
（ 0  ， X  ）

Parameters

    

**inplace** – can optionally do the operation in-place. Default: `False`

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where * means, any number of additional dimensions

  * Output: (N,∗)(N, *)(N,∗) , same shape as the input

![_images/ReLU.png](_images/ReLU.png)

Examples:

    
    
      >>> m = nn.ReLU()
      >>> input = torch.randn(2)
      >>> output = m(input)
    
    
    An implementation of CReLU - https://arxiv.org/abs/1603.05201
    
      >>> m = nn.ReLU()
      >>> input = torch.randn(2).unsqueeze(0)
      >>> output = torch.cat((m(input),m(-input)))
    

###  ReLU6 

_class_`torch.nn.``ReLU6`( _inplace=False_
)[[source]](_modules/torch/nn/modules/activation.html#ReLU6)

    

Applies the element-wise function:

ReLU6(x)=min⁡(max⁡(0,x),6)\text{ReLU6}(x) = \min(\max(0,x), 6)
ReLU6(x)=min(max(0,x),6)

Parameters

    

**inplace** – can optionally do the operation in-place. Default: `False`

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where * means, any number of additional dimensions

  * Output: (N,∗)(N, *)(N,∗) , same shape as the input

![_images/ReLU6.png](_images/ReLU6.png)

Examples:

    
    
    >>> m = nn.ReLU6()
    >>> input = torch.randn(2)
    >>> output = m(input)
    

###  RReLU 

_class_`torch.nn.``RReLU`( _lower=0.125_ , _upper=0.3333333333333333_ ,
_inplace=False_ )[[source]](_modules/torch/nn/modules/activation.html#RReLU)

    

应用随机漏泄整流衬垫单元的功能，逐元素，如在论文中描述：

[整流的激活的实证评价卷积网络](https://arxiv.org/abs/1505.00853)。

该函数被定义为：

RReLU(x)={xif x≥0ax otherwise \text{RReLU}(x) = \begin{cases} x & \text{if } x
\geq 0 \\\ ax & \text{ otherwise } \end{cases} RReLU(x)={xax​if x≥0 otherwise
​

其中 一 一 一 被随机地从取样的均匀分布 U  （ 下 ， 上 ） \ mathcal 【U}（\文本{低}，\文本{上部}） U  （ 下 ， 上 ）
。

> 请参阅：[ https://arxiv.org/pdf/1505.00853.pdf
](https://arxiv.org/pdf/1505.00853.pdf)

Parameters

    

  * **下** \- 下界的均匀分布的。默认值： 1  8  \压裂{1 } {8}  8  1 

  * **上** \- 上限的均匀分布的。默认值： 1  3  \压裂{1 } {3}  3  1 

  * **inplace** – can optionally do the operation in-place. Default: `False`

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where * means, any number of additional dimensions

  * Output: (N,∗)(N, *)(N,∗) , same shape as the input

Examples:

    
    
    >>> m = nn.RReLU(0.1, 0.3)
    >>> input = torch.randn(2)
    >>> output = m(input)
    

### 九色鹿

_class_`torch.nn.``SELU`( _inplace=False_
)[[source]](_modules/torch/nn/modules/activation.html#SELU)

    

应用元素方面，如：

SELU(x)=scale∗(max⁡(0,x)+min⁡(0,α∗(exp⁡(x)−1)))\text{SELU}(x) = \text{scale} *
(\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))
SELU(x)=scale∗(max(0,x)+min(0,α∗(exp(x)−1)))

与 α =  1.6732632423543772848170429916717  \阿尔法=
1.6732632423543772848170429916717  α =  1  。  6  7  3  2  6  3  2  4  2  3  5
4  3  7  7  2  8  4  8  1  7  0  4  2  9  9  1  6  7  1  7  和 规模 =
1.0507009873554804934193349852946  \文本{规模} = 1.0507009873554804934193349852946
规模 =  1  。  0  5  0  7  0  0  9  8  7  3  5  5  4  8  0  4  9  3  4  1  9  3
3  4  9  8  5  2  9  4  6  。

更多详细信息可在本文中找到[自正火神经网络[HTG1。](https://arxiv.org/abs/1706.02515)

Parameters

    

**就地** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in
Python v3.7\)") _，_ _可选_ ） - 可任选地执行操作就地。默认值：`假 `

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where * means, any number of additional dimensions

  * Output: (N,∗)(N, *)(N,∗) , same shape as the input

![_images/SELU.png](_images/SELU.png)

Examples:

    
    
    >>> m = nn.SELU()
    >>> input = torch.randn(2)
    >>> output = m(input)
    

###  CELU 

_class_`torch.nn.``CELU`( _alpha=1.0_ , _inplace=False_
)[[source]](_modules/torch/nn/modules/activation.html#CELU)

    

Applies the element-wise function:

CELU(x)=max⁡(0,x)+min⁡(0,α∗(exp⁡(x/α)−1))\text{CELU}(x) = \max(0,x) + \min(0,
\alpha * (\exp(x/\alpha) - 1)) CELU(x)=max(0,x)+min(0,α∗(exp(x/α)−1))

更多细节可以在文献[的连续可微指数直线单元](https://arxiv.org/abs/1704.07483)中找到。

Parameters

    

  * **阿尔法** \- 的 α \阿尔法 α 为CELU制剂值。默认值：1.0

  * **inplace** – can optionally do the operation in-place. Default: `False`

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where * means, any number of additional dimensions

  * Output: (N,∗)(N, *)(N,∗) , same shape as the input

![_images/CELU.png](_images/CELU.png)

Examples:

    
    
    >>> m = nn.CELU()
    >>> input = torch.randn(2)
    >>> output = m(input)
    

### 乙状结肠

_class_`torch.nn.``Sigmoid`[[source]](_modules/torch/nn/modules/activation.html#Sigmoid)

    

Applies the element-wise function:

Sigmoid(x)=11+exp⁡(−x)\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}
Sigmoid(x)=1+exp(−x)1​

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where * means, any number of additional dimensions

  * Output: (N,∗)(N, *)(N,∗) , same shape as the input

![_images/Sigmoid.png](_images/Sigmoid.png)

Examples:

    
    
    >>> m = nn.Sigmoid()
    >>> input = torch.randn(2)
    >>> output = m(input)
    

###  Softplus 

_class_`torch.nn.``Softplus`( _beta=1_ , _threshold=20_
)[[source]](_modules/torch/nn/modules/activation.html#Softplus)

    

Applies the element-wise function:

Softplus(x)=1β∗log⁡(1+exp⁡(β∗x))\text{Softplus}(x) = \frac{1}{\beta} * \log(1
+ \exp(\beta * x)) Softplus(x)=β1​∗log(1+exp(β∗x))

SoftPlus是光滑逼近RELU功能，并且可以用于约束的机器的输出以始终是正的。

对于数值稳定性的执行恢复到线性函数对于高于某个值的输入。

Parameters

    

  * **的β** \- 的 β \的β β 为Softplus制剂值。默认值：1

  * **阈** \- 高于此值恢复到一个线性函数。默认值：20

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where * means, any number of additional dimensions

  * Output: (N,∗)(N, *)(N,∗) , same shape as the input

![_images/Softplus.png](_images/Softplus.png)

Examples:

    
    
    >>> m = nn.Softplus()
    >>> input = torch.randn(2)
    >>> output = m(input)
    

###  Softshrink 

_class_`torch.nn.``Softshrink`( _lambd=0.5_
)[[source]](_modules/torch/nn/modules/activation.html#Softshrink)

    

应用软收缩功能的elementwise：

SoftShrinkage(x)={x−λ, if x>λx+λ, if x<−λ0, otherwise \text{SoftShrinkage}(x)
= \begin{cases} x - \lambda, & \text{ if } x > \lambda \\\ x + \lambda, &
\text{ if } x < -\lambda \\\ 0, & \text{ otherwise } \end{cases}
SoftShrinkage(x)=⎩⎪⎨⎪⎧​x−λ,x+λ,0,​ if x>λ if x<−λ otherwise ​

Parameters

    

**lambd** \- 的 λ \拉姆达 λ 为Softshrink制剂值。默认值：0.5

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where * means, any number of additional dimensions

  * Output: (N,∗)(N, *)(N,∗) , same shape as the input

![_images/Softshrink.png](_images/Softshrink.png)

Examples:

    
    
    >>> m = nn.Softshrink()
    >>> input = torch.randn(2)
    >>> output = m(input)
    

###  Softsign 

_class_`torch.nn.``Softsign`[[source]](_modules/torch/nn/modules/activation.html#Softsign)

    

Applies the element-wise function:

SoftSign(x)=x1+∣x∣\text{SoftSign}(x) = \frac{x}{ 1 + |x|} SoftSign(x)=1+∣x∣x​

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where * means, any number of additional dimensions

  * Output: (N,∗)(N, *)(N,∗) , same shape as the input

![_images/Softsign.png](_images/Softsign.png)

Examples:

    
    
    >>> m = nn.Softsign()
    >>> input = torch.randn(2)
    >>> output = m(input)
    

### 双曲正切

_class_`torch.nn.``Tanh`[[source]](_modules/torch/nn/modules/activation.html#Tanh)

    

Applies the element-wise function:

Tanh(x)=tanh⁡(x)=ex−e−xex+e−x\text{Tanh}(x) = \tanh(x) = \frac{e^x - e^{-x}}
{e^x + e^{-x}} Tanh(x)=tanh(x)=ex+e−xex−e−x​

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where * means, any number of additional dimensions

  * Output: (N,∗)(N, *)(N,∗) , same shape as the input

![_images/Tanh.png](_images/Tanh.png)

Examples:

    
    
    >>> m = nn.Tanh()
    >>> input = torch.randn(2)
    >>> output = m(input)
    

###  Tanhshrink 

_class_`torch.nn.``Tanhshrink`[[source]](_modules/torch/nn/modules/activation.html#Tanhshrink)

    

Applies the element-wise function:

Tanhshrink(x)=x−Tanh(x)\text{Tanhshrink}(x) = x - \text{Tanh}(x)
Tanhshrink(x)=x−Tanh(x)

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where * means, any number of additional dimensions

  * Output: (N,∗)(N, *)(N,∗) , same shape as the input

![_images/Tanhshrink.png](_images/Tanhshrink.png)

Examples:

    
    
    >>> m = nn.Tanhshrink()
    >>> input = torch.randn(2)
    >>> output = m(input)
    

### 阈值

_class_`torch.nn.``Threshold`( _threshold_ , _value_ , _inplace=False_
)[[source]](_modules/torch/nn/modules/activation.html#Threshold)

    

阈值输入张量的每个元素。

阈值定义为：

y={x, if x>thresholdvalue, otherwise y = \begin{cases} x, &\text{ if } x >
\text{threshold} \\\ \text{value}, &\text{ otherwise } \end{cases}
y={x,value,​ if x>threshold otherwise ​

Parameters

    

  * **阈** \- 在该值的阈值

  * **值** \- 该值与替换

  * **inplace** – can optionally do the operation in-place. Default: `False`

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where * means, any number of additional dimensions

  * Output: (N,∗)(N, *)(N,∗) , same shape as the input

Examples:

    
    
    >>> m = nn.Threshold(0.1, 20)
    >>> input = torch.randn(2)
    >>> output = m(input)
    

## 非线性激活（其他）

###  Softmin 

_class_`torch.nn.``Softmin`( _dim=None_
)[[source]](_modules/torch/nn/modules/activation.html#Softmin)

    

施加Softmin功能的n维输入张量重新缩放它们，使得所述n维输出张量谎言的范围在 [0,1] 和总和为1的元素。

Softmin定义为：

Softmin(xi)=exp⁡(−xi)∑jexp⁡(−xj)\text{Softmin}(x_{i}) =
\frac{\exp(-x_i)}{\sum_j \exp(-x_j)} Softmin(xi​)=∑j​exp(−xj​)exp(−xi​)​

Shape:

    

  * 输入： （ *  ） （*）  （ *  ） 其中 * 手段，任意数量的附加维度的

  * 输出： （ *  ） （*）  （ *  ） ，相同形状的输入

Parameters

    

**暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in
Python v3.7\)")） - 沿其Softmin将被计算的尺寸（因此沿暗淡每片将总结为1）。

Returns

    

相同的尺寸和形状作为输入的范围内的张量，其值[0，1]

Examples:

    
    
    >>> m = nn.Softmin()
    >>> input = torch.randn(2, 3)
    >>> output = m(input)
    

### 使用SoftMax 

_class_`torch.nn.``Softmax`( _dim=None_
)[[source]](_modules/torch/nn/modules/activation.html#Softmax)

    

应用使用SoftMax功能的n维输入张量重新缩放它们，使得所述n维输出张量谎言在范围[0,1]和总和为1的元素。

使用SoftMax定义为：

Softmax(xi)=exp⁡(xi)∑jexp⁡(xj)\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j
\exp(x_j)} Softmax(xi​)=∑j​exp(xj​)exp(xi​)​

Shape:

    

  * Input: (∗)(*)(∗) where * means, any number of additional dimensions

  * Output: (∗)(*)(∗) , same shape as the input

Returns

    

相同的尺寸和形状作为输入的与值的范围内的张量[0，1]

Parameters

    

**暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in
Python v3.7\)")） - 沿其使用SoftMax将被计算的尺寸（因此沿暗淡每片将总结为1）。

Note

此模块不与NLLLoss，其预计使用SoftMax和自身之间要计算日志直接工作。使用 LogSoftmax 代替（它的速度更快，具有更好的数值属性）。

Examples:

    
    
    >>> m = nn.Softmax(dim=1)
    >>> input = torch.randn(2, 3)
    >>> output = m(input)
    

###  Softmax2d 

_class_`torch.nn.``Softmax2d`[[source]](_modules/torch/nn/modules/activation.html#Softmax2d)

    

适用使用SoftMax在功能，每个空间位置。

当给定的`的图像频道 × 高度 × 宽度 `，它将应用使用SoftMax 到每个位置 （ C  H  一 n的 n的 E  L  S  ， H  i的 ，
W  [HTG51：J  ） （频道，h_i，w_j） （ ç  H  一 n的 n的 E  L  S  ， H  i的 ， W  [HTG131：J  ）

Shape:

    

  * 输入： （ N  ， C  ， H  ， W  ） （N，C，H，W） （ N  ， C  ， H  ， W  ） 

  * 输出： （ N  ， C  ， H  ， W  ） （N，C，H，W） （ N  ， C  ， H  ， W  ）  （相同形状的输入）

Returns

    

a Tensor of the same dimension and shape as the input with values in the range
[0, 1]

Examples:

    
    
    >>> m = nn.Softmax2d()
    >>> # you softmax over the 2nd dimension
    >>> input = torch.randn(2, 3, 12, 13)
    >>> output = m(input)
    

###  LogSoftmax 

_class_`torch.nn.``LogSoftmax`( _dim=None_
)[[source]](_modules/torch/nn/modules/activation.html#LogSoftmax)

    

应用 日志 ⁡ （ 使用SoftMax  （ × ） ） \日志（\文本{使用SoftMax}（X）） LO  G  （ 使用SoftMax  （ × ）
） 功能的n维输入张量。所述LogSoftmax制剂可以被简化为：

LogSoftmax(xi)=log⁡(exp⁡(xi)∑jexp⁡(xj))\text{LogSoftmax}(x_{i}) =
\log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)
LogSoftmax(xi​)=log(∑j​exp(xj​)exp(xi​)​)

Shape:

    

  * Input: (∗)(*)(∗) where * means, any number of additional dimensions

  * Output: (∗)(*)(∗) , same shape as the input

Parameters

    

**暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in
Python v3.7\)")） - 沿其LogSoftmax将被计算的尺寸。

Returns

    

相同的尺寸和形状作为输入的与值的范围内的张量[-INF，0）

Examples:

    
    
    >>> m = nn.LogSoftmax()
    >>> input = torch.randn(2, 3)
    >>> output = m(input)
    

###  AdaptiveLogSoftmaxWithLoss 

_class_`torch.nn.``AdaptiveLogSoftmaxWithLoss`( _in_features_ , _n_classes_ ,
_cutoffs_ , _div_value=4.0_ , _head_bias=False_
)[[source]](_modules/torch/nn/modules/adaptive.html#AdaptiveLogSoftmaxWithLoss)

    

作为用于GPU的由爱德华墓，阿芒Joulin穆斯塔法西塞，大卫Grangier和埃尔韦Jégou在[高效SOFTMAX近似描述高效SOFTMAX近似。](https://arxiv.org/abs/1609.04309)

自适应SOFTMAX是与产量大空间的培训模式近似的策略。当标签分布极不平衡，例如，在自然语言建模，这里所说的频率分布大致如下[齐普夫定律](https://en.wikipedia.org/wiki/Zipf%27s_law)这是最有效的。

自适应SOFTMAX划分标签分成几个集群，根据自己的频率。这些集群可以包含不同数量的每一个目标。另外，将含有较不频繁的标签的集群分配低维的嵌入到这些标签，从而加快了计算。对于每个minibatch，只为其中至少一个目标是本簇进行评估。

这个想法是，这是经常访问的集群（如第一个，包含最常见的标签），也应该是廉价的计算 - 也就是，包含少量分配的标签。

我们强烈建议您考虑看看原来的文件的更多细节。

  * `截断值 `应在增加顺序排序整数的有序序列。它控制集群的数量和目标分割成集群。例如设置`截断值 =  [10， 100， 1000]`表示第一 10 目标将被分配到所述自适应SOFTMAX的 '头部'，目标 11，12，...，100 将被分配到所述第一群集，和目标 101 ，102，...，1000 将被分配到所述第二群集，而目标 1001，1002，...，n_classes - 1 将被分配到最后，第三集群。

  * `div_value`被用来计算每个附加簇的大小，它被给定为 ⌊  i的 n的 _  F  E  一 T  U  R  E  S  d  i的 [HTG42】v  _  [HTG46】v 一 L  U  E  i的 d  × ⌋ \左\ lfloor \压裂{在\ _features} {DIV \ _value ^ {IDX}} \右\ rfloor  ⌊ d  i的 [HTG101】V  _  [HTG105】V  一 L  U  E  i的 d  × i的 n的 _  F  E  一 T  U  R  E  S  [HTG19 0]  ⌋ ，其中 i的 d  × IDX  i的 d  × 是群集索引（与集群以较少具有较大索引频繁字和索引起始从 1  1  1  ）。

  * `head_bias`如果设置为True，增加了一个偏项到自适应SOFTMAX的“头”。有关详细信息，请参阅纸张。设置为False正式执行。

Warning

作为输入传递给此模块的标签应被分类accoridng自己的频率。这意味着，最频繁的标签应该由指数 0 ，和至少频繁标签应该由指数
n_classes来表示来表示 - 1 。

Note

这个模块返回`NamedTuple`与`输出 `和`损失 `字段。详情请参见更多文档。

Note

为了计算对数概率的所有类，则`log_prob`可以使用的方法。

Parameters

    

  * 在输入张量的特征数 - **in_features** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")）

  * 在数据集的类的数量 - **n_classes** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")）

  * **截断值** （ _序号_ ） - 用于分配的目标，以他们的水桶保险丝

  * **div_value** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 用来作为指数来计算集群的大小值。默认值：4.0

  * **head_bias** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，增加了一个偏项到自适应SOFTMAX的“头”。默认值：`假 `

Returns

    

  * **输出** 是大小的张量`N  [HTG5含有计算目标数概率对于每个实施例`

  * **损失** 是表示所计算的负对数似然损耗的标量

Return type

    

`NamedTuple`与`输出 `和`损失 `字段

Shape:

    

  * 输入： （ N  ， i的 n的 _  F  E  一 T  U  [R  E  S  ） （N，在\ _features） （ N  ， i的 n的 _  F  E  一 T  U  R  E  S  ）

  * 目标： （ N  ） （N）  （ N  ） 其中，每个值满足 0  & LT ;  =  T  一 R  克 E  T  [ i的 & LT ;  =  n的 _  C  L  一 S  S  E  S  0 & LT ; =目标[I] & LT ; = N \ _classes  0  [H TG97]  & LT ;  =  T  一 R  克 E  T  [  i的 & LT ;  =  n的 _  C  升 一 S  S  E  S 

  * 输出1： （ N  ） （N）  （ N  ）

  * OUTPUT2：`[HTG1标量 `

`log_prob`( _input_
)[[source]](_modules/torch/nn/modules/adaptive.html#AdaptiveLogSoftmaxWithLoss.log_prob)

    

计算对数概率对于所有 n的 _  C  L  一 S  S  E  S  n的\ _classes  n的 _  C  L  一个 S  S  E  S

Parameters

    

**输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 的例子的minibatch

Returns

    

对数概率为每个类 C  C  C  在范围 0  & LT ;  =  C  & LT ;  =  n的 _  C  L  一 S  S  E  S  0
& LT ; = C & LT ; = N \ _classes  0  & LT ;  =  C  & LT ;  =  n的 _  C  L  一 S
S  E  S  ，其中 n的 _  C  升 一 S  S  E  S  n的\ _classes  n的 _  C  L  一 S  S  E  S
的参数传递为`AdaptiveLogSoftmaxWithLoss [HTG1 86] `构造。

Shape:

    

  * 输入： （ N  ， i的 n的 _  F  E  一 T  U  [R  E  S  ） （N，在\ _features） （ N  ， i的 n的 _  F  E  一 T  U  R  E  S  ）

  * 输出： （ N  ， n的 _  C  L  一 S  S  E  S  ） （N，N- \ _classes） （ N  ， n的 _  C  升 一 S  S  E  S  ）

`predict`( _input_
)[[source]](_modules/torch/nn/modules/adaptive.html#AdaptiveLogSoftmaxWithLoss.predict)

    

这等同于 self.log_pob（输入）.argmax（暗= 1），但是在某些情况下更为有效。

Parameters

    

**input** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – a
minibatch of examples

Returns

    

用对于每个实施例的概率最高的类

Return type

    

输出（[张量](tensors.html#torch.Tensor "torch.Tensor")）

Shape:

    

  * Input: (N,in_features)(N, in\\_features)(N,in_features)

  * 输出： （ N  ） （N）  （ N  ）

## 归一化的层

###  BatchNorm1d 

_class_`torch.nn.``BatchNorm1d`( _num_features_ , _eps=1e-05_ , _momentum=0.1_
, _affine=True_ , _track_running_stats=True_
)[[source]](_modules/torch/nn/modules/batchnorm.html#BatchNorm1d)

    

适用批标准化在2D或如在文献[批标准化描述3D输入（带有可选的附加的信道尺寸的小批量的1D输入）：通过减少内部协变量移位](https://arxiv.org/abs/1502.03167)加速深网络训练。

y=x−E[x]Var[x]+ϵ∗γ+βy = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] +
\epsilon}} * \gamma + \betay=Var[x]+ϵ​x−E[x]​∗γ+β

的平均值和标准偏差是每个尺寸来计算放置在迷你批次和 γ \伽马 γ 和 β \的β β 是大小的可学习的参数矢量 C （其中 C
是输入大小）。默认情况下，γ  \伽马 的 的元素 γ 被设置为1和 的元素 β \的β β 被设置为0。

此外，默认情况，在培训过程中这层继续运行其计算的均值和方差，然后再评估期间用于标准化的估计。正在运行的估计是保持了默认的`势头 `0.1。

如果`track_running_stats`设置为`假 `，这一层则不会继续运行的估计，和批量统计过程中的评估时间，而不是作为好。

Note

此`动量 `参数是从一个在优化器中使用的类和动量的常规概念不同。在数学上，这里运行统计数据的更新规则为 × ^  新 =  （ 1  \-  动量 ）
× × ^  \+  动量 × × T  \帽子{X} _ \文本{新} =（1 - \文本{动量}）\倍\帽子{X} + \文本{动量} \倍X_T  ×
^  新 =  （ 1  \-  动量 ） × × ^  \+  动量 × × T  ，其中 × ^  \帽子{X}  × ^  ​​  是估计的统计量和
X  T  X_T  × T  是新的观测值。

因为批标准化是在 C 维完成的，在（N，L）切片计算统计数据，这是共同的术语来调用这个时空批标准化。

Parameters

    

  * **NUM_FEATURES** \-  C  C  C  从大小的预期输入 （ N  ， C  ， L  ） （N，C，L） （ N  ， C  ， L  ） 或 L  L  L  从的输入大小 （ N  ， L  ） （N，L） （ N  ， L  ）

  * **EPS** \- 的值添加到分母数值稳定性。默认值：1E-5

  * **动量** \- 用于running_mean和running_var计算的值。可以被设置为`无 [HTG5用于累积移动平均（即简单平均）。默认值：0.1`

  * **仿射** \- 一个布尔值，当设置为`真 `，该模块具有可学习的仿射参数。默认值：`真 `

  * **track_running_stats** \- 当设置为`真 `，此模块跟踪的运行均值和方差，和一个布尔值，当设置为`假 `，该模块不跟踪这样的统计并始终使用在训练和eval模式批次的统计数据。默认值：`真 `

Shape:

    

  * 输入： （ N  ， C  ） （N，C） （ N  ， C  ） 或 （ N  ， C  ， L  ） （N，C，L） （ N  ， C  ， L  ）

  * 输出： （ N  ， C  ） （N，C） （ N  ， C  ） 或 （ N  ， C  ， L  ） （N，C，L） （ N  ， C  ， L  ） （相同形状的输入）

Examples:

    
    
    >>> # With Learnable Parameters
    >>> m = nn.BatchNorm1d(100)
    >>> # Without Learnable Parameters
    >>> m = nn.BatchNorm1d(100, affine=False)
    >>> input = torch.randn(20, 100)
    >>> output = m(input)
    

###  BatchNorm2d 

_class_`torch.nn.``BatchNorm2d`( _num_features_ , _eps=1e-05_ , _momentum=0.1_
, _affine=True_ , _track_running_stats=True_
)[[source]](_modules/torch/nn/modules/batchnorm.html#BatchNorm2d)

    

适用作为纸[批标准化描述在4D输入批标准化（用另外的通道尺寸的小批量的2D输入）：通过减少内部协变量移位](https://arxiv.org/abs/1502.03167)加速深网络训练。

y=x−E[x]Var[x]+ϵ∗γ+βy = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] +
\epsilon}} * \gamma + \betay=Var[x]+ϵ​x−E[x]​∗γ+β

The mean and standard-deviation are calculated per-dimension over the mini-
batches and γ\gammaγ and β\betaβ are learnable parameter vectors of size C
(where C is the input size). By default, the elements of γ\gammaγ are set to 1
and the elements of β\betaβ are set to 0.

Also by default, during training this layer keeps running estimates of its
computed mean and variance, which are then used for normalization during
evaluation. The running estimates are kept with a default `momentum`of 0.1.

If `track_running_stats`is set to `False`, this layer then does not keep
running estimates, and batch statistics are instead used during evaluation
time as well.

Note

This `momentum`argument is different from one used in optimizer classes and
the conventional notion of momentum. Mathematically, the update rule for
running statistics here is x^new=(1−momentum)×x^+momentum×xt\hat{x}_\text{new}
= (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times
x_tx^new​=(1−momentum)×x^+momentum×xt​ , where x^\hat{x}x^ is the estimated
statistic and xtx_txt​ is the new observed value.

因为批标准化是在 C 维完成的，在（N，H，W）切片计算统计数据，这是共同的术语来调用这个空间批标准化。

Parameters

    

  * **NUM_FEATURES** \-  C  C  C  从大小的预期输入 （ N  ， C  ， H  ， W  ） （N，C，H，W） （ N  ， C  ， H  ， W  ）

  * **eps** – a value added to the denominator for numerical stability. Default: 1e-5

  * **momentum** – the value used for the running_mean and running_var computation. Can be set to `None`for cumulative moving average (i.e. simple average). Default: 0.1

  * **affine** – a boolean value that when set to `True`, this module has learnable affine parameters. Default: `True`

  * **track_running_stats** – a boolean value that when set to `True`, this module tracks the running mean and variance, and when set to `False`, this module does not track such statistics and always uses batch statistics in both training and eval modes. Default: `True`

Shape:

    

  * Input: (N,C,H,W)(N, C, H, W)(N,C,H,W)

  * Output: (N,C,H,W)(N, C, H, W)(N,C,H,W) (same shape as input)

Examples:

    
    
    >>> # With Learnable Parameters
    >>> m = nn.BatchNorm2d(100)
    >>> # Without Learnable Parameters
    >>> m = nn.BatchNorm2d(100, affine=False)
    >>> input = torch.randn(20, 100, 35, 45)
    >>> output = m(input)
    

###  BatchNorm3d 

_class_`torch.nn.``BatchNorm3d`( _num_features_ , _eps=1e-05_ , _momentum=0.1_
, _affine=True_ , _track_running_stats=True_
)[[source]](_modules/torch/nn/modules/batchnorm.html#BatchNorm3d)

    

适用作为纸[批标准化描述在5D输入批标准化（用另外的通道尺寸的小批量的3D输入）：通过减少内部协变量移位](https://arxiv.org/abs/1502.03167)加速深网络训练。

y=x−E[x]Var[x]+ϵ∗γ+βy = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] +
\epsilon}} * \gamma + \betay=Var[x]+ϵ​x−E[x]​∗γ+β

The mean and standard-deviation are calculated per-dimension over the mini-
batches and γ\gammaγ and β\betaβ are learnable parameter vectors of size C
(where C is the input size). By default, the elements of γ\gammaγ are set to 1
and the elements of β\betaβ are set to 0.

Also by default, during training this layer keeps running estimates of its
computed mean and variance, which are then used for normalization during
evaluation. The running estimates are kept with a default `momentum`of 0.1.

If `track_running_stats`is set to `False`, this layer then does not keep
running estimates, and batch statistics are instead used during evaluation
time as well.

Note

This `momentum`argument is different from one used in optimizer classes and
the conventional notion of momentum. Mathematically, the update rule for
running statistics here is x^new=(1−momentum)×x^+momentum×xt\hat{x}_\text{new}
= (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times
x_tx^new​=(1−momentum)×x^+momentum×xt​ , where x^\hat{x}x^ is the estimated
statistic and xtx_txt​ is the new observed value.

因为批标准化是在 C 维，在计算统计数据（N，d，H，W）切片，它是常见的术语做调用此体积批标准化或时空批标准化。

Parameters

    

  * **NUM_FEATURES** \-  C  C  C  从大小的预期输入 （ N  ， C  ， d  ， H  ， W  ） （N，C，d，H，W ） （ N  ， C  ， d  ， H  ， W  ）

  * **eps** – a value added to the denominator for numerical stability. Default: 1e-5

  * **momentum** – the value used for the running_mean and running_var computation. Can be set to `None`for cumulative moving average (i.e. simple average). Default: 0.1

  * **affine** – a boolean value that when set to `True`, this module has learnable affine parameters. Default: `True`

  * **track_running_stats** – a boolean value that when set to `True`, this module tracks the running mean and variance, and when set to `False`, this module does not track such statistics and always uses batch statistics in both training and eval modes. Default: `True`

Shape:

    

  * 输入： （ N  ， C  ， d  ， H  ， W  ） （N，C，d，H，W） （ N  ， C  ， d  ， H  ， W  ）

  * 输出： （ N  ， C  ， d  ， H  ， W  ） （N，C，d，H，W） （ N  ， C  ， d  ， H  ， W  ） （相同形状的输入）

Examples:

    
    
    >>> # With Learnable Parameters
    >>> m = nn.BatchNorm3d(100)
    >>> # Without Learnable Parameters
    >>> m = nn.BatchNorm3d(100, affine=False)
    >>> input = torch.randn(20, 100, 35, 45, 10)
    >>> output = m(input)
    

###  GroupNorm 

_class_`torch.nn.``GroupNorm`( _num_groups_ , _num_channels_ , _eps=1e-05_ ,
_affine=True_
)[[source]](_modules/torch/nn/modules/normalization.html#GroupNorm)

    

如在文献[组规范化](https://arxiv.org/abs/1803.08494)中描述的应用组规范化在小批量的输入。

y=x−E[x]Var[x]+ϵ∗γ+βy = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] +
\epsilon}} * \gamma + \beta y=Var[x]+ϵ​x−E[x]​∗γ+β

输入通道分离成`NUM_GROUPS`组，每组包含`NUM_CHANNELS  /  NUM_GROUPS
`通道。的平均值和标准偏差在各组分别计算。  γ \伽马 γ 和 β \的β β 是可学习的每个信道的仿射变换大小的参数矢量`NUM_CHANNELS
`如果`仿射 `是`真 `。

该层使用在训练和评价模式从输入数据计算的统计信息。

Parameters

    

  * **NUM_GROUPS** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 基团的数目的信道分离成

  * **NUM_CHANNELS** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 预计在输入信道数

  * **EPS** \- 的值添加到分母数值稳定性。默认值：1E-5

  * **仿射** \- 一个布尔值，当设置为`真 `，该模块具有初始化为一（用于权重）和零可学习每个信道的仿射参数（偏差）。默认值：`真 [HTG9。`

Shape:

    

  * 输入： （ N  ， C  ， *  ） （N，C，*） （ N  ， C  ， *  ） 其中 C  =  NUM_CHANNELS  C = \文本{NUM \ _channels}  C  =  NUM_CHANNELS 

  * 输出： （ N  ， C  ， *  ） （N，C，*） （ N  ， C  ， *  ） （相同形状的输入）

Examples:

    
    
    >>> input = torch.randn(20, 6, 10, 10)
    >>> # Separate 6 channels into 3 groups
    >>> m = nn.GroupNorm(3, 6)
    >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
    >>> m = nn.GroupNorm(6, 6)
    >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
    >>> m = nn.GroupNorm(1, 6)
    >>> # Activating the module
    >>> output = m(input)
    

###  SyncBatchNorm 

_class_`torch.nn.``SyncBatchNorm`( _num_features_ , _eps=1e-05_ ,
_momentum=0.1_ , _affine=True_ , _track_running_stats=True_ ,
_process_group=None_
)[[source]](_modules/torch/nn/modules/batchnorm.html#SyncBatchNorm)

    

适用在N维输入批标准化（一小批量的[N-2]用另外的通道尺寸d的输入），如文献[批标准化描述：通过减少内部协变量移加速深网络训练](https://arxiv.org/abs/1502.03167)
。

y=x−E[x]Var[x]+ϵ∗γ+βy = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] +
\epsilon}} * \gamma + \betay=Var[x]+ϵ​x−E[x]​∗γ+β

平均值和标准偏差每个维度在同一过程组的所有小批量计算。  γ \伽马 γ 和 β \的β β 是大小的可学习的参数矢量 C （其中 C
被输入大小）。默认情况下，γ  \伽马 的 的元素 γ 是从 取样 U  （ 0  ， 1  ） \ mathcal {U】（0，1） U  （ 0  ，
1  ） 和 的元素β \的β β 被设置为0。

Also by default, during training this layer keeps running estimates of its
computed mean and variance, which are then used for normalization during
evaluation. The running estimates are kept with a default `momentum`of 0.1.

If `track_running_stats`is set to `False`, this layer then does not keep
running estimates, and batch statistics are instead used during evaluation
time as well.

Note

此`动量 `参数是从一个在优化器中使用的类和动量的常规概念不同。在数学上，这里运行统计数据的更新规则为 × ^  新 =  （ 1  \-  动量 ）
× × ^  \+  momemtum  × × T  \帽子{X} _ \文本{新} =（1 - \文本{动量}）\倍\帽子{X} +
\文本{momemtum} \倍X_T  × ^  新 =  （ 1  \-  动量 ） × × ^  \+  momemtum  × × T  ，其中 ×
^  \帽子{X}  × ^  ​​  是估计的统计量和 X  T  X_T  × T  是新的观测值。

因为批标准化是在 C 维完成的，在（N，+）切片计算统计数据，这是共同的术语来调用这个体积批标准化或时空批标准化。

目前SyncBatchNorm仅支持DistributedDataParallel每个进程单GPU。使用torch.nn.SyncBatchNorm.convert_sync_batchnorm（）与DDP包装网前BatchNorm层转换为SyncBatchNorm。

Parameters

    

  * **NUM_FEATURES** \-  C  C  C  从大小的预期输入 （ N  ， C  ， \+  ） （N，C，+） （ N  ， C  ， \+  ）

  * **eps** – a value added to the denominator for numerical stability. Default: 1e-5

  * **momentum** – the value used for the running_mean and running_var computation. Can be set to `None`for cumulative moving average (i.e. simple average). Default: 0.1

  * **affine** – a boolean value that when set to `True`, this module has learnable affine parameters. Default: `True`

  * **track_running_stats** – a boolean value that when set to `True`, this module tracks the running mean and variance, and when set to `False`, this module does not track such statistics and always uses batch statistics in both training and eval modes. Default: `True`

  * **process_group** \- 统计的同步每个进程组内发生独立。默认行为是在整个世界同步

Shape:

    

  * 输入： （ N  ， C  ， \+  ） （N，C，+） （ N  ， C  ， \+  ）

  * 输出： （ N  ， C  ， \+  ） （N，C，+） （ N  ， C  ， \+  ） （相同形状的输入）

Examples:

    
    
    >>> # With Learnable Parameters
    >>> m = nn.SyncBatchNorm(100)
    >>> # creating process group (optional)
    >>> # process_ids is a list of int identifying rank ids.
    >>> process_group = torch.distributed.new_group(process_ids)
    >>> # Without Learnable Parameters
    >>> m = nn.BatchNorm3d(100, affine=False, process_group=process_group)
    >>> input = torch.randn(20, 100, 35, 45, 10)
    >>> output = m(input)
    
    >>> # network is nn.BatchNorm layer
    >>> sync_bn_network = nn.SyncBatchNorm.convert_sync_batchnorm(network, process_group)
    >>> # only single gpu per process is currently supported
    >>> ddp_sync_bn_network = torch.nn.parallel.DistributedDataParallel(
    >>>                         sync_bn_network,
    >>>                         device_ids=[args.local_rank],
    >>>                         output_device=args.local_rank)
    

_classmethod_`convert_sync_batchnorm`( _module_ , _process_group=None_
)[[source]](_modules/torch/nn/modules/batchnorm.html#SyncBatchNorm.convert_sync_batchnorm)

    

辅助函数来在模型为 torch.nn.SyncBatchNorm 层 torch.nn.BatchNormND 层转换。

Parameters

    

  * **模块** （ _nn.Module_ ） - 包含模块

  * **process_group** （ _可选_ ） - 处理组范围的同步，

默认是整个世界

Returns

    

原始模块与转化 torch.nn.SyncBatchNorm 层

Example:

    
    
    >>> # Network with nn.BatchNorm layer
    >>> module = torch.nn.Sequential(
    >>>            torch.nn.Linear(20, 100),
    >>>            torch.nn.BatchNorm1d(100)
    >>>          ).cuda()
    >>> # creating process group (optional)
    >>> # process_ids is a list of int identifying rank ids.
    >>> process_group = torch.distributed.new_group(process_ids)
    >>> sync_bn_module = convert_sync_batchnorm(module, process_group)
    

###  InstanceNorm1d 

_class_`torch.nn.``InstanceNorm1d`( _num_features_ , _eps=1e-05_ ,
_momentum=0.1_ , _affine=False_ , _track_running_stats=False_
)[[source]](_modules/torch/nn/modules/instancenorm.html#InstanceNorm1d)

    

适用实例正常化了作为在纸[实例规范化描述的3D输入（带有可选的附加的信道尺寸的小批量的1D输入）：用于快速程式化失踪的成分](https://arxiv.org/abs/1607.08022)。

y=x−E[x]Var[x]+ϵ∗γ+βy = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] +
\epsilon}} * \gamma + \betay=Var[x]+ϵ​x−E[x]​∗γ+β

的平均值和标准偏差是每个维度分别计算用于在小批量的每个对象。  γ \伽马 γ 和 β \的β β 是大小的可学习的参数矢量 C （其中 C
被输入尺寸）如果`仿射 `是`真 `。

默认情况下，该层使用在训练和评价模式从输入数据计算实例的统计数据。

如果`track_running_stats`被设定为`真
`，在训练期间该层保持运行而其计算均值和方差，然后将其用于估计评估期间正常化。正在运行的估计是保持了默认的`势头 `0.1。

Note

This `momentum`argument is different from one used in optimizer classes and
the conventional notion of momentum. Mathematically, the update rule for
running statistics here is x^new=(1−momentum)×x^+momemtum×xt\hat{x}_\text{new}
= (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times
x_tx^new​=(1−momentum)×x^+momemtum×xt​ , where x^\hat{x}x^ is the estimated
statistic and xtx_txt​ is the new observed value.

Note

`InstanceNorm1d`和 `LayerNorm`非常相似，但有一些细微的差别。`InstanceNorm1d`
加到等多维时间序列引导数据的每个信道，但 `LayerNorm`通常施加在整个样本并经常在NLP任务。 Additionaly， `
LayerNorm`适用的elementwise仿射变换，而 `InstanceNorm1d`通常不应用仿射变换。

Parameters

    

  * **num_features** – CCC from an expected input of size (N,C,L)(N, C, L)(N,C,L) or LLL from input of size (N,L)(N, L)(N,L)

  * **eps** – a value added to the denominator for numerical stability. Default: 1e-5

  * **动量** \- 用于running_mean和running_var计算的值。默认值：0.1

  * **仿射** \- 为完成一个布尔值，当设置为`真 `，该模块具有可学习仿射参数，初始化的相同的方式进行批量标准化。默认值：`假 [HTG9。`

  * **track_running_stats** \- 当设置为`真 `，此模块跟踪的运行均值和方差，和一个布尔值，当设置为`假 `，该模块不跟踪这样的统计并始终使用在训练和eval模式批次的统计数据。默认值：`假 `

Shape:

    

  * 输入： （ N  ， C  ， L  ） （N，C，L） （ N  ， C  ， L  ）

  * 输出： （ N  ， C  ， L  ） （N，C，L） （ N  ， C  ， L  ） （相同形状的输入）

Examples:

    
    
    >>> # Without Learnable Parameters
    >>> m = nn.InstanceNorm1d(100)
    >>> # With Learnable Parameters
    >>> m = nn.InstanceNorm1d(100, affine=True)
    >>> input = torch.randn(20, 100, 40)
    >>> output = m(input)
    

###  InstanceNorm2d 

_class_`torch.nn.``InstanceNorm2d`( _num_features_ , _eps=1e-05_ ,
_momentum=0.1_ , _affine=False_ , _track_running_stats=False_
)[[source]](_modules/torch/nn/modules/instancenorm.html#InstanceNorm2d)

    

适用实例正常化了作为在纸[实例规范化描述的4D输入（用另外的通道尺寸的小批量的2D输入）：用于快速程式化失踪的成分](https://arxiv.org/abs/1607.08022)。

y=x−E[x]Var[x]+ϵ∗γ+βy = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] +
\epsilon}} * \gamma + \betay=Var[x]+ϵ​x−E[x]​∗γ+β

The mean and standard-deviation are calculated per-dimension separately for
each object in a mini-batch. γ\gammaγ and β\betaβ are learnable parameter
vectors of size C (where C is the input size) if `affine`is `True`.

By default, this layer uses instance statistics computed from input data in
both training and evaluation modes.

If `track_running_stats`is set to `True`, during training this layer keeps
running estimates of its computed mean and variance, which are then used for
normalization during evaluation. The running estimates are kept with a default
`momentum`of 0.1.

Note

This `momentum`argument is different from one used in optimizer classes and
the conventional notion of momentum. Mathematically, the update rule for
running statistics here is x^new=(1−momentum)×x^+momemtum×xt\hat{x}_\text{new}
= (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times
x_tx^new​=(1−momentum)×x^+momemtum×xt​ , where x^\hat{x}x^ is the estimated
statistic and xtx_txt​ is the new observed value.

Note

`InstanceNorm2d`和 `LayerNorm`非常相似，但有一些细微的差别。`InstanceNorm2d`
加到像RGB图像引导数据的每个信道，但 `LayerNorm`通常施加在整个样本并经常在NLP任务。 Additionaly， `
LayerNorm`适用的elementwise仿射变换，而 `InstanceNorm2d`通常不应用仿射变换。

Parameters

    

  * **num_features** – CCC from an expected input of size (N,C,H,W)(N, C, H, W)(N,C,H,W)

  * **eps** – a value added to the denominator for numerical stability. Default: 1e-5

  * **momentum** – the value used for the running_mean and running_var computation. Default: 0.1

  * **affine** – a boolean value that when set to `True`, this module has learnable affine parameters, initialized the same way as done for batch normalization. Default: `False`.

  * **track_running_stats** – a boolean value that when set to `True`, this module tracks the running mean and variance, and when set to `False`, this module does not track such statistics and always uses batch statistics in both training and eval modes. Default: `False`

Shape:

    

  * Input: (N,C,H,W)(N, C, H, W)(N,C,H,W)

  * Output: (N,C,H,W)(N, C, H, W)(N,C,H,W) (same shape as input)

Examples:

    
    
    >>> # Without Learnable Parameters
    >>> m = nn.InstanceNorm2d(100)
    >>> # With Learnable Parameters
    >>> m = nn.InstanceNorm2d(100, affine=True)
    >>> input = torch.randn(20, 100, 35, 45)
    >>> output = m(input)
    

###  InstanceNorm3d 

_class_`torch.nn.``InstanceNorm3d`( _num_features_ , _eps=1e-05_ ,
_momentum=0.1_ , _affine=False_ , _track_running_stats=False_
)[[source]](_modules/torch/nn/modules/instancenorm.html#InstanceNorm3d)

    

适用实例正常化了作为在纸[实例规范化描述的5D输入（用另外的通道尺寸的小批量的3D输入）：用于快速程式化失踪的成分](https://arxiv.org/abs/1607.08022)。

y=x−E[x]Var[x]+ϵ∗γ+βy = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] +
\epsilon}} * \gamma + \betay=Var[x]+ϵ​x−E[x]​∗γ+β

的平均值和标准偏差是每个维度分别计算用于在小批量的每个对象。  γ \伽马 γ 和 β \的β β 是尺寸为C的可学习的参数向量（其中C是输入大小）如果`
仿射 `是`真 `。

By default, this layer uses instance statistics computed from input data in
both training and evaluation modes.

If `track_running_stats`is set to `True`, during training this layer keeps
running estimates of its computed mean and variance, which are then used for
normalization during evaluation. The running estimates are kept with a default
`momentum`of 0.1.

Note

This `momentum`argument is different from one used in optimizer classes and
the conventional notion of momentum. Mathematically, the update rule for
running statistics here is x^new=(1−momentum)×x^+momemtum×xt\hat{x}_\text{new}
= (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times
x_tx^new​=(1−momentum)×x^+momemtum×xt​ , where x^\hat{x}x^ is the estimated
statistic and xtx_txt​ is the new observed value.

Note

`InstanceNorm3d`和 `LayerNorm`非常相似，但有一些细微的差别。`InstanceNorm3d`
加到像的三维模型与RGB彩色引导数据的每个信道，但 `LayerNorm`通常施加在整个样本并经常在NLP任务。 Additionaly， `
LayerNorm`适用的elementwise仿射变换，而 `InstanceNorm3d`通常不应用仿射变换。

Parameters

    

  * **num_features** – CCC from an expected input of size (N,C,D,H,W)(N, C, D, H, W)(N,C,D,H,W)

  * **eps** – a value added to the denominator for numerical stability. Default: 1e-5

  * **momentum** – the value used for the running_mean and running_var computation. Default: 0.1

  * **affine** – a boolean value that when set to `True`, this module has learnable affine parameters, initialized the same way as done for batch normalization. Default: `False`.

  * **track_running_stats** – a boolean value that when set to `True`, this module tracks the running mean and variance, and when set to `False`, this module does not track such statistics and always uses batch statistics in both training and eval modes. Default: `False`

Shape:

    

  * Input: (N,C,D,H,W)(N, C, D, H, W)(N,C,D,H,W)

  * Output: (N,C,D,H,W)(N, C, D, H, W)(N,C,D,H,W) (same shape as input)

Examples:

    
    
    >>> # Without Learnable Parameters
    >>> m = nn.InstanceNorm3d(100)
    >>> # With Learnable Parameters
    >>> m = nn.InstanceNorm3d(100, affine=True)
    >>> input = torch.randn(20, 100, 35, 45, 10)
    >>> output = m(input)
    

###  LayerNorm 

_class_`torch.nn.``LayerNorm`( _normalized_shape_ , _eps=1e-05_ ,
_elementwise_affine=True_
)[[source]](_modules/torch/nn/modules/normalization.html#LayerNorm)

    

作为纸张[图层规范化](https://arxiv.org/abs/1607.06450)上述适用图层规范化在小批量的输入。

y=x−E[x]Var[x]+ϵ∗γ+βy = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] +
\epsilon}} * \gamma + \beta y=Var[x]+ϵ​x−E[x]​∗γ+β

的平均值和标准偏差都在其必须由`normalized_shape`中指定的形状的最后若干尺寸分别计算。  γ \伽马 γ 和 β \的β β
是可学习的仿射变换的`参数normalized_shape`如果`elementwise_affine`是`真 `。

Note

不像批量规范化与实例标准化，它适用标量比例和偏压用于与`仿射 `选项，层正常化应用于每个元素的规模和偏压`[每个整个信道/平面HTG5]
elementwise_affine `。

This layer uses statistics computed from input data in both training and
evaluation modes.

Parameters

    

  * **normalized_shape** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)") _或_ _torch.Size_ ） - 

从尺寸的期望的输入的输入形状

[∗×normalized_shape[0]×normalized_shape[1]×…×normalized_shape[−1]][* \times
\text{normalized\\_shape}[0] \times \text{normalized\\_shape}[1] \times \ldots
\times \text{normalized\\_shape}[-1]]
[∗×normalized_shape[0]×normalized_shape[1]×…×normalized_shape[−1]]

如果使用一个整数，它被视为一个单独列表，并且该模块将在正常化，预计将特定大小的最后一个维度。

  * **eps** – a value added to the denominator for numerical stability. Default: 1e-5

  * **elementwise_affine** \- 一个布尔值，当设置为`真 `，该模块具有初始化为一（用于权重）和零可学习每个元素的仿射参数（偏差）。默认值：`真 [HTG9。`

Shape:

    

  * 输入： （ N  ， *  ） （N，*） （ N  ， *  ）

  * 输出： （ N  ， *  ） （N，*） （ N  ， *  ） （相同形状的输入）

Examples:

    
    
    >>> input = torch.randn(20, 5, 10, 10)
    >>> # With Learnable Parameters
    >>> m = nn.LayerNorm(input.size()[1:])
    >>> # Without Learnable Parameters
    >>> m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
    >>> # Normalize over last two dimensions
    >>> m = nn.LayerNorm([10, 10])
    >>> # Normalize over last dimension of size 10
    >>> m = nn.LayerNorm(10)
    >>> # Activating the module
    >>> output = m(input)
    

###  LocalResponseNorm 

_class_`torch.nn.``LocalResponseNorm`( _size_ , _alpha=0.0001_ , _beta=0.75_ ,
_k=1.0_
)[[source]](_modules/torch/nn/modules/normalization.html#LocalResponseNorm)

    

适用在几个输入平面，其中信道占用所述第二维组成的输入信号响应的本地归一化。适用跨渠道正常化。

bc=ac(k+αn∑c′=max⁡(0,c−n/2)min⁡(N−1,c+n/2)ac′2)−βb_{c} = a_{c}\left(k +
\frac{\alpha}{n} \sum_{c'=\max(0,
c-n/2)}^{\min(N-1,c+n/2)}a_{c'}^2\right)^{-\beta}
bc​=ac​⎝⎛​k+nα​c′=max(0,c−n/2)∑min(N−1,c+n/2)​ac′2​⎠⎞​−β

Parameters

    

  * **大小** \- 用于标准化相邻信道的量

  * **阿尔法** \- 乘法因子。默认值：0.0001

  * **的β** \- 指数。默认值：0.75

  * **K** \- 加法因子。默认值：1

Shape:

    

  * Input: (N,C,∗)(N, C, *)(N,C,∗)

  * Output: (N,C,∗)(N, C, *)(N,C,∗) (same shape as input)

Examples:

    
    
    >>> lrn = nn.LocalResponseNorm(2)
    >>> signal_2d = torch.randn(32, 5, 24, 24)
    >>> signal_4d = torch.randn(16, 5, 7, 7, 7, 7)
    >>> output_2d = lrn(signal_2d)
    >>> output_4d = lrn(signal_4d)
    

## 复发性层

###  RNN 

_class_`torch.nn.``RNN`( _*args_ , _**kwargs_
)[[source]](_modules/torch/nn/modules/rnn.html#RNN)

    

适用的多层埃尔曼RNN与 T  一 n的 H  的tanh  T  一 n的 H  或 R  E  L  U  RELU  R  E  L  U
非线性到输入序列。

在输入序列中的每个元件中，每个层计算下面的函数：

ht=tanh(Wihxt+bih+Whhh(t−1)+bhh)h_t = \text{tanh}(W_{ih} x_t + b_{ih} + W_{hh}
h_{(t-1)} + b_{hh}) ht​=tanh(Wih​xt​+bih​+Whh​h(t−1)​+bhh​)

其中 H  T  h_t  H  T  是在时刻 T 隐藏状态， × T  X_T  × T  是在时刻 T 的输入，并 H  （ T  \-  1  ）
[HTG135 1 H _ {（T-1）}  H  （ T  \-  1  ） 是以前的层中的时间的隐藏状态 T-1 或在时间的初始隐藏状态 0  。如果`
非线性 `是`'RELU' `，则使用 RELU 而非的tanh  。

Parameters

    

  * **input_size** \- 的预期功能在输入×个数

  * **hidden_​​size** \- 的特征在隐藏状态 h将数

  * **num_layers** \- 复发层数。例如，设置`num_layers = 2`将意味着堆叠两个RNNs在一起以形成层叠RNN ，与第二RNN取入第一RNN的输出和计算所述最后的结果。默认值：1

  * **非线性** \- 非线性使用。可以是`'的tanh' `或`'RELU' `。默认值：`'的tanh' `

  * **偏压** \- 若`假 `，则该层不使用偏压权重 b_ih 和 b_hh 。默认值：`真 `

  * **batch_first** \- 若`真 `，则输入和输出张量被设置为（分批，SEQ，特征）。默认值：`假 `

  * **差** \- 如果不为零，介绍等于`漏失一个降上除了最后层各RNN层的输出层，用差概率 `。默认值：0

  * **双向** \- 若`真 `，成为双向RNN。默认值：`假 `

Inputs: input, h_0

    

  * **输入** 的形状（seq_len，分批，input_size）：张量包含输入序列的特征。输入也可以是填充可变长度序列。参见 `torch.nn.utils.rnn.pack_padded_sequence（） `或 `torch.nn.utils.rnn.pack_sequence（ ） `的详细信息。

  * **H_0** 的形状（num_layers * num_directions，分批，hidden_​​size）：张量包含所述初始状态隐藏在批次中的每个元件。默认为零，如果不提供。如果RNN是双向的，num_directions应该是2，否则它应该是1。

Outputs: output, h_n

    

  * **输出** 形状的（seq_len，分批，num_directions * hidden_​​size）：张量含有来自RNN的最后一层的输出特征（ h_t ），对于每一个 T 。如果 `torch.nn.utils.rnn.PackedSequence`已被给定为输入，输出也将是一个拥挤的序列。

对于解压缩的情况下，该方向可使用`output.view分离（seq_len， 批次， num_directions， hidden_​​size）
`，与向前和向后方向为 0 和 1 分别。类似地，方向可以在堆积的情况下被分离。

  * **h_n** 的形状（num_layers * num_directions，分批，hidden_​​size）：张量含有 T = seq_len 隐藏状态。

像 _输出_ 时，层可使用`h_n.view（num_layers分离， num_directions， 批次， hidden_​​size ） `。

Shape:

    

  * 输入1： （ L  ， N  ， H  i的 n的 ） （L，N，H_ {在}） （ L  ， N  ， H  i的 n的 ） [HTG89含有张量输入功能，其中 H  i的 [HT G102] n的  =  input_size  H_ {IN} = \文本{输入\ _size}  H  i的 n的 =  input_size  和 L 表示序列长度。

  * 输入2： （ S  ， N  ， H  O  U  T  ） （S，N，H_ {出}） （ S  ， N  ， H  O  U  T  ）  [HTG93含有初始隐藏状态在批次中的每个元件张量。  H  O  U  T  =  hidden_​​size  H_ {出} = \文本{隐藏\ _size}  H  O  U  T  =  hidden_​​size  如果不设置缺省值为零。其中 S  =  num_layers  *  num_directions  S = \文本{NUM \ _layers} * \文本{NUM \ _directions}  S  =  num_layers  *  num_directions  [HTG237如果RNN是双向的，num_directions应该是2，否则它应为1。

  * 输出1： （ L  ， N  ， H  一 L  L  ） （L，N，H_ {所有}） （ L  ， N  ， H  一 L  L  ）  其中 H  一个[H TG105]  L  L  =  num_directions  *  hidden_​​size  H_ {所有} = \文本{NUM \ _directions} * \文本{隐藏\ _size}  H  一 L  L  =  num_directions  *  hidden_​​size  [HT G193] 

  * OUTPUT2： （ S  ， N  ， H  O  U  T  ） （S，N，H_ {出}） （ S  ， N  ， H  O  U  T  ）  包含下一个隐藏状态在批次中的每个元件张量

Variables

    

  * **〜RNN.weight_ih_l [k]的** \- 第k层的可学习输入隐藏重量，形状（hidden_​​size，input_size）的为 K = 0  。否则，该形状是（hidden_​​size，num_directions * hidden_​​size）

  * **〜RNN.weight_hh_l [k]的** \- 第k层的可学习隐藏的权重，形状的（hidden_​​size，hidden_​​size）

  * **〜RNN.bias_ih_l [k]的** \- 第k层的可学习输入隐藏偏压，形状的（hidden_​​size）

  * **〜RNN.bias_hh_l [k]的** \- 第k层的可学习隐藏偏压，形状的（hidden_​​size）

Note

所有的重量和偏见从 初始化U  （ \-  K  ， K  ） \ mathcal【U}（ - \ SQRT {K}，\ SQRT {K}） U  （ \-
K  ， K  ） 其中 K  =  1  hidden_​​size  K = \压裂{1} {\文本{隐藏\ _size}}  K  =
hidden_​​size  1  [HTG19 3]

Note

如果满足以下条件：1）使能cudnn，2）输入的数据是在GPU上3）的输入数据已经DTYPE `torch.float16`4）V100
GPU时，5 ）输入数据不是在`PackedSequence`格式持续算法可经选择以提高性能。

Examples:

    
    
    >>> rnn = nn.RNN(10, 20, 2)
    >>> input = torch.randn(5, 3, 10)
    >>> h0 = torch.randn(2, 3, 20)
    >>> output, hn = rnn(input, h0)
    

###  LSTM 

_class_`torch.nn.``LSTM`( _*args_ , _**kwargs_
)[[source]](_modules/torch/nn/modules/rnn.html#LSTM)

    

适用的多层长短期记忆（LSTM）RNN到输入序列。

For each element in the input sequence, each layer computes the following
function:

it=σ(Wiixt+bii+Whih(t−1)+bhi)ft=σ(Wifxt+bif+Whfh(t−1)+bhf)gt=tanh⁡(Wigxt+big+Whgh(t−1)+bhg)ot=σ(Wioxt+bio+Whoh(t−1)+bho)ct=ft∗c(t−1)+it∗gtht=ot∗tanh⁡(ct)\begin{array}{ll}
\\\ i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\\ f_t =
\sigma(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\\ g_t = \tanh(W_{ig}
x_t + b_{ig} + W_{hg} h_{(t-1)} + b_{hg}) \\\ o_t = \sigma(W_{io} x_t + b_{io}
+ W_{ho} h_{(t-1)} + b_{ho}) \\\ c_t = f_t * c_{(t-1)} + i_t * g_t \\\ h_t =
o_t * \tanh(c_t) \\\ \end{array}
it​=σ(Wii​xt​+bii​+Whi​h(t−1)​+bhi​)ft​=σ(Wif​xt​+bif​+Whf​h(t−1)​+bhf​)gt​=tanh(Wig​xt​+big​+Whg​h(t−1)​+bhg​)ot​=σ(Wio​xt​+bio​+Who​h(t−1)​+bho​)ct​=ft​∗c(t−1)​+it​∗gt​ht​=ot​∗tanh(ct​)​

其中 H  T  h_t  H  T  是在时刻 T 隐藏状态， C  T  C_T  C  T  是在时刻 T 小区状态， × T  X_T  × T
是输入时刻 T ， H  （ T  \-  1  ） [HTG191 1 H _ {（T-1）}  [HTG19 6]  H  （ T  \-  1  ）
是的隐藏状态层时刻 T-1 或时刻 0 初始隐藏状态，和 i的 T  I_T  i的​​  T  [HTG2 85]  ， F  T  F_T  F  T
， 克 T  G_T  克 [HT G383]  T  ， O  T  O_t同 O  T  是输入，忘记，细胞，和输出门，分别。  σ \西格玛 σ
是S形函数，并 *  *  *  为Hadamard乘积。

在多层LSTM，输入 × T  （ L  ） ×^ {（L）} _ T  × T  （ L  ） 的 L  L  L  第层（ L  & GT ;  =
2  L & GT ; = 2  L  & GT ;  =  2  ）是隐藏状态 H  T  （ L  \-  1  ） H ^ {（L-1）} _ T
H  T  （ L  \-  1  ） 先前层乘以的脱落 δ T  （ L  \-  1  ） \增量^ {（L-1）} _ T  δ ​​  T  （ L
\-  1  ） 其中各HTG316 ]  δ T  （ L  \-  1  ） \增量^ {（L-1）} _ T  δ T  （ L  \-  1  ）
是伯努利随机变量，它是 0  0  0  的概率`漏失 `。

Parameters

    

  * **input_size** – The number of expected features in the input x

  * **hidden_size** – The number of features in the hidden state h

  * **num_layers** \- 复发层数。例如，设置`num_layers = 2`将意味着堆叠两个LSTMs在一起以形成层叠LSTM ，与第二LSTM取入输出第一LSTM的和计算的最后的结果。默认值：1

  * **bias** – If `False`, then the layer does not use bias weights b_ih and b_hh. Default: `True`

  * **batch_first** \- 若`真 `，则输入和输出张量被设置为（批次，SEQ，特征）。默认值：`假 `

  * **差** \- 如果不为零，介绍等于`漏失一个降上除了最后层各LSTM层的输出层，用差概率 `。默认值：0

  * **双向** \- 若`真 `，成为双向LSTM。默认值：`假 `

Inputs: input, (h_0, c_0)

    

  * **输入** 的形状（seq_len，分批，input_size）：张量包含输入序列的特征。输入也可以是填充可变长度序列。参见 `torch.nn.utils.rnn.pack_padded_sequence（） `或 `torch.nn.utils.rnn.pack_sequence（ ） `的详细信息。

  * **H_0** 的形状（num_layers * num_directions，分批，hidden_​​size）：张量包含所述初始状态隐藏在批次中的每个元件。如果LSTM是双向的，num_directions应该是2，否则它应该是1。

  * 的形状（num_layers * num_directions，分批，hidden_​​size） **C_0** ：张量包含所述初始小区状态为批中每个元件。

如果（H_0，C_0）不设置，二者 **H_0** 和 **C_0** 默认为零。

Outputs: output, (h_n, c_n)

    

  * **输出** 形状的（seq_len，分批，num_directions * hidden_​​size）：张量包含输出特征（h_t）从LSTM的最后一层，对于每个 T 。如果 `torch.nn.utils.rnn.PackedSequence`已被给定为输入，输出也将是一个拥挤的序列。

For the unpacked case, the directions can be separated using
`output.view(seq_len, batch, num_directions, hidden_size)`, with forward and
backward being direction 0 and 1 respectively. Similarly, the directions can
be separated in the packed case.

  * **h_n** of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t = seq_len.

像 _输出_ 时，层可使用`h_n.view（num_layers分离， num_directions， 批次， hidden_​​size ）
`同样地，对于 _C_N_ 。

  * **C_N** 的形状（num_layers * num_directions，分批，hidden_​​size）：张量含有 T = seq_len 小区状态。

Variables

    

  * **〜LSTM.weight_ih_l [k]的** \- 的 k是可学习输入隐藏权重 T  H  \文本{K} ^ {第}  K  T  H  层（W_ii | W_if | W_ig | W_io），形状的（4 * hidden_​​size，input_size）为 K = 0 。否则，该形状是（4 * hidden_​​size，num_directions * hidden_​​size）

  * **〜LSTM.weight_hh_l [k]的** \- 的 k是可学习隐藏权重 T  H  \文本{K} ^ {第}  K  T  H  层（W_hi | W_hf | W_hg | W_ho），形状的（4 * hidden_​​size，hidden_​​size）

  * **〜LSTM.bias_ih_l [k]的** \- 的可学习输入隐藏偏置 K  T  H  \文本{K} ^ {第}  K  T  H  层形状的（b_ii | b_if | | b_ig b_io），（4 * hidden_​​size）

  * **〜LSTM.bias_hh_l [k]的** \- 的可学习隐藏偏压 K  T  H  \文本{K} ^ {第}  K  T  H  层形状的（b_hi | b_hf | | b_hg b_ho），（4 * hidden_​​size）

Note

All the weights and biases are initialized from U(−k,k)\mathcal{U}(-\sqrt{k},
\sqrt{k})U(−k​,k​) where k=1hidden_sizek =
\frac{1}{\text{hidden\\_size}}k=hidden_size1​

Note

If the following conditions are satisfied: 1) cudnn is enabled, 2) input data
is on the GPU 3) input data has dtype `torch.float16`4) V100 GPU is used, 5)
input data is not in `PackedSequence`format persistent algorithm can be
selected to improve performance.

Examples:

    
    
    >>> rnn = nn.LSTM(10, 20, 2)
    >>> input = torch.randn(5, 3, 10)
    >>> h0 = torch.randn(2, 3, 20)
    >>> c0 = torch.randn(2, 3, 20)
    >>> output, (hn, cn) = rnn(input, (h0, c0))
    

###  GRU 

_class_`torch.nn.``GRU`( _*args_ , _**kwargs_
)[[source]](_modules/torch/nn/modules/rnn.html#GRU)

    

适用门控重复单元（GRU）RNN到输入序列的多层。

For each element in the input sequence, each layer computes the following
function:

rt=σ(Wirxt+bir+Whrh(t−1)+bhr)zt=σ(Wizxt+biz+Whzh(t−1)+bhz)nt=tanh⁡(Winxt+bin+rt∗(Whnh(t−1)+bhn))ht=(1−zt)∗nt+zt∗h(t−1)\begin{array}{ll}
r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\\ z_t =
\sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\\ n_t = \tanh(W_{in}
x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\\ h_t = (1 - z_t) * n_t +
z_t * h_{(t-1)} \end{array}
rt​=σ(Wir​xt​+bir​+Whr​h(t−1)​+bhr​)zt​=σ(Wiz​xt​+biz​+Whz​h(t−1)​+bhz​)nt​=tanh(Win​xt​+bin​+rt​∗(Whn​h(t−1)​+bhn​))ht​=(1−zt​)∗nt​+zt​∗h(t−1)​​

其中 H  T  h_t  H  T  是在时刻 T 隐藏状态， × T  X_T  × T  是在时刻 T 的输入， H  （ T  \-  1  ）
[HTG135 1 H _ {（T-1）}  H  （ T  \-  1  ） 是该层的在时间的隐藏状态 T-1 或在时间的初始隐藏状态 0 和[HTG19
0]  R  T  r_t  R  T  ， Z  吨 z_t  Z  ​​  T  [HT G288]  ， n的 T  N_T  n的 T
是复位，更新和新的大门，分别。  σ \西格玛 σ 是S形函数，并 *  *  *  为Hadamard乘积。

在多层GRU，输入 × T  （ L  ） ×^ {（L）} _ T  × T  （ L  ） 的 L  L  L  第层（ L  & GT ;  =  2
升& GT ; = 2  L  & GT ;  =  2  ）是隐藏状态 [HTG155 1 H  T  （ L  \-  1  ） H ^ {（L-1）}
_ T  H  T [HTG1 94]  （ L  \-  1  ） 先前层乘以的脱落 δ T  （ L  \-  1  ） \增量^ {（L-1）} _
T  δ ​​  T  （ L  \-  1  ） 其中各HTG316 ]  δ T  （ L  \-  1  ） \增量^ {（L-1）} _ T  δ
T  （ L  \-  1  ） 是伯努利随机变量，它是 0  0  0  的概率`差 `。

Parameters

    

  * **input_size** – The number of expected features in the input x

  * **hidden_size** – The number of features in the hidden state h

  * **num_layers** \- 复发层数。例如，设置`num_layers = 2`将意味着堆叠两个越冬在一起以形成层叠GRU ，与第二GRU取入第一GRU的输出和计算所述最后的结果。默认值：1

  * **bias** – If `False`, then the layer does not use bias weights b_ih and b_hh. Default: `True`

  * **batch_first** – If `True`, then the input and output tensors are provided as (batch, seq, feature). Default: `False`

  * **差** \- 如果不为零，介绍了除了最后层各GRU层的输出降层，用差概率等于`差 `。默认值：0

  * **双向** \- 若`真 `，成为双向的GRU。默认值：`假 `

Inputs: input, h_0

    

  * **输入** 的形状（seq_len，分批，input_size）：张量包含输入序列的特征。输入也可以是填充可变长度序列。参见 `对于细节torch.nn.utils.rnn.pack_padded_sequence（） `。

  * **h_0** of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch. Defaults to zero if not provided. If the RNN is bidirectional, num_directions should be 2, else it should be 1.

Outputs: output, h_n

    

  * **输出** 形状（seq_len，分批，num_directions * hidden_​​size）的：张量包含输出特征从GRU的最后一层h_t，对于每个 T 。如果 `torch.nn.utils.rnn.PackedSequence`已被给定为输入，输出也将是一个拥挤的序列。对于解压缩的情况下，该方向可使用`output.view分离（seq_len， 批次， num_directions， hidden_​​size） `，与向前和向后方向为 0 和 1 分别。

类似地，方向可以在堆积的情况下被分离。

  * **h_n** 形状（num_layers * num_directions，分批，hidden_​​size）的：张量包含隐藏状态 T = seq_len 

Like _output_ , the layers can be separated using `h_n.view(num_layers,
num_directions, batch, hidden_size)`.

Shape:

    

  * Input1: (L,N,Hin)(L, N, H_{in})(L,N,Hin​) tensor containing input features where Hin=input_sizeH_{in}=\text{input\\_size}Hin​=input_size and L represents a sequence length.

  * Input2: (S,N,Hout)(S, N, H_{out})(S,N,Hout​) tensor containing the initial hidden state for each element in the batch. Hout=hidden_sizeH_{out}=\text{hidden\\_size}Hout​=hidden_size Defaults to zero if not provided. where S=num_layers∗num_directionsS=\text{num\\_layers} * \text{num\\_directions}S=num_layers∗num_directions If the RNN is bidirectional, num_directions should be 2, else it should be 1.

  * Output1: (L,N,Hall)(L, N, H_{all})(L,N,Hall​) where Hall=num_directions∗hidden_sizeH_{all}=\text{num\\_directions} * \text{hidden\\_size}Hall​=num_directions∗hidden_size

  * Output2: (S,N,Hout)(S, N, H_{out})(S,N,Hout​) tensor containing the next hidden state for each element in the batch

Variables

    

  * **〜GRU.weight_ih_l [k]的** \- 的 k是可学习输入隐藏权重 T  H  \文本{K} ^ {第}  K  T  H  层（W_ir | W_iz | W_IN），形状（3 * hidden_​​size，input_size）为中的k = 0 。否则，该形状是（3 * hidden_​​size，num_directions * hidden_​​size）

  * **〜GRU.weight_hh_l [k]的** \- 的 k是可学习隐藏权重 T  H  \文本{K} ^ {第}  K  T  H  层（W_hr | W_hz | W_hn），形状的（3 * hidden_​​size，hidden_​​size）

  * **〜GRU.bias_ih_l [k]的** \- 的可学习输入隐藏偏置 K  T  H  \文本{K} ^ {第}  K  T  H  层形状的（b_ir | | b_iz B_IN），（3 * hidden_​​size）

  * **〜GRU.bias_hh_l [k]的** \- 的可学习隐藏偏压 K  T  H  \文本{K} ^ {第}  K  T  H  层形状的（b_hr | | b_hz b_hn），（3 * hidden_​​size）

Note

All the weights and biases are initialized from U(−k,k)\mathcal{U}(-\sqrt{k},
\sqrt{k})U(−k​,k​) where k=1hidden_sizek =
\frac{1}{\text{hidden\\_size}}k=hidden_size1​

Note

If the following conditions are satisfied: 1) cudnn is enabled, 2) input data
is on the GPU 3) input data has dtype `torch.float16`4) V100 GPU is used, 5)
input data is not in `PackedSequence`format persistent algorithm can be
selected to improve performance.

Examples:

    
    
    >>> rnn = nn.GRU(10, 20, 2)
    >>> input = torch.randn(5, 3, 10)
    >>> h0 = torch.randn(2, 3, 20)
    >>> output, hn = rnn(input, h0)
    

###  RNNCell 

_class_`torch.nn.``RNNCell`( _input_size_ , _hidden_size_ , _bias=True_ ,
_nonlinearity='tanh'_ )[[source]](_modules/torch/nn/modules/rnn.html#RNNCell)

    

一个埃尔曼RNN细胞与双曲正切或RELU非线性。

h′=tanh⁡(Wihx+bih+Whhh+bhh)h' = \tanh(W_{ih} x + b_{ih} + W_{hh} h +
b_{hh})h′=tanh(Wih​x+bih​+Whh​h+bhh​)

如果`非线性 `是“RELU” ，然后RELU代替的tanh的使用。

Parameters

    

  * **input_size** – The number of expected features in the input x

  * **hidden_size** – The number of features in the hidden state h

  * **bias** – If `False`, then the layer does not use bias weights b_ih and b_hh. Default: `True`

  * **nonlinearity** – The non-linearity to use. Can be either `'tanh'`or `'relu'`. Default: `'tanh'`

Inputs: input, hidden

    

  * **输入** 形状的（分批，input_size）：张量含有输入特征

  * **隐藏形状 （分批，hidden_​​size）的**：张量包含所述初始状态隐藏在批次中的每个元件。默认为零，如果不提供。

Outputs: h’

    

  * **H”** 形状（分批，hidden_​​size）的：张量包含下一隐藏状态为批中每个元件

Shape:

    

  * 输入1： （ N  ， H  i的 n的 ） （N，H_ {在}） （ N  ， H  i的 n的 ）  [HTG79含有张量输入功能，其中 H  i的 n的 H_ {IN}  [HT G102]  H  i的 n的 =  input_size 

  * 输入2： （ N  ， H  O  U  T  ） （N，H_ {出}） （ N  ， H  O  U  T  ） [HTG83含有初始隐藏状态批中每个元件张量，其中 H  O  U  T  H_ {出}  H  O  U  T  =  hidden_​​size 如果不设置缺省值为零。

  * 输出： （ N  ， H  O  U  T  ） （N，H_ {出}） （ N  ， H  O  U  T  ） 包含下一个隐藏状态在批次中的每个元件张量

Variables

    

  * **〜RNNCell.weight_ih** \- 的可学习输入隐藏重量，形状的（hidden_​​size，input_size）

  * **〜RNNCell.weight_hh** \- 的可学习隐藏的权重，形状的（hidden_​​size，hidden_​​size）

  * **〜RNNCell.bias_ih** \- 的可学习输入隐藏偏压，形状的（hidden_​​size）

  * **〜RNNCell.bias_hh** \- 的可学习隐藏偏压，形状的（hidden_​​size）

Note

All the weights and biases are initialized from U(−k,k)\mathcal{U}(-\sqrt{k},
\sqrt{k})U(−k​,k​) where k=1hidden_sizek =
\frac{1}{\text{hidden\\_size}}k=hidden_size1​

Examples:

    
    
    >>> rnn = nn.RNNCell(10, 20)
    >>> input = torch.randn(6, 3, 10)
    >>> hx = torch.randn(3, 20)
    >>> output = []
    >>> for i in range(6):
            hx = rnn(input[i], hx)
            output.append(hx)
    

###  LSTMCell 

_class_`torch.nn.``LSTMCell`( _input_size_ , _hidden_size_ , _bias=True_
)[[source]](_modules/torch/nn/modules/rnn.html#LSTMCell)

    

长短期记忆（LSTM）细胞。

i=σ(Wiix+bii+Whih+bhi)f=σ(Wifx+bif+Whfh+bhf)g=tanh⁡(Wigx+big+Whgh+bhg)o=σ(Wiox+bio+Whoh+bho)c′=f∗c+i∗gh′=o∗tanh⁡(c′)\begin{array}{ll}
i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\\ f = \sigma(W_{if} x +
b_{if} + W_{hf} h + b_{hf}) \\\ g = \tanh(W_{ig} x + b_{ig} + W_{hg} h +
b_{hg}) \\\ o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\\ c' = f * c +
i * g \\\ h' = o * \tanh(c') \\\
\end{array}i=σ(Wii​x+bii​+Whi​h+bhi​)f=σ(Wif​x+bif​+Whf​h+bhf​)g=tanh(Wig​x+big​+Whg​h+bhg​)o=σ(Wio​x+bio​+Who​h+bho​)c′=f∗c+i∗gh′=o∗tanh(c′)​

其中 σ \西格玛 σ 是S形函数，并 *  *  *  为Hadamard乘积。

Parameters

    

  * **input_size** – The number of expected features in the input x

  * **hidden_size** – The number of features in the hidden state h

  * **偏压** \- 若`假 `，则该层不使用偏压权重 b_ih 和 b_hh 。默认值：`真 `

Inputs: input, (h_0, c_0)

    

  * **input** of shape (batch, input_size): tensor containing input features

  * **H_0** 形状（分批，hidden_​​size）的：张量包含所述初始状态隐藏在批次中的每个元件。

  * **C_0** 形状（分批，hidden_​​size）的：张量包含所述初始小区状态为批中每个元件。

If (h_0, c_0) is not provided, both **h_0** and **c_0** default to zero.

Outputs: (h_1, c_1)

    

  * **H_1** 的形状（分批，hidden_​​size）：张量含有批处理每个元素的下一个隐藏状态

  * **C_1** 形状（分批，hidden_​​size）的：张量含有批处理每个元素的下一个小区状态

Variables

    

  * **〜LSTMCell.weight_ih** \- 的可学习输入隐藏重量，形状的（4 * hidden_​​size，input_size）

  * **〜LSTMCell.weight_hh** \- 的可学习隐藏的权重，形状的（4 * hidden_​​size，hidden_​​size）

  * **〜LSTMCell.bias_ih** \- 的可学习输入隐藏偏压，形状的（4 * hidden_​​size）

  * **〜LSTMCell.bias_hh** \- 的可学习隐藏偏压，形状的（4 * hidden_​​size）

Note

All the weights and biases are initialized from U(−k,k)\mathcal{U}(-\sqrt{k},
\sqrt{k})U(−k​,k​) where k=1hidden_sizek =
\frac{1}{\text{hidden\\_size}}k=hidden_size1​

Examples:

    
    
    >>> rnn = nn.LSTMCell(10, 20)
    >>> input = torch.randn(6, 3, 10)
    >>> hx = torch.randn(3, 20)
    >>> cx = torch.randn(3, 20)
    >>> output = []
    >>> for i in range(6):
            hx, cx = rnn(input[i], (hx, cx))
            output.append(hx)
    

###  GRUCell 

_class_`torch.nn.``GRUCell`( _input_size_ , _hidden_size_ , _bias=True_
)[[source]](_modules/torch/nn/modules/rnn.html#GRUCell)

    

门控重复单元（GRU）细胞

r=σ(Wirx+bir+Whrh+bhr)z=σ(Wizx+biz+Whzh+bhz)n=tanh⁡(Winx+bin+r∗(Whnh+bhn))h′=(1−z)∗n+z∗h\begin{array}{ll}
r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\\ z = \sigma(W_{iz} x +
b_{iz} + W_{hz} h + b_{hz}) \\\ n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h +
b_{hn})) \\\ h' = (1 - z) * n + z * h
\end{array}r=σ(Wir​x+bir​+Whr​h+bhr​)z=σ(Wiz​x+biz​+Whz​h+bhz​)n=tanh(Win​x+bin​+r∗(Whn​h+bhn​))h′=(1−z)∗n+z∗h​

where σ\sigmaσ is the sigmoid function, and ∗*∗ is the Hadamard product.

Parameters

    

  * **input_size** – The number of expected features in the input x

  * **hidden_size** – The number of features in the hidden state h

  * **bias** – If `False`, then the layer does not use bias weights b_ih and b_hh. Default: `True`

Inputs: input, hidden

    

  * **input** of shape (batch, input_size): tensor containing input features

  * **hidden** of shape (batch, hidden_size): tensor containing the initial hidden state for each element in the batch. Defaults to zero if not provided.

Outputs: h’

    

  * **h’** of shape (batch, hidden_size): tensor containing the next hidden state for each element in the batch

Shape:

    

  * Input1: (N,Hin)(N, H_{in})(N,Hin​) tensor containing input features where HinH_{in}Hin​ = input_size

  * Input2: (N,Hout)(N, H_{out})(N,Hout​) tensor containing the initial hidden state for each element in the batch where HoutH_{out}Hout​ = hidden_size Defaults to zero if not provided.

  * Output: (N,Hout)(N, H_{out})(N,Hout​) tensor containing the next hidden state for each element in the batch

Variables

    

  * **〜GRUCell.weight_ih** \- 的可学习输入隐藏重量，形状的（3 * hidden_​​size，input_size）

  * **〜GRUCell.weight_hh** \- 的可学习隐藏的权重，形状的（3 * hidden_​​size，hidden_​​size）

  * **〜GRUCell.bias_ih** \- 的可学习输入隐藏偏压，形状的（3 * hidden_​​size）

  * **〜GRUCell.bias_hh** \- 的可学习隐藏偏压，形状的（3 * hidden_​​size）

Note

All the weights and biases are initialized from U(−k,k)\mathcal{U}(-\sqrt{k},
\sqrt{k})U(−k​,k​) where k=1hidden_sizek =
\frac{1}{\text{hidden\\_size}}k=hidden_size1​

Examples:

    
    
    >>> rnn = nn.GRUCell(10, 20)
    >>> input = torch.randn(6, 3, 10)
    >>> hx = torch.randn(3, 20)
    >>> output = []
    >>> for i in range(6):
            hx = rnn(input[i], hx)
            output.append(hx)
    

## 变压器层

### 变压器

_class_`torch.nn.``Transformer`( _d_model=512_ , _nhead=8_ ,
_num_encoder_layers=6_ , _num_decoder_layers=6_ , _dim_feedforward=2048_ ,
_dropout=0.1_ , _custom_encoder=None_ , _custom_decoder=None_
)[[source]](_modules/torch/nn/modules/transformer.html#Transformer)

    

变压器模型。用户可以根据需要修改其属性。该architechture是基于纸“注意是所有你需要”。阿希什瓦斯瓦尼，诺姆Shazeer，尼基帕尔马雅各布Uszkoreit，Llion琼斯，艾Ñ戈麦斯，卢卡斯凯泽，和Illia
Polosukhin。 2017注意力是你所需要的。在神经信息处理系统的进步，6000-6010页。

Parameters

    

  * **d_model** \- 的预期功能在编码器/解码器的输入的数量（缺省值= 512）。

  * **NHEAD** \- 的头在multiheadattention模型数（默认= 8）。

  * **num_encoder_layers** \- 在编码器（默认= 6）子编码器的层的数目。

  * **num_decoder_layers** \- 在解码器（默认= 6）子译码器层的数量。

  * **dim_feedforward** \- 前馈网络模型（缺省值= 2048）的维数。

  * **差** \- 漏失值（缺省值= 0.1）。

  * **custom_encoder** \- 定制的编码器（默认=无）。

  * **custom_decoder** \- 定制解码器（默认=无）。

Examples::

    
    
    
    >>> transformer_model = nn.Transformer(src_vocab, tgt_vocab)
    >>> transformer_model = nn.Transformer(src_vocab, tgt_vocab, nhead=16, num_encoder_layers=12)
    

`forward`( _src_ , _tgt_ , _src_mask=None_ , _tgt_mask=None_ ,
_memory_mask=None_ , _src_key_padding_mask=None_ , _tgt_key_padding_mask=None_
, _memory_key_padding_mask=None_
)[[source]](_modules/torch/nn/modules/transformer.html#Transformer.forward)

    

采取在与过程掩蔽源/靶序列。

Parameters

    

  * **SRC** \- （必需）序列与编码器。

  * **TGT** \- 的顺序向解码器（必需）。

  * **src_mask** \- 为对SRC序列（任选的）添加剂掩模。

  * **tgt_mask** \- 添加剂掩码TGT序列（可选）。

  * **memory_mask** \- 为对编码器输出（可选）添加剂掩模。

  * **src_key_padding_mask** \- 对于每批次（可选）SRC键ByteTensor掩模。

  * **tgt_key_padding_mask** \- 的ByteTensor掩模每批（可选）TGT密钥。

  * **memory_key_padding_mask** \- 对于每批次（可选）存储器键ByteTensor掩模。

Shape:

    

  * SRC： （ S  ， N  ， E  ） （S，N，E） （ S  ， N  ， E  ） 。

  * TGT： （ T  ， N  ， E  ） （T，N，E） （ T  ， N  ， E  ） 。

  * src_mask： （ S  ， S  ） （S，S） （ S  ， S  ） 。

  * tgt_mask： （ T  ， T  ） （T，T） （ T  ， T  ） 。

  * memory_mask： （ T  ， S  ） （T，S） （ T  ， S  ） 。

  * src_key_padding_mask： （ N  ， S  ） （N，S） （ N  ， S  ） 。

  * tgt_key_padding_mask： （ N  ， T  ） （N，T） （ N  ， T  ） 。

  * memory_key_padding_mask： （ N  ， S  ） （N，S） （ N  ， S  ） 。

注意：[SRC / TGT /存储器] _mask应与浮子（“ -
INF”）被填充在掩蔽位置和浮动（0.0）其他。这些面具确保了位置预测我只取决于东窗事发j和是相同的批处理中的每个序列应用。 [SRC / TGT
/存储器] _key_padding_mask应该是一个ByteTensor其中真值应与浮子（“ -
INF”）被掩蔽的位置值和假值将保持不变。该掩模可确保没有信息将从位置采取i如果它被屏蔽，并且具有用于批处理中的每个序列的单独的掩模。

  * 输出： （ T  ， N  ， E  ） （T，N，E） （ T  ， N  ， E  ） 。

注意：由于在变压器模型中的多磁头关注架构中，变压器的输出序列的长度是一样的解码器的输入序列（即目标）的长度。

其中，S是源序列长度，T是所述靶序列的长度，N是批量大小，E是功能号码

Examples

    
    
    >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
    

`generate_square_subsequent_mask`( _sz_
)[[source]](_modules/torch/nn/modules/transformer.html#Transformer.generate_square_subsequent_mask)

    

生成序列的方形面具。蒙面位置都充满了浮动（“ - INF”）。未掩蔽的位置被填充有浮子（0.0）。

###  TransformerEncoder 

_class_`torch.nn.``TransformerEncoder`( _encoder_layer_ , _num_layers_ ,
_norm=None_
)[[source]](_modules/torch/nn/modules/transformer.html#TransformerEncoder)

    

TransformerEncoder是N个编码器的层的叠层

Parameters

    

  * **encoder_layer** \- （必需）TransformerEncoderLayer（）的类的实例。

  * **num_layers** \- 在编码器中的子编码器的层的数量（需要）。

  * **规范** \- 层归一部件（可选）。

Examples::

    
    
    
    >>> encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
    >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    

`forward`( _src_ , _mask=None_ , _src_key_padding_mask=None_
)[[source]](_modules/torch/nn/modules/transformer.html#TransformerEncoder.forward)

    

通过依次endocder层传递输入。

Parameters

    

  * **SRC** \- （必需）序列给解码器。

  * **掩模** \- 为对SRC序列（任选的）掩模。

  * **src_key_padding_mask** \- 对于每批次（可选）在src键掩模。

Shape:

    

看到变压器类的文档。

###  TransformerDecoder 

_class_`torch.nn.``TransformerDecoder`( _decoder_layer_ , _num_layers_ ,
_norm=None_
)[[source]](_modules/torch/nn/modules/transformer.html#TransformerDecoder)

    

TransformerDecoder是N解码器的层的叠层

Parameters

    

  * **decoder_layer** \- 的TransformerDecoderLayer（）的类的实例（必需）。

  * **num_layers** \- 在解码器中的子译码器，层的数量（需要）。

  * **norm** – the layer normalization component (optional).

Examples::

    
    
    
    >>> decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
    >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
    

`forward`( _tgt_ , _memory_ , _tgt_mask=None_ , _memory_mask=None_ ,
_tgt_key_padding_mask=None_ , _memory_key_padding_mask=None_
)[[source]](_modules/torch/nn/modules/transformer.html#TransformerDecoder.forward)

    

通过依次解码器层传递的输入（和掩模）。

Parameters

    

  * **tgt** – the sequence to the decoder (required).

  * **存储器** \- 来自编码器的最后一层的序列（必需）。

  * **tgt_mask** \- 掩模为TGT序列（可选）。

  * **memory_mask** \- 为对存储器序列（任选的）掩模。

  * **tgt_key_padding_mask** \- 掩模每批次（可选）在tgt密钥。

  * **memory_key_padding_mask** \- 对于每批次（可选）存储键掩模。

Shape:

    

see the docs in Transformer class.

###  TransformerEncoderLayer 

_class_`torch.nn.``TransformerEncoderLayer`( _d_model_ , _nhead_ ,
_dim_feedforward=2048_ , _dropout=0.1_
)[[source]](_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer)

    

TransformerEncoderLayer由自经办人及前馈网络。该标准编码层是基于纸“注意是所有你需要”。阿希什瓦斯瓦尼，诺姆Shazeer，尼基帕尔马雅各布Uszkoreit，Llion琼斯，艾Ñ戈麦斯，卢卡斯凯泽，和Illia
Polosukhin。 2017注意力是你所需要的。在神经信息处理系统的进步，6000-6010页。用户可以修改或应用程序中以不同的方式实现。

Parameters

    

  * **d_model** \- 在输入预期特征的数量（需要）。

  * **NHEAD** \- （必需）在multiheadattention模型头的数目。

  * **dim_feedforward** – the dimension of the feedforward network model (default=2048).

  * **dropout** – the dropout value (default=0.1).

Examples::

    
    
    
    >>> encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
    

`forward`( _src_ , _src_mask=None_ , _src_key_padding_mask=None_
)[[source]](_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer.forward)

    

通过endocder层传递输入。

Parameters

    

  * **SRC** \- （必需）序列提供给编码器层。

  * **src_mask** \- 为对SRC序列（任选的）掩模。

  * **src_key_padding_mask** – the mask for the src keys per batch (optional).

Shape:

    

see the docs in Transformer class.

###  TransformerDecoderLayer 

_class_`torch.nn.``TransformerDecoderLayer`( _d_model_ , _nhead_ ,
_dim_feedforward=2048_ , _dropout=0.1_
)[[source]](_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer)

    

TransformerDecoderLayer由自经办人，多头经办人及前馈网络。该标准解码器层是基于纸“注意是所有你需要”。阿希什瓦斯瓦尼，诺姆Shazeer，尼基帕尔马雅各布Uszkoreit，Llion琼斯，艾Ñ戈麦斯，卢卡斯凯泽，和Illia
Polosukhin。 2017注意力是你所需要的。在神经信息处理系统的进步，6000-6010页。用户可以修改或应用程序中以不同的方式实现。

Parameters

    

  * **d_model** – the number of expected features in the input (required).

  * **nhead** – the number of heads in the multiheadattention models (required).

  * **dim_feedforward** – the dimension of the feedforward network model (default=2048).

  * **dropout** – the dropout value (default=0.1).

Examples::

    
    
    
    >>> decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
    

`forward`( _tgt_ , _memory_ , _tgt_mask=None_ , _memory_mask=None_ ,
_tgt_key_padding_mask=None_ , _memory_key_padding_mask=None_
)[[source]](_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer.forward)

    

通过解码器层传递的输入（和掩模）。

Parameters

    

  * **TGT** \- 的顺序向解码器层（必需）。

  * **memory** – the sequnce from the last layer of the encoder (required).

  * **tgt_mask** – the mask for the tgt sequence (optional).

  * **memory_mask** – the mask for the memory sequence (optional).

  * **tgt_key_padding_mask** – the mask for the tgt keys per batch (optional).

  * **memory_key_padding_mask** – the mask for the memory keys per batch (optional).

Shape:

    

see the docs in Transformer class.

## 线性层

### 身份

_class_`torch.nn.``Identity`( _*args_ , _**kwargs_
)[[source]](_modules/torch/nn/modules/linear.html#Identity)

    

占位符的身份操作符是参数不敏感。

Parameters

    

  * **ARGS** \- 任何参数（未使用）

  * **kwargs** \- 任何关键字参数（未使用）

Examples:

    
    
    >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
    >>> input = torch.randn(128, 20)
    >>> output = m(input)
    >>> print(output.size())
    torch.Size([128, 20])
    

### 线性

_class_`torch.nn.``Linear`( _in_features_ , _out_features_ , _bias=True_
)[[source]](_modules/torch/nn/modules/linear.html#Linear)

    

适用的线性变换，以将输入数据： Y  =  × A  T  \+  b  Y = XA ^ T + b  Y  =  × A  T  \+  b

Parameters

    

  * **in_features** \- 每个输入样本的大小

  * **out_features** \- 每个输出样本的大小

  * **偏压** \- 如果设置为`假 `，该层不会学添加剂偏压。默认值：`真 `

Shape:

    

  * 输入： （ N  ， *  ， H  i的 n的 ） （N，*，H_ {在}） （ N  ， *  ， H  i的 n的 ） 其中 *  *  *  是指任何数量的附加维度和 H  i的 n的 =  in_features  H_ {IN} = \文本{在\ _features}  H  i的 n的 =  in_features 

  * 输出： （ N  ， *  ， H  O  U  T  ） （N，*，H_ {出}） （ N  ， *  ， H  O  U  T  ）  其中除了最后尺寸是相同的形状的输入，并 H  O  U  T  =  out_features  H_ {出} = \文本{出\ _features}  H  O  U  T  =  out_features  。

Variables

    

  * **〜Linear.weight** \- 形状 （ out_features的模块[的可学习权重HTG11 ] ， in_features  ） （\文本{出\ _features}，\文本{在\ _features}） （ out_features  ， in_features  ） 。的值是从 初始化U  （ \-  K  ， K  ） \ mathcal【U}（ - \ SQRT { ķ}，\ SQRT {K}） U  （  \-  K  ， K  ） ，其中 K  =  1  in_features  K = \压裂{1} {\文本{在\ _features}}  K  =  in_features  1 [HT G237] 

  * **〜Linear.bias** \- 形状 （ out_features的模块[的可学习偏压HTG11 ] ） （\文本{出\ _features}） （ out_features  ） 。如果`偏压 `是`真 `时，值被初始化从 U  （ \-  K  ， ķ  ） \ mathcal【U}（ - \ SQRT {K}，\ SQRT {K}） U  （ \-  K  ， [HT G132] K  ） 其中 K  =  1  in_features  K = \压裂{1} {\文本{在\ _features}}  K  =  in_features  1 

Examples:

    
    
    >>> m = nn.Linear(20, 30)
    >>> input = torch.randn(128, 20)
    >>> output = m(input)
    >>> print(output.size())
    torch.Size([128, 30])
    

### 双线性

_class_`torch.nn.``Bilinear`( _in1_features_ , _in2_features_ , _out_features_
, _bias=True_ )[[source]](_modules/torch/nn/modules/linear.html#Bilinear)

    

适用双线性变换到传入数据： Y  =  × 1  A  × 2  \+  b  Y = X_1甲X_2 + b  Y  =  × 1  A  × 2  \+
b

Parameters

    

  * **in1_features** \- 每个第一输入样本的大小

  * **in2_features** \- 每个第二输入样本的大小

  * **out_features** – size of each output sample

  * **偏压** \- 如果设置为False，该层不会学添加剂偏压。默认值：`真 `

Shape:

    

  * 输入1： （ N  ， *  ， H  i的 n的 1  ） （N，*，H_ {IN1}） （ N  ， *  ， H  i的 n的 1  ）  其中 H  一世 n的 1  =  in1_features  H_ {IN1} = \文本{IN1 \ _features}  H  i的 n的 1  =  in1_features  和 *  *  * [HT G197]  是指任何数量的附加的维度。但所有的输入的最后一个维度应该是相同的。

  * 输入2： （ N  ， *  ， H  i的 n的 2  ） （N，*，H_ {平方英寸}） （ N  ， *  ， H  i的 n的 2  ）  其中 H  一世 n的 2  =  in2_features  H_ {平方英寸} = \文本{平方英寸\ _features}  H  i的 n的 2  =  in2_features  。

  * 输出： （ N  ， *  ， H  O  U  T  ） （N，*，H_ {出}） （ N  ， *  ， H  O  U  T  ）  其中 H  Ø  U  T  =  out_features  H_ {出} = \文本{出\ _features}  H  O  U  T  =  out_features  和所有，但最后一个维度是相同的形状的输入。

Variables

    

  * **〜Bilinear.weight** \- 形状 （ out_features的模块[的可学习权重HTG11 ] ， in1_features  ， in2_features  ） （\文本{出\ _features} ，\文本{IN1 \ _features}，\文本{平方英寸\ _features}） （ out_features  ， in1_features  ， in2_features  ） 。的值是从 初始化U  （ \-  K  ， K  ） \ mathcal【U}（ - \ SQRT { ķ}，\ SQRT {K}） U  （  \-  K  ， K  [HTG1 55]  ） ，其中 ķ  =  1  in1_features  K = \压裂{1} {\文本{IN1 \ _features }}  K  =  in1_features  1  ​​ 

  * **〜Bilinear.bias** \- 形状 （ out_features的模块[的可学习偏压HTG11 ] ） （\文本{出\ _features}） （ out_features  ） 。如果`偏压 `是`真 `时，值被初始化从 U  （ \-  K  ， ķ  ） \ mathcal【U}（ - \ SQRT {K}，\ SQRT {K}） U  （ \-  K  ， [HT G132] K  ） ，其中 K  =  1  in1_features  K = \压裂{1} {\文本{IN1 \ _features}}  K  =  in1_features  [HTG22 4]  1 

Examples:

    
    
    >>> m = nn.Bilinear(20, 30, 40)
    >>> input1 = torch.randn(128, 20)
    >>> input2 = torch.randn(128, 30)
    >>> output = m(input1, input2)
    >>> print(output.size())
    torch.Size([128, 40])
    

## 漏失层

### 降

_class_`torch.nn.``Dropout`( _p=0.5_ , _inplace=False_
)[[source]](_modules/torch/nn/modules/dropout.html#Dropout)

    

在训练期间，随机归零一些输入张量与概率`P`使用样品从贝努利分布元件。每个通道将独立于每前行调用清零。

这已被证明是用于正则化和防止神经元的共适应通过防止特征检测器的互相适应的文件[提高神经网络中描述的有效的技术。](https://arxiv.org/abs/1207.0580)

此外，输出由一个因素 1  1  [HTG12缩放] -  p  \压裂{1} {1-p}  1  \-  p  1
[HTG83训练期间。这意味着，在评估期间模块简单计算标识功能。

Parameters

    

  * **P** \- 元素的概率将被归零。默认值：0.5

  * **就地** \- 如果设置为`真 `，会做此操作就地。默认值：`假 `

Shape:

    

  * 输入： （ *  ） （*）  （ *  ） 。输入可以是任何形状的

  * 输出： （ *  ） （*）  （ *  ） 。输出是相同的形状作为输入

Examples:

    
    
    >>> m = nn.Dropout(p=0.2)
    >>> input = torch.randn(20, 16)
    >>> output = m(input)
    

###  Dropout2d 

_class_`torch.nn.``Dropout2d`( _p=0.5_ , _inplace=False_
)[[source]](_modules/torch/nn/modules/dropout.html#Dropout2d)

    

随机零出整个信道（信道是2D特征映射，例如， [HTG6：J  [HTG9：J  [HTG18：J  的第信道 i的 i的 i的 在成批输入第样品是二维张量
输入 [ i的 ， [HTG62：J  \文本{输入} [I，J]  输入 [  i的 ， [HTG88：J  [H TG91]
）。每个信道将与使用的样品从一个伯努利分布概率`P`独立地置零在每一个前向呼叫。

通常情况下，输入来自`nn.Conv2d`模块。

正如在论文[高效对象定位使用卷积网络](http://arxiv.org/abs/1411.4280)，如果特征映射内的相邻像素是强相关的描述（如通常在早期卷积层的情况下），那么独立同分布辍学不会正规化的激活和否则将只是导致一个有效的学习速度下降。

在这种情况下，`nn.Dropout2d（） `将有利于促进功能的地图之间的独立性，并应改为使用。

Parameters

    

  * **P** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 的元素的概率是零-ED。

  * **就地** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果设定为`真 `，会做就地这种操作

Shape:

    

  * Input: (N,C,H,W)(N, C, H, W)(N,C,H,W)

  * Output: (N,C,H,W)(N, C, H, W)(N,C,H,W) (same shape as input)

Examples:

    
    
    >>> m = nn.Dropout2d(p=0.2)
    >>> input = torch.randn(20, 16, 32, 32)
    >>> output = m(input)
    

###  Dropout3d 

_class_`torch.nn.``Dropout3d`( _p=0.5_ , _inplace=False_
)[[source]](_modules/torch/nn/modules/dropout.html#Dropout3d)

    

随机零出整个信道（信道是3D特征地图，例如， [HTG6：J  [HTG9：J  [HTG18：J  的第信道 i的 i的 i的 在成批输入第样品是三维张量
输入 [ i的 ， [HTG62：J  \文本{输入} [I，J]  输入 [  i的 ， [HTG88：J  [H TG91]
）。每个信道将与使用的样品从一个伯努利分布概率`P`独立地置零在每一个前向呼叫。

通常情况下，输入来自`nn.Conv3d`模块。

As described in the paper [Efficient Object Localization Using Convolutional
Networks](http://arxiv.org/abs/1411.4280) , if adjacent pixels within feature
maps are strongly correlated (as is normally the case in early convolution
layers) then i.i.d. dropout will not regularize the activations and will
otherwise just result in an effective learning rate decrease.

在这种情况下，`nn.Dropout3d（） `将有利于促进功能的地图之间的独立性，并应改为使用。

Parameters

    

  * **P** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 的元素的概率将被归零。

  * **inplace** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If set to `True`, will do this operation in-place

Shape:

    

  * Input: (N,C,D,H,W)(N, C, D, H, W)(N,C,D,H,W)

  * Output: (N,C,D,H,W)(N, C, D, H, W)(N,C,D,H,W) (same shape as input)

Examples:

    
    
    >>> m = nn.Dropout3d(p=0.2)
    >>> input = torch.randn(20, 16, 4, 32, 32)
    >>> output = m(input)
    

###  AlphaDropout 

_class_`torch.nn.``AlphaDropout`( _p=0.5_ , _inplace=False_
)[[source]](_modules/torch/nn/modules/dropout.html#AlphaDropout)

    

适用阿尔法差超过输入。

阿尔法差是一种差的维持自正火财产。对于具有零均值和单位标准差的输入，阿尔法差的输出保持原始平均值和输入的标准偏差。阿尔法降去手在手与活化九色鹿函数，这确保了输出具有零均值和单位标准偏差。

在训练期间，它随机掩模一些输入张量与概率 _使用样品从伯努利分布p_ 的元素。到屏蔽元件被随机化在每个前向呼叫，并缩放和移动以保持零均值和单位标准偏差。

在评估过程中的模块简单地计算一个身份功能。

More details can be found in the paper [Self-Normalizing Neural
Networks](https://arxiv.org/abs/1706.02515) .

Parameters

    

  * **P** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 的元素的概率被丢弃。默认值：0.5

  * **inplace** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If set to `True`, will do this operation in-place

Shape:

    

  * Input: (∗)(*)(∗) . Input can be of any shape

  * Output: (∗)(*)(∗) . Output is of the same shape as input

Examples:

    
    
    >>> m = nn.AlphaDropout(p=0.2)
    >>> input = torch.randn(20, 16)
    >>> output = m(input)
    

## 稀疏层

### 嵌入

_class_`torch.nn.``Embedding`( _num_embeddings_ , _embedding_dim_ ,
_padding_idx=None_ , _max_norm=None_ , _norm_type=2.0_ ,
_scale_grad_by_freq=False_ , _sparse=False_ , __weight=None_
)[[source]](_modules/torch/nn/modules/sparse.html#Embedding)

    

存储一个固定字典和尺寸的嵌入简单的查找表。

该模块通常用于存储字的嵌入，并使用索引进行检索。输入到模块是指数列表，并且输出是对应的字的嵌入。

Parameters

    

  * **num_embeddings** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 的嵌入的词典的大小

  * **embedding_dim** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 各嵌入矢量的大小

  * **padding_idx** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 如果给定的，垫在与嵌入矢量输出`padding_idx`（初始化为零）每当遇到的索引。

  * **max_norm** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 如果给定的，具有范数大于各嵌入矢量`max_norm`被重新归一化，以具有规范`max_norm`。

  * **norm_type** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 的p范数的p来计算用于`max_norm`选项。默认`2`。

  * **scale_grad_by_freq** （ _布尔_ _，_ _可选_ ） - 如果给出，这将通过的话频率在微型逆扩展梯度批量。默认的`假 [HTG11。`

  * **稀疏** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，梯度WRT `重量 `矩阵将是稀疏张量。请参阅有关稀疏梯度更多细节说明。

Variables

    

**〜Embedding.weight** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） -
形状的模块的可学习权重（num_embeddings，embedding_dim）从 初始化 N  （ 0  ， 1  ） \ mathcal
{N}（0，1） N  （ 0  ， 1  ）

Shape:

    

  * 输入： （ *  ） （*）  （ *  ） ，任意形状的LongTensor包含的索引来提取

  * 输出： （ *  ， H  ） （*，H） （ *  ， H  ） ，其中 * 是输入形状和 H  =  embedding_dim  H = \文本{嵌入\ _dim}  H  =  embedding_dim 

Note

请记住，只有优化的数量有限支持稀疏梯度：目前它的`optim.SGD`（ CUDA 和 CPU ），`optim.SparseAdam`（
CUDA 和 CPU ）和`optim.Adagrad`（ CPU ）

Note

与`padding_idx`组，在`的嵌入矢量padding_idx
`被初始化为全零。然而，注意，这载体可随后使用定制初始化方法来修饰，例如，从而改变用于垫的输出矢量。从 `嵌入 `本矢量梯度始终为零。

Examples:

    
    
    >>> # an Embedding module containing 10 tensors of size 3
    >>> embedding = nn.Embedding(10, 3)
    >>> # a batch of 2 samples of 4 indices each
    >>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
    >>> embedding(input)
    tensor([[[-0.0251, -1.6902,  0.7172],
             [-0.6431,  0.0748,  0.6969],
             [ 1.4970,  1.3448, -0.9685],
             [-0.3677, -2.7265, -0.1685]],
    
            [[ 1.4970,  1.3448, -0.9685],
             [ 0.4362, -0.4004,  0.9400],
             [-0.6431,  0.0748,  0.6969],
             [ 0.9124, -2.3616,  1.1151]]])
    
    
    >>> # example with padding_idx
    >>> embedding = nn.Embedding(10, 3, padding_idx=0)
    >>> input = torch.LongTensor([[0,2,0,5]])
    >>> embedding(input)
    tensor([[[ 0.0000,  0.0000,  0.0000],
             [ 0.1535, -2.0309,  0.9315],
             [ 0.0000,  0.0000,  0.0000],
             [-0.1655,  0.9897,  0.0635]]])
    

_classmethod_`from_pretrained`( _embeddings_ , _freeze=True_ ,
_padding_idx=None_ , _max_norm=None_ , _norm_type=2.0_ ,
_scale_grad_by_freq=False_ , _sparse=False_
)[[source]](_modules/torch/nn/modules/sparse.html#Embedding.from_pretrained)

    

创建一个从给定的2维FloatTensor嵌入实例。

Parameters

    

  * **的嵌入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - FloatTensor含有用于嵌入权重。第一维度被传递给嵌入为`num_embeddings`，第二为`embedding_dim`。

  * **冻结** （ _布尔_ _，_ _可选_ ） - 如果`真 `时，张量不在学习过程中得到更新。等价于`embedding.weight.requires_grad  =  假 `。默认值：`真 `

  * **padding_idx** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 参见模块的初始化文档。

  * **max_norm** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 参见模块的初始化文档。

  * **norm_type** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 参见模块的初始化文档。默认`2`。

  * **scale_grad_by_freq** （ _布尔_ _，_ _可选_ ） - 参见模块的初始化文档。默认的`假 [HTG11。`

  * **稀疏** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 参见模块的初始化文档。

Examples:

    
    
    >>> # FloatTensor containing pretrained weights
    >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
    >>> embedding = nn.Embedding.from_pretrained(weight)
    >>> # Get embeddings for index 1
    >>> input = torch.LongTensor([1])
    >>> embedding(input)
    tensor([[ 4.0000,  5.1000,  6.3000]])
    

###  EmbeddingBag 

_class_`torch.nn.``EmbeddingBag`( _num_embeddings_ , _embedding_dim_ ,
_max_norm=None_ , _norm_type=2.0_ , _scale_grad_by_freq=False_ , _mode='mean'_
, _sparse=False_ , __weight=None_
)[[source]](_modules/torch/nn/modules/sparse.html#EmbeddingBag)

    

计算的的嵌入的“袋”，和或装置没有实例的中间的嵌入。

恒定长度的袋和无`per_sample_weights`，这个类

>   * 与`模式= “总和” `等于 ``接着`炬嵌入 `。总和（暗= 0）

>

>   * 接着`炬与`模式= “意指” `等于 `嵌入 `。平均（暗淡= 0） `

>

>   * 与`模式= “最大” `等于 ``接着`炬嵌入 `。最大（暗= 0） 。

>

>

然而， `EmbeddingBag`做花费更多时间和存储器不是使用这些操作中的一个链高效。

EmbeddingBag还支持每个样品重量作为参数传递给直传。这个缩放嵌入的输出作为由`模式 `中指定执行的加权还原之前。如果`
per_sample_weights``通过，仅支持`模式 `是`“总和” `，其中根据`per_sample_weights
`计算的加权和。

Parameters

    

  * **num_embeddings** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – size of the dictionary of embeddings

  * **embedding_dim** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – the size of each embedding vector

  * **max_norm** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – If given, each embedding vector with norm larger than `max_norm`is renormalized to have norm `max_norm`.

  * **norm_type** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – The p of the p-norm to compute for the `max_norm`option. Default `2`.

  * **scale_grad_by_freq** （ _布尔_ _，_ _可选_ ） - 如果给定的，这将通过的话频率在微型逆扩展梯度批量。默认的`假 [HTG11。注意：不支持此选项时`模式= “MAX” [HTG15。``

  * **模式** （ _串_ _，_ _可选_ ） - `“总和” `，`“的意思是” `或`“最大” `。指定要降低袋的方式。 `“总和” `计算的加权和，以`per_sample_weights`考虑在内。 `“的意思是” `计算在袋中的值的平均值，`“最大” `计算在每个袋中的最大值。默认值：`“的意思是” `

  * **稀疏** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，梯度WRT `重量 `矩阵将是稀疏张量。请参阅有关稀疏梯度更多细节说明。注意：不支持此选项时`模式= “MAX” [HTG21。`

Variables

    

**EmbeddingBag.weight** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")）〜
- （num_embeddings，embedding_dim）从初始化形状的模块的可学习权重 N  （ 0  ， 1  ） \ mathcal
{N}（0，1） N  （ 0  ， 1  ） 。

Inputs: `input`(LongTensor), `offsets`(LongTensor, optional), and

    

`per_index_weights`（张量，可选）

  * 如果`输入 `是形状的2D （B，N）

它会被视为`B`袋（序列）各固定长度的`N`，这将返回`B`值在某种程度上取决于`模式 `聚合。 `偏移 `被忽略，并且需要为`无
`在这种情况下。

  * 如果`输入 `是形状的1D （N）

它会被视为多个袋（序列）的级联。 `偏移 `需要为含有`输入 `每个袋子的起始索引位置的1D张量。因此，对于`偏移 `形状的（B），`输入
`将被视为具有`B`袋。空袋通过零填充（即，具有长度为0的）将已经返回向量。

per_sample_weights (Tensor, optional): a tensor of float / double weights, or
None

    

以指示所有的权重应取为`1`。如果已指定，`per_sample_weights`必须具有完全相同的形状作为输入，被视为具有相同的`偏移
`，如果这些都没有`无 `。仅支持`模式= '总和' `。

输出形状：（B，embedding_dim）

Examples:

    
    
    >>> # an Embedding module containing 10 tensors of size 3
    >>> embedding_sum = nn.EmbeddingBag(10, 3, mode='sum')
    >>> # a batch of 2 samples of 4 indices each
    >>> input = torch.LongTensor([1,2,4,5,4,3,2,9])
    >>> offsets = torch.LongTensor([0,4])
    >>> embedding_sum(input, offsets)
    tensor([[-0.8861, -5.4350, -0.0523],
            [ 1.1306, -2.5798, -1.0044]])
    

_classmethod_`from_pretrained`( _embeddings_ , _freeze=True_ , _max_norm=None_
, _norm_type=2.0_ , _scale_grad_by_freq=False_ , _mode='mean'_ ,
_sparse=False_
)[[source]](_modules/torch/nn/modules/sparse.html#EmbeddingBag.from_pretrained)

    

从给定的2维FloatTensor创建EmbeddingBag实例。

Parameters

    

  * **的嵌入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - FloatTensor含有用于EmbeddingBag权重。第一维度被传递给EmbeddingBag为“num_embeddings”，第二为“embedding_dim”。

  * **冻结** （ _布尔_ _，_ _可选_ ） - 如果`真 `时，张量不在学习过程中得到更新。等价于`embeddingbag.weight.requires_grad  =  假 `。默认值：`真 `

  * **max_norm** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 参见模块的初始化文档。默认值：`无 `

  * **norm_type** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – See module initialization documentation. Default `2`.

  * **scale_grad_by_freq** ( _boolean_ _,_ _optional_ ) – See module initialization documentation. Default `False`.

  * **模式** （ _串_ _，_ _可选_ ） - 参见模块的初始化文档。默认值：`“的意思是” `

  * **稀疏** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 参见模块的初始化文档。默认值：`假 [HTG13。`

Examples:

    
    
    >>> # FloatTensor containing pretrained weights
    >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
    >>> embeddingbag = nn.EmbeddingBag.from_pretrained(weight)
    >>> # Get embeddings for index 1
    >>> input = torch.LongTensor([[1, 0]])
    >>> embeddingbag(input)
    tensor([[ 2.5000,  3.7000,  4.6500]])
    

## 距离函数

### 余弦相似性

_class_`torch.nn.``CosineSimilarity`( _dim=1_ , _eps=1e-08_
)[[source]](_modules/torch/nn/modules/distance.html#CosineSimilarity)

    

返回之间的余弦相似度 × 1  X_1  × 1  和 × 2  X_2  × 2  ，沿着昏暗计算。

similarity=x1⋅x2max⁡(∥x1∥2⋅∥x2∥2,ϵ).\text{similarity} = \dfrac{x_1 \cdot
x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}.
similarity=max(∥x1​∥2​⋅∥x2​∥2​,ϵ)x1​⋅x2​​.

Parameters

    

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 维其中余弦相似度进行计算。默认值：1

  * **EPS** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 小值由零避免分裂。默认值：1E-8

Shape:

    

  * 输入1： （ *  1  d  ， *  2  ） （\ ast_1，d，\ ast_2） （ *  1  ， d  ， *  2  ） 其中d是在位置暗淡

  * 输入2： （ *  1  d  ， *  2  ） （\ ast_1，d，\ ast_2） （ *  1  ， d  ， *  2  ） ，相同形状的输入1

  * 输出： （ *  1  *  2  ） （\ ast_1，\ ast_2） （ *  1  ， *  2  ）

Examples::

    
    
    
    >>> input1 = torch.randn(100, 128)
    >>> input2 = torch.randn(100, 128)
    >>> cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    >>> output = cos(input1, input2)
    

###  PairwiseDistance 

_class_`torch.nn.``PairwiseDistance`( _p=2.0_ , _eps=1e-06_ , _keepdim=False_
)[[source]](_modules/torch/nn/modules/distance.html#PairwiseDistance)

    

计算之间的载体 [HTG7】V  1  [HTG13分批成对距离] V_1  [HTG23】v  1  ， v  2  V_2  [HTG77 】v  2
使用 的p范数：

∥x∥p=(∑i=1n∣xi∣p)1/p.\Vert x \Vert _p = \left( \sum_{i=1}^n \vert x_i \vert ^
p \right) ^ {1/p}. ∥x∥p​=(i=1∑n​∣xi​∣p)1/p.

Parameters

    

  * **P** （ _真实_ ） - 范数度。默认值：2

  * **EPS** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 小值由零避免分裂。默认值：1E-6

  * **keepdim** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 确定是否要保持向量维度。默认值：false

Shape:

    

  * 输入1： （ N  ， d  ） （N，d） （ N  ， d  ） 其中 d =载体尺寸

  * 输入2： （ N  ， d  ） （N，d） （ N  ， d  ） ，相同形状的输入1

  * 输出： （ N  ） （N）  （ N  ） 。如果`keepdim`是`真 `，然后 （ N  ， 1  ） （N，1） （ N  ， 1  ） 。

Examples::

    
    
    
    >>> pdist = nn.PairwiseDistance(p=2)
    >>> input1 = torch.randn(100, 128)
    >>> input2 = torch.randn(100, 128)
    >>> output = pdist(input1, input2)
    

## 损失函数

###  L1Loss 

_class_`torch.nn.``L1Loss`( _size_average=None_ , _reduce=None_ ,
_reduction='mean'_ )[[source]](_modules/torch/nn/modules/loss.html#L1Loss)

    

创建在输入 × ×各元件之间测量的平均绝对误差（MAE）的标准 × 和目标 Y  Y  Y  。

未还原的（即，具有`还原 `设置为 `'无'）损耗可以被描述为：`

ℓ(x,y)=L={l1,…,lN}⊤,ln=∣xn−yn∣,\ell(x, y) = L = \\{l_1,\dots,l_N\\}^\top,
\quad l_n = \left| x_n - y_n \right|, ℓ(x,y)=L={l1​,…,lN​}⊤,ln​=∣xn​−yn​∣,

其中 N  N  N  是批量大小。如果`还原 `不是`'无' `（默认`'平均' `），然后：

ℓ(x,y)={mean⁡(L),if reduction=’mean’;sum⁡(L),if reduction=’sum’.\ell(x, y) =
\begin{cases} \operatorname{mean}(L), & \text{if reduction} =
\text{'mean';}\\\ \operatorname{sum}(L), & \text{if reduction} = \text{'sum'.}
\end{cases} ℓ(x,y)={mean(L),sum(L),​if reduction=’mean’;if reduction=’sum’.​

× × × 和 Y  Y  Y  是任意形状的张量，总的 n的 n的 n的 每个元件。

求和操作仍然工作在所有元素，并除以 n的 n的 n的 。

除以 n的 n的 n的 可避免如果一个集`还原 =  '和' `。

Parameters

    

  * **size_average** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 已过时（见`还原 `）。默认情况下，损失平均超过批中每个元素的损失。请注意，对于一些损失，有每个样品的多个元素。如果该字段`size_average`被设定为`假 `时，损失代替求和每个minibatch。当减少是`假 `忽略。默认值：`真 `

  * **减少** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 已过时（见`还原 `）。默认情况下，损耗进行平均或求和观测为视`size_average`每个minibatch。当`减少 `是`假 `，返回每批元件的损耗，而不是并忽略`size_average`。默认值：`真 `

  * **还原** （ _串_ _，_ _可选_ ） - 指定还原应用到输出：`'无' `| `'的意思是' `| `'和' `。 `'无' `：不降低将被应用，`'意味' `：将输出的总和将通过的数量来划分在输出中，`'和' `元素：输出将被累加。注意：`size_average`和`减少 `处于被淘汰，并且在此同时，指定是这两个参数的个数将覆盖`还原 `。默认值：`'平均' `

Shape:

    

  * 输入： （ N  ， *  ） （N，*） （ N  ， *  ） 其中 *  *  *  手段，任意数量的附加维度的

  * 目标： （ N  ， *  ） （N，*） （ N  ， *  ） ，相同形状的输入

  * 输出：标量。如果`还原 `是`'无' `，然后 （ N  ， *  ） （N，*） （ N  ， *  ） ，相同形状的输入

Examples:

    
    
    >>> loss = nn.L1Loss()
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> target = torch.randn(3, 5)
    >>> output = loss(input, target)
    >>> output.backward()
    

###  MSELoss 

_class_`torch.nn.``MSELoss`( _size_average=None_ , _reduce=None_ ,
_reduction='mean'_ )[[source]](_modules/torch/nn/modules/loss.html#MSELoss)

    

创建在输入 × [HTG9各元件之间测量均方误差（平方L2范数）的标准]×  × 和目标 Y  Y  Y  。

The unreduced (i.e. with `reduction`set to `'none'`) loss can be described
as:

ℓ(x,y)=L={l1,…,lN}⊤,ln=(xn−yn)2,\ell(x, y) = L = \\{l_1,\dots,l_N\\}^\top,
\quad l_n = \left( x_n - y_n \right)^2, ℓ(x,y)=L={l1​,…,lN​}⊤,ln​=(xn​−yn​)2,

where NNN is the batch size. If `reduction`is not `'none'`(default
`'mean'`), then:

ℓ(x,y)={mean⁡(L),if reduction=’mean’;sum⁡(L),if reduction=’sum’.\ell(x, y) =
\begin{cases} \operatorname{mean}(L), & \text{if reduction} =
\text{'mean';}\\\ \operatorname{sum}(L), & \text{if reduction} = \text{'sum'.}
\end{cases} ℓ(x,y)={mean(L),sum(L),​if reduction=’mean’;if reduction=’sum’.​

xxx and yyy are tensors of arbitrary shapes with a total of nnn elements each.

The sum operation still operates over all the elements, and divides by nnn .

The division by nnn can be avoided if one sets `reduction = 'sum'`.

Parameters

    

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where ∗*∗ means, any number of additional dimensions

  * Target: (N,∗)(N, *)(N,∗) , same shape as the input

Examples:

    
    
    >>> loss = nn.MSELoss()
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> target = torch.randn(3, 5)
    >>> output = loss(input, target)
    >>> output.backward()
    

###  CrossEntropyLoss 

_class_`torch.nn.``CrossEntropyLoss`( _weight=None_ , _size_average=None_ ,
_ignore_index=-100_ , _reduce=None_ , _reduction='mean'_
)[[source]](_modules/torch/nn/modules/loss.html#CrossEntropyLoss)

    

该标准结合`nn.LogSoftmax（） `和`nn.NLLLoss（） `在一个单独的类。

训练与 C 类分类问题时是有用的。如果提供的话，可选的参数`重量 `应该是一个1D 张量重量分配到每个类。当你有一个不平衡的训练集这是特别有用。

的输入预计包含生的，非归一化的分数为每个类。

输入必须是尺寸的张量为 （ M  I  n的 i的 b  一 T  C  [HTG26 1 H ， C  ） （minibatch，C） （ M  i的
n的 i的 b  一 T  C  H  ， C  ） 或 （ M  i的 n的 i的 b  一 T  C  H [HTG1 01] ， C  ， d  1
， d  2  ， 。  。  。  ， d  K  ） （minibatch，C，D_1， D_2，...，d_K） （ M  i的 n的 i的 b  一
T  C  H  ， C  ， d  1  ， d  [H TG223] 2  ， 。  。  。  ， d  K  ​​  ） 与 K  ≥ 1  ķ\
GEQ 1  ķ  ≥ 1  为 K 维情况下（后述）。

该标准需要一个类指数在范围 [ 0  ， C  \-  1  [0，C-1]  [ 0  ， C  \-  1  作为用于大小 minibatch
[的一维张量的每个值的目标;]如果 ignore_index 被指定时，该标准也接受这个类索引（此索引可以不一定是在类范围内）。

损失可以被描述为：

loss(x,class)=−log⁡(exp⁡(x[class])∑jexp⁡(x[j]))=−x[class]+log⁡(∑jexp⁡(x[j]))\text{loss}(x,
class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right) =
-x[class] + \log\left(\sum_j \exp(x[j])\right)
loss(x,class)=−log(∑j​exp(x[j])exp(x[class])​)=−x[class]+log(j∑​exp(x[j]))

或在`重量 `参数的情况下被指定的：

loss(x,class)=weight[class](−x[class]+log⁡(∑jexp⁡(x[j])))\text{loss}(x, class)
= weight[class] \left(-x[class] + \log\left(\sum_j \exp(x[j])\right)\right)
loss(x,class)=weight[class](−x[class]+log(j∑​exp(x[j])))

这些损失是整个观测平均每个minibatch。

也可以用于更高的尺寸的输入，如2D图像，通过提供尺寸 （ M [的输入HTG9]  i的 n的 i的 b  一 T  C  H  ， C  ， d  1
， d  2  ， 。 。 ， d  K  ） （minibatch，C，D_1 ，D_2，...，d_K） （ M  I  n的 i的 b  一 T  C
[HTG92 1 H  ， C  ， d  1  ， d  2  ， 。  。  。  ， d  K  ） 与 K  ≥ 1  ķ\ GEQ 1  ķ  ≥
1  ，其中 K  K  ​​ K  是维数，和适当形状的目标（见下文）。

Parameters

    

  * **重量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 给每个类的手动重新缩放权重。如果给定的，必须是尺寸℃的张量

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **ignore_index** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 指定将被忽略，并且不向目标值输入梯度。当`size_average`是`真 `，损失平均超过非忽略的目标。

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

Shape:

    

  * 输入： （ N  ， C  ） （N，C） （ N  ， C  ） 其中 C =号码类或 （ N  ， ç  ， d  1  ， d  2  ， 。 。 。 ， d  K  ） （N，C，D_1，D_2，...，d_K） （ N  ， C  ， d  1  ， d  2  ， 。  。  。  ， d  K  ） 与 K  ≥ 1  ķ\ GEQ 1  ķ  ≥ 1  如的情况下K 维损失。

  * 目标： （ N  ） （N）  （ N  ） 其中每个值是 0  ≤ 目标 [ i的 ≤ C  \-  1  0 \当量\文本{目标} [I] \当量C-1  0  ≤ 目标 [  i的 ≤ C  \-  1  或 （ N  ， d  1  ， d  2  ， 。  。  。  ， d  K  ） （N，D_1，D_2， ...，d_K） （ N  ， d  1  ， d  2  ， 。 [HTG246  。  ， d  K  ​​  ） 与 K  ≥ 1  ķ\ GEQ 1  ķ  ≥ 1  在K维损失的情况下。

  * 输出：标量。如果`还原 `是`'无' `，然后相同的尺寸为目标： （ N  ） （N） （ N  ） 或 （ N  ， d  1  ， d  2  ， 。 。  。 ， d  K  ） （N，D_1，D_2，...，d_K） （ N  ，[HT G99]  d  1  ， d  2  ， 。  。  。  ， d  K  ） 与 K  ≥ 1  ķ\ GEQ 1  ķ  ≥ 1  在K维损失的情况下。

Examples:

    
    
    >>> loss = nn.CrossEntropyLoss()
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> target = torch.empty(3, dtype=torch.long).random_(5)
    >>> output = loss(input, target)
    >>> output.backward()
    

###  CTCLoss 

_class_`torch.nn.``CTCLoss`( _blank=0_ , _reduction='mean'_ ,
_zero_infinity=False_
)[[source]](_modules/torch/nn/modules/loss.html#CTCLoss)

    

该联结颞分类损失。

计算连续的（不分段）的时间序列和靶序列之间的损耗。
CTCLoss总结以上输入的可能的对准目标的概率，产生其是可微分的相对于每个输入节点的损耗值。输入到目标的取向被假定为“多到一”，这限制了靶序列的长度，使得它必须是
≤ \当量 ≤ 输入长度。

Parameters

    

  * **空白** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 空白标签。默认 0  0  0  。

  * **还原** （ _串_ _，_ _可选_ ） - 指定还原应用到输出：`'无' `| `'的意思是' `| `'和' `。 `'无' `：不降低将被应用，`'意味' `：输出损耗将由目标长度，然后被划分平均超过该批次取。默认值：`'平均' `

  * **zero_infinity** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 是否为零无限损失和相关联的梯度。默认值：`假 `主要是当输入太短，无法对准目标出现无限损失。

Shape:

    

  * Log_probs：的张量大小 （ T  ， N  ， C  ） （T，N，C） （ T  ， N  ， C  ） ，其中 T  =  输入长度 T = \文本{输入长度}  T  =  输入长度 ， N  =  批量大小 N = \文本{批量大小}  N  =  批量大小 和 ç  =  的类（包括空白） C = \文本{的类（包括坯件）数}数 C  =  的类号码（包括空格） 。的输出的取对数概率（例如，用[ `获得torch.nn.functional.log_softmax（） `](nn.functional.html#torch.nn.functional.log_softmax "torch.nn.functional.log_softmax")）。

  * 目标：大小 （ N  ， S的张量 ） （N，S） （ N  ， S  ） 或 （ 总结 ⁡ （ target_lengths  ） ） （\ operatorname {总和}（\文本{目标\ _lengths}）） （ S  U  M  （ target_lengths  ） ） ，其中[H TG96]  N  =  批量大小 N = \文本{批量大小}  N  =  批量大小 和 S  =  最大目标长度，如果形状是 （ N  ， S  ） S = \文本{最大目标长度，如果形状为}（N，S） S  =  最大目标长度，如果形状是 （ N  ， S  ） 。它代表了靶序列。在靶序列中的每个元素是一个类的索引。和目标索引不能为空（缺省值= 0）。在 （ N  ， S  ） （N，S） （ N  ， S  ） 形式，目标被填充到最长的序列的长度，并堆叠。在 （ 总和 ⁡ （ target_lengths  ） ） （\ operatorname {总和}（\文本{目标\ _lengths}）） ​​  （ S  U  M  （ target_lengths  ） ） 的形式中，目标是假定为未填充的和1名维中串联。

  * Input_lengths：元组或的大小 （ N  ）张量 （N） （ N  ） ，其中 N  =  批次大小 N = \文本{批量大小}  N  =  批量大小 。它表示的输入长度（每一个都必须 ≤ T  \当量T  ≤ T  ）。和长度为每个序列，以实现该序列被填充到长度相等的假设下掩蔽指定。

  * Target_lengths：元组或的大小 （ N  ）张量 （N） （ N  ） ，其中 N  =  批次大小 N = \文本{批量大小}  N  =  批量大小 。它代表了目标的长度。长度对于每个序列，以实现该序列被填充到长度相等的假设下掩蔽指定。如果目标形状为 （ N  ， S  ） （N，S） （ N  ， S  ） ，target_lengths是有效的停止指数 S  n的 S_N  S  n的 [H TG167]  对于每个靶序列，从而使得`TARGET_N  =  目标[N，0：S_N]  [HTG177用于批处理中的每个目标。长度每一个都必须 ≤ S  \当量S  ≤ S  [HTG211如果目标被给定为一维张量是单个目标的级联，所述target_lengths必须加起来张量的总长度。`

  * 输出：标量。如果`还原 `是`'无' `，然后 （ N  ） （N） （ N  ） ，其中 N  =  批量大小 N = \文本{批量大小}  N  =  批量大小 。

Example:

    
    
    >>> T = 50      # Input sequence length
    >>> C = 20      # Number of classes (including blank)
    >>> N = 16      # Batch size
    >>> S = 30      # Target sequence length of longest target in batch
    >>> S_min = 10  # Minimum target length, for demonstration purposes
    >>>
    >>> # Initialize random batch of input vectors, for *size = (T,N,C)
    >>> input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
    >>>
    >>> # Initialize random batch of targets (0 = blank, 1:C = classes)
    >>> target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
    >>>
    >>> input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
    >>> target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
    >>> ctc_loss = nn.CTCLoss()
    >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
    >>> loss.backward()
    

Reference:

    

A. Graves等人：联结颞分类：与回归神经网络的标注未分段序列数据：[
https://www.cs.toronto.edu/~graves/icml_2006.pdf
](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

Note

为了使用CuDNN时，必须满足以下条件：HTG0] 目标 必须在级联格式，所有`input_lengths`必须 T 。  B  L  一 n的 ķ
=  0  空白= 0  b  L  一 n的 K  =  0  ，`target_lengths`≤ 256  \当量256  ≤ 2  5  6
，整数参数必须是D型细胞的`torch.int32`。

常规实现使用（多见于PyTorch） torch.long  D型。

Note

In some circumstances when using the CUDA backend with CuDNN, this operator
may select a nondeterministic algorithm to increase performance. If this is
undesirable, you can try to make the operation deterministic (potentially at a
performance cost) by setting `torch.backends.cudnn.deterministic = True`.
Please see the notes on [Reproducibility](notes/randomness.html) for
background.

###  NLLLoss 

_class_`torch.nn.``NLLLoss`( _weight=None_ , _size_average=None_ ,
_ignore_index=-100_ , _reduce=None_ , _reduction='mean'_
)[[source]](_modules/torch/nn/modules/loss.html#NLLLoss)

    

负对数似然的损失。重要的是要培养具有 C 类分类问题是有用的。

如果提供的话，可选的参数`重量 `应该是张量1D重量分配给每个类。当你有一个不平衡的训练集这是特别有用。

的输入通过前向呼叫给出预计包含每个类的对数概率。 输入必须是尺寸的张量为 （ M  I  n的 i的 b  一 T  C  [HTG28 1 H ， C
） （minibatch，C） （ M  i的 n的 i的 b  一 T  C  H  ， C  ） 或 （ M  i的 n的 i的 b  一 T  C
H  ， C  ， d  1  ， d  2  ， 。  。  。  ， d  K  ） （minibatch，C，D_1， D_2，...，d_K） （
M  i的 n的 i的 b  一 T  C  H  ， C  ， d  1  ， d  [H TG225] 2  ， 。  。  。  ， d  K  ​​
） 与 K  ≥ 1  ķ\ GEQ 1  ķ  ≥ 1  为 K 维情况下（后述）。

在神经网络中获取数的概率是很容易在你的网络的最后一层添加 LogSoftmax 层来实现。您可以使用 CrossEntropyLoss
相反，如果你不喜欢添加额外的层。

的目标，这一损失预计应该是在范围 [ 0 [一类索引HTG11] ， C  \-  1  [0，C-1]  [ 0  ， C  \-  1  ]  其中 C
=班数 ;如果 ignore_index 被指定时，这个损失也接受这个类别的索引（此索引可以不一定是在类范围内）。

The unreduced (i.e. with `reduction`set to `'none'`) loss can be described
as:

ℓ(x,y)=L={l1,…,lN}⊤,ln=−wynxn,yn,wc=weight[c]⋅1{c≠ignore_index},\ell(x, y) =
L = \\{l_1,\dots,l_N\\}^\top, \quad l_n = - w_{y_n} x_{n,y_n}, \quad w_{c} =
\text{weight}[c] \cdot \mathbb{1}\\{c \not= \text{ignore\\_index}\\},
ℓ(x,y)=L={l1​,…,lN​}⊤,ln​=−wyn​​xn,yn​​,wc​=weight[c]⋅1{c​=ignore_index},

其中 N  N  N  是批量大小。如果`还原 `不是`'无' `（默认`'平均' `），然后

ℓ(x,y)={∑n=1N1∑n=1Nwynln,if reduction=’mean’;∑n=1Nln,if
reduction=’sum’.\ell(x, y) = \begin{cases} \sum_{n=1}^N \frac{1}{\sum_{n=1}^N
w_{y_n}} l_n, & \text{if reduction} = \text{'mean';}\\\ \sum_{n=1}^N l_n, &
\text{if reduction} = \text{'sum'.} \end{cases}
ℓ(x,y)={∑n=1N​∑n=1N​wyn​​1​ln​,∑n=1N​ln​,​if reduction=’mean’;if
reduction=’sum’.​

也可以用于更高的尺寸的输入，如2D图像，通过提供尺寸 （ M [的输入HTG9]  i的 n的 i的 b  一 T  C  H  ， C  ， d  1
， d  2  ， 。 。 ， d  K  ） （minibatch，C，D_1 ，D_2，...，d_K） （ M  I  n的 i的 b  一 T  C
[HTG92 1 H  ， C  ， d  1  ， d  2  ， 。  。  。  ， d  K  ） 与 K  ≥ 1  ķ\ GEQ 1  ķ  ≥
1  ，其中 K  K  ​​ K  是维数，和适当形状的目标（见下文）。在图像的情况下，它计算每像素NLL损失。

Parameters

    

  * **重量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 给每个类的手动重新缩放权重。如果给定的，它必须是尺寸 C 的张量。否则，将被视为有，如果所有的人。

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **ignore_index** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 指定将被忽略，并且不向目标值输入梯度。当`size_average`是`真 `，损失平均超过非忽略的目标。

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

Shape:

    

  * Input: (N,C)(N, C)(N,C) where C = number of classes, or (N,C,d1,d2,...,dK)(N, C, d_1, d_2, ..., d_K)(N,C,d1​,d2​,...,dK​) with K≥1K \geq 1K≥1 in the case of K-dimensional loss.

  * Target: (N)(N)(N) where each value is 0≤targets[i]≤C−10 \leq \text{targets}[i] \leq C-10≤targets[i]≤C−1 , or (N,d1,d2,...,dK)(N, d_1, d_2, ..., d_K)(N,d1​,d2​,...,dK​) with K≥1K \geq 1K≥1 in the case of K-dimensional loss.

  * 输出：标量。如果`还原 `是`'无' `，然后相同的尺寸为目标： （ N  ） （N） （ N  ） 或 （ N  ， d  1  ， d  2  ， 。 。  。 ， d  K  ） （N，D_1，D_2，...，d_K） （ N  ，[HT G99]  d  1  ， d  2  ， 。  。  。  ， d  K  ） 与 K  ≥ 1  ķ\ GEQ 1  ķ  ≥ 1  在K维损失的情况下。

Examples:

    
    
    >>> m = nn.LogSoftmax(dim=1)
    >>> loss = nn.NLLLoss()
    >>> # input is of size N x C = 3 x 5
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> # each element in target has to have 0 <= value < C
    >>> target = torch.tensor([1, 0, 4])
    >>> output = loss(m(input), target)
    >>> output.backward()
    >>>
    >>>
    >>> # 2D loss example (used, for example, with image inputs)
    >>> N, C = 5, 4
    >>> loss = nn.NLLLoss()
    >>> # input is of size N x C x height x width
    >>> data = torch.randn(N, 16, 10, 10)
    >>> conv = nn.Conv2d(16, C, (3, 3))
    >>> m = nn.LogSoftmax(dim=1)
    >>> # each element in target has to have 0 <= value < C
    >>> target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
    >>> output = loss(m(conv(data)), target)
    >>> output.backward()
    

###  PoissonNLLLoss 

_class_`torch.nn.``PoissonNLLLoss`( _log_input=True_ , _full=False_ ,
_size_average=None_ , _eps=1e-08_ , _reduce=None_ , _reduction='mean'_
)[[source]](_modules/torch/nn/modules/loss.html#PoissonNLLLoss)

    

负对数似然损失与目标的泊松分布。

The loss can be described as:

target∼Poisson(input)loss(input,target)=input−target∗log⁡(input)+log⁡(target!)\text{target}
\sim \mathrm{Poisson}(\text{input}) \text{loss}(\text{input}, \text{target}) =
\text{input} - \text{target} * \log(\text{input}) \+
\log(\text{target!})target∼Poisson(input)loss(input,target)=input−target∗log(input)+log(target!)

的最后一项可被省略或用斯特林式近似。用于靶的近似值超过1.对于目标小于或等于1个零添加到损失。

Parameters

    

  * **log_input** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `损耗被计算为 EXP  ⁡ （ 输入 ） \-  目标 *  输入 \ EXP（\文本{输入}） - \文本{目标} * \文本{输入}  EXP  （ 输入 ） \-  目标 *  输入 时，如果`假 `损失 输入 \-  目标 *  日志 ⁡ （ 输入 \+  EPS  ）  \文本{输入} - \文本{目标} * \日志（\文本{输入} + \文本{EPS}） 输入 \-  目标 *  LO  G  （ 输入 \+  EPS  ） 。

  * **充满** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 

是否计算全部损失，我。即添加的斯特林近似术语

target∗log⁡(target)−target+0.5∗log⁡(2πtarget).\text{target}*\log(\text{target})
- \text{target} + 0.5 * \log(2\pi\text{target}).
target∗log(target)−target+0.5∗log(2πtarget).

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **EPS** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 小值，以避免的 [评价HTG12]  日志 ⁡ （ 0  ） \日志（0） LO  G  （ 0  ） 时`log_input  =  假 `。默认值：1E-8

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

Examples:

    
    
    >>> loss = nn.PoissonNLLLoss()
    >>> log_input = torch.randn(5, 2, requires_grad=True)
    >>> target = torch.randn(5, 2)
    >>> output = loss(log_input, target)
    >>> output.backward()
    

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where ∗*∗ means, any number of additional dimensions

  * Target: (N,∗)(N, *)(N,∗) , same shape as the input

  * 输出：在默认情况下标。如果`还原 `是`'无' `，然后 （ N  ， *  ） （N，*） （ N  ， *  ） ，相同的形状作为输入

###  KLDivLoss 

_class_`torch.nn.``KLDivLoss`( _size_average=None_ , _reduce=None_ ,
_reduction='mean'_ )[[source]](_modules/torch/nn/modules/loss.html#KLDivLoss)

    

在[相对熵](https://en.wikipedia.org/wiki/Kullback-Leibler_divergence)损失

KL散度是连续分布的有用距离度量和过（离散采样）连续输出分布的空间执行直接回归时是经常有益的。

与 `NLLLoss`时，输入中给出预计包含 _对数概率_ ，并且不限制于2D张量。各项指标均给定为 _概率_ （即没有取对数）。

该标准需要一个目标 张量相同尺寸与输入 张量的。

The unreduced (i.e. with `reduction`set to `'none'`) loss can be described
as:

l(x,y)=L={l1,…,lN},ln=yn⋅(log⁡yn−xn)l(x,y) = L = \\{ l_1,\dots,l_N \\}, \quad
l_n = y_n \cdot \left( \log y_n - x_n \right)
l(x,y)=L={l1​,…,lN​},ln​=yn​⋅(logyn​−xn​)

其中，索引 N  N  N  跨越`输入 `的所有尺寸和 L  L  L  具有相同的形状`输入 `。如果`还原 `不是`'无' `（默认`
'平均' `），然后：

ℓ(x,y)={mean⁡(L),if reduction=’mean’;sum⁡(L),if reduction=’sum’.\ell(x, y) =
\begin{cases} \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}
\\\ \operatorname{sum}(L), & \text{if reduction} = \text{'sum'.} \end{cases}
ℓ(x,y)={mean(L),sum(L),​if reduction=’mean’;if reduction=’sum’.​

在默认`还原 `模式`'平均' `时，损耗超过观测平均每个minibatch **以及** 在尺寸。 `'batchmean'
`模式给出正确的KL散度，其中损耗进行平均仅批次的尺寸。 `'平均' `模式的行为将被更改为与`'batchmean' `在接下来的主要版本。

Parameters

    

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **还原** （ _串_ _，_ _可选_ ） - 指定还原应用到输出：`'无' `| `'batchmean' `| `'和' `| `'意味' `。 `'无' `：不降低将被应用。 `'batchmean' `：将输出的总和将通过BATCHSIZE进行划分。 `'和' `：输出将被累加。 `“意味”`：输出将通过在输出元件的数目来划分。默认值：`'平均' `

Note

`size_average`和`减少 `处于被淘汰，并且在此同时，指定是这两个参数的个数将覆盖`还原 `。

Note

`还原 `= `'平均' `不回真正的KL散度值，请使用`还原 `= `'batchmean' `与KL数学定义对齐。在接下来的主要版本，`
'的意思是' `将改为同`'batchmean' [HTG23。`

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where ∗*∗ means, any number of additional dimensions

  * Target: (N,∗)(N, *)(N,∗) , same shape as the input

  * 输出：在默认情况下标。如果：ATTR：`还原 `是`'无' `，然后 （ N  ， *  ） （N，*） （ N  ， *  ） ，相同的形状作为输入

###  BCELoss 

_class_`torch.nn.``BCELoss`( _weight=None_ , _size_average=None_ ,
_reduce=None_ , _reduction='mean'_
)[[source]](_modules/torch/nn/modules/loss.html#BCELoss)

    

创建一个测量目标与输出之间的二进制交叉熵的标准：

The unreduced (i.e. with `reduction`set to `'none'`) loss can be described
as:

ℓ(x,y)=L={l1,…,lN}⊤,ln=−wn[yn⋅log⁡xn+(1−yn)⋅log⁡(1−xn)],\ell(x, y) = L =
\\{l_1,\dots,l_N\\}^\top, \quad l_n = - w_n \left[ y_n \cdot \log x_n + (1 -
y_n) \cdot \log (1 - x_n) \right],
ℓ(x,y)=L={l1​,…,lN​}⊤,ln​=−wn​[yn​⋅logxn​+(1−yn​)⋅log(1−xn​)],

where NNN is the batch size. If `reduction`is not `'none'`(default
`'mean'`), then

ℓ(x,y)={mean⁡(L),if reduction=’mean’;sum⁡(L),if reduction=’sum’.\ell(x, y) =
\begin{cases} \operatorname{mean}(L), & \text{if reduction} =
\text{'mean';}\\\ \operatorname{sum}(L), & \text{if reduction} = \text{'sum'.}
\end{cases} ℓ(x,y)={mean(L),sum(L),​if reduction=’mean’;if reduction=’sum’.​

这是用于测量一个重建的误差在例如一个自动编码器。请注意，目标 Y  Y  Y  应该是0和1之间的数字。

Parameters

    

  * **重量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 给予每个批次元素的损失的手动重新缩放权重。如果给定的，必须是尺寸 nbatch 的张量。

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where ∗*∗ means, any number of additional dimensions

  * Target: (N,∗)(N, *)(N,∗) , same shape as the input

  * 输出：标量。如果`还原 `是`'无' `，然后 （ N  ， *  ） （N，*） （ N  ， *  ） ，相同的形状作为输入。

Examples:

    
    
    >>> m = nn.Sigmoid()
    >>> loss = nn.BCELoss()
    >>> input = torch.randn(3, requires_grad=True)
    >>> target = torch.empty(3).random_(2)
    >>> output = loss(m(input), target)
    >>> output.backward()
    

###  BCEWithLogitsLoss 

_class_`torch.nn.``BCEWithLogitsLoss`( _weight=None_ , _size_average=None_ ,
_reduce=None_ , _reduction='mean'_ , _pos_weight=None_
)[[source]](_modules/torch/nn/modules/loss.html#BCEWithLogitsLoss)

    

这种损失结合了乙状结肠层和 BCELoss 在一个单独的类。这个版本比使用纯更数值稳定乙状结肠接着对数和-EXP特技进行数值的 BCELoss
如，通过操作组合为一个层，我们利用稳定性。

The unreduced (i.e. with `reduction`set to `'none'`) loss can be described
as:

ℓ(x,y)=L={l1,…,lN}⊤,ln=−wn[yn⋅log⁡σ(xn)+(1−yn)⋅log⁡(1−σ(xn))],\ell(x, y) = L =
\\{l_1,\dots,l_N\\}^\top, \quad l_n = - w_n \left[ y_n \cdot \log \sigma(x_n)
\+ (1 - y_n) \cdot \log (1 - \sigma(x_n)) \right],
ℓ(x,y)=L={l1​,…,lN​}⊤,ln​=−wn​[yn​⋅logσ(xn​)+(1−yn​)⋅log(1−σ(xn​))],

where NNN is the batch size. If `reduction`is not `'none'`(default
`'mean'`), then

ℓ(x,y)={mean⁡(L),if reduction=’mean’;sum⁡(L),if reduction=’sum’.\ell(x, y) =
\begin{cases} \operatorname{mean}(L), & \text{if reduction} =
\text{'mean';}\\\ \operatorname{sum}(L), & \text{if reduction} = \text{'sum'.}
\end{cases} ℓ(x,y)={mean(L),sum(L),​if reduction=’mean’;if reduction=’sum’.​

这是用于测量一个重建的误差在例如一个自动编码器。注意，目标 T [1] 应该是0和1之间的数字。

这可以通过增加配重，以积极的例子权衡召回和精度。在多标签分类的情况下的损失可以被描述为：

ℓc(x,y)=Lc={l1,c,…,lN,c}⊤,ln,c=−wn,c[pcyn,c⋅log⁡σ(xn,c)+(1−yn,c)⋅log⁡(1−σ(xn,c))],\ell_c(x,
y) = L_c = \\{l_{1,c},\dots,l_{N,c}\\}^\top, \quad l_{n,c} = - w_{n,c} \left[
p_c y_{n,c} \cdot \log \sigma(x_{n,c}) \+ (1 - y_{n,c}) \cdot \log (1 -
\sigma(x_{n,c})) \right],
ℓc​(x,y)=Lc​={l1,c​,…,lN,c​}⊤,ln,c​=−wn,c​[pc​yn,c​⋅logσ(xn,c​)+(1−yn,c​)⋅log(1−σ(xn,c​))],

其中 C  C  C  是类数（HTG24]  C  & GT ;  1  C & GT ; 1  C  & GT ;  1
[HTG63用于多标签二元分类， C  =  1  C = 1  C  =  1  为单标签二元分类）， n的 n的 n的 是在批处理的样品的数目和 p
C  P_C  p  C  是肯定的回答，为类 [重量HTG183]  C  C  C  。

P  C  & GT ;  1  P_C & GT ; 1  p  C  [ - - ] GT ;  1  增加召回，  p  C  & LT ;  1
P_C & LT ; 1  p  C  & LT ;  1  提高精度。

例如，如果数据集包含单个类的100个的正和300反面的例子，则 pos_weight 为类应等于 300  100  =  3  \压裂{300}
{100} = 3  1  0  0  3  0  0  =  3  。损失将充当如果数据集包含 3  × 100  =  300  3 \倍100 =
300  3  × 1  0  0  =  3  0  0  正例。

Examples:

    
    
    >>> target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
    >>> output = torch.full([10, 64], 0.999)  # A prediction (logit)
    >>> pos_weight = torch.ones([64])  # All weights are equal to 1
    >>> criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    >>> criterion(output, target)  # -log(sigmoid(0.999))
    tensor(0.3135)
    

Parameters

    

  * **weight** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – a manual rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size nbatch.

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

  * **pos_weight** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 正例的权重。必须与长度等于类的数量的矢量。

Shape:

    

>   * 输入： （ N  ， *  ） （N，*） （ N  ， *  ） 其中 *  *  *  手段，任意数量的附加维度的

>

>   * Target: (N,∗)(N, *)(N,∗) , same shape as the input

>

>   * Output: scalar. If `reduction`is `'none'`, then (N,∗)(N, *)(N,∗) , same
shape as input.

>

>

Examples:

    
    
    >>> loss = nn.BCEWithLogitsLoss()
    >>> input = torch.randn(3, requires_grad=True)
    >>> target = torch.empty(3).random_(2)
    >>> output = loss(input, target)
    >>> output.backward()
    

###  MarginRankingLoss 

_class_`torch.nn.``MarginRankingLoss`( _margin=0.0_ , _size_average=None_ ,
_reduce=None_ , _reduction='mean'_
)[[source]](_modules/torch/nn/modules/loss.html#MarginRankingLoss)

    

创建测量损耗给定的输入 × 1  X1 [HTG12一个标准]  × 1  ， × 2  X2  × 2  ，两个一维小批量张量，和标签1D小批量张量 Y
Y  Y  （含有1或-1）。

如果 Y  =  1  Y = 1  Y  =  1  然后将其假定第一输入应该被排名更高（具有更大的值）大于第二输入，反之亦然为 Y  =  \-  1
Y = -1  Y  =  \-  1  。

在小批量每个样品的损失函数是：

loss(x,y)=max⁡(0,−y∗(x1−x2)+margin)\text{loss}(x, y) = \max(0, -y * (x1 - x2)
+ \text{margin}) loss(x,y)=max(0,−y∗(x1−x2)+margin)

Parameters

    

  * **余量** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 具有的 默认值 0  0  0  。

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

Shape:

    

  * 输入： （ N  ， d  ） （N，d） （ N  ， d  ） 其中 N 是批量大小和 d 是一个样品的大小。

  * 目标： （ N  ） （N）  （ N  ）

  * 输出：标量。如果`还原 `是`'无' `，然后 （ N  ） （N） （ N  ） 。

###  HingeEmbeddingLoss 

_class_`torch.nn.``HingeEmbeddingLoss`( _margin=1.0_ , _size_average=None_ ,
_reduce=None_ , _reduction='mean'_
)[[source]](_modules/torch/nn/modules/loss.html#HingeEmbeddingLoss)

    

措施的损失给定的输入张量 × × × 和标签张量 Y  Y  Y  （含有1或-1）。这通常是用于测量两个输入是否是相似或不相似，例如使用L1成对距离为 ×
× × ，并且通常用于学习的嵌入的非线性或半监督学习。

损耗函数为 n的 n的 n的 在迷你批次号的抽样是

ln={xn,if yn=1,max⁡{0,Δ−xn},if yn=−1,l_n = \begin{cases} x_n, & \text{if}\;
y_n = 1,\\\ \max \\{0, \Delta - x_n\\}, & \text{if}\; y_n = -1, \end{cases}
ln​={xn​,max{0,Δ−xn​},​ifyn​=1,ifyn​=−1,​

和总损耗函数是

ℓ(x,y)={mean⁡(L),if reduction=’mean’;sum⁡(L),if reduction=’sum’.\ell(x, y) =
\begin{cases} \operatorname{mean}(L), & \text{if reduction} =
\text{'mean';}\\\ \operatorname{sum}(L), & \text{if reduction} = \text{'sum'.}
\end{cases} ℓ(x,y)={mean(L),sum(L),​if reduction=’mean’;if reduction=’sum’.​

其中 L  =  { L  1  ， ...  ， L  N  }  ⊤ L = \ {L_1，\点，L_N \\} ^ \顶 L  =  { L  1
， ...  ， L  N  }  ⊤ 。

Parameters

    

  * **余量** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 具有的 1 的默认值。

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

Shape:

    

  * 输入： （ *  ） （*）  （ *  ） 其中 *  *  *  手段，任意维数。总和操作过的所有元素进行操作。

  * 目标： （ *  ） （*）  （ *  ） ，相同形状的输入

  * 输出：标量。如果`还原 `是`'无' `，然后相同的形状作为输入

###  MultiLabelMarginLoss 

_class_`torch.nn.``MultiLabelMarginLoss`( _size_average=None_ , _reduce=None_
, _reduction='mean'_
)[[source]](_modules/torch/nn/modules/loss.html#MultiLabelMarginLoss)

    

创建优化输入 × 之间的多级多分类铰链损失（基于容限的损失）的标准× × （二维小批量张量）和输出 Y  Y  Y  （这是一个2D
张量目标类指数）。对于在小批量每个样品：

loss(x,y)=∑ijmax⁡(0,1−(x[y[j]]−x[i]))x.size(0)\text{loss}(x, y) =
\sum_{ij}\frac{\max(0, 1 - (x[y[j]] - x[i]))}{\text{x.size}(0)}
loss(x,y)=ij∑​x.size(0)max(0,1−(x[y[j]]−x[i]))​

其中 × ∈ { 0  ， ⋯ ， x.size  （ 0  ） \-  1  }  × \在\左\ {0，\ [] \ cdots，\ [] \ {文本}
x.size（0） - 1 \右\\}  × ∈ { 0  ， ⋯ ， x.size  （ 0  ） \-  1  }  ， Y  ∈ { 0  ， ⋯ ，
y.size  （ 0  ） \-  1  }  在\ Y \左\ {0，\ [] \ cdots，\ [] \ {文本} y.size（0） - 1
\右\\}  Y  ∈ { 0  ， [H TG186] ⋯ ， y.size  （ 0  ） \-  1  }  ， 0  ≤ Y  [
[HTG238：J  ≤ x.size  （ 0  ） \-  1  0 \当量Y [j]的\当量\文本{x.size}（0）-1  0  ​​≤ Y  [
[HTG282：J  ≤ x.size  （ 0  ） \-  1  和 i的 ≠ Y  [ [HTG336：J  I \ NEQ Y [j]的 i的 
=  Y  [ [HTG400：J  所有 i的 i的 i的 和 [HTG438：J  [HTG441：J  [HTG450：J  。

Y  Y  Y  和 × × × 必须具有相同的大小。

该标准仅考虑的非负目标的连续的块开始于前面。

这允许不同的样品，以具有可变的量的靶类。

Parameters

    

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

Shape:

    

  * 输入： （ C  ） （C）  （ C  ） 或 （ N  ， C  ） （N，C） （  N  ， C  ） 其中 N 是批量大小和 C 是类的数量。

  * 目标： （ C  ） （C）  （ C  ） 或 （ N  ， C  ） （N，C） （  N  ， C  ） 标签的目标-1保证相同形状的输入填充。

  * Output: scalar. If `reduction`is `'none'`, then (N)(N)(N) .

Examples:

    
    
    >>> loss = nn.MultiLabelMarginLoss()
    >>> x = torch.FloatTensor([[0.1, 0.2, 0.4, 0.8]])
    >>> # for target y, only consider labels 3 and 0, not after label -1
    >>> y = torch.LongTensor([[3, 0, -1, 1]])
    >>> loss(x, y)
    >>> # 0.25 * ((1-(0.1-0.2)) + (1-(0.1-0.4)) + (1-(0.8-0.2)) + (1-(0.8-0.4)))
    tensor(0.8500)
    

###  SmoothL1Loss 

_class_`torch.nn.``SmoothL1Loss`( _size_average=None_ , _reduce=None_ ,
_reduction='mean'_
)[[source]](_modules/torch/nn/modules/loss.html#SmoothL1Loss)

    

创建一个使用平方项如果绝对逐元素误差低于1和L1术语否则的标准。这是异常值比 MSELoss 在某些情况下较不敏感防止爆炸梯度（例如，请参见由Ross
Girshick快速R-CNN 纸）。又称胡贝尔损失：

loss(x,y)=1n∑izi\text{loss}(x, y) = \frac{1}{n} \sum_{i} z_{i}
loss(x,y)=n1​i∑​zi​

其中 Z  i的 Z_ {I}  Z  i的 由下式给出：

zi={0.5(xi−yi)2,if ∣xi−yi∣<1∣xi−yi∣−0.5,otherwise z_{i} = \begin{cases} 0.5
(x_i - y_i)^2, & \text{if } |x_i - y_i| < 1 \\\ |x_i - y_i| - 0.5, &
\text{otherwise } \end{cases} zi​={0.5(xi​−yi​)2,∣xi​−yi​∣−0.5,​if
∣xi​−yi​∣<1otherwise ​

× × × 和 Y  Y  Y  的任意形状，总的 n的 n的 n的 每个求和操作仍然工作在所有的元素，并且通过分割元素 n的 n的 n的 。

除以 n的 n的 n的 可避免如果集`还原 =  '和' `。

Parameters

    

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

Shape:

    

  * Input: (N,∗)(N, *)(N,∗) where ∗*∗ means, any number of additional dimensions

  * Target: (N,∗)(N, *)(N,∗) , same shape as the input

  * Output: scalar. If `reduction`is `'none'`, then (N,∗)(N, *)(N,∗) , same shape as the input

###  SoftMarginLoss 

_class_`torch.nn.``SoftMarginLoss`( _size_average=None_ , _reduce=None_ ,
_reduction='mean'_
)[[source]](_modules/torch/nn/modules/loss.html#SoftMarginLoss)

    

创建优化之间的二类别分类的物流损失的标准输入张量 × × × 和目标张量 Y  Y  Y  （含有1或-1）。

loss(x,y)=∑ilog⁡(1+exp⁡(−y[i]∗x[i]))x.nelement()\text{loss}(x, y) = \sum_i
\frac{\log(1 + \exp(-y[i]*x[i]))}{\text{x.nelement}()}
loss(x,y)=i∑​x.nelement()log(1+exp(−y[i]∗x[i]))​

Parameters

    

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

Shape:

    

  * 输入： （ *  ） （*）  （ *  ） 其中 *  *  *  手段，任意数量的附加维度的

  * Target: (∗)(*)(∗) , same shape as the input

  * Output: scalar. If `reduction`is `'none'`, then same shape as the input

###  MultiLabelSoftMarginLoss 

_class_`torch.nn.``MultiLabelSoftMarginLoss`( _weight=None_ ,
_size_average=None_ , _reduce=None_ , _reduction='mean'_
)[[source]](_modules/torch/nn/modules/loss.html#MultiLabelSoftMarginLoss)

    

创建优化的多标签一个抗所有基于最大熵损失的标准，之间输入 × × × 和目标 Y  Y  Y  的大小 （ N  ， C  ） （N，C） （ N  ， C
） 。对于在minibatch每个样品：

loss(x,y)=−1C∗∑iy[i]∗log⁡((1+exp⁡(−x[i]))−1)+(1−y[i])∗log⁡(exp⁡(−x[i])(1+exp⁡(−x[i])))loss(x,
y) = - \frac{1}{C} * \sum_i y[i] * \log((1 + \exp(-x[i]))^{-1}) \+ (1-y[i]) *
\log\left(\frac{\exp(-x[i])}{(1 + \exp(-x[i]))}\right)
loss(x,y)=−C1​∗i∑​y[i]∗log((1+exp(−x[i]))−1)+(1−y[i])∗log((1+exp(−x[i]))exp(−x[i])​)

其中 i的 ∈ { 0  ， ⋯ ， x.nElement  （ ） \-  1  }  i的\ \左\ {0 \ [] \ cdots，\ [] \
{文本} x.nElement（） - 1 \右\\}  i的 ∈ { 0  ， ⋯ ， x.nElement  （ ）​​ \-  1  }  ， Y
[ i的 ∈ { 0  ， 1  }  值Y [i] \在\左\ {0，\ [] 1 \右\\}  Y  [  i的 ∈ { 0  ， 1  }  。

Parameters

    

  * **weight** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – a manual rescaling weight given to each class. If given, it has to be a Tensor of size C. Otherwise, it is treated as if having all ones.

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

Shape:

    

  * 输入： （ N  ， C  ） （N，C） （ N  ， C  ） 其中 N 是批量大小和 C 是类的数量。

  * 目标： （ N  ， C  ） （N，C） （ N  ， C  ） ，标记目标填充由-1确保相同的形状的输入。

  * Output: scalar. If `reduction`is `'none'`, then (N)(N)(N) .

###  CosineEmbeddingLoss 

_class_`torch.nn.``CosineEmbeddingLoss`( _margin=0.0_ , _size_average=None_ ,
_reduce=None_ , _reduction='mean'_
)[[source]](_modules/torch/nn/modules/loss.html#CosineEmbeddingLoss)

    

创建一种测量标准的损失给定的输入张量 × 1  X_1  × 1  ， × 2  X_2  × 2  [HT G102]  和a 张量标签 Y  Y  Y
与值1或-1。此被用于测量两个输入是否是相似或不相似，使用余弦距离，并且通常用于学习的嵌入的非线性或半监督学习。

每个样品的损失函数是：

loss(x,y)={1−cos⁡(x1,x2),if y=1max⁡(0,cos⁡(x1,x2)−margin),if
y=−1\text{loss}(x, y) = \begin{cases} 1 - \cos(x_1, x_2), & \text{if } y = 1
\\\ \max(0, \cos(x_1, x_2) - \text{margin}), & \text{if } y = -1 \end{cases}
loss(x,y)={1−cos(x1​,x2​),max(0,cos(x1​,x2​)−margin),​if y=1if y=−1​

Parameters

    

  * **余量** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 应该是从 [HTG12的数]  \-  1  -1-  \-  1  至 1  1  1  ， 0  0  0  至 0.5  0.5  [H TG102]  0  。  5  建议。如果`余量 `缺失，则默认值为 0  0  0  。

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

###  MultiMarginLoss 

_class_`torch.nn.``MultiMarginLoss`( _p=1_ , _margin=1.0_ , _weight=None_ ,
_size_average=None_ , _reduce=None_ , _reduction='mean'_
)[[source]](_modules/torch/nn/modules/loss.html#MultiMarginLoss)

    

创建优化输入 × 之间的多类分类铰链损失（基于容限的损失）[HTG9的标准]×  × （二维微型批次张量）和输出 Y  Y  Y
（它是目标类索引的1D张量， 0  ≤ Y  ≤ x.size  （ 1  ） \-  1  0 \当量Y \当量\文本{x.size}（1）-1  0
≤ Y  ≤ x.size  （ 1  ） \-  1  ）：

对于每个小批量样品，在一维输入方面损失 × × × 和标量输出 Y  Y  Y  是：

loss(x,y)=∑imax⁡(0,margin−x[y]+x[i]))px.size(0)\text{loss}(x, y) =
\frac{\sum_i \max(0, \text{margin} - x[y] + x[i]))^p}{\text{x.size}(0)}
loss(x,y)=x.size(0)∑i​max(0,margin−x[y]+x[i]))p​

其中 × ∈ { 0  ， ⋯ ， x.size  （ 0  ） \-  1  }  × \在\左\ {0，\ [] \ cdots，\ [] \ {文本}
x.size（0） - 1 \右\\}  × ∈ { 0  ， ⋯ ， x.size  （ 0  ） \-  1  }  和 i的 ≠ Y  I \ NEQ
Y  i的  =  Y  。

可选地，可以通过使1D `重量 `张量到构造给出的类不相等的权重。

那么损失函数变为：

loss(x,y)=∑imax⁡(0,w[y]∗(margin−x[y]+x[i]))p)x.size(0)\text{loss}(x, y) =
\frac{\sum_i \max(0, w[y] * (\text{margin} - x[y] +
x[i]))^p)}{\text{x.size}(0)}
loss(x,y)=x.size(0)∑i​max(0,w[y]∗(margin−x[y]+x[i]))p)​

Parameters

    

  * **P** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 具有的 默认值 1  1  1  。  1  1  1  和 2  2  2  是唯一支持的值。

  * **余量** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 具有的 默认值 1  1  1  。

  * **weight** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _,_ _optional_ ) – a manual rescaling weight given to each class. If given, it has to be a Tensor of size C. Otherwise, it is treated as if having all ones.

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

###  TripletMarginLoss 

_class_`torch.nn.``TripletMarginLoss`( _margin=1.0_ , _p=2.0_ , _eps=1e-06_ ,
_swap=False_ , _size_average=None_ , _reduce=None_ , _reduction='mean'_
)[[source]](_modules/torch/nn/modules/loss.html#TripletMarginLoss)

    

创建测量三重损失的标准给定的输入张量 × 1  X1  × 1  ， × 2  X2  × 2  ， × 3  ×3  × 3  并以更大的值的裕度比 0
0  0  。此被用于测量样本之间的相对相似性。三元组是由一， P 和 n的（即锚，正例和[HTG118组成]反例分别地）。所有输入张量的形状应是 （ N
， d  ） （N，d） （  N  ， d  ） 。

距离交换中详细纸张[学习与三重态损耗](http://www.bmva.org/bmvc/2016/papers/paper119/index.html)由V.
Balntas，E.里巴等人浅卷积特征描述符描述。

The loss function for each sample in the mini-batch is:

L(a,p,n)=max⁡{d(ai,pi)−d(ai,ni)+margin,0}L(a, p, n) = \max \\{d(a_i, p_i) -
d(a_i, n_i) + {\rm margin}, 0\\} L(a,p,n)=max{d(ai​,pi​)−d(ai​,ni​)+margin,0}

哪里

d(xi,yi)=∥xi−yi∥pd(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i
\right\rVert_p d(xi​,yi​)=∥xi​−yi​∥p​

Parameters

    

  * **余量** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 默认值： 1  1  1  。

  * **P** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 范数度成对距离。默认值： 2  2  2  。

  * **交换** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 的距离交换被详细描述在本文中描述HTG10]学习浅卷积特征描述符与三重态损耗由V. Balntas，E.里巴等。默认值：`假 [HTG15。`

  * **size_average** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field `size_average`is set to `False`, the losses are instead summed for each minibatch. Ignored when reduce is `False`. Default: `True`

  * **reduce** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Deprecated (see `reduction`). By default, the losses are averaged or summed over observations for each minibatch depending on `size_average`. When `reduce`is `False`, returns a loss per batch element instead and ignores `size_average`. Default: `True`

  * **reduction** ( _string_ _,_ _optional_ ) – Specifies the reduction to apply to the output: `'none'`| `'mean'`| `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Note: `size_average`and `reduce`are in the process of being deprecated, and in the meantime, specifying either of those two args will override `reduction`. Default: `'mean'`

Shape:

    

  * 输入： （ N  ， d  ） （N，d） （ N  ， d  ） 其中 d  d  d  是矢量维数。

  * Output: scalar. If `reduction`is `'none'`, then (N)(N)(N) .

    
    
    >>> triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    >>> anchor = torch.randn(100, 128, requires_grad=True)
    >>> positive = torch.randn(100, 128, requires_grad=True)
    >>> negative = torch.randn(100, 128, requires_grad=True)
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()
    

## 视力层

###  PixelShuffle 

_class_`torch.nn.``PixelShuffle`( _upscale_factor_
)[[source]](_modules/torch/nn/modules/pixelshuffle.html#PixelShuffle)

    

重新排列的元件在形状 （ *  ， ℃的张量 × R  2  ， H  ， W  ） （*，C \倍R ^ 2，H，W） （ *  ， C  × R  2
， H  ， W  ） 到的张量定型 （ *  ， C  H  × R  ， W  × R  ） （*，C，H \倍R，W \次数R） （ *  ， C
， H  × R  W  × R  ） 。

这是用于实现高效的子像素卷积用的 1  /  R A步幅有用 1 / R  1  /  R  。

见文章：[实时单幅图像和视频超分辨率采用高效的子像素卷积神经网络](https://arxiv.org/abs/1609.05158)由Shi等。人，（2016）的更多细节。

Parameters

    

**upscale_factor** （[ _INT_
](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")）
- 因子以增加由空间分辨率

Shape:

    

  * 输入： （ N  ， L  ， H  i的 n的 ， W  i的 n的 ） （N，L，H_ {IN} ，W_ {在}） （ N  ， L  ， H  i的 n的 ， W  i的 n的 ） 其中 L  =  C  × upscale_factor  2  L = c ^ \倍\文本{高档\ _factor} ^ 2  L  =  C  × upscale_factor  [HTG19 6]  2 

  * 输出： （ N  ， C  ， H  O  U  T  ， W  O  U  T  ） （N，C，H_ {出}，W_ {出}） （ N  ， C  ， H  O  U  T  ， W  O  U  T  ） 其中 H  O  U  T  =  H  i的 n的 × upscale_factor  H_ {出} = H_ {在} \倍\文本{高档\ _factor}  H  [H TG196]  O  U  T  =  H  i的 n的 × ​​  upscale_factor  和 W¯¯ [HTG29 2]  O  U  T  =  W  i的 n的 × upscale_factor  W_ {出} = W_ {在} \倍\文本{高档\ _factor}  W  O  U  T  =  W  I  n的 × upscale_factor 

Examples:

    
    
    >>> pixel_shuffle = nn.PixelShuffle(3)
    >>> input = torch.randn(1, 9, 4, 4)
    >>> output = pixel_shuffle(input)
    >>> print(output.size())
    torch.Size([1, 1, 12, 12])
    

### 上采样

_class_`torch.nn.``Upsample`( _size=None_ , _scale_factor=None_ ,
_mode='nearest'_ , _align_corners=None_
)[[source]](_modules/torch/nn/modules/upsampling.html#Upsample)

    

上采样一个给定的多通道1D（时间），二维（空间）或3D（体积）的数据。

所述输入数据被假定为形式 minibatch
X通道×[可选深度]×[可选高度]×宽度的。因此，对于空间的投入，我们预计四维张量和体积的投入，我们预计5D张量。

可用于上采样的算法分别是3D，4D和5D输入张量最近邻和线性，双线性，双三次和三线性。

一个可以得到`scale_factor`或目标输出`大小 `来计算输出大小。 （你不能给双方，因为它是不明确）

Parameters

    

  * **大小** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ _元组_ _[_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _]或_ _元组_ _[_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _]或_ _元组_ _[_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") __ _，_ _可选的_ ） - 输出空间尺寸

  * **scale_factor** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ _元组_ _[_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _]或_ _元组_ _[_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _]或_ _元组_ _[_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") __ _，_ _可选的_ ） - 乘法器，用于空间尺寸。有，如果它是一个元组匹配输入的内容。

  * **模式** （[ _STR_ ](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)") _，_ _可选_ ） - 上采样算法：的`一个'最近”`，`'线性' `，`'双线性' `，`'双三次'`和`'三线性' `。默认值：`'最近' `

  * **align_corners** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，输入和输出张量的拐角像素被对齐，从而保持在那些像素的值。这仅具有效力时`模式 `是`'线性' `，`'双线性' `或`'三线性' `。默认值：`假 `

Shape:

    

  * 输入： （ N  ， C  ， W  i的 n的 ） （N，C，W_ {在}） （ N  ， C  ， W  i的 n的 ） ， （ N  ， C  ， H  i的 n的 ， W  i的 n的 ） （N，C，H_ {IN}，W_ {IN} ） （ N  ， C  ， H  i的 n的 ， W  [HTG20 1]  i的 n的 ） 或 （ N  ， C  ， d  i的 n的 ， H  i的 n的 ， W ​​  i的 n的 ） （N，C，D_ {IN}，H_ {IN}，W_ {在}） （ N  ，[HTG2 95]  C  ， d  i的 n的 ， H  i的 n的 ， W  i的 n的 [HTG39 3]  ）

  * 输出： （ N  ， C  ， W  O  U  T  ） （N，C，W_ {出}） （ N  ， C  ， W  O  U  T  ）  ， （ N  ， C  ， H  O  U  T  ， W  O  U  T  ） （N，C，H_ {出}，W_ {出}） （ N  ， C  ， H  O  U  T  ， W  O  U  T  ） 或 （ N  ， C  ， d  O  U  T  ， ​​  H  O  U  T  ， W  O  U  T  ）[HTG2 97]  （N，C，D_ {出}，H_ {出}，W_ {出}） （ N  ， C  ， d  O  U  T  ， H  O  U  T  [H TG392]  ， W  O  U  T  ） ，其中

Dout=⌊Din×scale_factor⌋D_{out} = \left\lfloor D_{in} \times
\text{scale\\_factor} \right\rfloor Dout​=⌊Din​×scale_factor⌋

Hout=⌊Hin×scale_factor⌋H_{out} = \left\lfloor H_{in} \times
\text{scale\\_factor} \right\rfloor Hout​=⌊Hin​×scale_factor⌋

Wout=⌊Win×scale_factor⌋W_{out} = \left\lfloor W_{in} \times
\text{scale\\_factor} \right\rfloor Wout​=⌊Win​×scale_factor⌋

Warning

与`align_corners  =  真 `时，线性地内插模式（线性，双线性
双三次和三线性）不按比例对齐的输出和输入的像素，和因此输出值可以依赖于输入的大小。这是这些模式可支持高达0.3.1版本的默认行为。此后，缺省行为是`
align_corners  =  假 `。见下面关于这将如何影响输出的具体例子。

Note

如果您想采样/一般大小调整，你应该使用`插值（） [HTG3。`

Examples:

    
    
    >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
    >>> input
    tensor([[[[ 1.,  2.],
              [ 3.,  4.]]]])
    
    >>> m = nn.Upsample(scale_factor=2, mode='nearest')
    >>> m(input)
    tensor([[[[ 1.,  1.,  2.,  2.],
              [ 1.,  1.,  2.,  2.],
              [ 3.,  3.,  4.,  4.],
              [ 3.,  3.,  4.,  4.]]]])
    
    >>> m = nn.Upsample(scale_factor=2, mode='bilinear')  # align_corners=False
    >>> m(input)
    tensor([[[[ 1.0000,  1.2500,  1.7500,  2.0000],
              [ 1.5000,  1.7500,  2.2500,  2.5000],
              [ 2.5000,  2.7500,  3.2500,  3.5000],
              [ 3.0000,  3.2500,  3.7500,  4.0000]]]])
    
    >>> m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    >>> m(input)
    tensor([[[[ 1.0000,  1.3333,  1.6667,  2.0000],
              [ 1.6667,  2.0000,  2.3333,  2.6667],
              [ 2.3333,  2.6667,  3.0000,  3.3333],
              [ 3.0000,  3.3333,  3.6667,  4.0000]]]])
    
    >>> # Try scaling the same data in a larger tensor
    >>>
    >>> input_3x3 = torch.zeros(3, 3).view(1, 1, 3, 3)
    >>> input_3x3[:, :, :2, :2].copy_(input)
    tensor([[[[ 1.,  2.],
              [ 3.,  4.]]]])
    >>> input_3x3
    tensor([[[[ 1.,  2.,  0.],
              [ 3.,  4.,  0.],
              [ 0.,  0.,  0.]]]])
    
    >>> m = nn.Upsample(scale_factor=2, mode='bilinear')  # align_corners=False
    >>> # Notice that values in top left corner are the same with the small input (except at boundary)
    >>> m(input_3x3)
    tensor([[[[ 1.0000,  1.2500,  1.7500,  1.5000,  0.5000,  0.0000],
              [ 1.5000,  1.7500,  2.2500,  1.8750,  0.6250,  0.0000],
              [ 2.5000,  2.7500,  3.2500,  2.6250,  0.8750,  0.0000],
              [ 2.2500,  2.4375,  2.8125,  2.2500,  0.7500,  0.0000],
              [ 0.7500,  0.8125,  0.9375,  0.7500,  0.2500,  0.0000],
              [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]]])
    
    >>> m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    >>> # Notice that values in top left corner are now changed
    >>> m(input_3x3)
    tensor([[[[ 1.0000,  1.4000,  1.8000,  1.6000,  0.8000,  0.0000],
              [ 1.8000,  2.2000,  2.6000,  2.2400,  1.1200,  0.0000],
              [ 2.6000,  3.0000,  3.4000,  2.8800,  1.4400,  0.0000],
              [ 2.4000,  2.7200,  3.0400,  2.5600,  1.2800,  0.0000],
              [ 1.2000,  1.3600,  1.5200,  1.2800,  0.6400,  0.0000],
              [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]]])
    

###  UpsamplingNearest2d 

_class_`torch.nn.``UpsamplingNearest2d`( _size=None_ , _scale_factor=None_
)[[source]](_modules/torch/nn/modules/upsampling.html#UpsamplingNearest2d)

    

施加2D最近邻上采样到几个输入信道组成的输入信号。

要指定规模，它需要要么`大小 `或`scale_factor`，因为它的构造函数的参数。

当`大小 `中给出，它是图像的输出尺寸（H，W）。

Parameters

    

  * **大小** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ _元组_ _[_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") __ _，_ _可选_ ） - 输出空间大小

  * **scale_factor** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ _元组_ _[_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") __ _，_ _可选_ ） - 乘法器，用于空间的大小。

Warning

这个类是有利于`插值（） `弃用。

Shape:

    

  * Input: (N,C,Hin,Win)(N, C, H_{in}, W_{in})(N,C,Hin​,Win​)

  * Output: (N,C,Hout,Wout)(N, C, H_{out}, W_{out})(N,C,Hout​,Wout​) where

Hout=⌊Hin×scale_factor⌋H_{out} = \left\lfloor H_{in} \times
\text{scale\\_factor} \right\rfloor Hout​=⌊Hin​×scale_factor⌋

Wout=⌊Win×scale_factor⌋W_{out} = \left\lfloor W_{in} \times
\text{scale\\_factor} \right\rfloor Wout​=⌊Win​×scale_factor⌋

Examples:

    
    
    >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
    >>> input
    tensor([[[[ 1.,  2.],
              [ 3.,  4.]]]])
    
    >>> m = nn.UpsamplingNearest2d(scale_factor=2)
    >>> m(input)
    tensor([[[[ 1.,  1.,  2.,  2.],
              [ 1.,  1.,  2.,  2.],
              [ 3.,  3.,  4.,  4.],
              [ 3.,  3.,  4.,  4.]]]])
    

###  UpsamplingBilinear2d 

_class_`torch.nn.``UpsamplingBilinear2d`( _size=None_ , _scale_factor=None_
)[[source]](_modules/torch/nn/modules/upsampling.html#UpsamplingBilinear2d)

    

施加2D双线性上采样到几个输入信道组成的输入信号。

To specify the scale, it takes either the `size`or the `scale_factor`as it’s
constructor argument.

When `size`is given, it is the output size of the image (h, w).

Parameters

    

  * **size** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _or_ _Tuple_ _[_[ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _,_[ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _]_ _,_ _optional_ ) – output spatial sizes

  * **scale_factor** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _or_ _Tuple_ _[_[ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_[ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _]_ _,_ _optional_ ) – multiplier for spatial size.

Warning

这个类是有利于`插值（） `弃用。它等同于`nn.functional.interpolate（...， 模式= '双线性'，
align_corners =真） `。

Shape:

    

  * Input: (N,C,Hin,Win)(N, C, H_{in}, W_{in})(N,C,Hin​,Win​)

  * Output: (N,C,Hout,Wout)(N, C, H_{out}, W_{out})(N,C,Hout​,Wout​) where

Hout=⌊Hin×scale_factor⌋H_{out} = \left\lfloor H_{in} \times
\text{scale\\_factor} \right\rfloor Hout​=⌊Hin​×scale_factor⌋

Wout=⌊Win×scale_factor⌋W_{out} = \left\lfloor W_{in} \times
\text{scale\\_factor} \right\rfloor Wout​=⌊Win​×scale_factor⌋

Examples:

    
    
    >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
    >>> input
    tensor([[[[ 1.,  2.],
              [ 3.,  4.]]]])
    
    >>> m = nn.UpsamplingBilinear2d(scale_factor=2)
    >>> m(input)
    tensor([[[[ 1.0000,  1.3333,  1.6667,  2.0000],
              [ 1.6667,  2.0000,  2.3333,  2.6667],
              [ 2.3333,  2.6667,  3.0000,  3.3333],
              [ 3.0000,  3.3333,  3.6667,  4.0000]]]])
    

## 数据并行层（多GPU，分布式）

### 数据并行

_class_`torch.nn.``DataParallel`( _module_ , _device_ids=None_ ,
_output_device=None_ , _dim=0_
)[[source]](_modules/torch/nn/parallel/data_parallel.html#DataParallel)

    

实现了在模块级数据并行。

这个容器通过在批尺寸分块分割在整个指定的设备上的输入并行化给定的`模块
`的应用（其他目的将每一次设备中复制）。在正向通，该模块被复制在每个设备上，和每个复制品处理输入的一部分。在向后传递，从每个副本梯度求和成原来的模块。

将批料尺寸应比使用GPU的数量大。

另请参见：[ 使用nn.DataParallel而不是多处理 ](notes/cuda.html#cuda-nn-dataparallel-instead)

任意位置和关键字输入被允许被传递到数据并行但某些类型是特殊处理的。张量将 **散**
上暗淡指定（默认为0）。元组，列表和字典类型将是浅复制。其他类型将不同的线程之间共享，并且可以如果模型中的直传写入被破坏。

并行化`模块 `必须对`它的参数和缓冲剂device_ids [0]`运行前此 `数据并行 `模块。

Warning

在每一个前向，`模块 `是 **复制** 在每台设备上，因此任何更新在`向前 `运行模块会迷路。例如，如果`模块 `具有在每个`递增计数器属性向前
`，它会一直停留在初始值，因为更新其上后`转发 `破坏了副本完成。然而， `数据并行 `保证`设备上的副本[0]
`将具有其参数和缓冲器共享存储与基部并行`模块 `。所以 **就地** [0] 将被记录`设备上更新的参数或缓冲剂。例如， `BatchNorm2d
`和 `spectral_norm（） `依靠这种行为来更新缓冲区。`

Warning

上`向前和向后的钩子定义的模块 `及其子模块将被调用`LEN（device_ids）
`次，每次用定位于特定的输入设备。特别地，钩只保证以正确的顺序相对于操作上的相应装置来执行。例如，它不能保证通过 ``被所有 `
len个之前执行register_forward_pre_hook（）设置挂钩（ device_ids） ``向前（） `
电话，但每一个这样的钩之前执行相应的 `该设备的前向（） `呼叫。

Warning

当`模块 `在`向前返回一个标（即，0维张量）（） `，此包装将返回长度的矢量等于在数据并行使用的设备，包含来自每个设备的结果的数量。

Note

有在使用`收拾 序列的微妙 - & GT ;  复发 网络 - & GT ;  解压 序列 `在 `模块图案 `包裹在 `数据并行 `。参见[
我经常性的网络不与数据并行 ](notes/faq.html#pack-rnn-unpack-with-data-
parallelism)部分中常见问题的细节工作。

Parameters

    

  * **模块** （ _模块_ ） - 模块可以并行

  * **device_ids** （ _蟒的列表：INT_ _或_ [ _torch.device_ ](tensor_attributes.html#torch.torch.device "torch.torch.device")） - CUDA设备（默认值：所有设备）

  * **output_device** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _torch.device_ ](tensor_attributes.html#torch.torch.device "torch.torch.device")） - 输出的设备位置（默认值：device_ids [0]）

Variables

    

**〜DataParallel.module** （ _模块_ ） - 该模块可以并行

Example:

    
    
    >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
    >>> output = net(input_var)  # input_var can be on any device, including CPU
    

###  DistributedDataParallel 

_class_`torch.nn.parallel.``DistributedDataParallel`( _module_ ,
_device_ids=None_ , _output_device=None_ , _dim=0_ , _broadcast_buffers=True_
, _process_group=None_ , _bucket_cap_mb=25_ , _find_unused_parameters=False_ ,
_check_reduction=False_
)[[source]](_modules/torch/nn/parallel/distributed.html#DistributedDataParallel)

    

分布式数据并行实现了基于`torch.distributed`封装在模块级。

这个容器通过在批尺寸分块分割在整个指定的设备上的输入并行化给定的模块的应用。该模块被复制每台机器和每个设备上，并且每个这样的副本处理输入的一部分。在向后传递，从每个节点的梯度进行平均。

将批料尺寸应比本地使用GPU的数量大。

参见：[ 基础 ](distributed.html#distributed-basics)和[ 使用nn.DataParallel而不是多处理
[HTG7。如](notes/cuda.html#cuda-nn-dataparallel-instead) `
上输入相同的约束torch.nn.DataParallel`适用。

这个类的创作要求`torch.distributed`被已经初始化，通过调用[ `
torch.distributed.init_process_group（） `
](distributed.html#torch.distributed.init_process_group
"torch.distributed.init_process_group")。

`DistributedDataParallel`可在以下两种方式使用：

  1. 单进程多GPU

在这种情况下，一个单一的过程将每个主机/节点上催生了每个进程将在它的运行节点的所有GPU的操作。要使用`DistributedDataParallel
[HTG3以这种方式，你可以简单的构建模型，如下：`

    
    
    >>> torch.distributed.init_process_group(backend="nccl")
    >>> model = DistributedDataParallel(model) # device_ids will include all GPU devices by default
    

  2. 多进程单GPU

这是使用`高度推荐的方法DistributedDataParallel
`，其中多个进程，其中的每一个在单个GPU工作。这是目前使用PyTorch做数据并行训练最快的方法和适用于单节点（多GPU）和多节点数据并行训练。它被证明比
`torch.nn.DataParallel`为单节点多GPU数据并行训练显著更快。

这里是如何使用它：用N
GPU的每台主机上，你应该产卵N个流程，同时确保每道工序上的单个GPU单独工作从0到N-1。因此，它是你的工作，以确保您的培训脚本通过调用一个给定的GPU工作：

    
    
    >>> torch.cuda.set_device(i)
    

其中i是从0到N-1。在每一个过程中，你应该参考以下构造此模块：

    
    
    >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
    >>> model = DistributedDataParallel(model, device_ids=[i], output_device=i)
    

为了产卵了每个节点的多个进程，则可以使用`torch.distributed.launch`或`
torch.multiprocessing.spawn`

Note

`NCCL`后端是目前与多进程单GPU分布式训练中使用最快和高度推荐的后端，这既适用于单节点和多节点分布式训练

Note

该模块还支持混合精度分布式训练。这意味着，你的模型可以有不同类型的参数，如混合类型FP16和FP32的，对这些混合类型的参数梯度减少将只是正常工作。还要注意的是`
NCCL`后端是目前FP16 / FP32混合精度训练最快，强烈推荐后端。

Note

如果您使用`torch.save`在一个进程设置检查点模块，以及`torch.load`一些其他进程来恢复它，确保`map_location
`正确每个进程配置。无`map_location`，`torch.load`将恢复模块，其中所述模块是从保存器件。

Warning

此模块只能与`GLOO`和`NCCL`后端。

Warning

构造，向前方法，以及输出（或该模块的输出的函数）的分化是一个分布式同步点。考虑到这一点的情况下，不同的过程可能会执行不同的代码。

Warning

该模块假定所有的参数都在被它创建的时候模型上注册。没有参数应增加，也没有晚删除。同样适用于缓冲区。

Warning

该模块假定在每个分布式过程的模型被注册的所有参数都以相同的顺序。模块本身将以下模型的登记参数的相反顺序进行梯度所有还原。换句话说，它是用户的责任，以确保每个分布式过程具有完全相同的模型，因此完全相同的参数登记顺序。

Warning

该模块假定所有的缓冲区和梯度密集。

Warning

此模块不一起工作[ `torch.autograd.grad（） `](autograd.html#torch.autograd.grad
"torch.autograd.grad")（即，它将仅当梯度在`[HTG7要累积的工作] .grad `的参数属性）。

Warning

如果打算使用该模块带有`NCCL`后端或`GLOO
`后端（即使用的Infiniband），具有的DataLoader在一起使用多个工人，请改变多处理开始方法`forkserver`（Python
3中只）或`菌种 `。不幸的是GLOO（使用的Infiniband）和NCCL2不是叉安全，你可能会经历死锁，如果你不更改此设置。

Warning

上`模块 `及其子模块将不再被调用，除非钩在`向前初始化（） [HTG7向前和向后的钩子限定] 方法。`

Warning

你不应该尝试DistributedDataParallel结束了你的模型后，改变你的模型参数。换句话说，结束了与DistributedDataParallel你的模型时，DistributedDataParallel的构造函数将登记于模型本身的所有参数在施工时的附加梯度降低功能。如果您在DistributedDataParallel施工后改变模型的参数，这是不支持的和意想不到的行为可能发生，因为有些参数的梯度减少功能可能不会被调用。

Note

参数从不进程之间播出。该模块对梯度全减少步骤，并且假设它们将被优化器中以相同的方式的所有过程被修改。缓冲液（例如BatchNorm数据）是从列0的过程中的模块广播，给系统中的在每一个迭代中的所有其它复制品。

Parameters

    

  * **module** ( _Module_) – module to be parallelized

  * **device_ids** （ _蟒的列表：INT_ _或_ [ _torch.device_ ](tensor_attributes.html#torch.torch.device "torch.torch.device")） - CUDA设备。当输入模块驻留在单个CUDA设备上这应该只被提供。对于单器件模块时，`i``th  ：ATTR：`module` 副本 是 放置 在 ``device_ids [I]`。对于多设备模块和CPU模块，device_ids必须是无或一个空列表，并且输入数据为直传必须放置在正确的设备上。 （默认值：单器件模块的所有设备）

  * **output_device** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _或_ [ _torch.device_ ](tensor_attributes.html#torch.torch.device "torch.torch.device")） - 用于输出的装置位置单设备CUDA模块。对于多设备模块和CPU模块，它必须是无，模块本身决定了输出位置。 （默认值：device_ids [0]为单器件模块）

  * **broadcast_buffers** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 标志，使在前进功能的开始同步的模块的（广播）的缓冲器。 （默认值：`真 `）

  * **process_group** \- 用于分布式数据所有还原处理组。如果`无 `，默认处理组，它是由`建立`torch.distributed.init_process_group``，将被使用。 （默认值：`无 `）

  * **bucket_cap_mb** \- DistributedDataParallel将桶的参数分成多个存储桶，使得每个桶的梯度减小可以潜在地与向后计算重叠。 `bucket_cap_mb`控制以兆字节为桶大小（MB）（默认值：25）

  * **find_unused_pa​​rameters** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 遍历包含在返回值的所有张量的autograd曲线图中的包装的模块的`向前 `功能。不接收梯度，因为这图的一部分参数被抢先标记为准备减少。请注意，所有`向前 `从模块参数导出的输出必须参与计算损失和更高的梯度计算。如果他们不这样做，这个包装将挂起等待autograd产生这些参数的梯度。其他未使用的从模块的参数导出的任何输出可从autograd图表使用`torch.Tensor.detach`脱离。 （默认值：`假 `）

  * **check_reduction** \- 设置为`真 `，它使DistributedDataParallel如果前一次迭代落后的减少是在每次迭代的正向功能年初成功发行自动检查时。你通常不需要启用这个选项，除非你正在观察怪异行为，如不同等级越来越不同梯度，如果DistributedDataParallel正确使用应该不会发生。 （默认值：`假 `）

Variables

    

**〜DistributedDataParallel.module** （ _模块_ ） - 该模块可以并行

Example:

    
    
    >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
    >>> net = torch.nn.DistributedDataParallel(model, pg)
    

`no_sync`()[[source]](_modules/torch/nn/parallel/distributed.html#DistributedDataParallel.no_sync)

    

上下文管理器跨DDP过程禁用梯度同步。在这方面，将梯度上模块的变量，这将在后面的第一个前后通离开上下文同步进行累积。

Example:

    
    
    >>> ddp = torch.nn.DistributedDataParallel(model, pg)
    >>> with ddp.no_sync():
    ...   for input in inputs:
    ...     ddp(input).backward()  # no synchronization, accumulate grads
    ... ddp(another_input).backward()  # synchronize grads
    

## 公用事业

###  clip_grad_norm_ 

`torch.nn.utils.``clip_grad_norm_`( _parameters_ , _max_norm_ , _norm_type=2_
)[[source]](_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_)

    

的参数可迭代的视频梯度范数。

范数上计算所有梯度在一起，如同它们连接成一个单一的载体中。梯度就地修改。

Parameters

    

  * **参数** （ _可迭代_ _[_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _]或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 张量的一个可迭代或单个张量，将有梯度归一化

  * **max_norm** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 最大梯度的范数

  * **norm_type** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 所使用的p-范数的类型。可以用`'INF' `为无穷规范。

Returns

    

的参数的总范数（视为单个矢量）。

###  clip_grad_value_ 

`torch.nn.utils.``clip_grad_value_`( _parameters_ , _clip_value_
)[[source]](_modules/torch/nn/utils/clip_grad.html#clip_grad_value_)

    

在指定的值的参数可迭代的视频梯度。

梯度就地修改。

Parameters

    

  * **parameters** ( _Iterable_ _[_[ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _] or_[ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – an iterable of Tensors or a single Tensor that will have gradients normalized

  * **clip_value** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 最大允许梯度的值。的梯度的范围限幅 [ -clip_value  ， clip_value  \左[\ {文本-clip \ _value}，\ {文本夹\ _value} \右]  [ -clip_value  ， clip_value 

###  parameters_to_vector 

`torch.nn.utils.``parameters_to_vector`( _parameters_
)[[source]](_modules/torch/nn/utils/convert_parameters.html#parameters_to_vector)

    

转换参数，以一个矢量

Parameters

    

**参数** （ _可迭代_ _[_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") __ ） -
张量的迭代器，是模型的参数。

Returns

    

由单个向量表示的参数

###  vector_to_parameters 

`torch.nn.utils.``vector_to_parameters`( _vec_ , _parameters_
)[[source]](_modules/torch/nn/utils/convert_parameters.html#vector_to_parameters)

    

将一个向量的参数

Parameters

    

  * **VEC** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 一个单一的矢量表示的模型的参数。

  * **parameters** ( _Iterable_ _[_[ _Tensor_](tensors.html#torch.Tensor "torch.Tensor") _]_ ) – an iterator of Tensors that are the parameters of a model.

###  weight_norm 

`torch.nn.utils.``weight_norm`( _module_ , _name='weight'_ , _dim=0_
)[[source]](_modules/torch/nn/utils/weight_norm.html#weight_norm)

    

适用重量归一化到给定模块中的参数。

w=gv∥v∥\mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|} w=g∥v∥v​

重量归一化是解耦其方向的张量重量的大小的重新参数化。这取代由`name指定的参数 `（例如`'重量' `）使用两个参数：一个指定的幅度（例如，`
'weight_g' `）和一个指定方向（例如，`'weight_v' `）。重量归一化是通过每一个`向前（）
`呼叫之前重新计算从所述幅度和方向的重量张量的钩实现。

默认情况下，与`暗淡= 0`，规范独立地，每个输出通道/平面来计算。为了计算在整个重量张量范数，用`暗淡=无 `。

参见[ https://arxiv.org/abs/1602.07868 ](https://arxiv.org/abs/1602.07868)

Parameters

    

  * **模块** （ _模块_ ） - 包含模块

  * **名称** （[ _STR_ ](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)") _，_ _可选_ ） - 权重参数的名称

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 维在其上计算标准

Returns

    

原始模块与重量规范钩

Example:

    
    
    >>> m = weight_norm(nn.Linear(20, 40), name='weight')
    >>> m
    Linear(in_features=20, out_features=40, bias=True)
    >>> m.weight_g.size()
    torch.Size([40, 1])
    >>> m.weight_v.size()
    torch.Size([40, 20])
    

###  remove_weight_norm 

`torch.nn.utils.``remove_weight_norm`( _module_ , _name='weight'_
)[[source]](_modules/torch/nn/utils/weight_norm.html#remove_weight_norm)

    

删除从一个模块的重量归一化重新参数化。

Parameters

    

  * **module** ( _Module_) – containing module

  * **name** ([ _str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)") _,_ _optional_ ) – name of weight parameter

例

    
    
    >>> m = weight_norm(nn.Linear(20, 40))
    >>> remove_weight_norm(m)
    

###  spectral_norm 

`torch.nn.utils.``spectral_norm`( _module_ , _name='weight'_ ,
_n_power_iterations=1_ , _eps=1e-12_ , _dim=None_
)[[source]](_modules/torch/nn/utils/spectral_norm.html#spectral_norm)

    

适用谱归一化到给定模块中的参数。

WSN=Wσ(W),σ(W)=max⁡h:h≠0∥Wh∥2∥h∥2\mathbf{W}_{SN} =
\dfrac{\mathbf{W}}{\sigma(\mathbf{W})}, \sigma(\mathbf{W}) = \max_{\mathbf{h}:
\mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}
WSN​=σ(W)W​,σ(W)=h:h​=0max​∥h∥2​∥Wh∥2​​

谱归一化通过重新缩放重量张量与谱范数 σ稳定在剖成对抗性网络（甘斯）鉴别器（影评）的训练 \西格玛 σ
使用幂迭代方法计算的权重矩阵的。如果重量张量的维数大于2，它被整形以幂迭代法在二维获得谱范数。这是通过计算光谱范数和再缩放重量之前每`向前
`（）调用的钩实现。

见剖成对抗性网络 [谱归。](https://arxiv.org/abs/1802.05957)

Parameters

    

  * **module** ( _nn.Module_) – containing module

  * **name** ([ _str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)") _,_ _optional_ ) – name of weight parameter

  * **n_power_iterations** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 功率的迭代次数来计算谱范

  * **EPS** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 小量在计算准则的数值稳定性

  * **暗淡** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 维对应的输出数，默认是`0`，除了模块，其ConvTranspose {1,2,3} d的情况下，当它是`1`

Returns

    

原始模块与所述谱范数钩

Example:

    
    
    >>> m = spectral_norm(nn.Linear(20, 40))
    >>> m
    Linear(in_features=20, out_features=40, bias=True)
    >>> m.weight_u.size()
    torch.Size([40])
    

###  remove_spectral_norm 

`torch.nn.utils.``remove_spectral_norm`( _module_ , _name='weight'_
)[[source]](_modules/torch/nn/utils/spectral_norm.html#remove_spectral_norm)

    

删除从一个模块的光​​谱归一化重新参数化。

Parameters

    

  * **module** ( _Module_) – containing module

  * **name** ([ _str_](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)") _,_ _optional_ ) – name of weight parameter

Example

    
    
    >>> m = spectral_norm(nn.Linear(40, 10))
    >>> remove_spectral_norm(m)
    

###  PackedSequence 

`torch.nn.utils.rnn.``PackedSequence`( _data_ , _batch_sizes=None_ ,
_sorted_indices=None_ , _unsorted_indices=None_
)[[source]](_modules/torch/nn/utils/rnn.html#PackedSequence)

    

保持数据和填充序列的batch_sizes 的`列表。`

所有RNN模块接受打包序列作为输入。

Note

这个类的实例绝不应手动创建。这意味着它们是用相同的功能被实例化`pack_padded_sequence（） `。

批量大小表示在批次中的每个序列步骤的数量的元件，而不是不同的序列长度传递到 `pack_padded_sequence（） `。例如，给定的数据`
ABC`和`× `中的 `PackedSequence`将包含与`batch_sizes = [数据`axbc`2,1,1]`。

Variables

    

  * **〜PackedSequence.data** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 张量含有包装序列

  * **〜PackedSequence.batch_sizes** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 在每个序列步骤保持约批量大小信息的整数的张量

  * **〜PackedSequence.sorted_indices** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 整数张量保持如何 `PackedSequence`是从序列构建。

  * **〜PackedSequence.unsorted_indices** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选_ ） - 整数张量保持如何以恢复原始与正确的顺序的序列。

Note

`数据 `可以在任意设备和任意的D型。 `sorted_indices`和`unsorted_indices`必须`torch.int64
`相同装置上张量`数据 `。

然而，`batch_sizes`应当总是一个CPU `torch.int64`张量。

这个不变的保持在整个 `PackedSequence`类，并且该构建体的所有功能的：类：PackedSequence
在PyTorch（即，它们只通过在张量符合此约束）。

###  pack_padded_sequence 

`torch.nn.utils.rnn.``pack_padded_sequence`( _input_ , _lengths_ ,
_batch_first=False_ , _enforce_sorted=True_
)[[source]](_modules/torch/nn/utils/rnn.html#pack_padded_sequence)

    

包含可变长度的填充序列的张量。

`输入 `可以是大小`T  × B  × *`其中 T 是最长的序列的长度（等于`长度[0]`），`B`是批量大小，和`*
`是任何数目的维度（包括0）的。如果`batch_first`是`真 `，`B  × T ...  × *``输入 `预计。

对于未排序的序列，用 enforce_sorted =假。如果`enforce_sorted`是`真 `，所述序列应通过长度以递减的顺序排序的，即`
输入[ ：0]`应该是最长的序列，和`输入[：，B-1]`最短的一个。  enforce_sorted =真仅用于ONNX出口必要的。

Note

该函数接受具有至少两个维度的任何输入。你可以运用它来包装标签，并使用RNN的输出与他们直接计算的损失。张量可以从 `PackedSequence`
对象通过访问其`。数据 `属性进行检索。

Parameters

    

  * **输入** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 填充料的可变长度的序列。

  * **长度** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 每批元件的序列长度的列表。

  * **batch_first** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，输入预计`B  × T  × *`格式。

  * **enforce_sorted** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，输入被预期含有由长度以递减的顺序排序的序列。如果`假 `，这种情况不检查。默认值：`真 [HTG21。`

Returns

    

一个 `PackedSequence`对象

###  pad_packed_sequence 

`torch.nn.utils.rnn.``pad_packed_sequence`( _sequence_ , _batch_first=False_ ,
_padding_value=0.0_ , _total_length=None_
)[[source]](_modules/torch/nn/utils/rnn.html#pad_packed_sequence)

    

垫的填充料的可变长度的序列。

这是一个逆运算为 `pack_padded_sequence（） `。

返回的张量的数据将是大小`T  × B  × *`的，其中 T 是最长的序列的长度和 B 是批量大小。如果`batch_first
`是True，则数据将被移位到`B  × T  X  *`格式。

批处理元素将通过逐渐降低它们的长度进行排序。

Note

`total_length`是有用的实施`收拾 序列 - & GT ;  复发 网络 - & GT ;  解压 序列 `在图案一个 `模块 `
包裹在 `数据并行 `。参见[ 这个常见问题解答部分 ](notes/faq.html#pack-rnn-unpack-with-data-
parallelism)了解详情。

Parameters

    

  * **序列** （ _PackedSequence_ ） - 批次到垫

  * **batch_first** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，输出将在`B  × T  × *`格式。

  * **padding_value** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 用于填充元素的值。

  * **total_length** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 如果不是`无 `，输出将被填充到具有长度`total_length`。此方法将抛出[ `ValueError异常 `](https://docs.python.org/3/library/exceptions.html#ValueError "\(in Python v3.7\)")如果`total_length`小于`最大序列长度序列 `。

Returns

    

张量的含元组中的填充序列，和包含在所述批次中的每个序列的长度的列表中的张量。

###  pad_sequence 

`torch.nn.utils.rnn.``pad_sequence`( _sequences_ , _batch_first=False_ ,
_padding_value=0_ )[[source]](_modules/torch/nn/utils/rnn.html#pad_sequence)

    

垫可变长度张量与`padding_value列表 `

`pad_sequence`堆叠张量的沿一个新的维度的列表，并把它们垫相等的长度。例如，如果输入是序列的大小为列表`L  × *
`并且如果batch_first是False，并且`T  × B  × *`否则。

B 是批量大小。它等于在`序列 `的元素数。  T 是最长的序列的长度。  L 是序列的长度。  * 是任意数量的尾随尺寸，包括没有的。

Example

    
    
    >>> from torch.nn.utils.rnn import pad_sequence
    >>> a = torch.ones(25, 300)
    >>> b = torch.ones(22, 300)
    >>> c = torch.ones(15, 300)
    >>> pad_sequence([a, b, c]).size()
    torch.Size([25, 3, 300])
    

Note

此函数返回的大小`T  × B  × *`或张量`B  × T  × *`其中 T
是最长的序列的长度。此函数假定尾随尺寸和序列的所有的张量的类型相同。

Parameters

    

  * **序列** （[ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)") _[_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") __ ） - 可变长度的序列的列表。

  * **batch_first** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 输出将处于`B  × T  × *`如果为True，或在`T  X  B  × *`否则

  * **padding_value** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 用于填充元件值。默认值：0。

Returns

    

的大小`T  × B  × *`如果`张量 batch_first`是`假 `。的张量大小`B  × T  × *`否则

###  pack_sequence 

`torch.nn.utils.rnn.``pack_sequence`( _sequences_ , _enforce_sorted=True_
)[[source]](_modules/torch/nn/utils/rnn.html#pack_sequence)

    

包长度可变张量的列表

`序列 `应该是大小`L  × *`，其中的张量的列表 L 是一个序列的长度和 * 是任意数量的尾随尺寸，包括零的。

对于未排序的序列，用 enforce_sorted =假。如果`enforce_sorted`是`真 `，所述序列应在降低长度的顺序进行排序。 `
enforce_sorted  =  真 `仅用于ONNX出口必要的。

Example

    
    
    >>> from torch.nn.utils.rnn import pack_sequence
    >>> a = torch.tensor([1,2,3])
    >>> b = torch.tensor([4,5])
    >>> c = torch.tensor([6])
    >>> pack_sequence([a, b, c])
    PackedSequence(data=tensor([ 1,  4,  6,  2,  5,  3]), batch_sizes=tensor([ 3,  2,  1]))
    

Parameters

    

  * **序列** （[ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)") _[_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") __ ） - 减小长度的序列的列表。

  * **enforce_sorted** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，检查该输入包含由长度以递减的顺序排序的序列。如果`假 `，这种情况不检查。默认值：`真 [HTG21。`

Returns

    

a `PackedSequence`object

### 拼合

[Next ![](_static/images/chevron-right-orange.svg)](nn.functional.html
"torch.nn.functional") [![](_static/images/chevron-right-orange.svg)
Previous](storage.html "torch.Storage")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * torch.nn 
    * 参数
    * 集装箱
      * 模块
      * 序贯
      * ModuleList 
      * ModuleDict 
      * 参数列表
      * ParameterDict 
    * 卷积层
      * Conv1d 
      * Conv2d 
      * Conv3d 
      * ConvTranspose1d 
      * ConvTranspose2d 
      * ConvTranspose3d 
      * 展开
      * 折叠
    * 池层
      * MaxPool1d 
      * MaxPool2d 
      * MaxPool3d 
      * MaxUnpool1d 
      * MaxUnpool2d 
      * MaxUnpool3d 
      * AvgPool1d 
      * AvgPool2d 
      * AvgPool3d 
      * FractionalMaxPool2d 
      * LPPool1d 
      * LPPool2d 
      * AdaptiveMaxPool1d 
      * AdaptiveMaxPool2d 
      * AdaptiveMaxPool3d 
      * AdaptiveAvgPool1d 
      * AdaptiveAvgPool2d 
      * AdaptiveAvgPool3d 
    * 填充层
      * ReflectionPad1d 
      * ReflectionPad2d 
      * ReplicationPad1d 
      * ReplicationPad2d 
      * ReplicationPad3d 
      * ZeroPad2d 
      * ConstantPad1d 
      * ConstantPad2d 
      * ConstantPad3d 
    * 非线性激活（加权和，非线性）
      * ELU 
      * Hardshrink 
      * Hardtanh 
      * LeakyReLU 
      * LogSigmoid 
      * MultiheadAttention 
      * PReLU 
      * RELU 
      * ReLU6 
      * RReLU 
      * 九色鹿
      * CELU 
      * 乙状结肠
      * Softplus 
      * Softshrink 
      * Softsign 
      * 双曲正切
      * Tanhshrink 
      * 阈值
    * 非线性激活（其他）
      * Softmin 
      * 使用SoftMax 
      * Softmax2d 
      * LogSoftmax 
      * AdaptiveLogSoftmaxWithLoss 
    * 规范化层
      * BatchNorm1d 
      * BatchNorm2d 
      * BatchNorm3d 
      * GroupNorm 
      * SyncBatchNorm 
      * InstanceNorm1d 
      * InstanceNorm2d 
      * InstanceNorm3d 
      * LayerNorm 
      * LocalResponseNorm 
    * 复发性层
      * RNN 
      * LSTM 
      * GRU 
      * RNNCell 
      * LSTMCell 
      * GRUCell 
    * 变压器层
      * 变压器
      * TransformerEncoder 
      * TransformerDecoder 
      * TransformerEncoderLayer 
      * TransformerDecoderLayer 
    * 线性层
      * 身份
      * 线性
      * 双线性
    * 漏失层
      * 降
      * Dropout2d 
      * Dropout3d 
      * AlphaDropout 
    * 稀疏层
      * 嵌入
      * EmbeddingBag 
    * 距离函数
      * 余弦相似性
      * PairwiseDistance 
    * 损失函数
      * L1Loss 
      * MSELoss 
      * CrossEntropyLoss 
      * CTCLoss 
      * NLLLoss 
      * PoissonNLLLoss 
      * KLDivLoss 
      * BCELoss 
      * BCEWithLogitsLoss 
      * MarginRankingLoss 
      * HingeEmbeddingLoss 
      * MultiLabelMarginLoss 
      * SmoothL1Loss 
      * SoftMarginLoss 
      * MultiLabelSoftMarginLoss 
      * CosineEmbeddingLoss 
      * MultiMarginLoss 
      * TripletMarginLoss 
    * 视觉层
      * PixelShuffle 
      * 上采样
      * UpsamplingNearest2d 
      * UpsamplingBilinear2d 
    * 数据并行层（多GPU，分布式）
      * 数据并行
      * DistributedDataParallel 
    * 公用设施
      * clip_grad_norm_ 
      * clip_grad_value_ 
      * parameters_to_vector 
      * vector_to_parameters 
      * weight_norm 
      * remove_weight_norm 
      * spectral_norm 
      * remove_spectral_norm 
      * PackedSequence 
      * pack_padded_sequence 
      * pad_packed_sequence 
      * pad_sequence 
      * pack_sequence 
      * 拼合

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

