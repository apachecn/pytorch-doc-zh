# TorchScript

  * 创建TorchScript代码

  * 混合跟踪和脚本

  * TorchScript语言参考

    * 类型

      * 默认类型

      * 可选类型细化

      * 用户定义类型

    * 表达式

      * 字面

        * 列表构造

        * 元组建筑

        * 字典建筑

      * 变量

      * 算术运算符

      * 比较运算符

      * 逻辑运算符

      * 下标

      * 函数调用

      * 方法调用

      * 三元表达式

      * 粪中

      * 访问模块参数

    * 下列

    * 分辨率可变

    * Python中值的使用

      * 功能

      * 属性查找有关python模块

      * Python的定义的常量

      * 模块属性

    * 调试

      * [HTG0用于调试禁用JIT 

      * 检查代码

      * 解释图表

      * 跟踪边缘情况

      * 自动跟踪检查

      * 示踪剂警告

  * 常见问题

    * 内建函数

TorchScript是创建PyTorch代码序列化和优化的模型的方式。任何TorchScript程序可以从一个Python程序被保存，并在过程中不存在的Python依赖加载。

我们提供工具来递增地转变从一个纯Python程序模型，以能够独立地从Python中运行，诸如在一个独立的C
++程序的程序TorchScript。这使得训练使用Python中熟悉的工具在PyTorch模型，然后通过TorchScript导出模型的生产环境下的Python程序可能是不利的。性能和多线程的原因。

## 创建TorchScript代码

`torch.jit.``script`( _obj_ , _optimize=None_ , __frames_up=0_ , __rcb=None_
)[[source]](_modules/torch/jit.html#script)

    

脚本编写的函数或`nn.Module`将检查源代码，使用编译器TorchScript编译为TorchScript代码，并返回一个`
ScriptModule`或`torch._C.Function`。

**Scripting a function**

    

的`@ torch.jit.script`装饰将构造一个`torch._C.Function`。

实施例（脚本的函数）：

    
    
    import torch
    @torch.jit.script
    def foo(x, y):
        if x.max() > y.max():
            r = x
        else:
            r = y
        return r
    

**Scripting an nn.Module**

    

编写脚本的`nn.Module`缺省将编译`向前 `方法和递归编译任何方法，子模块，并且功能由[HTG8称为] 向前 。如果`nn.Module
`只使用在TorchScript支持的功能，以原模块代码没有改变应该是必要的。

实施例（脚本用参数a单模）：

    
    
    import torch
    
    class MyModule(torch.nn.Module):
        def __init__(self, N, M):
            super(MyModule, self).__init__()
            # This parameter will be copied to the new ScriptModule
            self.weight = torch.nn.Parameter(torch.rand(N, M))
    
            # When this submodule is used, it will be compiled
            self.linear = torch.nn.Linear(N, M)
    
        def forward(self, input):
            output = self.weight.mv(input)
    
            # This calls the `forward`method of the `nn.Linear`module, which will
            # cause the `self.linear`submodule to be compiled to a `ScriptModule`here
            output = self.linear(output)
            return output
    
    scripted_module = torch.jit.script(MyModule())
    

实施例（与脚本跟踪子模块的模块）：

    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class MyModule(nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            # torch.jit.trace produces a ScriptModule's conv1 and conv2
            self.conv1 = torch.jit.trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))
            self.conv2 = torch.jit.trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))
    
        def forward(self, input):
          input = F.relu(self.conv1(input))
          input = F.relu(self.conv2(input))
          return input
    
    scripted_module = torch.jit.script(MyModule())
    

要编译比`转发 `（和递归编译任何东西它调用）以外的方法，添加`@ torch.jit.export`装饰器方法。

`torch.jit.``trace`( _func_ , _example_inputs_ , _optimize=None_ ,
_check_trace=True_ , _check_inputs=None_ , _check_tolerance=1e-05_ ,
__force_outplace=False_ , __module_class=None_ , __compilation_unit=
<torch._C.CompilationUnit object>_)[[source]](_modules/torch/jit.html#trace)

    

跟踪的功能，并返回一个可执行`ScriptModule`或`torch.jit._C.Function`将使用刚刚在时间进行优化汇编。

警告

只有正确地跟踪记录功能，并且不依赖数据（例如，不具有在张量数据条件语句），并且没有任何未跟踪外部依赖（例如，执行输入/输出或访问全局变量）模块。如果您跟踪这些模型，你可以静静地坐上模型的后续调用不正确的结果。示踪将尝试发出做某事时警告，可能会导致产生不正确的轨迹。

Parameters

    

  * **FUNC** （ _可调用_ _或_ [ _torch.nn.Module_ ](nn.html#torch.nn.Module "torch.nn.Module")） - Python函数或`torch.nn.Module`将与`example_inputs`运行。参数，并返回至`FUNC`必须张量或包含张量（可能是嵌套）元组。

  * **example_inputs** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")） - 的同时跟踪将被传递给函数示例输入的元组。将得到的迹线可以与不同类型和形状假设跟踪操作的输入来运行支持这些类型和形状。 `example_inputs`也可以是单一的张量在这种情况下，它是自动包装在元组

Keyword Arguments

    

  * **check_trace** （[ _布尔_ ](storage.html#torch.FloatStorage.bool "torch.FloatStorage.bool") _，_ _可选_ ） - 检查是否相同的输入通过跟踪代码运行产生相同的输出。默认值：`真 [HTG13。你可能想禁用此，如果，例如，您的网络中包含非确定性OPS，或者如果你是确保网络尽管检查故障是否正确。`

  * **check_inputs** （ _元组的列表_ _，_ _可选[HTG7） - 应使用要检查的跟踪的输入参数的元组的名单是什么是期待。每个元组相当于一组输入参数，将在`example_inputs`来指定。为了达到最佳效果，通过一组检查输入代表性的形状和类型的你期望在网络上看到输入的空间的。如果未指定，则原始`example_inputs 用于检查`_

  * **check_tolerance** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 浮点比较耐受性检查过程中使用。这可以被用来放松中的结果发散数值为一个已知的原因，如操作者的融合事件的检查严格。

Returns

    

如果`可调用 `是`nn.Module`或`向前（ `的）`nn.Module`，`痕量 `返回与`ScriptModule
`对象的单个`向前（） [HTG27含有被跟踪代码方法。返回的`ScriptModule`将具有相同的组的子模块和参数作为原始`nn.Module
`。如果`可调用 `是一个独立的函数，`痕量 `返回`torch.jit._C.Function``

例：

    
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 1, 3)
    
        def forward(self, x):
            return self.conv(x)
    
        def weighted_kernel_sum(self, weight):
            return weight * self.conv.weight
    
    example_weight = torch.rand(1, 1, 3, 3)
    example_forward_input = torch.rand(1, 1, 3, 3)
    n = Net()
    # the following two calls are equivalent
    module = torch.jit.trace_module(n, example_forward_input)
    module = torch.jit.trace_module(n.forward, example_forward_input)
    

_class_`torch.jit.``ScriptModule`( _optimize=None_ , __qualified_name=None_ ,
__compilation_unit=None_ , __cpp_module=None_
)[[source]](_modules/torch/jit.html#ScriptModule)

    

在TorchScript核心数据结构是`ScriptModule`。它是火炬的`nn.Module
`类似物和表示整个模型作为子模块的一棵树。像正常模块，在`ScriptModule`可以有子模块，参数和方法每个单独模块。在`nn.Module
`S的方法被实现为Python函数，但在`ScriptModule`S的方法被实现为TorchScript功能，一个statically-
Python中的类型子集，它包含了所有PyTorch内置的张量操作。这种差异使您ScriptModules代码，而不需要一个Python解释器运行。

`ScriptModule`S以两种方式产生：

**跟踪：**

> 使用`torch.jit.trace`和`torch.jit.trace_module
`，可以把现有的模块或Python函数成TorchScript `torch._C.Function`或`ScriptModule
`。你必须提供例如输入，我们运行的功能，记录所有的张量所执行的操作。 *将所得的独立功能的记录产生`torch._C.Function`。
*将所得的`向前 `的`nn.Module`或`nn.Module 函数记录`生产`ScriptModule
`。该模块还包含了原来的模块有以及任何参数。

>

> 实施例（跟踪函数）：

>  
>  
>     import torch

>     def foo(x, y):

>         return 2 * x + y

>     traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))

>  
>

> 注意

>

> 跟踪一个独立的功能将构造一个`torch._C.Function`追踪`nn.Module``s``向前 `将构造一个`
ScriptModule`

>

> 实施例（跟踪现有模块）：

>  
>  
>     import torch

>     class Net(nn.Module):

>         def __init__(self):

>             super(Net, self).__init__()

>             self.conv = nn.Conv2d(1, 1, 3)

>  
>         def forward(self, x):

>             return self.conv(x)

>  
>         def weighted_kernel_sum(self, weight):

>             return weight * self.conv.weight

>  
>  
>     n = Net()

>     example_weight = torch.rand(1, 1, 3, 3)

>     example_forward_input = torch.rand(1, 1, 3, 3)

>  
>     # all three trace calls below are equivalent

>     # and construct `ScriptModule`with a single `forward`method

>     module = torch.jit.trace(n.forward, example_forward_input) # produces
ScriptModule with `forward`

>     module = torch.jit.trace(n, example_forward_input) # produces
ScriptModule with `forward`

>     module = torch.jit.trace_module(n, inputs) # produces ScriptModule with
`forward`

>  
>     inputs = {'forward' : example_forward_input, 'weighted_kernel_sum' :
example_weight}

>     # trace_module produces `ScriptModule`with two methods:

>     # `forward`and `weighted_kernel_sum`

>     module = torch.jit.trace_module(n, inputs, True, True)

>  
>

> Note

>

>   * 前三跟踪/ trace_module调用是等效的，并返回`ScriptModule`

>

>

>

> 与单一`向前 `方法。 *最后`trace_module`呼叫产生一个`ScriptModule
`用两种方法。当给定函数在给定的张量运行跟踪完成只记录操作。因此，返回的`ScriptModule
`将始终运行于任何输入相同的跟踪图。这有一些重要的启示，当你的模块预计将运行不同操作的集合，根据输入和/或模块的状态。例如，

>

>>   *
跟踪不会记录任何控制流一样，如果语句或循环。当这个控制流程是在您的模块不变，这是好的，它往往内联的控制流决策。但有时控制流实际上是模型本身的一部分。例如，一个经常性的网络是一个环上的输入序列的（可能是动态的）长度。

>>

>>   * 在返回`ScriptModule`，操作是在`训练 `和不同的行为`EVAL`模式将总是执行如它是在它在跟踪期间在模式，无论`
ScriptModule`是其中模式。

>>

>>

>

> 在这样的情况下，跟踪是不恰当的和脚本是一个更好的选择。

**脚本：**

> 您可以直接使用Python语法编写TorchScript代码。你这样做使用`@ torch.jit.script
`装饰的功能和模块。您也可以直接与要编译功能或模块调用`torch.jit.script
[HTG7。在功能方面，函数体被编译成TorchScript。如果施加到`nn.Module`，默认情况下`向前
`方法，它调用被编译的任何方法，并且所有缓冲器和参数原始模块的被复制到一个新的`ScriptModule`。你不应该需要手动构建`
ScriptModule  [HTG23。
TorchScript本身是Python语言的一个子集，因此不会在Python工作的所有功能，但我们提供足够的功能来计算的张量和做相关的控制操作。``

`torch.jit.``save`( _m_ , _f_ , __extra_files=ExtraFilesMap{}_
)[[source]](_modules/torch/jit.html#save)

    

在一个单独的进程保存此模块中使用的离线版本。保存的模块序列化的所有方法，子模块，参数和该模块的属性。它可以使用`炬:: JIT ::负载（文件名） `或与
`负载Python API中被加载到C ++ API`。

为了能够保存模块，它必须不使本机Python功能的任何电话。这意味着，所有的子模块必须的`torch.jit.ScriptModule`亚类。

危险

所有模块，不管他们的设备，总是被载入到CPU加载过程中。这是从 `不同加载的 `语义和在未来可能改变。

Parameters

    

  * **M** \- 一个ScriptModule保存

  * **F** \- 一个类文件对象（必须实现写和flush）或包含文件名的字符串

  * **_extra_files** \- 从文件名映射到内容将被存储为“F”的一部分

Warning

如果您在使用Python 2，`torch.save`不支持`StringIO.StringIO
`为有效的类文件对象。这是因为写方法应该返回写入的字节数; `StringIO.write（） `不执行此操作。

请使用类似`io.BytesIO`代替。

Example:

    
    
    import torch
    import io
    
    
    class MyModule(torch.nn.Module):
        def forward(self, x):
            return x + 10
    
    m = torch.jit.script(MyModule())
    
    # Save to file
    torch.jit.save(m, 'scriptmodule.pt')
    
    # Save to io.BytesIO buffer
    buffer = io.BytesIO()
    torch.jit.save(m, buffer)
    
    # Save with extra files
    extra_files = torch._C.ExtraFilesMap()
    extra_files['foo.txt'] = 'bar'
    torch.jit.save(m, 'scriptmodule.pt', _extra_files=extra_files)
    

`torch.jit.``load`( _f_ , _map_location=None_ , __extra_files=ExtraFilesMap{}_
)[[source]](_modules/torch/jit.html#load)

    

负载的`ScriptModule`以前保存的与 `保存 `

之前保存的所有模块，不管他们的设备，首先被加载到CPU，然后移动到他们从已保存的设备。如果失败（例如，由于运行时系统不具有一定的设备），将引发一个例外。然而，存储器可以被动态地重新映射到一组替代使用
map_location 参数的设备。比较[ `torch.load（） `](torch.html#torch.load
"torch.load")，在此功能map_location 被简化，只接受一个字符串（例如， 'CPU'
'CUDA：0'），或torch.device（例如，torch.device（ 'CPU'））

Parameters

    

  * **F** \- 一个类文件对象（必须实现读，readline的，告诉，求），或包含文件名的字符串

  * **map_location** \- 可以一个字符串（例如，“CPU”，“CUDA：0”），设备（例如，torch.device（“CPU”））

  * **_extra_files** \- 从文件名映射到的内容。在图中给出的额外的文件名会被加载和内容将被存储在所提供的地图。

Returns

    

A `ScriptModule`对象。

Example:

    
    
    torch.jit.load('scriptmodule.pt')
    
    # Load ScriptModule from io.BytesIO object
    with open('scriptmodule.pt', 'rb') as f:
        buffer = io.BytesIO(f.read())
    
    # Load all tensors to the original device
    torch.jit.load(buffer)
    
    # Load all tensors onto CPU, using a device
    torch.jit.load(buffer, map_location=torch.device('cpu'))
    
    # Load all tensors onto CPU, using a string
    torch.jit.load(buffer, map_location='cpu')
    
    # Load with extra files.
    files = {'metadata.json' : ''}
    torch.jit.load('scriptmodule.pt', _extra_files = files)
    print (files['metadata.json'])
    

## 混合跟踪和脚本

在许多情况下，不是跟踪或脚本为模型转换到TorchScript一个更简单的方法。跟踪和脚本可以组成以适应模型的一部分的特殊要求。

脚本函数可以调用追踪功能。当你需要使用围绕一个简单的前馈模型对照流这是特别有用。例如，所述波束搜索的序列序列模型通常将写在脚本但可以调用使用跟踪产生的编码器模块。

Example:

    
    
    import torch
    
    def foo(x, y):
        return 2 * x + y
    traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))
    
    @torch.jit.script
    def bar(x):
        return traced_foo(x, x)
    

究其根源，函数可以调用脚本函数。当一个模型的一小部分需要一些控制流尽管大多数模型仅仅是一个前馈网络，这非常有用。通过跟踪函数调用的脚本功能的内部控制流程正确保存：

Example:

    
    
    import torch
    
    @torch.jit.script
    def foo(x, y):
        if x.max() > y.max():
            r = x
        else:
            r = y
        return r
    
    
    def bar(x, y, z):
        return foo(x, y) + z
    
    traced_bar = torch.jit.trace(bar, (torch.rand(3), torch.rand(3), torch.rand(3)))
    

该组合物也适用于`nn.Module`S以及，在那里它可被用于产生使用跟踪，可以从脚本模块的方法中被称为子模块：

Example:

    
    
    import torch
    import torchvision
    
    class MyScriptModule(torch.nn.Module):
        def __init__(self):
            super(MyScriptModule, self).__init__()
            self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779, 123.68])
                                            .resize_(1, 3, 1, 1))
            self.resnet = torch.jit.trace(torchvision.models.resnet18(),
                                          torch.rand(1, 3, 224, 224))
    
        def forward(self, input):
            return self.resnet(input - self.means)
    
    my_script_module = torch.jit.script(MyScriptModule())
    

##  TorchScript语言参考

TorchScript是Python一个静态类型子集，它可以直接（使用`@ torch.jit.script
`装饰）从Python代码经由跟踪书面或自动生成。当使用追踪，代码自动被上张量仅记录实际运营商和简单地执行并丢弃其它周围Python代码转换成Python的该子集。

当写TorchScript直接使用`@ torch.jit.script
`装饰，程序员必须只使用Python的子集，支持TorchScript。本节介绍什么是TorchScript支持就​​好像它是一个独立的语言的语言参考。在此引用未提及的Python的任何功能都不会TorchScript的一部分。

对于Python的一个子集的任何有效TorchScript功能也是一个有效的Python功能。这使得它能够去除`@ torch.jit.script
`装饰，并使用标准的Python工具，如`PDB
`调试功能。相反的是不正确的：有无效的TorchScript程序许多有效的Python程序。相反，TorchScript特别侧重于那些需要代表火炬神经网络模型的Python的特点。

`PYTORCH_JIT=1`

    

设置环境变量`PYTORCH_JIT = 0
`将禁用所有脚本和追踪注解。如果有难以调试错误在你ScriptModules之一，您可以使用此标志强制所有使用本地Python来运行。这使得像`
调试代码PDB`工具的使用。

### 类型

TorchScript和完整Python语言之间最大的区别是，TorchScript只支持一小部分的所需要表达的神经网络模型的类型。特别是，TorchScript支持：

类型

|

描述  
  
---|---  
  
`张量 `

|

任何D型，尺寸，或后端的PyTorch张量  
  
`元组[T0， T1， ...]`

|

将含有元组亚型`T0`，`T1`等（例如，`元组[张量， 张量]`）  
  
`布尔 `

|

一个布尔值  
  
`INT`

|

标量整型  
  
`浮动 `

|

标量浮点数  
  
`列出[T]`

|

其列表中的所有成员都是类型`T` 
  
`可选[T]`

|

其是无或键入`T`的值  
  
`字典[K， V]`

|

与密钥类型`K`和值的类型`[HTG5】V `一个字典。只有`STR`，`INT`和`浮动 `被允许作为密钥类型。  
  
不像Python中，在TorchScript函数每个变量必须有一个单一的静态类型。这使得更容易优化TorchScript功能。

实施例（类型不匹配）：

    
    
    @torch.jit.script
    def an_error(x):
        if x:
            r = torch.rand(1)
        else:
            r = 4
        return r # Type mismatch: r is set to type Tensor in the true branch
                 # and type int in the false branch
    

#### 默认类型

默认情况下，在TorchScript函数的所有参数都假定为张量。要指定的参数时TorchScript功能是另一种类型，也可以使用利用上述列出的类型MyPy风格类型注释：

Example:

    
    
    @torch.jit.script
    def foo(x, tup):
        # type: (int, Tuple[Tensor, Tensor]) -> Tensor
        t0, t1 = tup
        return t0 + t1 + x
    
    print(foo(3, (torch.rand(3), torch.rand(3))))
    

Note

也可以来注释类型与Python 3类型的注释。在我们的例子中，我们使用基于注释的注释，以确保Python的2兼容性为好。

假定一个空列表是`列表[张量]`和空类型的字典`字典[STR， 张量]`。实例化的其他类型的空列表或字典中，用`
torch.jit.annotate`。

Example:

    
    
    import torch
    from torch.jit import Tensor
    from typing import List, Tuple
    
    class EmptyDataStructures(torch.jit.ScriptModule):
        def __init__(self):
            super(EmptyDataStructures, self).__init__()
    
        @torch.jit.script_method
        def forward(self, x):
            # type: (Tensor) -> Tuple[List[Tuple[int, float]], Dict[str, int]]
    
            # This annotates the list to be a `List[Tuple[int, float]]`
            my_list = torch.jit.annotate(List[Tuple[int, float]], [])
            for i in range(10):
                my_list.append((x, x))
    
            my_dict = torch.jit.annotate(Dict[str, int], {})
            return my_list, my_dict
    

#### 可选类型细化

TorchScript熬炼类型`可选[T]的变量的类型 `时为`比较无 `的条件的内部由if语句。编译器可以推理多个`无 `将检查相结合，与`和
`，`或 `和`不 `。细化也将发生未明确书面if语句的else块。

表达式必须在有条件的射出;分配`无 `检查一个变量，并使用它在条件不会缩小的类型。像的属性self.x 将不会完善，但在分配 self.x
一个局部变量第一会工作。

Example:

    
    
    @torch.jit.script_method
    def optional_unwrap(self, x, y):
      # type: (Optional[int], Optional[int]) -> int
      if x is None:
        x = 1
      x = x + 1
    
      z = self.z
      if y is not None and z is not None:
        x = y + z
      return x
    

#### 用户定义类型

Python类可以TorchScript使用，如果它们与`注释@ torch.jit.script`，类似于你会怎么声明TorchScript功能：

    
    
    @torch.jit.script
    class Foo:
      def __init__(self, x, y):
        self.x = x
    
      def aug_add_x(self, inc):
        self.x += inc
    

这个子集的限制：

  * 所有的功能必须是有效TorchScript功能（包括__init `__（） `）

  * 类必须是新样式类，我们用`__new __（） `与pybind11构建它们

  * TorchScript类是静态类型的。部件由在`__init __（） `方法分配给自声明

> 例如，分配`__init __（） `方法之外：

>  
>     >     @torch.jit.script

>     class Foo:

>       def assign_x(self):

>         self.x = torch.rand(2, 3)

>  
>

> 将导致：

>  
>     >     RuntimeError:

>     Tried to set nonexistent attribute: x. Did you forget to initialize it
in __init__()?:

>     def assign_x(self):

>       self.x = torch.rand(2, 3)

>       ~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE

>  

  * 除了方法定义没有表情被允许在类的主体

  * 继承或任何其他多态性战略，除了从对象继承不支持指定一个新式类

一类被定义之后，它可以在两个TorchScript和Python可互换使用像任何其他TorchScript类型：

    
    
    @torch.jit.script
    class Pair:
      def __init__(self, first, second):
        self.first = first
        self.second = second
    
    @torch.jit.script
    def sum_pair(p):
      # type: (Pair) -> Tensor
      return p.first + p.second
    
    p = Pair(torch.rand(2, 3), torch.rand(2, 3))
    print(sum_pair(p))
    

### 表达式

下面的Python表达式支持

#### 字面

> `真 `，`假 `，`无 `，`“串 文字 `，`“串 文字” `，数字面值`3`（解释为INT）`3.4`（解释为float）

##### 列表构造

> `[3， 4]`，`[]`，`[torch.rand（ 3）中， torch.rand（4）]`

>

> Note

>

> 空列表假设有型`列表[张量]  [HTG3。类型其他列表文字的从构件的类型的。表示另一种类型的空列表，用`torch.jit.annotate
`。`

##### 元组建筑

> `（3， 4） `，`（3） `

##### 字典建筑

> `{ '你好'： 3}`，`{}`，`{'一“： torch.rand（3）， 'b'： torch.rand（4）}`

>

> Note

>

> 一个空的字典假设有式`字典[STR， 张量]`。类型的dict其他文字的从构件的类型的。表示另一种类型的空字典，用`
torch.jit.annotate`。

#### 变量

> `my_variable_name`

>

> Note

>

> 请参阅变量是如何解决可变分辨率[HTG1。

#### 算术运算符

> `一 +  B`

>

> `一 -  B`

>

> `一 *  B`

>

> `一 /  B`

>

> `一 ^  B`

>

> `一 @  B`

#### 比较运算符

> `一 ==  B`

>

> `一 ！=  B`

>

> `一 & LT ;  B`

>

> `一 & GT ;  B`

>

> `一 & LT ; =  B`

>

> `一 & GT ; =  B`

#### 逻辑运算符

> `一 和 B`

>

> `一 或 B`

>

> `不 B`

#### 下标

> `T [0]`

>

> `T [-1]`

>

> `T [0：2]`

>

> `T [1：]`

>

> `T [1]`

>

> `T [：]`

>

> `T [0， 1]`

>

> `T [0， 1：2]`

>

> `T [0， ：1]`

>

> `T [-1， 1 :,  0]`

>

> `T [1 :,  -1， 0]`

>

> `T [1：J-， I]`

#### 函数调用

> 调用内置函数：`torch.rand（3， D型= torch.int） `

>

> 调用其它脚本函数：

>  
>  
>     import torch

>  
>     @torch.jit.script

>     def foo(x):

>       return x + 1

>  
>     @torch.jit.script

>     def bar(x):

>       return foo(x)

>  

#### 方法调用

> 调用等张量内建类型的方法：`x.mm（Y） `

>

> 当定义一个ScriptModule内部的脚本的方法，所述@script_method 注释用于`
。里面的这些方法可能调用的子模块这一类或访问方法的其他方法。`

>

> 直接调用的子模块（例如，`self.resnet（输入） `）等效于调用其`向前 `方法（例如`self.resnet.forward（输入）
`）

>  
>  
>     import torch

>  
>     class MyScriptModule(torch.jit.ScriptModule):

>         def __init__(self):

>             super(MyScriptModule, self).__init__()

>             self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779,
123.68])

>                                             .resize_(1, 3, 1, 1))

>             self.resnet = torch.jit.trace(torchvision.models.resnet18(),

>                                           torch.rand(1, 3, 224, 224))

>  
>         @torch.jit.script_method

>         def helper(self, input):

>           return self.resnet(input - self.means)

>  
>         @torch.jit.script_method

>         def forward(self, input):

>             return self.helper(input)

>  

#### 三元表达式

> `× 如果 × & GT ;  Y  别的 Y`

#### 施放

> `浮子（10） `

>

> `INT（3.5） `

>

> `布尔（10） `

#### 访问模块参数

> `self.my_parameter`

>

> `self.my_submodule.my_parameter`

### 下列

TorchScript支持下列类型的语句：

Simple Assignments

    
    
    
    a = b
    a += b # short-hand for a = a + b, does not operate in-place on a
    a -= b
    

Pattern Matching Assignments

    
    
    
    a, b = tuple_or_list
    a, b, *c = a_tuple
    

打印报表

> `打印（以下简称 “ 导致 的 的 添加：”， 一 +  [HTG15 b） `

如果语句

>

>        如果 一  & LT ;   4  ：  R   =    -   一 的elif  一  & LT ;   3  ：  R   =  一
+  一 别的 ：  R   =   3   *  一

除了布尔变量，浮点数，整型，和张量可以在有条件的使用，将被隐式浇铸为布尔值。

While循环

>

>        一  =   0  ，而 一  & LT ;   4  ： 打印 （ 一 ） 一  + =   1  

对于`范围 `环

>

>        ×  =   0  为  i的 在 范围 （  10  ）： ×  * =   i的

在过去的元组循环：

>

>     tup = (3, torch.rand(4))

>     for x in tup:

>         print(x)

>  
>

> Note

>

> 用于通过元组循环将展开循环，从而产生体为元组的每个成员。身体必须正确键入检查每个成员。

对于超过环路常数`torch.nn.ModuleList`

>

>     class SubModule(torch.jit.ScriptModule):

>         def __init__(self):

>             super(Sub, self).__init__()

>             self.weight = nn.Parameter(torch.randn(2))

>  
>         @torch.jit.script_method

>         def forward(self, input):

>             return self.weight + input

>  
>     class MyModule(torch.jit.ScriptModule):

>         __constants__ = ['mods']

>  
>         def __init__(self):

>             super(MyModule, self).__init__()

>             self.mods = torch.nn.ModuleList([SubModule() for i in
range(10)])

>  
>         @torch.jit.script_method

>         def forward(self, v):

>             for module in self.mods:

>                 v = m(v)

>             return v

>  
>

> Note

>

> 使用`nn.ModuleList`内的`@script_method`必须通过添加属性的名称到[标示恒定HTG8 ]
__constants__  列表的类型。用于在环的`nn.ModuleList`将展开循环的主体在编译时，与恒定模块列表的每个成员。

break和continue

>

>        为  i的 在 范围 （   5  ）： 如果  i的  ==   1   ： 继续 如果  i的  ==   3  ： 打破 打印 （
i的 ）

Return

    

`返回 A， B`

Note

TorchScript allows returns in the following circumstances:

    

  1. 在一个函数的结尾

  2. 在if语句，其中& LT ;真& GT ;和& LT ;假& GT ;都返回

  3. 在if语句，其中& LT ;真& GT ;回报和& LT ;假& GT ;是空的（一个提前返回）

### 分辨率可变

TorchScript支持Python的可变分辨率（即作用域）规则的子集。局部变量行为相同，在Python中，除了一个变量必须一起通过函数的所有路径相同类型的限制。如果一个变量对if语句的不同侧面不同的类型，它是if语句结束后，使用它的一个错误。

类似地，变量是不允许被使用，如果它仅仅是 _沿着通过函数一些路径定义_ 。

Example:

    
    
    @torch.jit.script
    def foo(x):
        if x < 0:
            y = 4
        print(y) # Error: undefined value y
    

定义函数时非本地变量有决心在编译时的Python值。然后，将这些值转换为使用的Python值的用途中描述的规则TorchScript值。

###  Python中值的使用

为了使编写TorchScript更方便，我们允许脚本代码来引用的Python值在周边范围。例如，任何时候有至`炬
`参考，TorchScript编译器实际上是它解决到`炬 `
Python模块当函数声明。这些Python的值不TorchScript的第一类部分。相反，他们去糖在编译时成TorchScript支持原始类型。这依赖于动态类型值当编译时引用了Python的。本节介绍了TorchScript访问Python的值时使用的规则。

#### 功能

>
TorchScript可以调用Python函数。这个功能是非常有用的，当一种模式逐步转化为TorchScript。该模型可以移动的功能按功能TorchScript，留在地方调用Python函数。这样，您就可以逐步检查模型的正确性，当您去。

>

> Example:

>  
>  
>     def foo(x):

>       print("I am called with {}".format(x))

>       import pdb; pdb.set_trace()

>       return x

>  
>     @torch.jit.script

>     def bar(x)

>       return foo(x + 1)

>  
>

> 试图调用`保存
`在包含到Python函数的调用将失败ScriptModule。目的是使这一途径用于调试和调用删除或保存之前变成脚本功能。如果你想导出模块Python的功能，添加`
@ torch.jit.ignore`装饰的功能，这将替换异常这些函数调用时保存的模型：

>  
>  
>     class M(torch.jit.ScriptModule):

>       def __init__(self):

>         super(M, self).__init__()

>  
>       @torch.jit.script_method

>       def forward(self, x):

>         self.ignored_code(x)

>         return x + 2

>  
>       @torch.jit.ignore

>       def ignored_code(self, x):

>         # non-TorchScript code

>         import pdb; pdb.set_trace()

>  
>     m = M()

>     # Runs, makes upcall to Python to run `ignored_code`

>     m(torch.ones(2, 2))

>  
>     # Replaces all calls to `ignored_code`with a `raise`

>     m.save("m.pt")

>     loaded = torch.jit.load("m.pt")

>  
>     # This runs `ignored_code`after saving which will raise an Exception!

>     loaded(torch.ones(2, 2))

>  

#### 属性查找有关python模块

> TorchScript可以查找在模块属性。像`内建函数torch.add`被访问这种方式。这允许TorchScript调用其他模块中定义的功能。

####  Python的定义的常量

>
TorchScript还提供了一种使用在Python中定义的常量。这些可用于硬编码超参数到函数，或定义通用常数。有指定一个Python值应为一个常数进行治疗的方法有两种。

>

>   1. 价值观抬头作为一个模块的属性被认为是恒定的。例如：`math.pi`

>

>   2. 一个ScriptModule的属性可以被标记通过列出它们作为类的`__constants__`属性的成员常数：

>

> Example:

>  
>     >     class Foo(torch.jit.ScriptModule):

>         __constants__ = ['a']

>  
>         def __init__(self):

>             super(Foo, self).__init__(False)

>             self.a = 1 + 4

>  
>        @torch.jit.script_method

>        def forward(self, input):

>            return self.a + input

>  
>

>

>

> 支持不断Python的价值观是

>

>   * `int`

>

>   * `float`

>

>   * `bool`

>

>   * `torch.device`

>

>   * `torch.layout`

>

>   * `torch.dtype`

>

>   * 包含支持的类型的元组

>

>   * `torch.nn.ModuleList`，它被在TorchScript用于循环

>

>

#### 模块属性

的`torch.nn.Parameter`包装和`register_buffer`可用于指定张量的`ScriptModule
`。与此类似，任何类型的属性可以在`ScriptModule`通过用它们包裹`torch.jit.Attribute
`并指定分配方式。所有可用的类型TorchScript的支持。这些属性是可变的，并在序列化模型二进制保存在单独的存档。张量的属性在语义上是一样的缓冲区。

Example:

    
    
    class Foo(torch.jit.ScriptModule):
      def __init__(self, a_dict):
        super(Foo, self).__init__(False)
        self.words = torch.jit.Attribute([], List[str])
        self.some_dict = torch.jit.Attribute(a_dict, Dict[str, int])
    
      @torch.jit.script_method
      def forward(self, input):
        # type: (str) -> int
        self.words.append(input)
        return self.some_dict[input]
    

### 调试

#### 停用JIT用于调试

> 如果你想禁用所有的JIT模式（跟踪和脚本），所以您可以用原始的Python调试程序，你可以使用`PYTORCH_JIT`环境变量。 `
PYTORCH_JIT`可用于全局禁用通过将其值设置为`0`的JIT。给定一个示例脚本：

>  
>  
>     @torch.jit.script

>     def scripted_fn(x : torch.Tensor):

>         for i in range(12):

>             x = x + x

>         return x

>  
>  
>     def fn(x):

>         x = torch.neg(x)

>         import pdb; pdb.set_trace()

>         return scripted_fn(x)

>  
>     traced_fn = torch.jit.trace(fn, (torch.rand(4, 5),))

>  
>     traced_fn(torch.rand(3, 4))

>  
>

> 除了当我们调用`@ torch.jit.script`功能调试这个脚本PDB工作。我们可以在全球范围禁用JIT，这样我们就可以称之为`@
torch.jit.script`功能作为一个正常的Python函数，而不是编译它。如果上面的脚本被称为`disable_jit_example.py
`，我们可以调用它像这样：

>  
>  
>     $ PYTORCH_JIT=0 python disable_jit_example.py

>  
>

> 和我们将能够torch.jit.script 功能步入`@作为一个正常的Python函数。`

#### 检查代码

> TorchScript为所有`ScriptModule
`实例代码漂亮打印机。这个漂亮的打印机给人的脚本方法的是有效的Python语法代码的解释。例如：

>  
>  
>     @torch.jit.script

>     def foo(len):

>         # type: (int) -> torch.Tensor

>         rv = torch.zeros(3, 4)

>         for i in range(len):

>             if i < 10:

>                 rv = rv - 1.0

>             else:

>                 rv = rv + 1.0

>             return rv

>  
>     print(foo.code)

>  
>

> A `ScriptModule`与单一`向前 `方法将具有属性`代码 `，它你可以用它来检查`ScriptModule的 `代码。如果`
ScriptModule`有一个以上的方法，你将需要访问`.CODE`的方法本身，而不是该模块。我们可以检查名为`方法的代码通过访问`
.bar.code`酒吧 `上的ScriptModule。

>

> 上面的示例脚本生成的代码：

>  
>  
>     def forward(self,

>                 len: int) -> Tensor:

>         rv = torch.zeros([3, 4], dtype=None, layout=None, device=None)

>         rv0 = rv

>         for i in range(len):

>             if torch.lt(i, 10):

>                 rv1 = torch.sub(rv0, 1., 1)

>             else:

>                 rv1 = torch.add(rv0, 1., 1)

>             rv0 = rv1

>         return rv0

>  
>

> 这是`转发 `方法的代码的TorchScript的编译。你可以用它来确保TorchScript（跟踪或脚本）已正确捕获你的模型代码。

#### 解释图表

> TorchScript还具有在比码pretty-打印机较低水平的表示，在IR图的形式。

>

> TorchScript使用静态单赋值（SSA）的中间表示（IR）来表示的计算。在此格式的指令包括阿坦（PyTorch的C
++后端）运营商和其他原始运营商，包括控制流运营商循环和条件的。举个例子：

>  
>  
>     @torch.jit.script

>     def foo(len):

>       # type: (int) -> torch.Tensor

>       rv = torch.zeros(3, 4)

>       for i in range(len):

>         if i < 10:

>             rv = rv - 1.0

>         else:

>             rv = rv + 1.0

>       return rv

>  
>     print(foo.graph)

>  
>

> `.graph`如下在检查代码部分关于`向前 `方法查找所描述的相同的规则。

>

> 上面的例子中的脚本产生的曲线图：

>  
>  
>     graph(%len : int) {

>       %15 : int = prim::Constant[value=1]()

>       %9 : bool = prim::Constant[value=1]()

>       %7 : Device = prim::Constant[value="cpu"]()

>       %6 : int = prim::Constant[value=0]()

>       %5 : int = prim::Constant[value=6]()

>       %1 : int = prim::Constant[value=3]()

>       %2 : int = prim::Constant[value=4]()

>       %11 : int = prim::Constant[value=10]()

>       %14 : float = prim::Constant[value=1]()

>       %4 : int[] = prim::ListConstruct(%1, %2)

>       %rv.1 : Tensor = aten::zeros(%4, %5, %6, %7)

>       %rv : Tensor = prim::Loop(%len, %9, %rv.1)

>         block0(%i : int, %13 : Tensor) {

>           %12 : bool = aten::lt(%i, %11)

>           %rv.4 : Tensor = prim::If(%12)

>             block0() {

>               %rv.2 : Tensor = aten::sub(%13, %14, %15)

>               -> (%rv.2)

>             }

>             block1() {

>               %rv.3 : Tensor = aten::add(%13, %14, %15)

>               -> (%rv.3)

>             }

>           -> (%9, %rv.4)

>         }

>       return (%rv);

>     }

>  
>

> 取指令`％rv.1  ： 动态 =  ATEN ::零（％3， ％4， ％5， ％6） `例如。 `％rv.1  ： 动态
`意味着我们分配输出到名为`RV一个（唯一的）值。 1`，并且该值是`动态 `类型，即，我们还不知道其具体形状。 `ATEN ::零
`是操作者（相当于`torch.zeros`）和输入列表`（％ 3， ％4， ％5， ％6）
`指定哪个范围值应该作为输入传递。该架构内建的功能，如`ATEN ::零 `可以在内置函数研究发现。

>

> 注意，运营商也可以具有相关联`嵌段 `，即`呆板::环 `和`呆板::如果
`运算符。在该图中打印出的，这些运营商被格式化，以反映它们的等效源代码形式，以便于容易调试。

>

> 如下所述图形可以检查如图以确认由`ScriptModule`所述的计算是正确的，在自动和手动方式。

#### 跟踪边缘情况

> 有迹象表明，存在其中给定Python函数/模块的轨迹不会代表底层代码的一些边缘情况。这些情况可能包括：

>

>   * 控制流程的跟踪是依赖于输入（例如张量形状）

>

>   * 就地的张量的观点操作的跟踪（在赋值的左侧例如索引）

>

>

>

> 请注意，这些情况可能实际上是在未来的可追溯。

#### 自动跟踪检查

> 自动赶上痕迹许多错误的一种方法是通过使用`check_inputs`关于`torch.jit.trace（） `API。 `
check_inputs`需要的将被用于重新跟踪计算和验证结果输入元组的列表。例如：

>  
>  
>     def loop_in_traced_fn(x):

>         result = x[0]

>         for i in range(x.size(0)):

>             result = result * x[i]

>         return result

>  
>     inputs = (torch.rand(3, 4, 5),)

>     check_inputs = [(torch.rand(4, 5, 6),), (torch.rand(2, 3, 4),)]

>  
>     traced = torch.jit.trace(loop_in_traced_fn, inputs,
check_inputs=check_inputs)

>  
>

> Gives us the following diagnostic information::

>  
>

> 错误：图形跨越调用不同！图DIFF：

>  
>  
>       graph(%x : Tensor) {

>         %1 : int = prim::Constant[value=0]()

>         %2 : int = prim::Constant[value=0]()

>         %result.1 : Tensor = aten::select(%x, %1, %2)

>         %4 : int = prim::Constant[value=0]()

>         %5 : int = prim::Constant[value=0]()

>         %6 : Tensor = aten::select(%x, %4, %5)

>         %result.2 : Tensor = aten::mul(%result.1, %6)

>         %8 : int = prim::Constant[value=0]()

>         %9 : int = prim::Constant[value=1]()

>         %10 : Tensor = aten::select(%x, %8, %9)

>     -   %result : Tensor = aten::mul(%result.2, %10)

>     +   %result.3 : Tensor = aten::mul(%result.2, %10)

>     ?          ++

>         %12 : int = prim::Constant[value=0]()

>         %13 : int = prim::Constant[value=2]()

>         %14 : Tensor = aten::select(%x, %12, %13)

>     +   %result : Tensor = aten::mul(%result.3, %14)

>     +   %16 : int = prim::Constant[value=0]()

>     +   %17 : int = prim::Constant[value=3]()

>     +   %18 : Tensor = aten::select(%x, %16, %17)

>     -   %15 : Tensor = aten::mul(%result, %14)

>     ?     ^                                 ^

>     +   %19 : Tensor = aten::mul(%result, %18)

>     ?     ^                                 ^

>     -   return (%15);

>     ?             ^

>     +   return (%19);

>     ?             ^

>       }

>  
>

> 此消息表明我们的计算，当我们第一次追查之间，当我们与`check_inputs追查 `不同。的确，`在身体内循环loop_in_traced_fn
`取决于输入`× `，并因此，当我们试图另一[HTG12的形状] × 具有不同形状，跟踪不同。

>

> 在这种情况下，像这样的数据有关的控制流可以使用脚本，而不是被捕获：

>  
>  
>     def fn(x):

>         result = x[0]

>         for i in range(x.size(0)):

>             result = result * x[i]

>         return result

>  
>     inputs = (torch.rand(3, 4, 5),)

>     check_inputs = [(torch.rand(4, 5, 6),), (torch.rand(2, 3, 4),)]

>  
>     scripted_fn = torch.jit.script(fn)

>     print(scripted_fn.graph)

>  
>     for input_tuple in [inputs] + check_inputs:

>         torch.testing.assert_allclose(fn(*input_tuple),
scripted_fn(*input_tuple))

>  
>

> 主要生产：

>  
>  
>     graph(%x : Tensor) {

>       %5 : bool = prim::Constant[value=1]()

>       %1 : int = prim::Constant[value=0]()

>       %result.1 : Tensor = aten::select(%x, %1, %1)

>       %4 : int = aten::size(%x, %1)

>       %result : Tensor = prim::Loop(%4, %5, %result.1)

>         block0(%i : int, %7 : Tensor) {

>           %10 : Tensor = aten::select(%x, %1, %i)

>           %result.2 : Tensor = aten::mul(%7, %10)

>           -> (%5, %result.2)

>         }

>       return (%result);

>     }

>  

#### 示踪剂警告

> 示踪产生警告，在追溯计算若干问题的模式。作为一个例子，取包含在张量的一个切片（视图）的就地分配的功能的跟踪：

>  
>  
>     def fill_row_zero(x):

>         x[0] = torch.rand(*x.shape[1:2])

>         return x

>  
>     traced = torch.jit.trace(fill_row_zero, (torch.rand(3, 4),))

>     print(traced.graph)

>  
>

> 产生多次警告，它简单的返回输入的图形：

>  
>  
>     fill_row_zero.py:4: TracerWarning: There are 2 live references to the
data region being modified when tracing in-place operator copy_ (possibly due
to an assignment). This might cause the trace to be incorrect, because all
other views that also reference this data will not reflect this change in the
trace! On the other hand, if all other views use the same memory chunk, but
are disjoint (e.g. are outputs of torch.split), this might still be safe.

>       x[0] = torch.rand(*x.shape[1:2])

>     fill_row_zero.py:6: TracerWarning: Output nr 1. of the traced function
does not match the corresponding output of the Python function. Detailed
error:

>     Not within tolerance rtol=1e-05 atol=1e-05 at input[0, 1]
(0.09115803241729736 vs. 0.6782537698745728) and 3 other locations (33.00%)

>       traced = torch.jit.trace(fill_row_zero, (torch.rand(3, 4),))

>     graph(%0 : Float(3, 4)) {

>       return (%0);

>     }

>  
>

> 我们可以通过修改代码不使用就地更新解决这个问题，而是建立结果张外的地方用 torch.cat ：

>  
>  
>     def fill_row_zero(x):

>         x = torch.cat((torch.rand(1, *x.shape[1:2]), x[1:2]), dim=0)

>         return x

>  
>     traced = torch.jit.trace(fill_row_zero, (torch.rand(3, 4),))

>     print(traced.graph)

>  

## 常见问题

问：我想在训练GPU的模型，做CPU的推论。什么是最好的做法？

> 首先从GPU转换你的模型到CPU，然后保存它，就像这样：

>  
>  
>     cpu_model = gpu_model.cpu()

>     sample_input_cpu = sample_input_gpu.cpu()

>     traced_cpu = torch.jit.trace(traced_cpu, sample_input_cpu)

>     torch.jit.save(traced_cpu, "cpu.pth")

>  
>     traced_gpu = torch.jit.trace(traced_gpu, sample_input_gpu)

>     torch.jit.save(traced_gpu, "gpu.pth")

>  
>     # ... later, when using the model:

>  
>     if use_gpu:

>       model = torch.jit.load("gpu.pth")

>     else:

>       model = torch.jit.load("cpu.pth")

>  
>     model(input)

>  
>

> 这是推荐的，因为示踪剂可以在特定设备上看到张量创建，使铸造一个已经载入的模型可能会有意想不到的效果。保存它确保示踪剂具有正确的设备信息之前，铸造模型
_。_

问：我如何保存在`ScriptModule`属性？

> 假设我们有一个像模型：

>  
>  
>     class Model(torch.jit.ScriptModule):

>       def __init__(self):

>         super(Model, self).__init__()

>         self.x = 2

>  
>       @torch.jit.script_method

>       def forward(self):

>         return self.x

>  
>

> 如果`型号 `被实例化会导致编译错误，因为编译器不知道`× [HTG7。有4种方式告知属性的编译器上`ScriptModule`：`

>

> 1\. `nn.Parameter`\- 包裹在`值nn.Parameter`将作为它们的`NN做工作。模块 `S

>

> 2\. `register_buffer`\- 值包裹在`register_buffer`，因为它们在`nn.Module做将工作 `
S

>

> 3 HTG0]  __constants__  \- 加入了一个名为`列表__constants__
`在类定义级别将迎来包含名称为常数。常量在模型的代码直接保存。参见 Python的定义的常量。

>

> 4\. `torch.jit.Attribute`\- 包裹在`值torch.jit.Attribute`可以是任何`
TorchScript`类型，待突变并保存了该模型的代码的外部。参见模块属性。

问：我想跟踪模块的方法，但我不断收到此错误：

`RuntimeError： 不能 插入 A  张量 是 需要 GRAD  如 一 常数。  考虑 制备 它 一 参数 或 输入，  或 拆卸 中的 梯度
`

> 此错误通常意味着，你正在跟踪的方法中，使用模块的参数和要传递一个模块实例的模块的方法，而不是（例如，`
my_module_instance.forward`对`my_module_instance`）。

>

>>   * 调用`痕量 `与模块的方法捕获模块参数（可能需要的梯度）作为 **常数** 。

>>

>>   * 在另一方面，调用`追踪 `与模块的实例（例如`my_module
`）创建一个新的模块，并正确地拷贝参数到新模块，因此，如果需要，他们可以积累梯度。

>>

>>

>

> 鉴于`痕量 `对待`my_module_instance.forward`作为一个独立的功能，这也意味着有 **不**
目前一方法来跟踪任意方法在模块中除了`转发 `在使用模块的参数。版本 **1.1.1** 将增加一个新的API `trace_module
`，将允许用户跟踪模块中的任何方法不止一种方法

>  
>  
>     class Net(nn.Module):

>         def __init__(self):

>             super(Net, self).__init__()

>             self.conv = nn.Conv2d(1, 1, 3)

>  
>         def forward(self, x):

>             return self.conv(x)

>  
>         def weighted_kernel_sum(self, weight):

>             return weight * self.conv.weight

>  
>     example_weight = torch.rand(1, 1, 3, 3)

>     example_forward_input = torch.rand(1, 1, 3, 3)

>     n = Net()

>     inputs = {'forward' : example_forward_input, 'weighted_kernel_sum' :
example_weight}

>     module = torch.jit.trace_module(n, inputs)

>  

### 内置函数

TorchScript支持内建张量和神经网络函数PyTorch提供的一个子集。上张量大多数方法以及在`炬功能 `命名空间，在`
torch.nn.functional`的所有功能，并从所有模块`torch.nn
`在支持TorchScript，不包括在下面的表中。对于不支持的模块，建议使用 `torch.jit.trace（） `。

不支持`torch.nn`模块

    
    
    torch.nn.modules.adaptive.AdaptiveLogSoftmaxWithLoss
    torch.nn.modules.normalization.CrossMapLRN2d
    torch.nn.modules.fold.Fold
    torch.nn.modules.fold.Unfold
    torch.nn.modules.rnn.GRU
    torch.nn.modules.rnn.RNN
    

[Next ![](_static/images/chevron-right-orange.svg)](multiprocessing.html
"Multiprocessing package - torch.multiprocessing")
[![](_static/images/chevron-right-orange.svg) Previous](hub.html "torch.hub")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * TorchScript 
    * 创建TorchScript代码
    * 混合跟踪和脚本
    * TorchScript语言参考
      * 类型
        * 默认类型
        * 可选类型细化
        * 用户定义类型
      * 表达式
        * 字面
          * 列表构造
          * 元组建筑
          * 字典建筑
        * 变量
        * 算术运算符
        * 比较运算符
        * 逻辑运算符
        * 下标
        * 函数调用
        * 方法调用
        * 三元表达式
        * 粪中
        * 访问模块参数
      * 下列
      * 分辨率可变
      * Python中值的使用
        * 功能
        * 属性查找有关python模块
        * Python的定义的常量
        * 模块属性
      * 调试
        * [HTG0用于调试禁用JIT 
        * 检查代码
        * 解释图表
        * 跟踪边缘情况
        * 自动跟踪检查
        * 示踪剂警告
    * 常见问题
      * 内建函数

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

