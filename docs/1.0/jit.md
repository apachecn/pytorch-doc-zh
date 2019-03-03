

# Torch Script

*   [Creating Torch Script Code](#creating-torch-script-code)
*   [Mixing Tracing and Scripting](#mixing-tracing-and-scripting)
*   [Torch Script Language Reference](#torch-script-language-reference)
    *   [Types](#types)
    *   [Expressions](#expressions)
    *   [Statements](#statements)
    *   [Variable Resolution](#variable-resolution)
    *   [Use of Python Values](#use-of-python-values)
    *   [Debugging](#debugging)
    *   [Builtin Functions](#builtin-functions)
*   [创建Torch Script代码](#creating-torch-script-code)
*   [将追踪和脚本化结合起来](#mixing-tracing-and-scripting)
*   [Torch Script语言参考](#torch-script-language-reference)
    *   [类型](#types)
    *   [表达式](#expressions)
    *   [Statements](#statements)
    *   [Variable Resolution](#variable-resolution)
    *   [Use of Python Values](#use-of-python-values)
    *   [Debugging](#debugging)
    *   [Builtin Functions](#builtin-functions)
    
Torch Script is a way to create serializable and optimizable models from PyTorch code. Any code written in Torch Script can be saved from your Python process and loaded in a process where there is no Python dependency.

We provide tools to incrementally transition a model from being a pure Python program to a Torch Script program that can be run independently from Python, for instance, in a standalone C++ program. This makes it possible to train models in PyTorch using familiar tools and then export the model to a production environment where it is not a good idea to run models as Python programs for performance and multi-threading reasons.

Torch脚本是一种从PyTorch代码创建可序列化和可优化模型的方法。用Torch脚本编写的代码可以从Python进程中保存，并在没有Python依赖的进程中加载。

我们提供了一些工具帮助我们将模型从纯Python程序逐步转换为可以独立于Python运行的Torch脚本程序。Torch脚本程序可以在其他语言的程序中运行（例如，在独立的C ++程序中）。这使得我们可以使用熟悉的工具在PyTorch中训练模型，而将模型导出到出于性能和多线程原因不能将模型作为Python程序运行的生产环境中去。
```py
class torch.jit.ScriptModule(optimize=True)
```

The core data structure in Torch Script is the `ScriptModule`. It is an analogue of torch’s nn.Module and represents an entire model as a tree of submodules. Like normal modules, each individual module in a ScriptModule can have submodules, parameters, and methods. In nn.Modules methods are implemented as Python functions, but in ScriptModules methods typically implemented as _Torch Script_ functions, a statically-typed subset of Python that contains all of PyTorch’s built-in Tensor operations. This difference allows your ScriptModules code to run without the need for a Python interpreter.

Torch脚本中的核心数据结构是`ScriptModule`。它与Torch的nn.Module类似，用子模块树代表整个模型。与普通模块一样，ScriptModule中每个模块都可以包含子模块，参数和方法。不同的是，nn.Modules中的方法是用Python函数实现的，而ScriptModules中的方法通常由 _Torch脚本_ 函数实现，这种函数是Python的一个静态类型子集，包含PyTorch的所有内置Tensor操作。这种差异使得ScriptModules代码的运行不依赖于Python解释器。

ScriptModules and the Torch Script functions inside of them can be created in two ways:

**Tracing:**

ScriptModule与其内部的Torch脚本函数可以通过两种方式创建：

**追踪：**

> 使用`torch.jit.trace`。torch.jit.trace以现有模块或python函数和样例输入作为参数，它会运行该python函数，记录函数在所有张量上执行的操作，并将记录转换为Torch脚本方法以作为ScriptModule的forward方法。创建的模块包含原始模块的所有参数。
>
> 例：
> 
> ```py
> import torch
> def foo(x, y):
>     return 2*x + y
> traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))
> 
> ```
>
> 注意
>
> 追踪一个 _函数_ 将生成一个`ScriptModule`，该ScriptModule中包含一个实现被追踪函数的`forward`方法，但不包含任何参数。
>
> 例：
> 
> ```py
> import torch
> import torchvision
> traced_net = torch.jit.trace(torchvision.models.resnet18(),
>                              torch.rand(1, 3, 224, 224))
> 
> ```
>
> 注意
>
> 追踪仅记录在给定张量上运行给定函数时执行的操作。因此，返回的`ScriptModule`在任何输入上将运行相同的追踪图。当你的模块需要根据输入和/或模块状态运行不同的操作集时，这会产生一些重要的影响。例如，
> &gt;* 追踪不会记录if语句或循环之类的控制流。当这个控制流在你的模块中保持不变时，这很好，它通常只是内联配置决策。但有时控制流实际上是模型本身的一部分。例如，序列到序列转换中的beam搜索是对（可变）输入序列长度的循环。
> 
> &gt;*在返回的`ScriptModule`中，在`training`和`eval`模式中具有不同行为的操作将始终表现为处于追踪期间的模式。
>
> 在上述情况下，脚本化是一个比追踪更好的选择。

> Using `torch.jit.trace`, you can take an existing module or python function, provide example inputs, and we run the function, recording the operations performed on all the tensors. We turn the resulting recording into a Torch Script method that is installed as the `forward` method of a ScriptModule. This module also contains any parameters that the original module had as well.
> 
> Example:
> 
> ```py
> import torch
> def foo(x, y):
>     return 2*x + y
> traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))
> 
> ```
> 
> Note
> 
> Tracing a _function_ will produce a `ScriptModule` with a single `forward` method that implements that function, and that contains no parameters.
> 
> Example:
> 
> ```py
> import torch
> import torchvision
> traced_net = torch.jit.trace(torchvision.models.resnet18(),
>                              torch.rand(1, 3, 224, 224))
> 
> ```
> 
> Note
> 
> Tracing only records operations done when the given function is run on the given tensors. Therefore, the returned `ScriptModule` will always run the same traced graph on any input. This has some important implications when your module is expected to run different sets of operations, depending on the input and/or the module state. For example,
> 
> &gt; *   Tracing will not record any control-flow like if statements or loops. When this control-flow is constant across your module, this is fine and it often just inlines configuration decisions. But sometimes the control-flow is actually part of the model itself. For instance, a beam search in sequence-to-sequence translation is a loop over the (varying) sequence length of inputs.
> &gt; *   In the returned `ScriptModule`, operations that have different behaviors in `training` and `eval` modes will always behave as if it is in the mode it was in during tracing, no matter which mode the `ScriptModule` is in.
> 
> In cases like these, tracing would not be appropriate and scripting is a better choice.

**Scripting:**

> You can write Torch Script code directly using Python syntax. You do this using the `torch.jit.script` annotation (for functions) or `torch.jit.script_method` annotation (for methods) on subclasses of ScriptModule. With this annotation the body of the annotated function is directly translated into Torch Script. Torch Script itself is a subset of the Python language, so not all features in python work, but we provide enough functionality to compute on tensors and do control-dependent operations.
> 
> Example:
> 
> ```py
> import torch
> @torch.jit.script
> def foo(x, y):
>     if x.max() &gt; y.max():
>         r = x
>     else:
>         r = y
>     return r
> 
> ```
> 
> Note
> 
> A script _function_ annotation will construct a ScriptModule with a single `forward` method that implements that function, and that contains no parameters.
> 
> Example:
> 
> ```py
> import torch
> class MyModule(torch.jit.ScriptModule):
>     def __init__(self, N, M):
>         super(MyModule, self).__init__()
>         self.weight = torch.nn.Parameter(torch.rand(N, M))
> 
>     @torch.jit.script_method
>     def forward(self, input):
>         return self.weight.mv(input)
> 
> ```
> 
> Example:
> 
> ```py
> import torch
> import torch.nn as nn
> import torch.nn.functional as F
> from torch.jit import ScriptModule, script_method, trace
> 
> class MyScriptModule(ScriptModule):
>     def __init__(self):
>         super(MyScriptModule, self).__init__()
>         # trace produces a ScriptModule's conv1 and conv2
>         self.conv1 = trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))
>         self.conv2 = trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))
> 
>     @script_method
>     def forward(self, input):
>       input = F.relu(self.conv1(input))
>       input = F.relu(self.conv2(input))
>       return input
> 
> ```

**脚本化**

> 你可以使用Python语法直接编写Torch脚本代码。你可以使用`torch.jit.script`注释（对于函数）或`torch.jit.script_method`注释（对于ScriptModule子类的方法）来编写Torch脚本代码。通过注释，被注释函数的主体将直接转换为Torch脚本。 Torch脚本本身只是Python语言的一个子集，因此不是python中的所有特性都可以使用，但我们提供了足够的功能来计算张量并执行与控制相关的操作。
> 
> 例:
> 
> ```py
> import torch
> @torch.jit.script
> def foo(x, y):
>     if x.max() &gt; y.max():
>         r = x
>     else:
>         r = y
>     return r
> 
> ```
> 
> 注意
> 
> 脚本 _函数_ 注释将构造带有一个`forward`方法的ScriptModule，该forward方法实现被注释函数，并且不包含任何参数。
> 
> 例：
> 
> ```py
> import torch
> class MyModule(torch.jit.ScriptModule):
>     def __init__(self, N, M):
>         super(MyModule, self).__init__()
>         self.weight = torch.nn.Parameter(torch.rand(N, M))
> 
>     @torch.jit.script_method
>     def forward(self, input):
>         return self.weight.mv(input)
> 
> ```
> 
> 例：
> 
> ```py
> import torch
> import torch.nn as nn
> import torch.nn.functional as F
> from torch.jit import ScriptModule, script_method, trace
> 
> class MyScriptModule(ScriptModule):
>     def __init__(self):
>         super(MyScriptModule, self).__init__()
>         # trace produces a ScriptModule's conv1 and conv2
>         self.conv1 = trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))
>         self.conv2 = trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))
> 
>     @script_method
>     def forward(self, input):
>       input = F.relu(self.conv1(input))
>       input = F.relu(self.conv2(input))
>       return input
> 
> ```

```py
save(filename)
```

Save an offline version of this module for use in a separate process. The saved module serializes all of the methods and parameters of this module. It can be loaded into the C++ API using `torch::jit::load(filename)` or into the Python API with `torch.jit.load(filename)`.

To be able to save a module, it must not make any calls to native python functions. This means that all submodules must be subclasses of ScriptModules as well.

Danger

All modules, no matter their device, are always loaded onto the CPU during loading. This is different from [`torch.load()`](torch.html#torch.load "torch.load")’s semantics and may change in the future.

```py
torch.jit.load(f, map_location=None)
```

Load a `ScriptModule` previously saved with `save`

All previously saved modules, no matter their device, are first loaded onto CPU, and then are moved to the devices they were saved from. If this fails (e.g. because the run time system doesn’t have certain devices), an exception is raised. However, storages can be dynamically remapped to an alternative set of devices using the `map_location` argument. Comparing to [`torch.load()`](torch.html#torch.load "torch.load"), `map_location` in this function is simplified, which only accepts a string (e.g., ‘cpu’, ‘cuda:0’), or torch.device (e.g., torch.device(‘cpu’))

```py
save(filename)
```

保存离线版本的模块，以便将来在其他的进程中使用。保存的模块会序列化当前模块的所有方法和参数。保存的模块可以使用`torch :: jit :: load（filename）`加载到C ++ API中，也可以使用`torch.jit.load（filename）`加载到Python API中。

为了能够保存模块，当前模块不能调用原生python函数。也就是说要保存模块的所有子模块也必须是ScriptModules的子类。

危险

所有模块，不论其设备，在加载过程中始终都会加载到CPU中。这与`torch.load()`的语义不同，将来可能会发生变化。


```py
torch.jit.load(f, map_location=None)
```

使用`load`加载之前用`save`保存的`ScriptModule`。

所有先前保存的模块，不论其设备，首先加载到CPU上，然后移动到之前保存它们的设备上。如果此操作失败（例如，运行时系统没有某些设备），则会引发异常。此时可以使用`map_location`参数将存储重新映射到另一组设备。与`torch.load()`相比，此函数中的`map_location`被简化为只接受字符串（例如'cpu'，'cuda：0'）或torch.device（例如，torch.device（'cpu'））


Parameters: 

*   **f** – a file-like object (has to implement read, readline, tell, and seek), or a string containing a file name
*   **map_location** – can a string (e.g., ‘cpu’, ‘cuda:0’), a device (e.g., torch.device(‘cpu’))


| Returns: | A `ScriptModule` object. |
| --- | --- |
参数：

*   **f** – 文件类对象（必须实现read，readline，tell和seek），或为文件名的字符串
*   **map_location** – 可以是一个字符串（例如，'cpu'，'cuda：0'），一个设备（例如，torch.device（'cpu'））


| 返回值: |  `ScriptModule` 对象. |
| --- | --- |

例

```py
>>> torch.jit.load('scriptmodule.pt')
# 从io.BytesIO对象加载ScriptModule
>>> with open('scriptmodule.pt', 'rb') as f:
 buffer = io.BytesIO(f.read())
# 将所有张量加载到原来的设备上
>>> torch.jit.load(buffer)
# 用设备将所有张量加载到CPU上
>>> torch.jit.load(buffer, map_location=torch.device('cpu'))
# 用字符串将所有张量加载到CPU上
>>> torch.jit.load(buffer, map_location='cpu')

```

```py
torch.jit.trace(func, example_inputs, optimize=True, check_trace=True, check_inputs=None, check_tolerance=1e-05, _force_outplace=False)
```

追踪一个函数并返回一个使用即时编译优化过的可执行追踪。

警告

追踪仅正确记录不依赖于数据的函数和模块（例如，对张量中的数据进行条件判断），并且没有任何未追踪的外部依赖性（例如，执行输入/输出或访问全局变量）。如果你追踪此类模型，则可能会在随后的模型调用中静默获取不正确的结果。当执行可能生成错误追踪的内容时，追踪器将尝试发出警告。

参数： 

*   **func** (_callable_ _or_ [_torch.nn.Module_](nn.html#torch.nn.Module "torch.nn.Module")) – 将使用example_inputs作为输入运行的python函数或torch.nn.Module。参数和返回值必须是Tensor或（嵌套的）包含张量的元组。
*   **example_inputs** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")) – 在追踪时将传递给函数的示例输入元组。假设被追踪操作支持这些类型和形状的情况下，生成的追踪可以在不同类型和形状的输入下运行。 example_inputs也可以是单个Tensor，这种情况下，它会自动包装到元组中。


| 关键字参数： |
| --- |
|   | 

*   **optimize** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – 是否应用优化。默认值：`True`。
*   **check_trace** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – 检查被追踪代码在相同输入下输出是否相同。默认值：`True`。你可以在某些情况下禁用此功能。例如，你的网络包含非确定性操作，或者你确定网络正确。
*   **check_inputs** (_list of tuples__,_ _optional_) – 应该用于根据预期检查追踪的输入参数元组列表。每个元组相当于一个将在`args`中指定的输入参数集合。为获得最佳结果，请传递一组检查输入表示你期望网络接受的形状和输入类型范围。如果未指定，则用原来的`args`检查。
*   **check_tolerance** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 在检查过程中使用的浮点比较容差。用于放松检查严格性。 

 
| 返回值： | 含有`forward（）`方法的`ScriptModule`对象，该方法包含被追踪代码。当func是`torch.nn.Module`时，返回的`ScriptModule`具有与原始模块相同的子模块和参数集。|
| --- | --- |

例

```py
>>> def f(x):
...     return x * 2
>>> traced_f = torch.jit.trace(f, torch.rand(1))

```

在许多情况下，追踪或脚本是转换模型的更简单方法。我们允许你将追踪和脚本组合使用以满足模型特定部分的特定要求。

脚本函数可以调用被追踪函数。当你需要使用控制流控制简单的前馈模型时，这尤其有用。例如，序列到序列模型的beam搜索通常将以脚本编写，但可以调用使用追踪生成的编码器模块。

例：

```py
import torch

def foo(x, y):
    return 2 * x + y
traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))

@torch.jit.script
def bar(x):
    return traced_foo(x, x)

```

被追踪函数也可以调用脚本函数。当模型大体是一个前馈网络，只有模型的一小部分需要一些控制流时，这也很有用。由追踪函数调用的脚本函数内部的控制流会被正确地保留。

例：

```py
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

traced_bar = torch.jit.trace(bar, (torch.rand(3), torch.rand(3), torch.rand(3))

```

组合也适用于模块，例如可以从脚本模块的方法调用追踪来生成子模块：

例：

```py
import torch
import torchvision

class MyScriptModule(torch.jit.ScriptModule):
    def __init__(self):
        super(MyScriptModule, self).__init__()
        self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779, 123.68])
                                        .resize_(1, 3, 1, 1))
        self.resnet = torch.jit.trace(torchvision.models.resnet18(),
                                      torch.rand(1, 3, 224, 224))

    @torch.jit.script_method
    def forward(self, input):
        return self.resnet(input - self.means)

```

Torch Script is a subset of Python that can either be written directly (using the @script annotations) or generated automatically from Python code via tracing. When using tracing, code is automatically converted into this subset of Python by recording only the actual operators on tensors and simply executing and discarding the other surrounding Python code.

When writing Torch Script directly using @script annotations, the programmer must only use the subset of Python supported in Torch Script. This section documents what is supported in Torch Script as if it were a language reference for a stand alone language. Any features of Python not mentioned in this reference are not part of Torch Script.

As a subset of Python any valid Torch Script function is also a valid Python function. This makes it possible to remove the @script annotations and debug the function using standard Python tools like pdb. The reverse is not true: there are many valid python programs that are not valid Torch Script programs. Instead, Torch Script focuses specifically on the features of Python that are needed to represent neural network models in Torch.

```py
PYTORCH_JIT=1
```

Setting the environment variable `PYTORCH_JIT=0` will disable all script and tracing annotations. If there is hard-to-debug error in one of your ScriptModules, you can use this flag to force everything to run using native Python. This allows the use of tools like `pdb` to debug code.

The largest difference between Torch Script and the full Python language is that Torch Script only support a small set of types that are needed to express neural net models. In particular Torch Script supports:

```py
Tensor
```

A PyTorch tensor of any dtype, dimension, or backend.

```py
Tuple[T0, T1, ...]
```

A tuple containing subtypes `T0`, `T1`, etc. (e.g. `Tuple[Tensor, Tensor]`)

```py
int
```

A scalar integer

```py
float
```

A scalar floating point number

```py
List[T]
```

A list of which all members are type `T`

Unlike Python, each variable in Torch Script function must have a single static type. This makes it easier to optimize Torch Script functions.

Example:

```py
@torch.jit.script
def an_error(x):
    if x:
        r = torch.rand(1)
    else:
        r = 4
    return r # Type mismatch: r is set to type Tensor in the true branch
             # and type int in the false branch

```

By default, all parameters to a Torch Script function are assumed to be Tensor because this is the most common type used in modules. To specify that an argument to a Torch Script function is another type, it is possible to use MyPy-style type annotations using the types listed above:

Example:

```py
@torch.jit.script
def foo(x, tup):
    # type: (int, Tuple[Tensor, Tensor]) -> Tensor
    t0, t1 = tup
    return t0 + t1 + x

print(foo(3, (torch.rand(3), torch.rand(3))))

```

Note

It is also possible to annotate types with Python 3 type annotations. In our examples, we use comment-based annotations to ensure Python 2 compatibility as well.

The following Python Expressions are supported

```py
Literals
```

`True`, `False`, `None`, `'string literals'`, `"string literals"`, number literals `3` (interpreted as int) `3.4` (interpreter as a float)

```py
Variables
```

`a`

Note

See [Variable Resolution](#variable-resolution) for how variables are resolved.

```py
Tuple Construction
```

`(3, 4)`, `(3,)`

```py
List Construction
```

`[3, 4]`, `[]`, `[torch.rand(3), torch.rand(4)]`

Note

an empty list is assumed have type `List[Tensor]`. The types of other list literals are derived from the type of the members.

```py
Arithmetic Operators
```

`a + b` `a - b` `a * b` `a / b` `a ^ b` `a @ b`

```py
Comparison Operators
```

`a == b` `a != b` `a < b` `a > b` `a <= b` `a >= b`

```py
Logical Operators
```

`a and b` `a or b` `not b`

```py
Subscripts
```

`t[0]` `t[-1]` `t[0:2]` `t[1:]` `t[:1]` `t[:]` `t[0, 1]` `t[0, 1:2]` `t[0, :1]` `t[-1, 1:, 0]` `t[1:, -1, 0]` `t[i:j, i]`

Note

Torch Script currently does not support mutating tensors in place, so any tensor indexing can only appear on the right-hand size of an expression.

```py
Function calls
```

Calls to built-in functions: `torch.rand(3, dtype=torch.int)`

Calls to other script functions:

```py
import torch

@torch.jit.script
def foo(x):
  return x + 1

@torch.jit.script
def bar(x):
  return foo(x)

```

```py
Method calls
```

Calls to methods of builtin types like tensor: `x.mm(y)`

When defining a Script method inside of a ScriptModule, the `@script_method` annotation is used. Inside of these methods it is possible to call other methods of this class or access methods on the submodules.

Calling a submodule directly (e.g. `self.resnet(input)`) is equivalent to calling its `forward` method (e.g. `self.resnet.forward(input)`)

```py
import torch

class MyScriptModule(torch.jit.ScriptModule):
    def __init__(self):
        super(MyScriptModule, self).__init__()
        self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779, 123.68])
                                        .resize_(1, 3, 1, 1))
        self.resnet = torch.jit.trace(torchvision.models.resnet18(),
                                      torch.rand(1, 3, 224, 224))

    @torch.jit.script_method
    def helper(self, input):
      return self.resnet(input - self.means)

    @torch.jit.script_method
    def forward(self, input):
        return self.helper(input)

```

```py
If expressions
```

`x if x > y else y`

```py
Casts
```

`float(ten)`, `int(3.5)`, `bool(ten)`

```py
Accessing Module Parameters
```

`self.my_parameter` `self.my_submodule.my_parameter`

Torch Script supports the following types of statements:

Simple Assignments

> ```py
> a = b
> a += b # short-hand for a = a + b, does not operate in-place on a
> a -= b
> 
> ```

Pattern Matching Assignments

> ```py
> a, b = tuple_or_list
> a, b, *c = a_tuple
> 
> ```

Print Statements

> `print("the result of an add:", a + b)`

If Statements

> ```py
> if a &lt; 4:
>     r = -a
> elif a &lt; 3:
>     r = a + a
> else:
>     r = 3 * a
> 
> ```

While Loops

> ```py
> a = 0
> while a &lt; 4:
>     print(a)
>     a += 1
> 
> ```

For loops with `range`

> ```py
> x = 0
> for i in range(10):
>     x *= i
> 
> ```
> 
> Note
> 
> Script currently does not support iterating over generic iterable objects like lists or tensors. Script currently does not support start or increment parameters to range. These will be added in a future version.

For loops over tuples:

> ```py
> tup = (3, torch.rand(4))
> for x in tup:
>     print(x)
> 
> ```
> 
> Note
> 
> for loops over tuples will unroll the loop, generating a body for each member of the tuple. The body must type-check correctly for each member.

For loops over constant `torch.nn.ModuleList`

> ```py
> class SubModule(torch.jit.ScriptModule):
>     def __init__(self):
>         super(Sub, self).__init__()
>         self.weight = nn.Parameter(torch.randn(2))
> 
>     @torch.jit.script_method
>     def forward(self, input):
>         return self.weight + input
> 
> class MyModule(torch.jit.ScriptModule):
>     __constants__ = ['mods']
> 
>     def __init__(self):
>         super(MyModule, self).__init__()
>         self.mods = torch.nn.ModuleList([SubModule() for i in range(10)])
> 
>     @torch.jit.script_method
>     def forward(self, v):
>         for module in self.mods:
>             v = m(v)
>         return v
> 
> ```
> 
> Note
> 
> To use a module list inside a `@script_method` it must be marked constant by adding the name of the attribute to the `__constants__` list for the type. For loops over a ModuleList will unroll the body of the loop at compile time, with each member of the constant module list.

```py
Return
```

`return a, b`

Note

there must be a return statement as the last member of the function and return statements cannot appear anywhere else in the function. This restriction will be removed in the future.

Torch Script supports a subset of Python’s variable resolution (i.e. scoping) rules. Local variables behave the same as in Python, except for the restriction that a variable must have the same type along all paths through a function. If a variable has a different type on different sides of an if statement, it is an error to use it after the end of the if statement.

Similarly, a variable is not allowed to be used if it is only _defined_ along some paths through the function.

Example:

```py
@torch.jit.script
def foo(x):
    if x < 0:
        y = 4
    print(y) # Error: undefined value y

```

Non-local variables are resolved to Python values at compile time when the function is defined. These values are then converted into Torch Script values using the rules described in [Use of Python Values](#use-of-python-values).

To make writing Torch Script more convenient, we allow script code to refer to Python values in the surrounding scope. For instance, any time there is a reference to `torch`, the Torch Script compiler is actually resolving it to the `torch` Python module when the function is declared. These Python values are not a first class part of Torch Script. Instead they are desugared at compile-time into the primitive types that Torch Script supports. This section describes the rules that are used when accessing Python values in Torch Script. They depend on the dynamic type of the python valued referenced.

```py
Functions
```

Torch Script can call python functions. This functionality is very useful when incrementally converting a model into script. The model can be moved function-by-function to script, leaving calls to Python functions in place. This way you can incrementally check the correctness of the model as you go.

Example:

```py
def foo(x):
  print("I am called with {}".format(x))
  import pdb; pdb.set_trace()
  return x

@torch.jit.script
def bar(x)
  return foo(x + 1)

```

Note

Attempting to call `save` on a ScriptModule that contains calls to Python functions will fail. The intention is that this pathway is used for debugging and the calls removed or turned into script functions before saving.

```py
Attribute Lookup On Python Modules
```

Torch Script can lookup attributes on modules. Builtin functions like `torch.add` are accessed this way. This allows Torch Script to call functions defined in other modules.

```py
Python-defined Constants
```

Torch Script also provides a way to use constants that are defined in Python. These can be used to hard-code hyper-parameters into the function, or to define universal constants. There are two ways of specifying that a Python value should be treated as a constant.

1.  Values looked up as attributes of a module are assumed to be constant. Example: `math.pi`

2.  Attributes of a ScriptModule can be marked constant by listing them as a member of the `__constants__` property of the class:

    Example:

    ```py
    class Foo(torch.jit.ScriptModule):
        __constants__ = ['a']

        def __init__(self):
            super(Foo, self).__init__(False)
            self.a = 1 + 4

       @torch.jit.ScriptModule
       def forward(self, input):
           return self.a + input

    ```

Supported constant Python Values are

*   `int`
*   `bool`
*   `torch.device`
*   `torch.layout`
*   `torch.dtype`
*   tuples containing supported types
*   `torch.nn.ModuleList` which can be used in a TorchScript for loop

```py
Disable JIT for Debugging
```

If you want to disable all JIT modes (tracing and scripting) so you can debug your program in raw Python, you can use the `PYTORCH_JIT` environment variable. `PYTORCH_JIT` can be used to globally disable the JIT by setting its value to `0`. Given an example script:

```py
@torch.jit.script
def scripted_fn(x : torch.Tensor):
    for i in range(12):
        x = x + x
    return x

def fn(x):
    x = torch.neg(x)
    import pdb; pdb.set_trace()
    return scripted_fn(x)

traced_fn = torch.jit.trace(fn, (torch.rand(4, 5),))

traced_fn(torch.rand(3, 4))

```

Debugging this script with PDB works except for when we invoke the @script function. We can globally disable JIT, so that we can call the @script function as a normal python function and not compile it. If the above script is called `disable_jit_example.py`, we can invoke it like so:

```py
$ PYTORCH_JIT=0 python disable_jit_example.py

```

and we will be able to step into the @script function as a normal Python function.

```py
Interpreting Graphs
```

TorchScript uses a static single assignment (SSA) intermediate representation (IR) to represent computation. The instructions in this format consist of ATen (the C++ backend of PyTorch) operators and other primitive operators, including control flow operators for loops and conditionals. As an example:

```py
@torch.jit.script
def foo(len):
  # type: (int) -> torch.Tensor
  rv = torch.zeros(3, 4)
  for i in range(len):
    if i < 10:
        rv = rv - 1.0
    else:
        rv = rv + 1.0
  return rv

print(foo.graph)

```

A `ScriptModule` with a single `forward` method will have an attribute `graph`, which you can use to inspect the IR representing the computation. If the ScriptModule has more than one method, you will need to access `.graph` on the method itself and not the module. We can inspect the graph of a method named `bar` on a ScriptModule by accessing `.bar.graph`.

The example script above produces the graph:

```py
graph(%len : int) {
  %13 : float = prim::Constant[value=1]()
  %10 : int = prim::Constant[value=10]()
  %2 : int = prim::Constant[value=4]()
  %1 : int = prim::Constant[value=3]()
  %3 : int[] = prim::ListConstruct(%1, %2)
  %4 : int = prim::Constant[value=6]()
  %5 : int = prim::Constant[value=0]()
  %6 : int[] = prim::Constant[value=[0, -1]]()
  %rv.1 : Dynamic = aten::zeros(%3, %4, %5, %6)
  %8 : int = prim::Constant[value=1]()
  %rv : Dynamic = prim::Loop(%len, %8, %rv.1)
    block0(%i : int, %12 : Dynamic) {
      %11 : int = aten::lt(%i, %10)
      %rv.4 : Dynamic = prim::If(%11)
        block0() {
          %14 : int = prim::Constant[value=1]()
          %rv.2 : Dynamic = aten::sub(%12, %13, %14)
          -> (%rv.2)
        }
        block1() {
          %16 : int = prim::Constant[value=1]()
          %rv.3 : Dynamic = aten::add(%12, %13, %16)
          -> (%rv.3)
        }
      %19 : int = prim::Constant[value=1]()
      -> (%19, %rv.4)
    }
  return (%rv);
}

```

Take the instruction `%rv.1 : Dynamic = aten::zeros(%3, %4, %5, %6)` for example. `%rv.1 : Dynamic` means we assign the output to a (unique) value named `rv.1`, and that value is of `Dynamic` type, i.e. we do not know its concrete shape. `aten::zeros` is the operator (equivalent to `torch.zeros`) and the input list `(%3, %4, %5, %6)` specifies which values in scope should be passed as inputs. The schema for built-in functions like `aten::zeros` can be found at [Builtin Functions](#builtin-functions).

Notice that operators can also have associated `blocks`, namely the `prim::Loop` and `prim::If` operators. In the graph print-out, these operators are formatted to reflect their equivalent source code forms to facilitate easy debugging.

Graphs can be inspected as shown to confirm that the computation described by a `ScriptModule` is correct, in both automated and manual fashion, as described below.

```py
Tracing Edge Cases
```

There are some edge cases that exist where the trace of a given Python function/module will not be representative of the underlying code. These cases can include:

*   Tracing of control flow that is dependent on inputs (e.g. tensor shapes)
*   Tracing of in-place operations of tensor views (e.g. indexing on the left-hand side of an assignment)

Note that these cases may in fact be traceable in the future.

```py
Automatic Trace Checking
```

One way to automatically catch many errors in traces is by using `check_inputs` on the `torch.jit.trace()` API. `check_inputs` takes a list of tuples of inputs that will be used to re-trace the computation and verify the results. For example:

```py
def loop_in_traced_fn(x):
    result = x[0]
    for i in range(x.size(0)):
        result = result * x[i]
    return result

inputs = (torch.rand(3, 4, 5),)
check_inputs = [(torch.rand(4, 5, 6),), (torch.rand(2, 3, 4),)]

traced = torch.jit.trace(loop_in_traced_fn, inputs, check_inputs=check_inputs)

```

Gives us the following diagnostic information:

```py
ERROR: Graphs differed across invocations!
Graph diff:
    graph(%0 : Dynamic) {
          %1 : int = prim::Constant[value=0]()
          %2 : int = prim::Constant[value=0]()
          %3 : Dynamic = aten::select(%0, %1, %2)
          %4 : int = prim::Constant[value=0]()
          %5 : int = prim::Constant[value=0]()
          %6 : Dynamic = aten::select(%0, %4, %5)
          %7 : Dynamic = aten::mul(%3, %6)
          %8 : int = prim::Constant[value=0]()
          %9 : int = prim::Constant[value=1]()
          %10 : Dynamic = aten::select(%0, %8, %9)
          %11 : Dynamic = aten::mul(%7, %10)
          %12 : int = prim::Constant[value=0]()
          %13 : int = prim::Constant[value=2]()
          %14 : Dynamic = aten::select(%0, %12, %13)
          %15 : Dynamic = aten::mul(%11, %14)
      +   %16 : int = prim::Constant[value=0]()
      +   %17 : int = prim::Constant[value=3]()
      +   %18 : Dynamic = aten::select(%0, %16, %17)
      +   %19 : Dynamic = aten::mul(%15, %18)
      -   return (%15);
      ?             ^
      +   return (%19);
      ?             ^
    }

```

This message indicates to us that the computation differed between when we first traced it and when we traced it with the `check_inputs`. Indeed, the loop within the body of `loop_in_traced_fn` depends on the shape of the input `x`, and thus when we try another `x` with a different shape, the trace differs.

In this case, data-dependent control flow like this can be captured using script instead:

```py
def fn(x):
    result = x[0]
    for i in range(x.size(0)):
        result = result * x[i]
    return result

inputs = (torch.rand(3, 4, 5),)
check_inputs = [(torch.rand(4, 5, 6),), (torch.rand(2, 3, 4),)]

scripted_fn = torch.jit.script(fn)
print(scripted_fn.graph)

for input_tuple in [inputs] + check_inputs:
    torch.testing.assert_allclose(fn(*input_tuple), scripted_fn(*input_tuple))

```

Which produces:

```py
graph(%x : Dynamic) {
  %1 : int = prim::Constant[value=0]()
  %2 : int = prim::Constant[value=0]()
  %result.1 : Dynamic = aten::select(%x, %2, %1)
  %4 : int = aten::size(%x, %1)
  %5 : int = prim::Constant[value=1]()
  %result : Dynamic = prim::Loop(%4, %5, %result.1)
    block0(%i : int, %7 : Dynamic) {
      %9 : int = prim::Constant[value=0]()
      %10 : Dynamic = aten::select(%x, %9, %i)
      %result.2 : Dynamic = aten::mul(%7, %10)
      %12 : int = prim::Constant[value=1]()
      -> (%12, %result.2)
    }
  return (%result);
}

```

```py
Tracer Warnings
```

The tracer produces warnings for several problematic patterns in traced computation. As an example, take a trace of a function that contains an in-place assignment on a slice (a view) of a Tensor:

```py
def fill_row_zero(x):
    x[0] = torch.rand(*x.shape[1:2])
    return x

traced = torch.jit.trace(fill_row_zero, (torch.rand(3, 4),))
print(traced.graph)

```

Produces several warnings and a graph which simply returns the input:

```py
fill_row_zero.py:4: TracerWarning: There are 2 live references to the data region being modified when tracing in-place operator copy_ (possibly due to an assignment). This might cause the trace to be incorrect, because all other views that also reference this data will not not reflect this change in the trace! On the other hand, if all other views use the same memory chunk, but are disjoint (e.g. are outputs of torch.split), this might still be safe.
  x[0] = torch.rand(*x.shape[1:2])
fill_row_zero.py:6: TracerWarning: Output nr 1\. of the traced function does not match the corresponding output of the Python function. Detailed error:
Not within tolerance rtol=1e-05 atol=1e-05 at input[0, 1] (0.09115803241729736 vs. 0.6782537698745728) and 3 other locations (33.00%)
  traced = torch.jit.trace(fill_row_zero, (torch.rand(3, 4),))
graph(%0 : Float(3, 4)) {
  return (%0);
}

```

We can fix this by modifying the code to not use the in-place update, but rather build up the result tensor out-of-place with `torch.cat`:

```py
def fill_row_zero(x):
    x = torch.cat((torch.rand(1, *x.shape[1:2]), x[1:2]), dim=0)
    return x

traced = torch.jit.trace(fill_row_zero, (torch.rand(3, 4),))
print(traced.graph)

```

Torch Script supports a subset of the builtin tensor and neural network functions that PyTorch provides. Most methods on Tensor as well as functions in the `torch` namespace are available. Many functions in `torch.nn.functional` are also availiable.

We currently do not provide any builtin ScriptModules e.g. a `Linear` or `Conv` module. This functionality is something that will be developed in the future. For now we suggest using `torch.jit.trace` to transform standard `torch.nn` modules into ScriptModules on construction.

