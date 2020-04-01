# TorchScript

> 译者：[keyianpai](https://github.com/keyianpai)

*   [创建 Torch 脚本代码](#creating-torch-script-code)
*   [将追踪和脚本化结合起来](#mixing-tracing-and-scripting)
*   [Torch 脚本语言参考](#torch-script-language-reference)
    *   [类型](#types)
    *   [表达式](#expressions)
    *   [语句](#statements)
    *   [变量解析](#variable-resolution)
    *   [python值的使用](#use-of-python-values)
    *   [调试](#debugging)
    *   [内置函数](#builtin-functions)
   
Torch脚本是一种从PyTorch代码创建可序列化和可优化模型的方法。用Torch脚本编写的代码可以从Python进程中保存，并在没有Python依赖的进程中加载。

我们提供了一些工具帮助我们将模型从纯Python程序逐步转换为可以独立于Python运行的Torch脚本程序。Torch脚本程序可以在其他语言的程序中运行(例如，在独立的C ++程序中）。这使得我们可以使用熟悉的工具在PyTorch中训练模型，而将模型导出到出于性能和多线程原因不能将模型作为Python程序运行的生产环境中去。

```py
class torch.jit.ScriptModule(optimize=True)
```

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
> &gt;* 追踪不会记录if语句或循环之类的控制流。当这个控制流在你的模块中保持不变时，这很好，它通常只是内联配置决策。但有时控制流实际上是模型本身的一部分。例如，序列到序列转换中的beam搜索是对(可变）输入序列长度的循环。
> 
> &gt;*在返回的`ScriptModule`中，在`training`和`eval`模式中具有不同行为的操作将始终表现为处于追踪期间的模式。
>
> 在上述情况下，脚本化是一个比追踪更好的选择。


**脚本化**

> 你可以使用Python语法直接编写Torch脚本代码。你可以使用`torch.jit.script`注释(对于函数）或`torch.jit.script_method`注释(对于ScriptModule子类的方法）来编写Torch脚本代码。通过注释，被注释函数的主体将直接转换为Torch脚本。 Torch脚本本身只是Python语言的一个子集，因此不是python中的所有特性都可以使用，但我们提供了足够的功能来计算张量并执行与控制相关的操作。
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
>         # 通过追踪产生ScriptModule的 conv1和conv2
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

保存离线版本的模块，以便将来在其他的进程中使用。保存的模块会序列化当前模块的所有方法和参数。保存的模块可以使用`torch :: jit :: load(filename）`加载到C ++ API中，也可以使用`torch.jit.load(filename）`加载到Python API中。

为了能够保存模块，当前模块不能调用原生python函数。也就是说要保存模块的所有子模块也必须是ScriptModules的子类。

危险

所有模块，不论其设备，在加载过程中始终都会加载到CPU中。这与`torch.load()`的语义不同，将来可能会发生变化。


```py
torch.jit.load(f, map_location=None)
```

使用`load`加载之前用`save`保存的`ScriptModule`。

所有先前保存的模块，不论其设备，首先加载到CPU上，然后移动到之前保存它们的设备上。如果此操作失败(例如，运行时系统没有某些设备），则会引发异常。此时可以使用`map_location`参数将存储重新映射到另一组设备。与`torch.load()`相比，此函数中的`map_location`被简化为只接受字符串(例如'cpu'，'cuda：0'）或torch.device(例如，torch.device('cpu'））

参数：

*   **f** – 文件类对象(必须实现read，readline，tell和seek），或为文件名的字符串
*   **map_location** – 可以是一个字符串(例如，'cpu'，'cuda：0'），一个设备(例如，torch.device('cpu'））


| 返回值: |  `ScriptModule` 对象. |
| --- | --- |

例：

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

追踪仅正确记录不依赖于数据的函数和模块(例如，对张量中的数据进行条件判断），并且没有任何未追踪的外部依赖性(例如，执行输入/输出或访问全局变量）。如果你追踪此类模型，则可能会在随后的模型调用中静默获取不正确的结果。当执行可能生成错误追踪的内容时，追踪器将尝试发出警告。

参数： 

*   **func** (_callable_ _or_ [_torch.nn.Module_](nn.html#torch.nn.Module "torch.nn.Module")) – 将使用example_inputs作为输入运行的python函数或torch.nn.Module。参数和返回值必须是Tensor或(嵌套的）包含张量的元组。
*   **example_inputs** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")) – 在追踪时将传递给函数的示例输入元组。假设被追踪操作支持这些类型和形状的情况下，生成的追踪可以在不同类型和形状的输入下运行。 example_inputs也可以是单个Tensor，这种情况下，它会自动包装到元组中。


| 关键字参数： |
| --- |

*   **optimize** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – 是否应用优化。默认值：`True`。
*   **check_trace** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – 检查被追踪代码在相同输入下输出是否相同。默认值：`True`。你可以在某些情况下禁用此功能。例如，你的网络包含非确定性操作，或者你确定网络正确。
*   **check_inputs** (_list of tuples__,_ _optional_) – 应该用于根据预期检查追踪的输入参数元组列表。每个元组相当于一个将在`args`中指定的输入参数集合。为获得最佳结果，请传递一组检查输入表示你期望网络接受的形状和输入类型范围。如果未指定，则用原来的`args`检查。
*   **check_tolerance** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – 在检查过程中使用的浮点比较容差。用于放松检查严格性。 

 
| 返回值： | 含有`forward()`方法的`ScriptModule`对象，该方法包含被追踪代码。当func是`torch.nn.Module`时，返回的`ScriptModule`具有与原始模块相同的子模块和参数集。|
| --- | --- |

例：

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

Torch脚本是Python的一个子集，可以直接编写(使用@script注释），也可以通过追踪从Python代码自动生成。使用追踪时，代码会自动转换为Python的这个子集，方法是仅记录和执行张量上的实际运算符，并丢弃其他Python代码。

使用@script注释直接编写Torch脚本时，程序员必须只使用Torch脚本支持的Python子集。本节以语言参考的形式介绍Torch脚本支持的功能。本参考中未提及的Python的其他功能都不是Torch脚本的一部分。

作为Python的一个子集，任何有效的Torch脚本函数也是一个有效的Python函数。因此你可以删除@script注释后使用标准Python工具(如pdb）调试函数。反之则不然：有许多有效的python程序不是有效的Torch脚本程序。Torch脚本专注于在Torch中表示神经网络模型所需的Python特性。



```py
PYTORCH_JIT=1
```

设置环境变量`PYTORCH_JIT = 0`将禁用所有脚本和追踪注释。如果在ScriptModule中遇到难以调试的错误，则可以使用此标志强制使用原生Python运行所有内容。此时可使用`pdb`之类的工具调试代码。
 
 Torch脚本与完整Python语言之间的最大区别在于Torch脚本仅支持表达神经网络模型所需的一些类型。特别地，Torch脚本支持：


```py
Tensor
```

具有任何dtype，维度或backend的PyTorch张量。

```py
Tuple[T0, T1, ...]
```

包含子类型`T0`，`T1`等的元组(例如`Tuple [Tensor，Tensor]`）。


```py
int
```

标量整数

```py
float
```
标量浮点数

```py
List[T]
```

所有成员都是T类型的列表`T`

与Python不同，Torch脚本函数中的每个变量都必须具有一个静态类型。这样以便于优化Torch脚本功能。


例：

```py
@torch.jit.script
def an_error(x):
    if x:
        r = torch.rand(1)
    else:
        r = 4
    return r # 类型不匹配：在条件为真时r为Tensor类型
             # 而为假时却是int类型

```

默认情况下，Torch脚本函数的所有参数都为Tensor类型，因为这是模块中最常用的类型。要将Torch脚本函数的参数指定为另一种类型，可以通过MyPy风格的注释使用上面列出的类型：

例：

```py
@torch.jit.script
def foo(x, tup):
    # type: (int, Tuple[Tensor, Tensor]) -> Tensor
    t0, t1 = tup
    return t0 + t1 + x

print(foo(3, (torch.rand(3), torch.rand(3))))

```

注意

也可以使用Python 3类型注释来注释类型。在示例中，我们使用基于注释的注释来确保对Python 2的兼容性。

Torch脚本支持以下Python表达式

```py
字面常量
```

`True`, `False`, `None`, `'string literals'`, `"string literals"`,  字面值`3`(解释为int）`3.4`(解释为float）

```py
变量
```

`a`

注意

请参阅[变量解析](#variable-resolution)，了解变量的解析方式。

```py
元组构造
```

`(3, 4)`, `(3,)`

```py
列表构造
```

`[3, 4]`, `[]`, `[torch.rand(3), torch.rand(4)]`

注意

空列表具有类型`List[Tensor]` 。其他列表字面常量的类型由成员的类型推出。


```py
算术运算符
```

`a + b` `a - b` `a * b` `a / b` `a ^ b` `a @ b`

```py
比较运算符
```

`a == b` `a != b` `a < b` `a > b` `a <= b` `a >= b`

```py
逻辑运算符
```

`a and b` `a or b` `not b`

```py
索引
```

`t[0]` `t[-1]` `t[0:2]` `t[1:]` `t[:1]` `t[:]` `t[0, 1]` `t[0, 1:2]` `t[0, :1]` `t[-1, 1:, 0]` `t[1:, -1, 0]` `t[i:j, i]`

注意

Torch脚本目前不支持原地修改张量，因此对张量索引只能出现在表达式的右侧。


```py
函数调用
```

调用内置函数：`torch.rand(3, dtype=torch.int)`

调用其他脚本函数：

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
方法调用
```

调用内置类型的方法，如tensor: `x.mm(y)`

在ScriptModule中定义Script方法时，使用`@script_method`批注。Script方法可以调用模块内其他方法或子模块的方法。

直接调用子模块(例如`self.resnet(input）`）等同于调用其`forward`方法(例如`self.resnet.forward(input）`）

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
If 表达式
```

`x if x > y else y`

```py
类型转换
```

`float(ten)`, `int(3.5)`, `bool(ten)`

```py
访问模块参数
```

`self.my_parameter` `self.my_submodule.my_parameter`

Torch脚本支持以下类型的语句：

简单赋值

> ```py
> a = b
> a += b # short-hand for a = a + b, does not operate in-place on a
> a -= b
> 
> ```

模式匹配赋值

> ```py
> a, b = tuple_or_list
> a, b, *c = a_tuple
> 
> ```

Print 语句

> `print("the result of an add:", a + b)`

If 语句

> ```py
> if a &lt; 4:
>     r = -a
> elif a &lt; 3:
>     r = a + a
> else:
>     r = 3 * a
> 
> ```

While 循环

> ```py
> a = 0
> while a &lt; 4:
>     print(a)
>     a += 1
> 
> ```

带 `range` 的for循环

> ```py
> x = 0
> for i in range(10):
>     x *= i
> 
> ```
> 
> 注意
> 
> 脚本目前不支持对一般可迭代对象(如列表或张量）进行迭代，也不支持range起始与增量参数，这些将在未来版本中添加。

对元组的for循环：

> ```py
> tup = (3, torch.rand(4))
> for x in tup:
>     print(x)
> 
> ```
> 
> 注意
> 
> 对于元组循环将展开循环，为元组的每个成员生成一个循环体。循环体内必须确保每个成员类型正确。

对常量 `torch.nn.ModuleList` 的for循环

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
> 注意
> 
> 要在`@script_method`中使用模块列表，必须通过将属性的名称添加到类型的`__constants__`列表来将其标记为常量。ModuleList上的for循环在编译时使用常量模块列表的每个成员展开循环体。

```py
Return 语句
```

`return a, b`

注意

return语句必须作为函数的最后一个成员，而不能出现在函数的其他位置。此限制将在以后删除。

Torch脚本支持Python变量解析(即作用域）规则的子集。局部变量的行为与Python中的相同，但变量必须在函数的所有路径中具有相同类型。如果变量在if语句的不同侧具有不同的类型，则在if语句结束后使用它会抱错。

类似地，如果仅在函数的某些执行路径上定义变量也会出错。

例：

```py
@torch.jit.script
def foo(x):
    if x < 0:
        y = 4
    print(y) # 错误: y 值未定义

```

定义函数的非局部变量在编译时解析为Python值。然后，用[Python值的使用](#use-of-python-values)中的规则将这些值转换为Torch脚本值。

为了使编写Torch脚本更方便，我们允许脚本代码引用周围的Python值。例如，当我们引用`torch`时，Torch脚本编译器实际上在声明函数时将其解析为Python的`torch`模块。这些Python值不是Torch脚本的一部分，它们在编译时被转换成Torch脚本支持的原始类型。本节介绍在Torch脚本中访问Python值时使用的规则。它们依赖于引用的python值的动态类型。

```py
函数
```

Torch脚本可以调用python函数。此功能在将模型逐步转换为脚本时非常有用。可以将模型中的函数逐个转成脚本，保留对其余Python函数的调用。这样，在逐步转换的过程中你可以随时检查模型的正确性。

例：

```py
def foo(x):
  print("I am called with {}".format(x))
  import pdb; pdb.set_trace()
  return x

@torch.jit.script
def bar(x)
  return foo(x + 1)

```

注意

不能在包含Python函数调用的ScriptModule上调用`save`。该功能仅用于调试，应在保存之前删除调用或将其转换为脚本函数。

```py
Python模块的属性查找
```

Torch脚本可以在模块上查找属性。像`torch.add`这样的内置函数就以这种方式访问。这允许Torch脚本调用其他模块中定义的函数。

```py
Python 中定义的常量
```

Torch脚本还提供了使用Python常量的方法。这可用于将超参数硬编码到函数中，或用于定义通用常量。有两种方法可以指定Python值为常量。

1.  查找的值为模块的属性,例如：`math.pi`

2.  可以将ScriptModule的属性标记为常量，方法是将其列为类的`__constants__`属性成员：

    例：

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

支持的Python常量值有

*   `int`
*   `bool`
*   `torch.device`
*   `torch.layout`
*   `torch.dtype`
*   包含支持类型的元组
*   `torch.nn.ModuleList` ，可以将其用在Torch 脚本for循环中

```py
禁用JIT以方便调试
```

可以通过将`PYTORCH_JIT`环境变量值设置为`0`禁用所有`JIT`模式(追踪和脚本化）以便在原始Python中调试程序。下面是一个示例脚本：

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

为了使用PDB调试此脚本。我们可以全局禁用JIT，这样我们就可以将@script函数作为普通的python函数调用而不会编译它。如果上面的脚本名为`disable_jit_example.py`，我们这样调用它：

```py
$ PYTORCH_JIT=0 python disable_jit_example.py

```

这样,我们就能够作为普通的Python函数步入@script函数。

```py
解释图
```

TorchScript使用静态单一指派(SSA）中间表示(IR）来表示计算。这种格式的指令包括ATen(PyTorch的C ++后端）运算符和其他原始运算符，包括循环和条件的控制流运算符。举个例子：

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

具有单个`forward`方法的`ScriptModule`具有`graph`属性，你可以使用该图来检查表示计算的IR。如果ScriptModule有多个方法，则需要访问方法本身的`.graph`属性。例如我们可以通过访问`.bar.graph`来检查ScriptModule上名为`bar`的方法的图。

上面的示例脚本生成图形：

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

以指令`％rv.1：Dynamic = aten :: zeros(％3，％4，％5，％6）`为例。` ％rv.1：Dynamic`将输出分配给名为`rv.1`的(唯一）值，该值是动态类型，即我们不知道它的具体形状。` aten :: zeros`是运算符(相当于`torch.zeros`），它的输入列表`(％3，％4，％5，％6）`指定范围中的哪些值应作为输入传递。内置函数(如`aten :: zeros`）的模式可以在[内置函数](#builtin-functions)中找到。

注意，运算符也可以有关联的`block`，如`prim :: Loop`和`prim :: If`运算符。在图形打印输出中，这些运算符被格式化以反映与其等价的源代码形式，以便于调试。

可以检查图以确认`ScriptModule`描述的计算是正确的，方法如下所述。

```py
追踪的边缘情况
```

在一些边缘情况下一些Python函数/模块的追踪不能代表底层代码。这些情况可以包括：

*   追踪依赖于输入的控制流(例如张量形状）
*   追踪张量视图的就地操作(例如，在分配的左侧进行索引）

请注意，这些情况在将来版本中可能是可追踪的。

```py
自动追踪检查
```

通过在`torch.jit.trace()`API上使用`check_inputs`，是自动捕获追踪中错误的一种方法。 `check_inputs`是用于重新追踪计算并验证结果的输入元组列表。例如：

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

上面代码会为我们提供以下诊断信息：

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

此消息表明，我们第一次追踪函数和使用`check_inputs`追踪函数时的计算存在差异。事实上，`loop_in_traced_fn`体内的循环取决于输入x的形状，因此当我们输入不同形状的`x`时，轨迹会有所不同。

在这种情况下，可以使用脚本捕获此类数据相关控制流：

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

上面代码会为我们提供以下信息：

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
追踪器警告
```

追踪器会在追踪计算中对有问题的模式生成警告。例如，追踪包含在Tensor的切片(视图）上就地赋值操作的函数：

```py
def fill_row_zero(x):
    x[0] = torch.rand(*x.shape[1:2])
    return x

traced = torch.jit.trace(fill_row_zero, (torch.rand(3, 4),))
print(traced.graph)

```

这会出现如下警告和一个简单返回输入的图：

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

我们可以通过使用`torch.cat`返回结果张量避免就地更新问题：

```py
def fill_row_zero(x):
    x = torch.cat((torch.rand(1, *x.shape[1:2]), x[1:2]), dim=0)
    return x

traced = torch.jit.trace(fill_row_zero, (torch.rand(3, 4),))
print(traced.graph)

```

Torch脚本支持部分PyTorch内置张量和神经网络函数。 Tensor上的大多数方法以及`torch`命名空间中的函数都可用。 `torch.nn.functional`中的许多函数也可用。


我们目前不提供像 `Linear` 或 `Conv` 模块之类内置ScriptModule,此功能将在未来开发。目前我们建议使用`torch.jit.trace`将标准的`torch.nn`模块转换为ScriptModule。
