# TORCHSCRIPT简介

> 译者：[masteryi-0018](https://github.com/masteryi-0018)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/Intro_to_TorchScript_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html>

作者： James Reed (jamesreed@fb.com), Michael Suo (suo@fb.com), 修订版2

本教程是对 TorchScript 的介绍，是 PyTorch 模型`nn.Module`（的子类）的一个中级表示，该模型然后可以在高性能环境（如C++）中运行。

在本教程中，我们将介绍：

1.在 PyTorch 中创作模型的基础知识，包括：

- 模块

- 定义函数`forward`

- 将模块组合到模块层次结构中

2.将 PyTorch 模块转换为 TorchScript 的具体方法，我们的高性能部署运行时

- Trace现有模块

- 使用script直接编译模块

- 如何组合这两种方法

- 保存和加载 TorchScript 模块

我们希望在完成本教程后，您将继续学习后续教程，该[教程](https://pytorch.org/tutorials/advanced/cpp_export.html)将引导您完成实际调用 TorchScript 的C++的模型的示例。 

```python
import torch  # This is all you need to use both PyTorch and TorchScript!
print(torch.__version__)
torch.manual_seed(191009)  # set the seed for reproducibility
```

输出：
```
2.0.1+cu117

<torch._C.Generator object at 0x7efffc5a6db0>
```

## PyTorch 模型创作的基础知识

让我们从定义一个简单的`Module`实例开始，`Module` 是 PyTorch 中的基本组成单位。它包含：

1. 一个构造函数，它为调用准备模块

2. 一组`Parameters`和子`Modules`。由构造函数初始化，并且可以在调用期间由模块使用。

3. 一个`forward`函数。这是模块运行时被调用代码。

让我们看一个小例子：

```python
class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()

    def forward(self, x, h):
        new_h = torch.tanh(x + h)
        return new_h, new_h

my_cell = MyCell()
x = torch.rand(3, 4)
h = torch.rand(3, 4)
print(my_cell(x, h))
```

输出：
```
(tensor([[0.8219, 0.8990, 0.6670, 0.8277],
        [0.5176, 0.4017, 0.8545, 0.7336],
        [0.6013, 0.6992, 0.2618, 0.6668]]), tensor([[0.8219, 0.8990, 0.6670, 0.8277],
        [0.5176, 0.4017, 0.8545, 0.7336],
        [0.6013, 0.6992, 0.2618, 0.6668]]))
```

因此，我们：

1. 创建了一个`torch.nn.Module`的子类。

2. 定义了一个构造函数。构造函数不做太多，只是调用super（父类）的构造函数。

3. 定义了一个`forward`函数，该函数接受两个输入并返回两个输出。`forward`函数的实际内容不是很重要，但它有点像假的 [RNN 单元](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) —— 意味着它是一个应用于循环的函数。

我们实例化了模块，并制作了 `x` 和 `h`，它们只是 3x4 随机值的矩阵。然后我们使用 `my_cell(x, h)`调用单元格。这反过来又调用了我们的`forward`函数。

让我们做一些更有趣的事情：

```python
class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

my_cell = MyCell()
print(my_cell)
print(my_cell(x, h))
```

输出：
```
MyCell(
  (linear): Linear(in_features=4, out_features=4, bias=True)
)
(tensor([[ 0.8573,  0.6190,  0.5774,  0.7869],
        [ 0.3326,  0.0530,  0.0702,  0.8114],
        [ 0.7818, -0.0506,  0.4039,  0.7967]], grad_fn=<TanhBackward0>), tensor([[ 0.8573,  0.6190,  0.5774,  0.7869],
        [ 0.3326,  0.0530,  0.0702,  0.8114],
        [ 0.7818, -0.0506,  0.4039,  0.7967]], grad_fn=<TanhBackward0>))
```

我们重新定义了我们的模块`MyCell`，但这次我们添加了一个属性`self.linear`，我们在`forward`函数调用了`self.linear`的功能。

这里到底发生了什么？是一个来自 PyTorch 标准库`torch.nn.Linear`。就像`Module`一样，它可以被`MyCell`使用调用语句调用。我们正在构建`Module`的层次结构。

`Module`上的`print`将直观地表示`Module`的子类层次结构。 在我们的示例中，我们可以看到`Linear`子类及其参数。

通过以这种方式组成`Module`，我们可以简洁易读地编写具有可重用组件的模型。

您可能已经在输出上注意到了`grad_fn`这个细节。这是 PyTorch 的自动微分方法，称为[autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)。 简而言之，该系统允许我们通过以下方式计算导数 可能复杂的程序。该设计允许大量的模型创作的灵活性。

现在让我们检查一下所说的灵活性：

```python
class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.dg = MyDecisionGate()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

my_cell = MyCell()
print(my_cell)
print(my_cell(x, h))
```

输出：
```
MyCell(
  (dg): MyDecisionGate()
  (linear): Linear(in_features=4, out_features=4, bias=True)
)
(tensor([[ 0.8346,  0.5931,  0.2097,  0.8232],
        [ 0.2340, -0.1254,  0.2679,  0.8064],
        [ 0.6231,  0.1494, -0.3110,  0.7865]], grad_fn=<TanhBackward0>), tensor([[ 0.8346,  0.5931,  0.2097,  0.8232],
        [ 0.2340, -0.1254,  0.2679,  0.8064],
        [ 0.6231,  0.1494, -0.3110,  0.7865]], grad_fn=<TanhBackward0>))
```

我们再次重新定义了我们的类`MyCell`，但在这里我们定义了`MyDecisionGate`，此模块利用**控制流**。控制流由循环和`if`语句等内容组成。

许多框架采用计算符号导数的方法给出完整的程序表示。但是，在 PyTorch 中，我们使用梯度tape。我们记录发生的操作，并对其进行回放计算导数的倒退。这样，框架就不会必须显式定义语言。

![自动grad的工作原理](../../img/dynamic_graph.gif)

## TORCHSCRIPT基础知识

现在让我们以我们的运行示例为例，看看如何应用 TorchScript。

简而言之，TorchScript提供了捕获您的定义的工具模型，即使考虑到 PyTorch 的灵活性和动态性。 让我们从我们所说的**Tracing**开始。


### Tracing模块

```python
class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell = torch.jit.trace(my_cell, (x, h))
print(traced_cell)
traced_cell(x, h)
```

输出：
```
MyCell(
  original_name=MyCell
  (linear): Linear(original_name=Linear)
)

(tensor([[-0.2541,  0.2460,  0.2297,  0.1014],
        [-0.2329, -0.2911,  0.5641,  0.5015],
        [ 0.1688,  0.2252,  0.7251,  0.2530]], grad_fn=<TanhBackward0>), tensor([[-0.2541,  0.2460,  0.2297,  0.1014],
        [-0.2329, -0.2911,  0.5641,  0.5015],
        [ 0.1688,  0.2252,  0.7251,  0.2530]], grad_fn=<TanhBackward0>))
```

我们回退了一点，并采用了我们教程的第二个版本`MyCell`。和以前一样，我们已经实例化了它，但这一次，我们调用了`torch.jit.trace`，传入了待trace的`Module`，并传入了示例网络可能看到的输入。

这到底做了什么？它调用了`Module`，记录了运行`Module`时发生的操作，并创建了`torch.jit.ScriptModule`的实例（其中`TracedModule`是实例）

TorchScript 以中间表示形式记录其定义（或IR），在深度学习中通常称为*图*。我们可以检查带有`.graph`属性的图：

```python
print(traced_cell.graph)
```

输出：
```
graph(%self.1 : __torch__.MyCell,
      %x : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu),
      %h : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %linear : __torch__.torch.nn.modules.linear.Linear = prim::GetAttr[name="linear"](%self.1)
  %20 : Tensor = prim::CallMethod[name="forward"](%linear, %x)
  %11 : int = prim::Constant[value=1]() # /var/lib/jenkins/workspace/beginner_source/Intro_to_TorchScript_tutorial.py:189:0
  %12 : Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu) = aten::add(%20, %h, %11) # /var/lib/jenkins/workspace/beginner_source/Intro_to_TorchScript_tutorial.py:189:0
  %13 : Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu) = aten::tanh(%12) # /var/lib/jenkins/workspace/beginner_source/Intro_to_TorchScript_tutorial.py:189:0
  %14 : (Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu), Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu)) = prim::TupleConstruct(%13, %13)
  return (%14)
```

但是，这是一个非常低级别的表示形式，大多数图表中包含的信息对最终用户没有用处。相反我们可以使用该`.code`属性给出 Python 语法解释的代码：


```python
print(traced_cell.code)
```

输出：
```
def forward(self,
    x: Tensor,
    h: Tensor) -> Tuple[Tensor, Tensor]:
  linear = self.linear
  _0 = torch.tanh(torch.add((linear).forward(x, ), h))
  return (_0, _0)
```

那么我们为什么要做这一切呢？有几个原因：

1. TorchScript 代码可以在它自己的解释器中调用，即基本上是一个受限制的Python解释器。此解释器不获取全局解释器锁，可以有很多请求同时在同一实例上处理。

2. 这种格式允许我们将整个模型保存到磁盘并加载它进入另一个环境，例如在除了python的用某种语言编写的服务器中

3. TorchScript 为我们提供了一个表示，我们可以在其中进行编译器优化代码以提供更高效的执行

4. TorchScript 允许我们与许多后端/设备运行时接口，这需要比单个运营商更广泛的程序视图。

我们可以看到，调用`traced_cell`产生的结果与 Python 模块：

```python
print(my_cell(x, h))
print(traced_cell(x, h))
```

输出：
```
(tensor([[-0.2541,  0.2460,  0.2297,  0.1014],
        [-0.2329, -0.2911,  0.5641,  0.5015],
        [ 0.1688,  0.2252,  0.7251,  0.2530]], grad_fn=<TanhBackward0>), tensor([[-0.2541,  0.2460,  0.2297,  0.1014],
        [-0.2329, -0.2911,  0.5641,  0.5015],
        [ 0.1688,  0.2252,  0.7251,  0.2530]], grad_fn=<TanhBackward0>))
(tensor([[-0.2541,  0.2460,  0.2297,  0.1014],
        [-0.2329, -0.2911,  0.5641,  0.5015],
        [ 0.1688,  0.2252,  0.7251,  0.2530]],
       grad_fn=<DifferentiableGraphBackward>), tensor([[-0.2541,  0.2460,  0.2297,  0.1014],
        [-0.2329, -0.2911,  0.5641,  0.5015],
        [ 0.1688,  0.2252,  0.7251,  0.2530]],
       grad_fn=<DifferentiableGraphBackward>))
```

## 使用脚本转换模块

我们使用模块的第二个版本是有原因的，而不是带有充满控制流的子模块。现在让我们检查一下：

```python
class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

my_cell = MyCell(MyDecisionGate())
traced_cell = torch.jit.trace(my_cell, (x, h))

print(traced_cell.dg.code)
print(traced_cell.code)
```

输出：
```
/var/lib/jenkins/workspace/beginner_source/Intro_to_TorchScript_tutorial.py:261: TracerWarning:

Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!

def forward(self,
    argument_1: Tensor) -> NoneType:
  return None

def forward(self,
    x: Tensor,
    h: Tensor) -> Tuple[Tensor, Tensor]:
  dg = self.dg
  linear = self.linear
  _0 = (linear).forward(x, )
  _1 = (dg).forward(_0, )
  _2 = torch.tanh(torch.add(_0, h))
  return (_2, _2)
```

查看`.code`输出，可以发现找不到`if-else`分支！为什么？跟踪完全按照我们所说的去做：运行代码，记录发生的操作，并构造一个执行此操作的`ScriptModule`。不幸的是，诸如控制流之类的东西被擦除了。

我们如何在 TorchScript 中忠实地表示此模块？我们提供了**script编译器**，它可以直接分析您的 Python 源代码以将其转换为 TorchScript。让我们使用脚本编译器转换`MyDecisionGate`：

```python
scripted_gate = torch.jit.script(MyDecisionGate())

my_cell = MyCell(scripted_gate)
scripted_cell = torch.jit.script(my_cell)

print(scripted_gate.code)
print(scripted_cell.code)
```

输出：
```
def forward(self,
    x: Tensor) -> Tensor:
  if bool(torch.gt(torch.sum(x), 0)):
    _0 = x
  else:
    _0 = torch.neg(x)
  return _0

def forward(self,
    x: Tensor,
    h: Tensor) -> Tuple[Tensor, Tensor]:
  dg = self.dg
  linear = self.linear
  _0 = torch.add((dg).forward((linear).forward(x, ), ), h)
  new_h = torch.tanh(_0)
  return (new_h, new_h)
```

万岁！现在，我们已经忠实地捕获了我们程序中 TorchScript 的行为。现在让我们尝试运行该程序：

```python
# New inputs
x, h = torch.rand(3, 4), torch.rand(3, 4)
print(scripted_cell(x, h))
```

输出：
```
(tensor([[ 0.5679,  0.5762,  0.2506, -0.0734],
        [ 0.5228,  0.7122,  0.6985, -0.0656],
        [ 0.6187,  0.4487,  0.7456, -0.0238]], grad_fn=<TanhBackward0>), tensor([[ 0.5679,  0.5762,  0.2506, -0.0734],
        [ 0.5228,  0.7122,  0.6985, -0.0656],
        [ 0.6187,  0.4487,  0.7456, -0.0238]], grad_fn=<TanhBackward0>))
```

### 混合Script and Trace

在某些情况下需要使用Trace而不是Script（例如模块有许多基于常量的架构决策，我们希望不出现在 TorchScript 中的 Python 值）。在这种情况下，Script可以用Trace来组合：`torch.jit.script`将内联被跟踪模块的代码，而跟踪将内联脚本模块的代码。

第一种情况的示例：

```python
class MyRNNLoop(torch.nn.Module):
    def __init__(self):
        super(MyRNNLoop, self).__init__()
        self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))

    def forward(self, xs):
        h, y = torch.zeros(3, 4), torch.zeros(3, 4)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y, h

rnn_loop = torch.jit.script(MyRNNLoop())
print(rnn_loop.code)
```

输出：
```
def forward(self,
    xs: Tensor) -> Tuple[Tensor, Tensor]:
  h = torch.zeros([3, 4])
  y = torch.zeros([3, 4])
  y0 = y
  h0 = h
  for i in range(torch.size(xs, 0)):
    cell = self.cell
    _0 = (cell).forward(torch.select(xs, 0, i), h0, )
    y1, h1, = _0
    y0, h0 = y1, h1
  return (y0, h0)
```

还有第二种情况的例子：

```python
class WrapRNN(torch.nn.Module):
    def __init__(self):
        super(WrapRNN, self).__init__()
        self.loop = torch.jit.script(MyRNNLoop())

    def forward(self, xs):
        y, h = self.loop(xs)
        return torch.relu(y)

traced = torch.jit.trace(WrapRNN(), (torch.rand(10, 3, 4)))
print(traced.code)
```

输出：
```
def forward(self,
    xs: Tensor) -> Tensor:
  loop = self.loop
  _0, y, = (loop).forward(xs, )
  return torch.relu(y)
```

这样，当情况需要时，可以将Trace和Script一起使用。

## 保存和加载模型

我们提供API来保存TorchScript模块并将其加载到磁盘/从磁盘加载存档格式。此格式包括代码、参数、属性和调试信息，这意味着存档是独立的，可以在完全独立的模型中加载的模型的表示形式过程。让我们保存并加载包装的 RNN 模块：

```python
traced.save('wrapped_rnn.pt')

loaded = torch.jit.load('wrapped_rnn.pt')

print(loaded)
print(loaded.code)
```

输出：
```
RecursiveScriptModule(
  original_name=WrapRNN
  (loop): RecursiveScriptModule(
    original_name=MyRNNLoop
    (cell): RecursiveScriptModule(
      original_name=MyCell
      (dg): RecursiveScriptModule(original_name=MyDecisionGate)
      (linear): RecursiveScriptModule(original_name=Linear)
    )
  )
)
def forward(self,
    xs: Tensor) -> Tensor:
  loop = self.loop
  _0, y, = (loop).forward(xs, )
  return torch.relu(y)
```

如您所见，序列化保留了模块层次结构和我们一直在研究的代码。也可以[将模型加载到 C++ 中](https://pytorch.org/tutorials/advanced/cpp_export.html)，以实现不依赖 Python 的执行。

### 延伸阅读

我们已经完成了教程！有关更多涉及证明的演示，请查看来自 NeurIPS 演示，使用 TorchScript 转换机器翻译模型：
[https://colab.research.google.com/drive/1HiICg6jRkBnr5hvK2-VnMi88Vi9pUzEJ](https://colab.research.google.com/drive/1HiICg6jRkBnr5hvK2-VnMi88Vi9pUzEJ)
