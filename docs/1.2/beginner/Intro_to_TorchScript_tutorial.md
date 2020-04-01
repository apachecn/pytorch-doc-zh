# 介绍TorchScript

> **作者**：James Reed (jamesreed@fb.com), Michael Suo(suo@fb.com), rev2
>
> **译者**：[松鼠](https://github.com/HelWireless)
>
> **校验**：[松鼠](https://github.com/HelWireless)

本教程是TorchScript的简介，TorchScript是PyTorch模型(子类nn.Module）的中间表示，可以在高性能环境(例如C ++）中运行。

在本教程中，我们将介绍：

1. PyTorch中的模型基础创建，包括：
    * 模块
    * 定义`forward`功能
    * 构成模块组成模块的层次结构

2. 将PyTorch模块转换为TorchScript(我们的高性能部署运行时）的特定方法
    * 跟踪现有模块
    * 使用脚本来直接编译的模块
    * 如何组合这两种方法
    * 保存和加载TorchScript模块

我们希望在完成本教程之后，您将继续阅读后续教程 ，该教程将引导您实际从C++调用TorchScript模型的示例。


```python    
import torch # This is all you need to use both
PyTorch and TorchScript!
print(torch.__version__)
```
输出：
```shell
 1.2.0
```    

## PyTorch模型制作的基础

让我们开始定义一个简单的`Module`。A`Module`是PyTorch中组成的基本单位。它包含：

1. 构造函数，为调用准备模块
2. 一组`Parameters`和`Modules`。这些由构造函数初始化，并且可以在调用期间由模块使用。
3. `forward`功能。这是调用模块时运行的代码。

让我们来看看一个小例子：

    
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

Out:
```python
    (tensor([[0.7853, 0.8882, 0.7137, 0.3746],
            [0.5265, 0.8508, 0.1487, 0.9144],
            [0.7057, 0.8217, 0.9575, 0.6132]]), tensor([[0.7853, 0.8882, 0.7137, 0.3746],
            [0.5265, 0.8508, 0.1487, 0.9144],
            [0.7057, 0.8217, 0.9575, 0.6132]]))
```    

因此，我们已经：

  1. 创建子类的类`torch.nn.Module`。
  2. 定义构造函数。构造函数没有做太多事情，只是调用的构造函数`super`。
  3. 定义了一个`forward`函数，该函数接受两个输入并返回两个输出。该`forward`函数的实际内容并不是很重要，但是它是一种伪造的RNN单元即，该函数应用于循环。

我们实例化了模块，并制作了`x`和`y`，它们只是3x4随机值矩阵。然后，我们使用调用单元,调用我们的`forward`函数`my_cell(x, h)`

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

Out:

    
```python

    MyCell(
      (linear): Linear(in_features=4, out_features=4, bias=True)
    )
    (tensor([[0.7619, 0.7761, 0.7476, 0.0897],
            [0.6886, 0.4990, 0.4495, 0.2021],
            [0.5849, 0.5874, 0.9256, 0.0460]], grad_fn=<TanhBackward>), tensor([[0.7619, 0.7761, 0.7476, 0.0897],
            [0.6886, 0.4990, 0.4495, 0.2021],
            [0.5849, 0.5874, 0.9256, 0.0460]], grad_fn=<TanhBackward>))
```   

我们已经重新定义了模块`MyCell`，但是这次我们添加了一个 `self.linear`属性，并`self.linear`在`forward`函数中调用。

这里到底发生了什么？`torch.nn.Linear`是`Module`来自PyTorch标准库的。就像一样`MyCell`，可以使用调用语法来调用它。我们正在建立的层次结构`Module`们。

`print`上的`Module`会直观地表示 `Module`的子类层次结构。在我们的示例中，我们可以看到 `Linear`子类及其参数。

通过这种方式构成`Module`们，我们可以简洁而易读地编写具有可复用组件的模型。

您可能已经注意到输出中的`grad_fn`了。这是PyTorch自动区分求导给出的信息，称为`autograd`。简而言之，该系统允许我们通过潜在的复杂程序来计算导数。该设计为模型创作提供了极大的灵活性。

现在让我们检查一下灵活性：
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

Out:

```python   
    
    MyCell(
      (dg): MyDecisionGate()
      (linear): Linear(in_features=4, out_features=4, bias=True)
    )
    (tensor([[ 0.9077,  0.5939,  0.6809,  0.0994],
            [ 0.7583,  0.7180,  0.0790,  0.6733],
            [ 0.9102, -0.0368,  0.8246, -0.3256]], grad_fn=<TanhBackward>), tensor([[ 0.9077,  0.5939,  0.6809,  0.0994],
            [ 0.7583,  0.7180,  0.0790,  0.6733],
            [ 0.9102, -0.0368,  0.8246, -0.3256]], grad_fn=<TanhBackward>))
```    

我们再次重新定义了MyCell类，但是在这里我们定义了 `MyDecisionGate`。该模块利用控制流程。控制流包括循环和if-statements之类的东西。

给定完整的程序表示形式，许多框架都采用计算符号派生的方法。但是，在PyTorch中，我们使用渐变色带。我们记录操作发生时的操作，并在计算衍生产品时向后回放。这样，框架不必为语言中的所有构造显式定义派生类。

![How autograd
works](https://github.com/pytorch/pytorch/raw/master/docs/source/_static/img/dynamic_graph.gif)

autograd的工作原理

## TorchScript的基础

现在，让我们以正在运行的示例为例，看看如何应用TorchScript。

简而言之，即使PyTorch具有灵活和动态的特性，TorchScript也提供了捕获模型定义的工具。让我们开始研究所谓的**跟踪**。



### 追踪`Modules `

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

Out:
```python   
    TracedModule[MyCell](
      (linear): TracedModule[Linear]()
    )
```    

我们来看看之前的例子。和以前一样，我们实例化了它，但是这次，我们使用`torch.jit.trace`方法调用了Module，并传入，然后传入了网络可能的示例输入。

这到底是做什么的？它已调用`Module`，记录了`Module`运行时发生的操作，并创建了`torch.jit.ScriptModule`(TracedModule的实例）

TorchScript将其定义记录在中间表示(或IR）中，在深度学习中通常称为图形。我们可以检查具有以下`.graph`属性的图形：
    
```python    
    print(traced_cell.graph)
```   

Out:
```python
    
    
    graph(%self : ClassType<MyCell>,
          %input : Float(3, 4),
          %h : Float(3, 4)):
      %1 : ClassType<Linear> = prim::GetAttr[name="linear"](%self)
      %weight : Tensor = prim::GetAttr[name="weight"](%1)
      %bias : Tensor = prim::GetAttr[name="bias"](%1)
      %6 : Float(4!, 4!) = aten::t(%weight), scope: MyCell/Linear[linear] # /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py:1369:0
      %7 : int = prim::Constant[value=1](), scope: MyCell/Linear[linear] # /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py:1369:0
      %8 : int = prim::Constant[value=1](), scope: MyCell/Linear[linear] # /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py:1369:0
      %9 : Float(3, 4) = aten::addmm(%bias, %input, %6, %7, %8), scope: MyCell/Linear[linear] # /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py:1369:0
      %10 : int = prim::Constant[value=1](), scope: MyCell # /var/lib/jenkins/workspace/beginner_source/Intro_to_TorchScript_tutorial.py:188:0
      %11 : Float(3, 4) = aten::add(%9, %h, %10), scope: MyCell # /var/lib/jenkins/workspace/beginner_source/Intro_to_TorchScript_tutorial.py:188:0
      %12 : Float(3, 4) = aten::tanh(%11), scope: MyCell # /var/lib/jenkins/workspace/beginner_source/Intro_to_TorchScript_tutorial.py:188:0
      %13 : (Float(3, 4), Float(3, 4)) = prim::TupleConstruct(%12, %12)
      return (%13)
```  
但是，这是一个非常低级的表示形式，图中包含的大多数信息对最终用户没有用。相反，我们可以使用`.code`属性为代码提供Python语法的解释：

    
```python   
    print(traced_cell.code)
```

Out:
```python 
    def forward(self,
        input: Tensor,
        h: Tensor) -> Tuple[Tensor, Tensor]:
      _0 = self.linear
      weight = _0.weight
      bias = _0.bias
      _1 = torch.addmm(bias, input, torch.t(weight), beta=1, alpha=1)
      _2 = torch.tanh(torch.add(_1, h, alpha=1))
      return (_2, _2)
```    

那么**为什么**我们要做所有这些呢？有以下几个原因：

1. TorchScript代码可以在其自己的解释器中调用，该解释器基本上是受限制的Python解释器。该解释器不获取全局解释器锁定，因此可以在同一实例上同时处理许多请求。
2. 这种格式使我们可以将整个模型保存到磁盘上，并将其加载到另一个环境中，例如在以Python以外的语言编写的服务器中
3. TorchScript为我们提供了一种表示形式，其中我们可以对代码进行编译器优化以提供更有效的执行
4. TorchScript允许我们与许多后端/设备运行时进行接口，这些运行时比单个操作员需要更广泛的程序视图。

我们可以看到调用`traced_cell`产生的结果与Python模块相同：：
```python    
    
    print(my_cell(x, h))
    print(traced_cell(x, h))
```

Out:
```python 
    (tensor([[ 0.0294,  0.2921,  0.5171,  0.2689],
            [ 0.5859,  0.8311,  0.2553,  0.8026],
            [-0.4138,  0.7641,  0.4251,  0.7217]], grad_fn=<TanhBackward>), tensor([[ 0.0294,  0.2921,  0.5171,  0.2689],
            [ 0.5859,  0.8311,  0.2553,  0.8026],
            [-0.4138,  0.7641,  0.4251,  0.7217]], grad_fn=<TanhBackward>))
    (tensor([[ 0.0294,  0.2921,  0.5171,  0.2689],
            [ 0.5859,  0.8311,  0.2553,  0.8026],
            [-0.4138,  0.7641,  0.4251,  0.7217]],
           grad_fn=<DifferentiableGraphBackward>), tensor([[ 0.0294,  0.2921,  0.5171,  0.2689],
            [ 0.5859,  0.8311,  0.2553,  0.8026],
            [-0.4138,  0.7641,  0.4251,  0.7217]],
           grad_fn=<DifferentiableGraphBackward>))
```
## 使用脚本来转换模块

我们使用模块的第二个版本是有原因的，而不是使用带有控制流的子模块的一个版本。现在让我们检查一下：
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
    print(traced_cell.code)
```

Out:
```python
    
    
    def forward(self,
        input: Tensor,
        h: Tensor) -> Tuple[Tensor, Tensor]:
      _0 = self.linear
      weight = _0.weight
      bias = _0.bias
      x = torch.addmm(bias, input, torch.t(weight), beta=1, alpha=1)
      _1 = torch.tanh(torch.add(torch.neg(x), h, alpha=1))
      return (_1, _1)
```

查看`.code`输出，可以看到`if-else`找不到分支！为什么？跟踪完全按照我们所说的去做：运行代码，记录发生的操作，并构造一个可以做到这一点的ScriptModule。不幸的是，诸如控制流之类的东西被抹去了。

我们如何在TorchScript中忠实地表示此模块？我们提供了一个脚本编译器，它可以直接分析您的Python源代码以将其转换为TorchScript。让我们`MyDecisionGate`使用脚本编译器进行转换：
```python 
    scripted_gate = torch.jit.script(MyDecisionGate())
    
    my_cell = MyCell(scripted_gate)
    traced_cell = torch.jit.script(my_cell)
    print(traced_cell.code)
```

Out:
```python 
    def forward(self,
        x: Tensor,
        h: Tensor) -> Tuple[Tensor, Tensor]:
      _0 = self.linear
      _1 = _0.weight
      _2 = _0.bias
      if torch.eq(torch.dim(x), 2):
        _3 = torch.__isnot__(_2, None)
      else:
        _3 = False
      if _3:
        bias = ops.prim.unchecked_unwrap_optional(_2)
        ret = torch.addmm(bias, x, torch.t(_1), beta=1, alpha=1)
      else:
        output = torch.matmul(x, torch.t(_1))
        if torch.__isnot__(_2, None):
          bias0 = ops.prim.unchecked_unwrap_optional(_2)
          output0 = torch.add_(output, bias0, alpha=1)
        else:
          output0 = output
        ret = output0
      _4 = torch.gt(torch.sum(ret, dtype=None), 0)
      if bool(_4):
        _5 = ret
      else:
        _5 = torch.neg(ret)
      new_h = torch.tanh(torch.add(_5, h, alpha=1))
      return (new_h, new_h)
```
万岁！现在，我们已经忠实地捕获了我们在TorchScript中程序的行为。现在让我们尝试运行该程序：

```python    
    # New inputs
    x, h = torch.rand(3, 4), torch.rand(3, 4)
    traced_cell(x, h)
```

### 混合脚本和跟踪

在某些情况下，我们只需要追踪的的结果而不需要全部脚本(例如，模块具有许多条件分支，这些分支我们并不希望展现在TorchScript中）。在这种情况下，脚本可以与用以下方法跟踪：`torch.jit.script`。他将只会追踪方法内的脚本，不会展示方法外的脚本情况。

第一种情况的一个示例：
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

Out:
```python 
    def forward(self,
        xs: Tensor) -> Tuple[Tensor, Tensor]:
      h = torch.zeros([3, 4], dtype=None, layout=None, device=None, pin_memory=None)
      y = torch.zeros([3, 4], dtype=None, layout=None, device=None, pin_memory=None)
      y0, h0 = y, h
      for i in range(torch.size(xs, 0)):
        _0 = self.cell
        _1 = torch.select(xs, 0, i)
        _2 = _0.linear
        weight = _2.weight
        bias = _2.bias
        _3 = torch.addmm(bias, _1, torch.t(weight), beta=1, alpha=1)
        _4 = torch.gt(torch.sum(_3, dtype=None), 0)
        if bool(_4):
          _5 = _3
        else:
          _5 = torch.neg(_3)
        _6 = torch.tanh(torch.add(_5, h0, alpha=1))
        y0, h0 = _6, _6
      return (y0, h0)
```

还有第二种情况的示例：

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

Out:
```python 
    def forward(self,
        argument_1: Tensor) -> Tensor:
      _0 = self.loop
      h = torch.zeros([3, 4], dtype=None, layout=None, device=None, pin_memory=None)
      h0 = h
      for i in range(torch.size(argument_1, 0)):
        _1 = _0.cell
        _2 = torch.select(argument_1, 0, i)
        _3 = _1.linear
        weight = _3.weight
        bias = _3.bias
        _4 = torch.addmm(bias, _2, torch.t(weight), beta=1, alpha=1)
        _5 = torch.gt(torch.sum(_4, dtype=None), 0)
        if bool(_5):
          _6 = _4
        else:
          _6 = torch.neg(_4)
        h0 = torch.tanh(torch.add(_6, h0, alpha=1))
      return torch.relu(h0)
```

这样，当情况需要它们时，可以使用脚本和跟踪并将它们一起使用。

## 保存和加载模型

我们提供API，以存档格式将TorchScript模块保存到磁盘或从磁盘加载TorchScript模块。这种格式包括代码，参数，属性和调试信息，这意味着归档文件是模型的独立表示形式，可以在完全独立的过程中加载。让我们保存并加载包装好的RNN模块：

```python 
    traced.save('wrapped_rnn.zip')
    
    loaded = torch.jit.load('wrapped_rnn.zip')
    
    print(loaded)
    print(loaded.code)
```

Out:
```python 
    ScriptModule(
      (loop): ScriptModule(
        (cell): ScriptModule(
          (dg): ScriptModule()
          (linear): ScriptModule()
        )
      )
    )
    def forward(self,
        argument_1: Tensor) -> Tensor:
      _0 = self.loop
      h = torch.zeros([3, 4], dtype=None, layout=None, device=None, pin_memory=None)
      h0 = h
      for i in range(torch.size(argument_1, 0)):
        _1 = _0.cell
        _2 = torch.select(argument_1, 0, i)
        _3 = _1.linear
        weight = _3.weight
        bias = _3.bias
        _4 = torch.addmm(bias, _2, torch.t(weight), beta=1, alpha=1)
        _5 = torch.gt(torch.sum(_4, dtype=None), 0)
        if bool(_5):
          _6 = _4
        else:
          _6 = torch.neg(_4)
        h0 = torch.tanh(torch.add(_6, h0, alpha=1))
      return torch.relu(h0)
```

正如你所看到的，序列化保留了模块层次结构和我们一直在研究的代码。例如，也可以将模型加载到C ++中以实现不依赖Python的执行。

### 进一步阅读

我们已经完成了教程！有关更多涉及的演示，请查看NeurIPS演示，以使用TorchScript转换[机器翻译模型](https://colab.research.google.com/drive/1HiICg6jRkBnr5hvK2-VnMi88Vi9pUzEJ)

**脚本的总运行时间：** (0分钟0.252秒）




