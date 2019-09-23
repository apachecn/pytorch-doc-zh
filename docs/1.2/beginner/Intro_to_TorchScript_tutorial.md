# 2.介绍TorchScript

_詹姆斯里德（jamesreed@fb.com），迈克尔琐（suo@fb.com）_ ，REV2

本教程是介绍TorchScript，一个PyTorch模型的中间表示（的`nn.Module`亚类），然后可以在一个高性能的环境，如C ++中运行。

在本教程中，我们将介绍：

  1. 在PyTorch，包括模型制作的基础知识：

  * 模块
  * 定义`向前 `功能
  * 构成模块到模块中的层次结构

  2. 转换PyTorch模块TorchScript具体的方法，我们的高性能运行时部署

  * 跟踪现有模块
  * 使用脚本来直接编译的模块
  * 如何撰写这两种方法
  * 保存和加载TorchScript模块

我们希望你完成本教程后，你继续去通过[在后续教程](https://pytorch.org/tutorials/advanced/cpp_export.html)这将引导您的实际调用从C
TorchScript模型++的例子。

    
    
    import torch # This is all you need to use both PyTorch and TorchScript!
    print(torch.__version__)
    

日期：

    
    
    1.2.0
    

## PyTorch模型制作的基础

让我们先来定义一个简单的`模块 [HTG3。 A `模块 `是在PyTorch组合物中的基本单元。它包含：`

  1. 构造函数，它准备模块，用于调用
  2. 一组参数的``和半`模块 `。这些由构造初始化并且可以由模块调用期间被使用。
  3. A `向前 `功能。这是被调用的模块时运行的代码。

让我们来看看一个小例子：

    
    
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
    

Out:

    
    
    (tensor([[0.7853, 0.8882, 0.7137, 0.3746],
            [0.5265, 0.8508, 0.1487, 0.9144],
            [0.7057, 0.8217, 0.9575, 0.6132]]), tensor([[0.7853, 0.8882, 0.7137, 0.3746],
            [0.5265, 0.8508, 0.1487, 0.9144],
            [0.7057, 0.8217, 0.9575, 0.6132]]))
    

所以我们：

  1. 创建子类`torch.nn.Module`的类。
  2. 定义构造函数。构造函数没有做太多，只是要求`超 `构造。
  3. 限定的`向前 `函数，它有两个输入端和返回两个输出。的实际内容`转发 `功能不是很重要，但是这有点假的[ RNN细胞](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) \- 即的IS-它是在应用功能环。

我们实例化的模块，和由`× `和`Y`，它是随机值只是3x4的矩阵。然后，我们来调用`my_cell（X， h）上的细胞
`。这反过来又要求我们的`转发 `功能。

让我们多一点有趣的做一些事情：

    
    
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
    

Out:

    
    
    MyCell(
      (linear): Linear(in_features=4, out_features=4, bias=True)
    )
    (tensor([[0.7619, 0.7761, 0.7476, 0.0897],
            [0.6886, 0.4990, 0.4495, 0.2021],
            [0.5849, 0.5874, 0.9256, 0.0460]], grad_fn=<TanhBackward>), tensor([[0.7619, 0.7761, 0.7476, 0.0897],
            [0.6886, 0.4990, 0.4495, 0.2021],
            [0.5849, 0.5874, 0.9256, 0.0460]], grad_fn=<TanhBackward>))
    

我们已经重新定义了我们的模块`了myCell`，但这次我们增加了`self.linear`属性，我们调用`self.linear
`在向前的功能。

究竟发生在这里？ `torch.nn.Linear`是`模块 `从PyTorch标准库。就像`了myCell
`，可以使用呼叫语法调用。我们正在建设的`模块 `个层次。

`打印 `在`模块 `将给出的`模块 `的视觉表示子类层次结构。在我们的例子中，我们可以看到我们的`线性 `子类及其参数。

通过这种方式组成`模块 `S，我们可以succintly和可读性很强笔者型号可重用的组件。

您可能已经注意到`在输出grad_fn  [HTG3。这是自动分化PyTorch的方法，称为[ autograd
](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)的细节。总之，该系统使我们能够通过潜在的复杂程序计算的衍生物。该设计允许的灵活性，在模型制作的巨量。`

现在，让我们来看看说的灵活性：

    
    
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
    

Out:

    
    
    MyCell(
      (dg): MyDecisionGate()
      (linear): Linear(in_features=4, out_features=4, bias=True)
    )
    (tensor([[ 0.9077,  0.5939,  0.6809,  0.0994],
            [ 0.7583,  0.7180,  0.0790,  0.6733],
            [ 0.9102, -0.0368,  0.8246, -0.3256]], grad_fn=<TanhBackward>), tensor([[ 0.9077,  0.5939,  0.6809,  0.0994],
            [ 0.7583,  0.7180,  0.0790,  0.6733],
            [ 0.9102, -0.0368,  0.8246, -0.3256]], grad_fn=<TanhBackward>))
    

我们再一次重新定义我们的了myCell类，但在这里我们定义`MyDecisionGate  [HTG3。这模块利用 **控制流** 。控制流由东西样环和`
如果 `-statements。`

许多框架搭给出一个完整的程序表示计算的符号衍生品的方法。然而，在PyTorch，我们使用渐变带。因为它们发生时，我们记录的操作，并且在计算衍生向后重放。通过这种方式，框架没有明确定义的衍生物在语言的所有构造。

![How autograd
works](https://github.com/pytorch/pytorch/raw/master/docs/source/_static/img/dynamic_graph.gif)

如何autograd作品

## TorchScript的基础

现在，让我们把我们运行的例子，看看我们如何可以申请TorchScript。

总之，TorchScript提供的工具捕捉到你的模型的定义，即使在PyTorch的灵活性和动态性的光。让我们先通过检查我们称之为 **追踪[HTG1。**

### 追踪`模块 `

    
    
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
    

Out:

    
    
    TracedModule[MyCell](
      (linear): TracedModule[Linear]()
    )
    

我们复卷一点，并采取了我们的`了myCell`类的第二个版本。和以前一样，我们实例化，但是，这个时候，我们称为`torch.jit.trace
`，在`模块 [HTG11通过]和in _例如输入_ 网络可能会看到通过。`

正是有这个做了什么？它调用`模块 `时，记录所发生当`模块已运行 `的操作，和产生的`[实例HTG9] torch.jit.ScriptModule
`（其中`TracedModule`是一个实例）

TorchScript记录其在中间表示（或IR）的定义，通常称为深学习作为 _图表_ 。我们可以检查与`.graph`属性图：

    
    
    print(traced_cell.graph)
    

Out:

    
    
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
    

然而，这是一个非常低的电平表示，大部分包含在图表中的信息不是为最终用户是有用的。相反，我们可以使用`.CODE
`属性给代码的一个Python语法的解释：

    
    
    print(traced_cell.code)
    

Out:

    
    
    def forward(self,
        input: Tensor,
        h: Tensor) -> Tuple[Tensor, Tensor]:
      _0 = self.linear
      weight = _0.weight
      bias = _0.bias
      _1 = torch.addmm(bias, input, torch.t(weight), beta=1, alpha=1)
      _2 = torch.tanh(torch.add(_1, h, alpha=1))
      return (_2, _2)
    

所以 **为什么** 我们做了这一切？有几个原因：

  1. TorchScript代码可以在其自己的解释，这基本上是受限制的Python解释被调用。这个解释并没有获得全局解释器锁，和这么多的请求可以在同一实例同时处理。
  2. 这种格式可以让我们整个模型保存到磁盘，并将其加载到另一个环境，如写在Python以外的语言的服务器
  3. TorchScript给了我们一个表示中，我们可以对代码做编译器优化，以提供更高效的执行
  4. TorchScript允许我们与所需要的程序比个体经营者的更广阔的视野许多后端/设备运行时的接口。

我们可以看到，调用`traced_cell`产生相同的结果Python模块：

    
    
    print(my_cell(x, h))
    print(traced_cell(x, h))
    

Out:

    
    
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
    

## 使用脚本来转换模块

还有我们用我们模块的两个版本的原因，而不是一个与控制流载货子模块。现在，让我们检查的是：

    
    
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
    

Out:

    
    
    def forward(self,
        input: Tensor,
        h: Tensor) -> Tuple[Tensor, Tensor]:
      _0 = self.linear
      weight = _0.weight
      bias = _0.bias
      x = torch.addmm(bias, input, torch.t(weight), beta=1, alpha=1)
      _1 = torch.tanh(torch.add(torch.neg(x), h, alpha=1))
      return (_1, _1)
    

综观`.CODE`输出，我们可以看到，`的if-else `分支无处可寻！为什么？跟踪不正是我们称将：运行代码，记录操作 _这种情况发生_
和构建ScriptModule这正是这么做的。不幸的是，像控制流程被删除。

我们如何能够忠实代表TorchScript这个模块？我们提供了一个 **编译脚本**
，它确实你的Python源代码分析直接将其转化为TorchScript。让我们使用脚本编译器转换`MyDecisionGate`：

    
    
    scripted_gate = torch.jit.script(MyDecisionGate())
    
    my_cell = MyCell(scripted_gate)
    traced_cell = torch.jit.script(my_cell)
    print(traced_cell.code)
    

Out:

    
    
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
    

万岁！现在，我们已经捕获忠实我们TorchScript程序的行为。现在，让我们试着运行该程序：

    
    
    # New inputs
    x, h = torch.rand(3, 4), torch.rand(3, 4)
    traced_cell(x, h)
    

### 混合脚本和跟踪

有些情况下需要使用跟踪，而不是脚本（例如一个模块是基于我们想不会出现在TorchScript不变的Python值做出了许多架构决策）。在这种情况下，脚本可以与由跟踪：`
torch.jit.script`将内联一个跟踪模块的代码，和跟踪将内联代码脚本模块。

第一种情况的一个示例：

    
    
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
    

Out:

    
    
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
    

和第二壳体的一个示例：

    
    
    class WrapRNN(torch.nn.Module):
      def __init__(self):
        super(WrapRNN, self).__init__()
        self.loop = torch.jit.script(MyRNNLoop())
    
      def forward(self, xs):
        y, h = self.loop(xs)
        return torch.relu(y)
    
    traced = torch.jit.trace(WrapRNN(), (torch.rand(10, 3, 4)))
    print(traced.code)
    

Out:

    
    
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
    

这样一来，脚本和跟踪可当形势需要每个人共同使用的使用。

## 保存和加载模型

我们提供的API来保存和从磁盘归档格式加载TorchScript模块/。此格式包括代码，参数，属性，和调试信息，这意味着该归档是可以在一个完全独立的过程来加载该模型的一个独立的表示。让我们保存和载入我们的包裹RNN模块：

    
    
    traced.save('wrapped_rnn.zip')
    
    loaded = torch.jit.load('wrapped_rnn.zip')
    
    print(loaded)
    print(loaded.code)
    

Out:

    
    
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
    

正如你所看到的，系列化保留了模块的层次结构，我们已经在整个检查代码。该模型也被加载，例如，[成C ++
](https://pytorch.org/tutorials/advanced/cpp_export.html)免费蟒-执行。

### 进一步阅读

我们已经完成了教程！对于更为复杂的论证，检查出NeurIPS演示转换使用TorchScript机器翻译模型：HTG0]
https://colab.research.google.com/drive/1HiICg6jRkBnr5hvK2-VnMi88Vi9pUzEJ

**脚本的总运行时间：** （0分钟0.252秒）

[`Download Python source code:
Intro_to_TorchScript_tutorial.py`](../_downloads/bf4ee4ef1ffde8b469d9ed4001a28ee8/Intro_to_TorchScript_tutorial.py)

[`Download Jupyter notebook:
Intro_to_TorchScript_tutorial.ipynb`](../_downloads/0fd9e9bc92ac80a422914e974021c007/Intro_to_TorchScript_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](../advanced/cpp_export.html "3. Loading a TorchScript Model in
C++") [![](../_static/images/chevron-right-orange.svg)
Previous](../intermediate/flask_rest_api_tutorial.html "1. Deploying PyTorch
in Python via a REST API with Flask")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。



  * [HTG0 2.介绍TorchScript 
    * PyTorch模型制作的基础
    * TorchScript的基础
      * 跟踪`模块 `
    * 使用脚本来转换器模块
      * 混合脚本和跟踪
    * 保存和载入模型
      * 进一步阅读

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)







 
[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)



