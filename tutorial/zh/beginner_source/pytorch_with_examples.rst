Learning PyTorch with Examples
******************************
**Author**: `Justin Johnson <https://github.com/jcjohnson/pytorch-examples>`_

This tutorial introduces the fundamental concepts of
`PyTorch <https://github.com/pytorch/pytorch>`__ through self-contained
examples.
这个教程通过一些单独的示例介绍了一些 `PyTorch <https://github.com/pytorch/pytorch>`__ 的基本概念。

At its core, PyTorch provides two main features:
PyTorch的核心部分提供了两个主要功能：

- An n-dimensional Tensor, similar to numpy but can run on GPUs
- Automatic differentiation for building and training neural networks
- 一个类似于numpy的n维张量，但可以在GPU上运行
- 为建立和训练神经网络自动分化

We will use a fully-connected ReLU network as our running example. The
network will have a single hidden layer, and will be trained with
gradient descent to fit random data by minimizing the Euclidean distance
between the network output and the true output.
我们将使用完全连接的ReLU网络作为我们的运行示例。该网络将有一个单一的隐藏层，
并将使用梯度下降训练去拟合随机数据通过最小化网络输出和真实输出之间的欧几里得距离。

.. Note::
.. 注意::
	You can browse the individual examples at the
	:ref:`end of this page <examples-download>`.
	你可以在这个 :ref:`页面的末尾 <examples-download>` 浏览单独的例子。

.. contents:: Table of Contents
	:local:
.. 内容:: 页面内容
  :本地:

Tensors 张量
=======

Warm-up: numpy 热身: numpy
--------------

Before introducing PyTorch, we will first implement the network using
numpy.
在介绍PyTorch之前，我们将会先实现使用numpy的网络。

Numpy provides an n-dimensional array object, and many functions for
manipulating these arrays. Numpy is a generic framework for scientific
computing; it does not know anything about computation graphs, or deep
learning, or gradients. However we can easily use numpy to fit a
two-layer network to random data by manually implementing the forward
and backward passes through the network using numpy operations:
Numpy提供了一个n维的数组对象，并提供了许多操纵这些数组的函数。 Numpy是科学计算的通用框架;
它不知道任何关于计算图的内容，或者深度学习，或梯度。然而我们可以很容易地使用numpy适应随机数据的
双层网络通过手工使用numpy操作实现网络正反向传输:

.. includenodoc:: /beginner/examples_tensor/two_layer_net_numpy.py


PyTorch: Tensors PyTorch: 张量
----------------

Numpy is a great framework, but it cannot utilize GPUs to accelerate its
numerical computations. For modern deep neural networks, GPUs often
provide speedups of `50x or
greater <https://github.com/jcjohnson/cnn-benchmarks>`__, so
unfortunately numpy won't be enough for modern deep learning.
Numpy是一个伟大的框架，但它不能利用GPU加速它数值计算。 对于现代的深度神经网络，
GPU往往是提供 `50倍或更大的加速 <https://github.com/jcjohnson/cnn-benchmarks>`__,
所以不幸的是，numpy不足以满足现在深度学习的需求。

Here we introduce the most fundamental PyTorch concept: the **Tensor**.
A PyTorch Tensor is conceptually identical to a numpy array: a Tensor is
an n-dimensional array, and PyTorch provides many functions for
operating on these Tensors. Like numpy arrays, PyTorch Tensors do not
know anything about deep learning or computational graphs or gradients;
they are a generic tool for scientific computing.
这里我们介绍一下最基本的PyTorch概念：** Tensor **。PyTorch张量在概念上与numpy数组相同：
张量是一个n维数组，PyTorch提供了很多能在这些张量上运行的功能。像numpy数组一样，PyTorch
张量不了解深度学习或计算图表或梯度;它们是科学计算的通用工具。

However unlike numpy, PyTorch Tensors can utilize GPUs to accelerate
their numeric computations. To run a PyTorch Tensor on GPU, you simply
need to cast it to a new datatype.
然而不像numpy，PyTorch张量可以利用GPU加速他们的数字计算。要在GPU上运行
PyTorch张量，只需将其转换为新的数据类型。

Here we use PyTorch Tensors to fit a two-layer network to random data.
Like the numpy example above we need to manually implement the forward
and backward passes through the network:
在这里，我们使用PyTorch张量来适应随机数据的双层网络。就像上面的numpy例子一样，
我们需要手动实现网络的正反向传输：

.. includenodoc:: /beginner/examples_tensor/two_layer_net_tensor.py


Autograd自动微分
========

PyTorch: Variables and autograd PyTorch: 变量和自动微分
-------------------------------

In the above examples, we had to manually implement both the forward and
backward passes of our neural network. Manually implementing the
backward pass is not a big deal for a small two-layer network, but can
quickly get very hairy for large complex networks.
在上面的例子中，我们不得不手动执行神经网络的正反向传输。手动执行反向传输对于一个
小型的双层网络来说是没什么大问题的，但是可以很快就会变得对大型复杂网络很棘手。

Thankfully, we can use `automatic
differentiation <https://en.wikipedia.org/wiki/Automatic_differentiation>`__
to automate the computation of backward passes in neural networks. The
**autograd** package in PyTorch provides exactly this functionality.
When using autograd, the forward pass of your network will define a
**computational graph**; nodes in the graph will be Tensors, and edges
will be functions that produce output Tensors from input Tensors.
Backpropagating through this graph then allows you to easily compute
gradients.
谢天谢地，我们可以使用`自动微分<https://en.wikipedia.org/wiki/Automatic_differentiation>`__
自动化神经网络中的后向通道计算。该PyTorch中的** autograd **包提供了这个功能。使用autograd时，
网络的正向传递将定义一个**计算图**; 图中的节点将是张量，边缘将是从输入张量产生输出张量的函数。
通过此图表的反向传播可以让您轻松计算梯度。

This sounds complicated, it's pretty simple to use in practice. We wrap
our PyTorch Tensors in **Variable** objects; a Variable represents a
node in a computational graph. If ``x`` is a Variable then ``x.data`` is
a Tensor, and ``x.grad`` is another Variable holding the gradient of
``x`` with respect to some scalar value.
这听起来很复杂，在实践中使用起来相当简单。我们将PyTorch的张量包裹在**变量**对象；
一个变量代表一个计算图中的节点。如果``x``是一个变量，则``x.data``是一个张量，
而`x.grad``是另外一个变量，其中包含``x``的梯度相对于一些标量值。

PyTorch Variables have the same API as PyTorch Tensors: (almost) any
operation that you can perform on a Tensor also works on Variables; the
difference is that using Variables defines a computational graph,
allowing you to automatically compute gradients.
PyTorch变量与PyTorch张量具有相同的API：（几乎）任何您可以在张量上执行的
操作也适用于变量;该区别在于使用变量定义了一个计算图，允许您自动计算梯度。

Here we use PyTorch Variables and autograd to implement our two-layer
network; now we no longer need to manually implement the backward pass
through the network:
这里我们使用PyTorch变量和自动微分来实现我们的双层网络; 现在我们不再需要手动
执行网络的反向传播：

.. includenodoc:: /beginner/examples_autograd/two_layer_net_autograd.py

PyTorch: Defining new autograd functions PyTorch: 定义新的自动微分方程
----------------------------------------

Under the hood, each primitive autograd operator is really two functions
that operate on Tensors. The **forward** function computes output
Tensors from input Tensors. The **backward** function receives the
gradient of the output Tensors with respect to some scalar value, and
computes the gradient of the input Tensors with respect to that same
scalar value.
在这层覆盖下，每个原始的autograd操作符实际上是两个函数在张量上运行。 **前向**
函数从输入张量计算输出张量。 **后向**函数收到输出张量相对于某个标量值的梯度，以
及计算输入张量相对于相同标量值的梯度。

In PyTorch we can easily define our own autograd operator by defining a
subclass of ``torch.autograd.Function`` and implementing the ``forward``
and ``backward`` functions. We can then use our new autograd operator by
constructing an instance and calling it like a function, passing
Variables containing input data.
在PyTorch中，我们可以通过定义一个自定义操作符来轻松定义自己的autograd操作符“torch.autograd.Function”的子类和执行“forward”和“后退”功能。 然后我们可以使用我们新的autograd操作符构造一个实例并将其作为一个函数调用，传递包含输入数据的变量

In this example we define our own custom autograd function for
performing the ReLU nonlinearity, and use it to implement our two-layer
network:
在这个例子中我们定义了我们自己定制的autograd函数执行ReLU非线性，并用它来实现我们的双层网络：

.. includenodoc:: /beginner/examples_autograd/two_layer_net_custom_function.py

TensorFlow: Static Graphs TensorFlow: 静态图
-------------------------

PyTorch autograd looks a lot like TensorFlow: in both frameworks we
define a computational graph, and use automatic differentiation to
compute gradients. The biggest difference between the two is that
TensorFlow's computational graphs are **static** and PyTorch uses
**dynamic** computational graphs.
PyTorch的autograd看起来很像TensorFlow：在这两个框架中定义一个计算图，并
使用自动微分计算梯度。两者之间最大的不同在于TensorFlow的计算图是 **静态的**
和PyTorch使用的 **动态** 计算图。

In TensorFlow, we define the computational graph once and then execute
the same graph over and over again, possibly feeding different input
data to the graph. In PyTorch, each forward pass defines a new
computational graph.
在TensorFlow中，我们只定义一次计算图然后重复执行同一个图表，可能会给不同的输入数据到图表。
在PyTorch中，每个正向传递定义一个新的计算图。

Static graphs are nice because you can optimize the graph up front; for
example a framework might decide to fuse some graph operations for
efficiency, or to come up with a strategy for distributing the graph
across many GPUs or many machines. If you are reusing the same graph
over and over, then this potentially costly up-front optimization can be
amortized as the same graph is rerun over and over.
静态图很好，因为您可以预先优化图；例如一个框架可能会为了效率决定融合一些图形操作，或想
出一个跨越许多GPU或许多机器的分配图的策略。如果您正在重复使用相同的图表，那么这个潜在的
昂贵的前期优化可以因同样的图形重复运行而得到摊销。

One aspect where static and dynamic graphs differ is control flow. For
some models we may wish to perform different computation for each data
point; for example a recurrent network might be unrolled for different
numbers of time steps for each data point; this unrolling can be
implemented as a loop. With a static graph the loop construct needs to
be a part of the graph; for this reason TensorFlow provides operators
such as ``tf.scan`` for embedding loops into the graph. With dynamic
graphs the situation is simpler: since we build graphs on-the-fly for
each example, we can use normal imperative flow control to perform
computation that differs for each input.
一方面来说，静态和动态图的控制流是不同的。对于有些模型我们可能希望对每个数据点执行不同
的计算；例如循环网络可能会被展开为对每个数据点的不同的数目的时间步数；这个展开可以用循
环来实现。循环结构的静态图需要成为图的一部分；为此TensorFlow提供 ``tf.scan`` 操作符
用于将循环嵌入到图形中。动态图形的情况比较简单：因为我们正在为之建立图形每个例子中，
我们可以使用正常的命令式流程控制为每个不同的输入执行计算。

To contrast with the PyTorch autograd example above, here we use
TensorFlow to fit a simple two-layer net:
为了与上面的PyTorch autograd例子进行对比，我们在这里使用TensorFlow适合简单的两层网络：

.. includenodoc:: /beginner/examples_autograd/tf_two_layer_net.py

`nn` module
===========

PyTorch: nn
-----------

Computational graphs and autograd are a very powerful paradigm for
defining complex operators and automatically taking derivatives; however
for large neural networks raw autograd can be a bit too low-level.
计算图和autograd是一个非常强大的定义复杂的运算符并自动地导出的范式；然而对于
大型的神经网络，原始的autograd可能有点太低级。

When building neural networks we frequently think of arranging the
computation into **layers**, some of which have **learnable parameters**
which will be optimized during learning.
当我们建立神经网络时，我们经常想到将计算安排 **层** 中，其中一些具有 **可学习参数**
这将在学习期间得到优化。

In TensorFlow, packages like
`Keras <https://github.com/fchollet/keras>`__,
`TensorFlow-Slim <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim>`__,
and `TFLearn <http://tflearn.org/>`__ provide higher-level abstractions
over raw computational graphs that are useful for building neural
networks.
在TensorFlow中，像 `Keras <https://github.com/fchollet/keras>`__,
`TensorFlow-Slim <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim>`__,
和 `TFLearn <http://tflearn.org/>`__ 通过构建对神经网络有用的原始计算图提供更高层次的抽象。

In PyTorch, the ``nn`` package serves this same purpose. The ``nn``
package defines a set of **Modules**, which are roughly equivalent to
neural network layers. A Module receives input Variables and computes
output Variables, but may also hold internal state such as Variables
containing learnable parameters. The ``nn`` package also defines a set
of useful loss functions that are commonly used when training neural
networks.
在PyTorch中，``nn`` 包起了同样的作用。 ``nn`` 包定义了一组 **模块** ，大致相当于神经网络层。
模块接收输入变量并进行计算输出变量，但也可以保持内部状态，如变量包含可学习的参数。 ``nn`` 包
也定义了一系列在训练神经网络时常用的有用的损失函数。

In this example we use the ``nn`` package to implement our two-layer
network:
在这个例子中，我们使用 ``nn`` 包来实现我们的双层网络：

.. includenodoc:: /beginner/examples_nn/two_layer_net_nn.py

PyTorch: optim
--------------

Up to this point we have updated the weights of our models by manually
mutating the ``.data`` member for Variables holding learnable
parameters. This is not a huge burden for simple optimization algorithms
like stochastic gradient descent, but in practice we often train neural
networks using more sophisticated optimizers like AdaGrad, RMSProp,
Adam, etc.
到目前为止，我们已经通过手动更新了模型的权重变异的``.data``成员保持可学习的变量参数。
这对于简单的优化算法像随机梯度下降来说不是一个巨大的负担，但实际上我们经常使用更巧妙的
优化器训练神经网络，如AdaGrad，RMSProp，Adam等。

The ``optim`` package in PyTorch abstracts the idea of an optimization
algorithm and provides implementations of commonly used optimization
algorithms.
PyTorch中的 ``optim`` 包简要包含了优化思想的算法并提供常用优化的实现算法。

In this example we will use the ``nn`` package to define our model as
before, but we will optimize the model using the Adam algorithm provided
by the ``optim`` package:
在这个例子中，我们将像之前一样使用``nn``包来定义我们的模型，但我们将使用由
 ``optim`` 包提供的Adam算法优化模型：

.. includenodoc:: /beginner/examples_nn/two_layer_net_optim.py

PyTorch: Custom nn Modules
--------------------------

Sometimes you will want to specify models that are more complex than a
sequence of existing Modules; for these cases you can define your own
Modules by subclassing ``nn.Module`` and defining a ``forward`` which
receives input Variables and produces output Variables using other
modules or other autograd operations on Variables.
有时你会想要指定比现有模块的顺序更复杂的模型；对于这些情况，你可以
通过继承 ``nn.Module`` 并定义一个 ``forward`` 来定义你自己的模块，
来实现模块接收输入变量并使用其他模块和变量上的autograd操作生成输出变量。

In this example we implement our two-layer network as a custom Module
subclass:
在这个例子中，我们实现了我们的双层网络作为一个自定义的模块子类：

.. includenodoc:: /beginner/examples_nn/two_layer_net_module.py

PyTorch: Control Flow + Weight Sharing
--------------------------------------

As an example of dynamic graphs and weight sharing, we implement a very
strange model: a fully-connected ReLU network that on each forward pass
chooses a random number between 1 and 4 and uses that many hidden
layers, reusing the same weights multiple times to compute the innermost
hidden layers.
作为一个动态图形和权重共享的例子，我们实现一个奇葩的模型：每个正向传递的完全连接
的ReLU网络选择1到4之间的一个随机数，并使用多隐藏层重复使用相同的权重来计算最内层
隐藏的图层。

For this model we can use normal Python flow control to implement the loop,
and we can implement weight sharing among the innermost layers by simply
reusing the same Module multiple times when defining the forward pass.
对于这个模型，我们可以使用普通的Python流量控制来实现循环，而且我们可以在定义前向传
输时通过简单地重复使用相同的模块实现最内层的权重共享。

We can easily implement this model as a Module subclass:
我们可以很容易地将这个模型作为Module子类来实现：

.. includenodoc:: /beginner/examples_nn/dynamic_net.py


.. _examples-download:

Examples 例子
========

You can browse the above examples here.

Tensors
-------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_tensor/two_layer_net_numpy
   /beginner/examples_tensor/two_layer_net_tensor

.. galleryitem:: /beginner/examples_tensor/two_layer_net_numpy.py

.. galleryitem:: /beginner/examples_tensor/two_layer_net_tensor.py

.. raw:: html

    <div style='clear:both'></div>

Autograd
--------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_autograd/two_layer_net_autograd
   /beginner/examples_autograd/two_layer_net_custom_function
   /beginner/examples_autograd/tf_two_layer_net


.. galleryitem:: /beginner/examples_autograd/two_layer_net_autograd.py

.. galleryitem:: /beginner/examples_autograd/two_layer_net_custom_function.py

.. galleryitem:: /beginner/examples_autograd/tf_two_layer_net.py

.. raw:: html

    <div style='clear:both'></div>

`nn` module
-----------

.. toctree::
   :maxdepth: 2
   :hidden:

   /beginner/examples_nn/two_layer_net_nn
   /beginner/examples_nn/two_layer_net_optim
   /beginner/examples_nn/two_layer_net_module
   /beginner/examples_nn/dynamic_net


.. galleryitem:: /beginner/examples_nn/two_layer_net_nn.py

.. galleryitem:: /beginner/examples_nn/two_layer_net_optim.py

.. galleryitem:: /beginner/examples_nn/two_layer_net_module.py

.. galleryitem:: /beginner/examples_nn/dynamic_net.py

.. raw:: html

    <div style='clear:both'></div>
