跟着例子学习 PyTorch
******************************
**Author**: `Justin Johnson <https://github.com/jcjohnson/pytorch-examples>`_

这个教程通过一些单独的示例介绍了 `PyTorch <https://github.com/pytorch/pytorch>`__ 的基本概念。

PyTorch 的核心部分提供了两个主要功能：

- 一个类似于 numpy 的n维张量，但可以在 GPU 上运行
- 为建立和训练神经网络自动微分

我们将使用完全连接的 ReLU 网络作为我们的运行示例。该网络将有一个单一的隐藏层，
并将使用梯度下降训练去拟合随机数据通过最小化网络输出和真实输出之间的欧几里得距离。

.. Note::
	You can browse the individual examples at the
	:ref:`end of this page <examples-download>`.

.. contents:: Table of Contents
  :local:

Tensors
=======

Warm-up: numpy
--------------

在介绍 PyTorch 之前，我们将会先使用 numpy 实现网络。

Numpy 提供了一个n维的数组对象，并提供了许多操纵这些数组的函数。 Numpy 是科学计算的通用框架;
它不知道任何关于计算图的内容，或者深度学习，或梯度。然而我们可以很容易地使用 numpy 适应随机数据的
双层网络通过手工使用 numpy 操作实现网络正反向传输:

.. includenodoc:: /beginner/examples_tensor/two_layer_net_numpy.py


PyTorch: Tensors
----------------

Numpy 是一个伟大的框架，但它不能利用GPU加速它数值计算。 对于现代的深度神经网络，
GPU往往是提供 `50倍或更大的加速 <https://github.com/jcjohnson/cnn-benchmarks>`__,
所以不幸的是，numpy 不足以满足现在深度学习的需求。

这里我们介绍一下最基本的 PyTorch 概念： **Tensor** 。PyTorch 张量在概念上与 numpy 数组相同：
张量是一个n维数组，PyTorch 提供了很多能在这些张量上运行的功能。像 numpy 数组一样，PyTorch 张量
不了解深度学习或计算图表或梯度；它们是科学计算的通用工具。

然而不像 numpy，PyTorch 张量可以利用GPU加速他们的数字计算。要在 GPU 上运行
PyTorch 张量，只需将其转换为新的数据类型。

在这里，我们使用 PyTorch 张量来适应随机数据的双层网络。就像上面的 numpy 例子一样，
我们需要手动实现网络的正反向传输：

.. includenodoc:: /beginner/examples_tensor/two_layer_net_tensor.py


Autograd
========

PyTorch: Variables and autograd
-------------------------------

在上面的例子中，我们不得不手动执行神经网络的正反向传输。手动执行反向传输对于一个
小型的双层网络来说是没什么大问题的，但是可以很快就会变得对大型复杂网络很棘手。

谢天谢地，我们可以使用 `自动微分<https://en.wikipedia.org/wiki/Automatic_differentiation>`__
自动化神经网络中的后向通道计算。该 PyTorch 中的 **autograd** 包提供了这个功能。使用 autograd 时，
网络的正向传递将定义一个 **计算图** ; 图中的节点将是张量，边缘将是从输入张量产生输出张量的函数。
通过此图表的反向传播可以让您轻松计算梯度。

这听起来很复杂，在实践中使用起来相当简单。我们将 PyTorch 的张量包裹在 **变量** 对象；
一个变量代表一个计算图中的节点。如果 ``x`` 是一个变量，则 ``x.data`` 是一个张量，
而 ``x.grad`` 是另外一个变量，其中包含 ``x`` 的梯度相对于一些标量值。

PyTorch 变量与 PyTorch 张量具有相同的 API：（几乎）任何您可以在张量上执行的
操作也适用于变量；该区别在于使用变量定义了一个计算图，允许您自动计算梯度。

这里我们使用 PyTorch 变量和自动微分来实现我们的双层网络；现在我们不再需要手动
执行网络的反向传播：

.. includenodoc:: /beginner/examples_autograd/two_layer_net_autograd.py

PyTorch: Defining new autograd functions
----------------------------------------

在这层覆盖下，每个原始的 autograd 操作符实际上是两个函数在张量上运行。 **前向**
函数从输入张量计算输出张量。 **后向** 函数收到输出张量相对于某个标量值的梯度，以
及计算输入张量相对于相同标量值的梯度。

在 PyTorch 中，我们可以通过定义一个 ``torch.autograd.Function`` 的子类和
执行 ``前向`` 和 ``后向`` 功能来轻松定义自己的 autograd 操作符。然后我们可以
使用我们新的 autograd 操作符构造一个实例并将其作为一个函数调用，传递包含输入数据的变量。

在这个例子中我们定义了我们自己定制的 autograd 函数执行 ReLU 非线性，并用它来实现我们的双层网络：

.. includenodoc:: /beginner/examples_autograd/two_layer_net_custom_function.py

TensorFlow: Static Graphs
-------------------------

使用自动微分计算梯度。两者之间最大的不同在于 TensorFlow 的计算图是 **静态的**
和 PyTorch 使用的 **动态** 计算图。

在 TensorFlow 中，我们只定义一次计算图然后重复执行同一个图表，可能会给不同的输入数据到图表。
在 PyTorch 中，每个正向传递定义一个新的计算图。

静态图很好，因为您可以预先优化图；例如一个框架可能会为了效率决定融合一些图形操作，或想
出一个跨越许多 GPU 或许多机器的分配图的策略。如果您正在重复使用相同的图表，那么这个潜在的
昂贵的前期优化可以因同样的图形重复运行而得到摊销。

一方面来说，静态和动态图的控制流是不同的。对于有些模型我们可能希望对每个数据点执行不同
的计算；例如循环网络可能会被展开为对每个数据点的不同的数目的时间步数；这个展开可以用循
环来实现。循环结构的静态图需要成为图的一部分；为此 TensorFlow 提供 ``tf.scan`` 操作符
用于将循环嵌入到图形中。动态图形的情况比较简单：因为我们正在为之建立图形每个例子中，
我们可以使用正常的命令式流程控制为每个不同的输入执行计算。

为了与上面的 PyTorch autograd 例子进行对比，我们在这里使用 TensorFlow 适合简单的两层网络：

.. includenodoc:: /beginner/examples_autograd/tf_two_layer_net.py

`nn` module
===========

PyTorch: nn
-----------

计算图和 autograd 是一个非常强大的定义复杂的运算符并自动地导出的范式；然而对于
大型的神经网络，原始的 autograd 可能有点太低级。

当我们建立神经网络时，我们经常想到将计算安排 **层** 中，其中一些具有 **可学习参数**
这将在学习期间得到优化。

在TensorFlow中，像 `Keras <https://github.com/fchollet/keras>`__,
`TensorFlow-Slim <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim>`__,
和 `TFLearn <http://tflearn.org/>`__ 通过构建对神经网络有用的原始计算图提供更高层次的抽象。

在 PyTorch 中，``nn`` 包起了同样的作用。 ``nn`` 包定义了一组 **模块** ，大致相当于神经网络层。
模块接收输入变量并进行计算输出变量，但也可以保持内部状态，如变量包含可学习的参数。 ``nn`` 包
也定义了一系列在训练神经网络时常用的有用的损失函数。

在这个例子中，我们使用 ``nn`` 包来实现我们的双层网络：

.. includenodoc:: /beginner/examples_nn/two_layer_net_nn.py

PyTorch: optim
--------------

到目前为止，我们已经通过手动更新了模型的权重变异的 ``.data`` 成员保持可学习的变量参数。
这对于简单的优化算法像随机梯度下降来说不是一个巨大的负担，但实际上我们经常使用更巧妙的
优化器训练神经网络，如 AdaGrad，RMSProp，Adam 等。

PyTorch 中的 ``optim`` 包简要包含了优化思想的算法并提供常用优化的实现算法。

在这个例子中，我们将像之前一样使用 ``nn`` 包来定义我们的模型，但我们将使用由 ``optim`` 包提供的Adam算法优化模型：

.. includenodoc:: /beginner/examples_nn/two_layer_net_optim.py

PyTorch: Custom nn Modules
--------------------------

有时你会想要指定比现有模块的顺序更复杂的模型；对于这些情况，你可以
通过继承 ``nn.Module`` 并定义一个 ``forward`` 来定义你自己的模块，
来实现模块接收输入变量并使用其他模块和变量上的 autograd 操作生成输出变量。

在这个例子中，我们实现了我们的双层网络作为一个自定义的模块子类：

.. includenodoc:: /beginner/examples_nn/two_layer_net_module.py

PyTorch: Control Flow + Weight Sharing
--------------------------------------

作为一个动态图形和权重共享的例子，我们实现一个奇葩的模型：每个正向传递的完全连接
的 ReLU 网络选择1到4之间的一个随机数，并使用多隐藏层重复使用相同的权重来计算最内层
隐藏的图层。

对于这个模型，我们可以使用普通的 Python 流量控制来实现循环，而且我们可以在定义前向传
输时通过简单地重复使用相同的模块实现最内层的权重共享。

我们可以很容易地将这个模型作为 Module 子类来实现：

.. includenodoc:: /beginner/examples_nn/dynamic_net.py


.. _examples-download:

Examples
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
