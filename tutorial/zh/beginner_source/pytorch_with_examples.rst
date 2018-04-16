跟着例子学习 PyTorch
******************************
**Author**: `Justin Johnson <https://github.com/jcjohnson/pytorch-examples>`_

这个教程通过一些单独的示例介绍了 `PyTorch <https://github.com/pytorch/pytorch>`__ 的基本概念. 

PyTorch 的核心部分提供了两个主要功能: 

- 一个类似于 numpy 的n维张量, 但可以在 GPU 上运行
- 为建立和训练神经网络自动微分

我们将使用全连接的 ReLU 网络作为我们的运行示例. 该网络将有一个隐藏层, 
并将使用梯度下降训练去最小化随机数字的预测输出和真实输出之间的欧式距离. 

.. Note::
	你可以下载这些单独的例子在页面的底端
	:ref:`<examples-download>`.

.. contents:: 本章内容目录
	:local:

Tensors
=======

Warm-up: numpy
--------------

在介绍 PyTorch 之前, 我们先使用 numpy 实现网络. 

Numpy 提供了一个n维的数组对象, 并提供了许多操纵这个数组对象的函数.  Numpy 是科学计算的通用框架;
Numpy 数组没有计算图, 也没有深度学习, 也没有梯度下降等方法实现的接口. 但是我们仍然可以很容易地使用 numpy 生成随机数据
并将产生的数据传入双层的神经网络, 并使用 numpy 来实现这个网络的正向传播和反向传播:

.. includenodoc:: /beginner/examples_tensor/two_layer_net_numpy.py


PyTorch: Tensors
----------------

Numpy 是一个伟大的框架, 但它不能利用 GPU 加速它数值计算.  对于现代的深度神经网络, 
GPU 往往是提供 `50倍或更大的加速 <https://github.com/jcjohnson/cnn-benchmarks>`__,
所以不幸的是, numpy 不足以满足现在深度学习的需求. 

这里我们介绍一下最基本的 PyTorch 概念:  **Tensor** . PyTorch Tensor 在概念上与 numpy 数组相同: 
Tensor 是一个n维数组, PyTorch 也提供了很多能在这些 Tensor 上操作的函数. 像 numpy 数组一样, PyTorch Tensor
也和numpy的数组对象一样不了解深度学习,计算图和梯度下降；它们只是科学计算的通用工具. 

然而不像 numpy, PyTorch Tensor 可以利用 GPU 加速他们的数字计算. 要在 GPU 上运行
PyTorch 张量, 只需将其转换为新的数据类型. 

在这里, 我们将 PyTorch Tensor 生成的随机数据传入双层的神经网络. 就像上面的 numpy 例子一样, 
我们需要手动实现网络的正向传播和反向传播: 

.. includenodoc:: /beginner/examples_tensor/two_layer_net_tensor.py


Autograd
========

PyTorch: Variables and autograd
-------------------------------

在上面的例子中, 我们不得不手写实现神经网络的正反向传播的代码. 而手写实现反向传播的代码对于一个
小型的双层网络来说是没什么大问题的, 但是在面对大型复杂网络手写方向传播代码就会变得很棘手. 

谢天谢地, 我们可以使用 `自动微分 <https://en.wikipedia.org/wiki/Automatic_differentiation>`__
来自动化的计算神经网络中的后向传播.  PyTorch 中的 **autograd** 包提供自动微分了这个功能. 使用 autograd 时, 
网络的正向传播将定义一个 **计算图** ; Tensor 将会成为图中的节点,从输入 Tensor 产生输出 Tensor 的函数将会用图中的( Edge )依赖边表示. 
通过计算图来反向传播可以让您轻松计算梯度. 

这听起来很复杂, 但是在实践中使用起来相当简单. 我们将 PyTorch 的 Tensor 包装成在 **Variable** 对象；
一个 Variable 代表一个计算图中的节点. 如果 ``x`` 是一个 Variable , 则 ``x.data`` 是一个 Tensor , 
而 ``x.grad`` 是另外一个包含关于 ``x`` 的梯度的 Variable .

PyTorch Variable 与 PyTorch Tensor 具有相同的 API:  (几乎) 任何您可以在 Tensor 上执行的
操作也适用于 Variable ；该区别在于如果你使用 Variable 定义了一个计算图, Pytorch 允许您自动计算梯度. 

这里我们使用 PyTorch 的 Variable 和自动微分来实现我们的双层网络；现在我们不再需要手写任何关于
计算网络反向传播的代码: 

.. includenodoc:: /beginner/examples_autograd/two_layer_net_autograd.py

PyTorch: Defining new autograd functions
----------------------------------------

在这层覆盖下, 每个原始的 autograd 操作符实际上是两个函数在张量上运行.  **前向传播**
函数从输入的 Tensor 计算将要输出的 Tensor .  **后向传播** 函数接收上一个 Tensor 关于 scalar 的梯度, 以
及计算当前输入 Tensor 对相同 scalar 值的梯度. 

在 PyTorch 中, 我们可以通过定义一个 ``torch.autograd.Function`` 的子类和
实现 ``前向传播`` 和 ``后向传播`` 函数来轻松定义自己的 autograd 操作符. 然后我们可以
使用我们新的 autograd 操作符构造一个实例并将其作为一个函数调用, 传递用 Variable 包装了的输入数据的. 

在这个例子中我们定义了我们自己的 autograd 函数来执行 ReLU 非线性函数, 并用它来实现我们的双层网络: 

.. includenodoc:: /beginner/examples_autograd/two_layer_net_custom_function.py

TensorFlow: Static Graphs
-------------------------

Pytorch 的 autograd 看上去有点像 TensorFlow .两个框架的共同点是他们都是定义了自己的计算图. 
和使用自动求微分的方法来计算梯度. 两者之间最大的不同在于 TensorFlow 的计算图是 **静态的**
和 PyTorch 的计算图是 **动态的** . 

在 TensorFlow 中, 我们只定义了一次计算图,然后重复执行同一张计算图, 只是输入计算图的数据不同而已. 
而在 PyTorch 中, 每个正向传播都会定义一个新的计算图. 

静态图很好, 因为您可以预先优化计算图；例如一个框架可能会为了计算效率决定融合一些计算图操作(像:Fused Graph), 或提出
一个多卡或者多机的分布式计算图的策略. 如果您正在重复使用相同的计算图, 那么这个潜在的
昂贵的前期优化可以使用静态图来得以减轻. 

一方面来说, 静态图和动态图的控制流是不同的. 对于有些模型我们可能希望对每个数据点执行不同
的计算；例如循环神经网络可能会被展开为对每个数据的不同的长度的时间步数；这个展开可以用循
环来实现. 循环结构的静态图需要成为计算图的一部分；为此 TensorFlow 提供 ``tf.scan`` 操作符
用于将重复的结构嵌入到计算图中. 而动态计算图的情况比较简单: 因为我们设计的计算图可以对每个不同长度的输入随机应变. 
我们可以使用正常的命令式代码对每个不同长度的输入执行计算. 

为了与上面的 PyTorch autograd 例子进行对比, 我们在这里也使用 TensorFlow 创建简单的两层神经网络: 

.. includenodoc:: /beginner/examples_autograd/tf_two_layer_net.py

`nn` module
===========

PyTorch: nn
-----------

计算图( Computational graphs )和 autograd 是一个非常强大的定义复杂的运算符并自动地导出的范式；然而对于
大型的神经网络, 原始的 autograd 仍然显得有点太低级. 

当我们创建神经网络时, 我们经常思考如何设计安排 ** layer ** , 以及一些在训练过程中网络会学习到的 ** learnable parameters **


在TensorFlow中, 像 `Keras <https://github.com/fchollet/keras>`__,
`TensorFlow-Slim <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim>`__,
和 `TFLearn <http://tflearn.org/>`__ 通过构建对神经网络有用的原始计算图提供更高层次的抽象. 

在 PyTorch 中, ``nn`` 包起了同样的作用.  ``nn`` 包定义了一组 ** Modules ** , 大致相当于神经网络层. 
模块接收输入变量并进行计算输出变量, 但也可以保持内部状态, 如 用 Variable 包装的 learnable parameters .  ``nn`` 包
也定义了一系列在训练神经网络时比较常用的损失函数. 

在这个例子中, 我们使用 ``nn`` 包来实现我们的双层神经网络: 

.. includenodoc:: /beginner/examples_nn/two_layer_net_nn.py

PyTorch: optim
--------------

到目前为止, 我们一直通过手动更新的方法更新模型的可学习参数( learnable parameters )的权重 ``.data``
这对于简单的优化算法像随机梯度下降来还算轻松, 但是在实际中我们经常使用更巧妙的
优化器来训练神经网络, 如 AdaGrad, RMSProp, Adam 等. 

PyTorch 中的 ``optim`` 包包含了一些优化器的算法, 并提供了一些常用优化器的使用. 

在这个例子中, 虽然我们将像之前一样使用 ``nn`` 包来定义我们的模型, 但是我们这次将使用由 ``optim`` 包提供的Adam算法来更新模型: 

.. includenodoc:: /beginner/examples_nn/two_layer_net_optim.py

PyTorch: Custom nn Modules
--------------------------

有时你会想要使用比现有模块组合更复杂的特殊模型；对于这些情况, 你可以
通过继承 ``nn.Module`` 来定义你自己的模块, 并定义一个 ``forward``
来实现模块接收输入 Variable 并使用其他模块输出的 Variable 和 其他 autograd 操作. 

在这个例子中, 我们使用了我们之前已经实现的双层网络来作为一个自定义的模块子类: 

.. includenodoc:: /beginner/examples_nn/two_layer_net_module.py

PyTorch: Control Flow + Weight Sharing
--------------------------------------

作为一个动态图和权值共享的例子, 我们实现一个奇葩的模型: 随机1-4次重复搭建同个正向传播的全连接
的 ReLU 网络, 并且多个隐藏层使用相同的权重来计算最内层隐藏层(译者注: 这里的相同权重,是指随机1-4次重复搭建的这个middle_linear). 

对于这个模型, 我们可以使用普通的 Python 流程控制语句来实现循环, 而且我们可以在定义前向传
播时通过简单地重复使用相同的模块实现 middle_linear 层的权重共享. 

我们可以很容易地将这个模型作为 Module 子类来实现: 

.. includenodoc:: /beginner/examples_nn/dynamic_net.py


.. _examples-download:

Examples
========

你可以在这里浏览上网提到的例子

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
