torch.onnx
============
.. automodule:: torch.onnx

示例:从Pytorch到Caffe2的端对端AlexNet模型
--------------------------------------------------

这里是一个简单的脚本程序,它将一个在 torchvision 中已经定义的预训练 AlexNet 模型导出到 ONNX 格式.
它会运行一次,然后把模型保存至 ``alexnet.proto``::

    from torch.autograd import Variable
    import torch.onnx
    import torchvision

    dummy_input = Variable(torch.randn(10, 3, 224, 224)).cuda()
    model = torchvision.models.alexnet(pretrained=True).cuda()
    torch.onnx.export(model, dummy_input, "alexnet.proto", verbose=True)

得到的 ``alexnet.proto`` 是一个 protobuf 二值文件，它包含所导出模型 ( 这里是 AlexNet )中网络架构和网络参数.
关键参数 ``verbose=True`` , 则会导出并打印出训练过程中该网络的可读表示::

    # All parameters are encoded explicitly as inputs.  By convention,
    # learned parameters (ala nn.Module.state_dict) are first, and the
    # actual inputs are last.
    graph(%1 : Float(64, 3, 11, 11)
          %2 : Float(64)
          # The definition sites of all variables are annotated with type
          # information, specifying the type and size of tensors.
          # For example, %3 is a 192 x 64 x 5 x 5 tensor of floats.
          %3 : Float(192, 64, 5, 5)
          %4 : Float(192)
          # ---- omitted for brevity ----
          %15 : Float(1000, 4096)
          %16 : Float(1000)
          %17 : Float(10, 3, 224, 224)) { # the actual input!
      # Every statement consists of some output tensors (and their types),
      # the operator to be run (with its attributes, e.g., kernels, strides,
      # etc.), its input tensors (%17, %1)
      %19 : UNKNOWN_TYPE = Conv[kernels=[11, 11], strides=[4, 4], pads=[2, 2, 2, 2], dilations=[1, 1], group=1](%17, %1), uses = [[%20.i0]];
      # UNKNOWN_TYPE: sometimes type information is not known.  We hope to eliminate
      # all such cases in a later release.
      %20 : Float(10, 64, 55, 55) = Add[broadcast=1, axis=1](%19, %2), uses = [%21.i0];
      %21 : Float(10, 64, 55, 55) = Relu(%20), uses = [%22.i0];
      %22 : Float(10, 64, 27, 27) = MaxPool[kernels=[3, 3], pads=[0, 0, 0, 0], dilations=[1, 1], strides=[2, 2]](%21), uses = [%23.i0];
      # ...
      # Finally, a network returns some tensors
      return (%58);
    }

你可以使用 `onnx <https://github.com/onnx/onnx/>`_ 库验证 protobuf,
并且用 conda 安装 ``onnx`` ::

    conda install -c conda-forge onnx

然后运行::

    import onnx

    # Load the ONNX model
    model = onnx.load("alexnet.proto")

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)

为了能够使用 `caffe2 <https://caffe2.ai/>`_ 运行脚本, 你需要三样东西:

1. 你需要安装 Caffe2. 如果你之前没有安装,请参照
   `安装指南 <https://caffe2.ai/docs/getting-started.html>`_. 

2. 你需要安装 `onnx-caffe2 <https://github.com/onnx/onnx-caffe2>`_ , 一个纯 Python 的库,它为 ONNX 提供了 Caffe2 的
   后端.你可以使用 pip 安装 ``onnx-caffe2``::

      pip install onnx-caffe2

一旦这些安装完成, 你就可以使用 Caffe2 的后端::

    # ...continuing from above
    import onnx_caffe2.backend as backend
    import numpy as np

    rep = backend.prepare(model, device="CUDA:0") # or "CPU"
    # For the Caffe2 backend:
    #     rep.predict_net is the Caffe2 protobuf for the network
    #     rep.workspace is the Caffe2 workspace for the network
    #       (see the class onnx_caffe2.backend.Workspace)
    outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))
    # To run networks with more than one input, pass a tuple
    # rather than a single numpy ndarray.
    print(outputs[0])

之后, 我们还会提供其它深度学习框架的后端支持.

局限
-----------

* ONNX 导出器是一个基于轨迹的导出器,这意味着它执行时需要运行一次模型,然后导出实际参与运算的运算符.
  这也意味着，如果你的模型是动态的,例如,改变一些依赖于输入数据的操作,这时的导出结果是不准确的. 同样,一
  个轨迹可能只对一个具体的输入尺寸有效（这是为什么我们在轨迹中需要有明确的输入的原因之一.）我们建议检查
  模型的轨迹,确保被追踪的运算符是合理的.

* Pytorch 和 Caffe2 中的一些运算符经常有着数值上的差异.根据模型的结构,这些差异可能是微小的, 但它们会在
  表现上产生很大的差别（尤其是对于未训练的模型.）之后,为了帮助你在准确度要求很高的情况中,能够轻松地避免这
  些差异带来的影响,我们计划让 Caffe2 能够直接调用 Torch 的运算符.


支持的运算符
-------------------

以下是已经被支持的运算符:

* add (nonzero alpha not supported)
* sub (nonzero alpha not supported)
* mul
* div
* cat
* mm
* addmm
* neg
* tanh
* sigmoid
* mean
* t
* expand (only when used before a broadcasting ONNX operator; e.g., add)
* transpose
* view
* split
* squeeze
* prelu (single weight shared among input channels not supported)
* threshold (non-zero threshold/non-zero value not supported)
* leaky_relu
* glu
* softmax
* avg_pool2d (ceil_mode not supported)
* log_softmax
* unfold (experimental support with ATen-Caffe2 integration)
* elu
* Conv
* BatchNorm
* MaxPool1d (ceil_mode not supported)
* MaxPool2d (ceil_mode not supported)
* MaxPool3d (ceil_mode not supported)
* Embedding (no optional arguments supported)
* RNN
* ConstantPadNd
* Dropout
* FeatureDropout (training mode not supported)
* Index (constant integer and tuple indices supported)
* Negate

上面的运算符足够导出下面的模型:

* AlexNet
* DCGAN
* DenseNet
* Inception (注意:该模型对操作符十分敏感)
* ResNet
* SuperResolution
* VGG
* `word_language_model <https://github.com/pytorch/examples/tree/master/word_language_model>`_

用于指定运算符定义的接口是高度实验性的, 并且还没有记录. 喜欢探索的用户应该注意, 这些API可能会在之后被修改.

Functions
--------------------------
.. autofunction:: export
