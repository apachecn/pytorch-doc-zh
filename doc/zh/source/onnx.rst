torch.onnx
============
.. automodule:: torch.onnx

示例:从Pytorch到Caffe2的端对端AlexNet模型
--------------------------------------------------

一下是一个简单的脚本程序,它将一个在 torchvision 中已经定义的预训练 AlexNet 模型导出到 ONNX 格式.
它会运行一次,然后把模型保存至 ``alexnet.proto``::

    from torch.autograd import Variable
    import torch.onnx
    import torchvision

    dummy_input = Variable(torch.randn(10, 3, 224, 224)).cuda()
    model = torchvision.models.alexnet(pretrained=True).cuda()

	#提供这些是可选的，但这样做会使你的转换模型更好用
    input_names = [ "learned_%d" % i for i in range(16) ] + [ "actual_input_1" ]
    output_names = [ "output1" ]

    torch.onnx.export(model, dummy_input, "alexnet.proto", verbose=True, input_names=input_names, output_names=output_names)

得到的 ``alexnet.proto`` 是一个 protobuf 二值文件，它包含所导出模型 ( 这里是 AlexNet )中网络架构和网络参数.
使用关键字参数 ``verbose=True`` 可以在输出端以可读的方式展现网络模型结构::

	# 所有参数都被显式地编码为输入。按照惯例，学习参数(ala nn.Module.state_dict)在前，
	# 实际的输入在后。
    graph(%learned_0 : Float(10, 3, 224, 224)
          %learned_1 : Float(64, 3, 11, 11)
		  # 所有变量的定义位置都标注了数据类型，指定了张量（tensors）的类型和大小。
		  # 举例来说，%learned_2 表示一个 192 x 64 x 5 x 5 的浮点型张量tensor.
          %learned_2 : Float(64)
          %learned_3 : Float(192, 64, 5, 5) 
          # ---- 缩略写法 ----
          %learned_14 : Float(4096)
          %learned_15 : Float(1000, 4096)
          %actual_input_1 : Float(1000)) { 
	  # 每个语句都包含：输出张量（及其类型）、需要运行的操作（及其属性，比如
	  # kernels、strides等）、输入张量（%learned_0, %learned_1, %learned_2）
      %17 : Float(10, 64, 55, 55) = Conv[dilations=[1, 1], group=1, kernel_shape=[11, 11], pads=[2, 2, 2, 2], strides=[4, 4]](%learned_0, %learned_1, %learned_2), scope: AlexNet/Sequential[features]/Conv2d[0]
      %18 : Float(10, 64, 55, 55) = Relu(%17), scope: AlexNet/Sequential[features]/ReLU[1]
      %19 : Float(10, 64, 27, 27) = MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2]](%18), scope: AlexNet/Sequential[features]/MaxPool2d[2] 
	  # ---- 缩略写法 ----
      %29 : Float(10, 256, 6, 6) = MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2]](%28), scope: AlexNet/Sequential[features]/MaxPool2d[12]
      %30 : Float(10, 9216) = Flatten[axis=1](%29), scope: AlexNet 
	  # 未知类型：有时类型信息不明确，我们期望在下个版本中消除这种情况。
      %31 : Float(10, 9216), %32 : UNKNOWN_TYPE = Dropout[is_test=1, ratio=0.5](%30), scope: AlexNet/Sequential[classifier]/Dropout[0]
      %33 : Float(10, 4096) = Gemm[alpha=1, beta=1, broadcast=1, transB=1](%31, %learned_11, %learned_12), scope: AlexNet/Sequential[classifier]/Linear[1] 
	  # ---- 缩略写法 ----
      %output1 : Float(10, 1000) = Gemm[alpha=1, beta=1, broadcast=1, transB=1](%38, %learned_15, %actual_input_1), scope: AlexNet/Sequential[classifier]/Linear[6]
      # Finally, a network returns some tensors
      # 最后, 神经网络将返回一些张量
      return (%output1);
    }

你可以使用 `onnx <https://github.com/onnx/onnx/>`_ 库验证 protobuf,
并且用 conda 安装 ``onnx`` :: 

    conda install -c conda-forge onnx

然后运行::

    import onnx

    # 加载 ONNX 模型
    model = onnx.load("alexnet.proto")

    # 检测 IR 生成情况
    onnx.checker.check_model(model)

    # 以可读的图形表示
    onnx.helper.printable_graph(model.graph)

为了能够使用 `caffe2 <https://caffe2.ai/>`_ 运行脚本,你需要安装 Caffe2. 如果你之前没有安装,请参照`安装指南 <https://caffe2.ai/docs/getting-started.html>`_.  

一旦这些安装完成, 你就可以使用 Caffe2 的后端::
 
    # ...从上面继续
    import caffe2.python.onnx.backend as backend
    import numpy as np

    rep = backend.prepare(model, device="CUDA:0") # or "CPU"
    # For the Caffe2 backend:
    #     rep.predict_net is the Caffe2 protobuf for the network
    #     rep.workspace is the Caffe2 workspace for the network
    #       (参看 类 caffe2.python.onnx.backend.Workspace)
    outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32)) 
    # 如果要使用多个输入，请传入tuple类型，而不是单个numpy数组 
    print(outputs[0])

之后, 我们还会提供其它深度学习框架的后端支持.

局限性
-----------

	*ONNX导出器是一个基于轨迹的导出器,这意味着它执行时需要运行一次模型,然后
	导出实际参与运算的运算符.这也意味着，如果你的模型是动态的,例如,改变一些
	依赖于输入数据的操作,这时的导出结果是不准确的. 同样,一个轨迹可能只对一个
	具体的输入尺寸有效（这是为什么我们在轨迹中需要有明确的输入的原因之一.）我
	们建议检查模型的轨迹,确保被追踪的运算符是合理的.

	* Pytorch 和 Caffe2 中的一些运算符经常有着数值上的差异.根据模型的结构,这些差异可能是微小的, 但它们会在表现上产生很大的差别（尤其是对于未训练的模型.）之后,为了帮助你在
	准确度要求很高的情况中,能够轻松地避免这些差异带来的影响,我们计划让 Caffe2
	能够直接调用 Torch 的运算符.

支持的运算符
-------------------

以下是已经被支持的运算符:

* add (不支持非零的alpha)
* sub (不支持非零的alpha)
* mul
* div
* cat
* mm
* addmm
* neg
* sqrt
* tanh
* sigmoid
* mean
* sum
* prod
* t
* expand (只在使用一个传播式 ONNX 运算符，比如add)
* transpose
* view
* split
* squeeze
* prelu (不支持输入通道中的单权重)
* threshold (不支持非零阈值和非零值)
* leaky_relu
* glu
* softmax (仅支持 dim=-1)
* avg_pool2d (不支持ceil_mode)
* log_softmax
* unfold (实验支持 ATen-Caffe2 集成)
* elu
* concat
* abs
* index_select
* pow
* clamp
* max
* min
* eq
* exp
* permute
* Conv
* BatchNorm
* MaxPool1d (不支持ceil_mode)
* MaxPool2d (不支持ceil_mode)
* MaxPool3d (不支持ceil_mode)
* Embedding (不支持可选参数)
* RNN
* ConstantPadNd
* Dropout
* FeatureDropout (不支持训练模型)
* Index (支持连续整数和 tuple 索引)

上面的运算符足够导出一下的模型:

* AlexNet
* DCGAN
* DenseNet
* Inception (注意:该模型对操作符十分敏感)
* ResNet
* SuperResolution
* VGG
* `word_language_model <https://github.com/pytorch/examples/tree/master/word_language_model>`_

给操作符添加输出支持是一种*高级用法*。
为实现这一功能，开发者需要获得 Pytorch 的源码。
请遵循 `说明 <https://github.com/pytorch/pytorch#from-source>`_
从源码安装PyTorch.
如果预期的操作符符合 ONNX 标准，添加操作符支持
应该相当容易(为操作符添加符号函数).
想确认操作符是否符合标准，请参阅
`ONNX 操作符列表 <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_.

如果是 ATen 操作符，也就是说你可以在
``torch/csrc/autograd/generated/VariableType.h``
（可在PyTorch安装目录中生成代码中找到）
找到函数的阐述，那么你应该先将符号函数添加到``torch/onnx/symbolic.py``
并按一下指令顺序操作：

* 在 `torch/onnx/symbolic.py <https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic.py>`_
  中定义符号函数。确保该函数和``VariableType.h``中定义的ATen 操作符/函数具有
  相同的名字。 
* 第一个参数永远是要输出的 ONNX 图表。
  参数名必须和``VariableType.h``中完全相同，
  因为函数使用关键字参数分配变量。
* 参数顺序并不需要``VariableType.h``中定义的相同，张量（输入）总在第一个，
  其他的非张量参数紧随其后。  
* 在符号函数中，如果操作符已经符合 ONNX 标准，那我们只需要创建一个节点，
  在图表中表示 ONNX 操作符。
* 如果输入参数是张量，而 ONNX 请求一个scalar，那么必须明确作出转换。帮助函数
  ``_scalar``可将scalar tensor转换为python scalar，函数``_if_scalar_type_as``
  可将Python scalar转换为PyTorch tensor。

如果要添加的不是ATen操作符，那么必须在对应的PyTorch函数类中添加。请阅读以下说明：
 
* 在对应的函数类中创建名称为``symbolic``的符号函数。 
* 第一个参数都是要输出的 ONNX 图表。 
* 除第一个参数外，参数名必须匹配``forward``中的名字。 
* 输出tuple的大小必须匹配``forward``中的输出。
* 在符号函数中，如果操作符已经符合 ONNX 标准，那我们只需要创建一个节点，
  在图表中表示 ONNX 操作符。

符号函数应当在Pyhton中实现。所有这些函数和C++和Python捆绑实现的Python方法进行交互，
直觉上交互界面看起来是这样的::


    def operator/symbolic(g, *inputs):
      """
	  修改图表（例：使用"op"），添加 ONNX 操作符以展示该PyTorch函数，并返回单个值或
	  tuple作为ONNX 输出，该输出对应原始autograd函数的PyTorch返回值
	  （如果输出值不被ONNX支持则返回None）。

      参数:
        g (Graph): 用来写入ONNX展示的图表
        inputs (Value...): 代表该函数输入变量的list
      """

    class Value(object):
      """代表ONNX计算的一个中间tensor数值"""
      def type(self):
        """返回数值的类型"""

    class Type(object):
      def sizes(self):
        """返回一个整型tuple，代表所描述的tenser的形状。"""

    class Graph(object):
      def op(self, opname, *inputs, **attrs):
        """
		创建一个ONNX操作符 'opname'，使用 'args' 作为输入和属性'kwargs'，
		并在当前图表中添加一个节点，返回该操作符的单一输出（参见`outputs`
		关键字参数用于多个返回节点）。

		操作符集合和他们需要的输入/属性记录在 https://github.com/onnx/onnx/blob/master/docs/Operators.md

        参数:
            opname (string): ONNX操作符名称, 例`Abs` 或 `Add`。
            args (Value...): 操作符的输入; 一般作为参数提供给`symbolic`定义。
            kwargs: ONNX操作符的属性，键名遵循如下规则：`alpha_f`表示`f`类型
				的`alpha`属性。合法的类型有`f` (浮点), `i` (整型), `s` (字符串)
				和 `t` (Tensor)。指定了浮点类型属性可以是单个浮点数，也可以是
				浮点list（比如：`dims_i`可以表示整型list的`dims`属性）。
            outputs (int, optional):  返回值的数量;
                默认返回单个输出.如果`outputs` 大于1，该函数返回一个tuple，其中
				每个元素代表ONNX操作符在当前位置的每个输出。 
        """
		
ONNX 图表的C++定义在``torch/csrc/jit/ir.h``文件. 

以下实例展示了如何处理``elu``操作符中缺失符号函数的情况。
当我们尝试输出模型，看到了如下信息:: 

    UserWarning: ONNX export failed on elu because torch.onnx.symbolic.elu does not exist
    RuntimeError: ONNX export failed: Couldn't export operator elu

由于PyTorch不支持``elu``操作符，导致输出失败。
我们在``VariableType.h``中找到了``virtual Tensor elu(const Tensor & input, Scalar alpha, bool inplace) const override;``这意味着``elu``是ATen操作符。
然后查看`ONNX 操作符列表 <http://https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_，
确认``Elu``符合ONNX标准。我们向``symbolic.py``添加以下几行:: 

    def elu(g, input, alpha, inplace=False):
        return g.op("Elu", input, alpha_f=_scalar(alpha))
  
现在 PyTorch 已支持输出``elu``操作符。 
 
更多的例子请参看：
`symbolic.py <https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic.py>`_,
`tensor.py <https://github.com/pytorch/pytorch/blob/99037d627da68cdf53d3d0315deceddfadf03bba/torch/autograd/_functions/tensor.py#L24>`_,
`padding.py <https://github.com/pytorch/pytorch/blob/99037d627da68cdf53d3d0315deceddfadf03bba/torch/nn/_functions/padding.py#L8>`_.


某些特定的操作符还处于试验阶段；喜欢创新的用户应该注意, 这些API可能会在之后被修改.

Functions
--------------------------
.. autofunction:: export
