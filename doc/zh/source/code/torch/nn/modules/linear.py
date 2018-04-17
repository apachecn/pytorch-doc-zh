import math

import torch
from torch.nn.parameter import Parameter
from .. import functional as F
from .module import Module


class Linear(Module):
    r"""对输入数据进行线性变换: :math:`y = Ax + b`

    Args:
        in_features: 每个输入样本的大小
        out_features: 每个输出样本的大小
        bias: 若设置为 False, 这层不会学习偏置. 默认值: True

    Shape:
        - Input: :math:`(N, *, in\_features)` 这里 `*` 意味着可以添加任意数量的其他维度
        - Output: :math:`(N, *, out\_features)` 除了最后一个维度外, 其余的都与输入相同

    Attributes:
        weight: 形状为 (out_features x in_features) 的模块中可学习的权值
        bias: 形状为 (out_features) 的模块中可学习的偏置
    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'


class Bilinear(Module):
    r"""对输入数据进行双线性变换:
    :math:`y = x_1 * A * x_2 + b`

    Args:
        in1_features: 输入一的每个输入样本的大小
        in2_features: 输入二的每个输入样本的大小
        out_features: 每个输出样本的大小
        bias: 若设置为False, 这层不会学习偏置. 默认值: True

    Shape:
        - Input: :math:`(N, in1\_features)`, :math:`(N, in2\_features)`
        - Output: :math:`(N, out\_features)`

    Attributes:
        weight: 形状为 (out_features x in1_features x in2_features) 的模块中可学习的权值
            
        bias: 形状为 (out_features) 的模块中可学习的偏置

    Examples::

        >>> m = nn.Bilinear(20, 30, 40)
        >>> input1 = autograd.Variable(torch.randn(128, 20))
        >>> input2 = autograd.Variable(torch.randn(128, 30))
        >>> output = m(input1, input2)
        >>> print(output.size())
    """

    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super(Bilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in1_features, in2_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2):
        return F.bilinear(input1, input2, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in1_features=' + str(self.in1_features) \
            + ', in2_features=' + str(self.in2_features) \
            + ', out_features=' + str(self.out_features) + ')'

# TODO: PartialLinear - maybe in sparse?
