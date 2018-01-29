import torch
from .module import Module
from .. import functional as F


class PairwiseDistance(Module):
    r"""计算向量 v1, v2 之间的 batchwise pairwise distance(分批成对距离):

    .. math ::
        \Vert x \Vert _p := \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}

    Args:
        p (real): norm degree(规范程度). Default: 2
        eps (float, optional): 小的值以避免被零除.
            Default: 1e-6

    Shape:
        - Input1: :math:`(N, D)`, 其中的 `D = vector dimension(向量维度)`
        - Input2: :math:`(N, D)`, 与 Input1 的 shape 一样
        - Output: :math:`(N, 1)`

    Examples::

    >>> pdist = nn.PairwiseDistance(p=2)
    >>> input1 = autograd.Variable(torch.randn(100, 128))
    >>> input2 = autograd.Variable(torch.randn(100, 128))
    >>> output = pdist(input1, input2)
    """
    def __init__(self, p=2, eps=1e-6):
        super(PairwiseDistance, self).__init__()
        self.norm = p
        self.eps = eps

    def forward(self, x1, x2):
        return F.pairwise_distance(x1, x2, self.norm, self.eps)


class CosineSimilarity(Module):
    r"""返回沿着 dim 方向计算的 x1 与 x2 之间的余弦相似度.

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}

    Args:
        dim (int, optional): 计算余弦相似度的维度. Default: 1
        eps (float, optional): 小的值以避免被零除.
            Default: 1e-8

    Shape:
        - Input1: :math:`(\ast_1, D, \ast_2)`, 其中的 D 表示 `dim` 的位置
        - Input2: :math:`(\ast_1, D, \ast_2)`, 与 Input1 一样的 shape
        - Output: :math:`(\ast_1, \ast_2)`

    Examples::

    >>> input1 = autograd.Variable(torch.randn(100, 128))
    >>> input2 = autograd.Variable(torch.randn(100, 128))
    >>> cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    >>> output = cos(input1, input2)
    >>> print(output)
    """
    def __init__(self, dim=1, eps=1e-8):
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return F.cosine_similarity(x1, x2, self.dim, self.eps)
