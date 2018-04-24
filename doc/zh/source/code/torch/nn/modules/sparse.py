import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from .module import Module
from .. import functional as F


class Embedding(Module):
    r""" 一个简单的查找表, 存储了固定字典和大小的 embedding.

    这个模块经常用来存储 word embeddings, 并通过索引来检索,
    模块的输入是索引构成的列表, 输出是对应的 word embeddings.


    Args:
        num_embeddings (int): embeddings 字典的大小
        embedding_dim (int): 每个 embedding 向量的大小
        padding_idx (int, optional): 如果给出, 在索引处, 输出补零
        max_norm (float, optional): 如果给出, 重新归一化 embeddings, 使其范数小于该值
        norm_type (float, optional): 为 max_norm 选项计算 p 范数时 P
        scale_grad_by_freq (boolean, optional): 如果给出, 会根据 words 在 mini-batch 中的频率缩放梯度                                              
        sparse (boolean, optional): 如果为 ``True``, 关于权重矩阵的梯度是一个稀疏张量, 详情请参考稀疏梯度
                                    

    Attributes:
        weight (Tensor): shape 为 (num_embeddings, embedding_dim) 的模块的可学习权重

    Shape:
        - Input: LongTensor `(N, W)`,  N = mini-batch,  W =  每个 mini-batch 中用来提取的索引数
        - Output: `(N, W, embedding_dim)`

    Notes:
        请注意, 只支持有限数量的优化器. 
        稀疏梯度:  当前是  (`cuda`  和  `cpu`) 版本的 `optim.SGD`, 和 (`cpu`) 版本的 `optim.Adagrad`.
        

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding = nn.Embedding(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
        >>> embedding(input)

        Variable containing:
        (0 ,.,.) =
         -1.0822  1.2522  0.2434
          0.8393 -0.6062 -0.3348
          0.6597  0.0350  0.0837
          0.5521  0.9447  0.0498

        (1 ,.,.) =
          0.6597  0.0350  0.0837
         -0.1527  0.0877  0.4260
          0.8393 -0.6062 -0.3348
         -0.8738 -0.9054  0.4281
        [torch.FloatTensor of size 2x4x3]

        >>> # example with padding_idx
        >>> embedding = nn.Embedding(10, 3, padding_idx=0)
        >>> input = Variable(torch.LongTensor([[0,2,0,5]]))
        >>> embedding(input)

        Variable containing:
        (0 ,.,.) =
          0.0000  0.0000  0.0000
          0.3452  0.4937 -0.9361
          0.0000  0.0000  0.0000
          0.0706 -2.1962 -0.6276
        [torch.FloatTensor of size 1x4x3]

    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.sparse = sparse

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def forward(self, input):
        padding_idx = self.padding_idx
        if padding_idx is None:
            padding_idx = -1
        return self._backend.Embedding.apply(
            input, self.weight,
            padding_idx, self.max_norm, self.norm_type,
            self.scale_grad_by_freq, self.sparse
        )

    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class EmbeddingBag(Module):
    r""" 计算一 个'bags' 里的 embedding s的均值或和, 不用实例化中间的 embeddings
    

    对于固定长度的 bags
        * nn.EmbeddingBag  和  `mode=sum`  相当于 nn.Embedding 与之后的 `torch.sum(dim=1)`
        * 其与 `mode=mean` 相当于 nn.Embedding 与之后的 `torch.mean(dim=1)`

    
    然而, 比起一连串这样的操作, nn.EmbeddingBag 在时间和内存上更加高效. 
    

    Args:
        num_embeddings (int): embeddings 字典的大小
        embedding_dim (int): 每个 embedding 向量的大小
        max_norm (float, optional): 如果给出, 重新归一化 embeddings, 使其范数小于该值
        norm_type (float, optional): 为 max_norm 选项计算 p 范数时的 P
        scale_grad_by_freq (boolean, optional): 如果给出, 会根据 words 在 mini-batch 中的频率缩放梯度
                                                
        mode (string, optional): 'sum' | 'mean'.  指定减少 bag 的方式.  默认: 'mean'

    Attributes:
        weight (Tensor): shape 为 (num_embeddings, embedding_dim) 的模块的可学习权重

    Inputs: input, offsets
        - **input** (N or BxN): LongTensor, 包括要提取的 embeddings 的索引, 
                                当 `input` 是形状为  `N` 的 1D 张量时, 
                                一个给出的 `offsets` 张量中包括:  mini-batch 中每个新序列的起始位置
                                
                                
        - **offsets** (B or None): LongTensor, 包括一个 mini-batch 的可变长度序列中的每个新样本的起始位置                                  
                                    如果 `input` 是 2D (BxN) 的,  offset 就不用再给出; 
                                    如果 `input` 是一个 mini-batch 的固定长度的序列, 每个序列的长度为 `N`


                                   


    Shape:
        - Input: LongTensor `N`,  N = 要提取的 embeddings 的数量, 
         或者是 LongTensor `BxN`,  B = mini-batch 中序列的数量,  N = 每个序列中 embeddings 的数量



        - Offsets: LongTensor `B`,  B = bags 的数量, 值为每个 bag 中 `input` 的 offset, i.e. 是长度的累加. 
                    Offsets 不会给出, 如果 Input是 2D 的 `BxN` 张量,  输入被认为是固定长度的序列

                   
        - Output: `(B, embedding_dim)`

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding_sum = nn.EmbeddingBag(10, 3, mode='sum')
        >>> # a batch of 2 samples of 4 indices each
        >>> input = Variable(torch.LongTensor([1,2,4,5,4,3,2,9]))
        >>> offsets = Variable(torch.LongTensor([0,4]))
        >>> embedding_sum(input, offsets)

        Variable containing:
        -0.7296 -4.6926  0.3295
        -0.5186 -0.5631 -0.2792
        [torch.FloatTensor of size 2x3]

    """

    def __init__(self, num_embeddings, embedding_dim,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 mode='mean'):
        super(EmbeddingBag, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.mode = mode

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)

    def forward(self, input, offsets=None):
        return F.embedding_bag(self.weight, input, offsets,
                               self.max_norm, self.norm_type,
                               self.scale_grad_by_freq, self.mode)

    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        s += ', mode={mode}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

# TODO: SparseLinear
