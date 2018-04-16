from .module import Module
from .. import functional as F


class Dropout(Module):
    r"""Dropout 在训练期间, 按照伯努利概率分布, 以概率 p 随机地将输入张量中的部分元素
	置为 0, 在每次调用时, 被置为 0 的元素是随机的.

	Dropout 已被证明是正则化的一个行之有效的技术, 并且在防止神经元之间互适应问题上
	也卓有成效.（神经元互适应问题详见论文 `Improving neural networks by preventing 
	co-adaptation of feature detectors`_ ）

	并且,  Dropout 的输出均与 *1/(1-p)* 的比例系数进行了相乘, 保证了求值时函数是归一化的.
    
	Args: 
		p: 元素被置为0的概率, 默认值: 0.5
		inplace: 如果为 True, 置0操作将直接发生在传入的元素上.默认值:  false
    
	Shape: 
		- Input:  any.输入数据可以是任何大小
		- Output:  Same.输出数据大小与输入相同
    
	Examples:: 

	    >>> m = nn.Dropout(p=0.2)
        >>> input = autograd.Variable(torch.randn(20, 16))
        >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout(input, self.p, self.training, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) \
            + inplace_str + ')'


class Dropout2d(Module):
    r"""Dropout2d 将输入张量的所有通道随机地置为 0.被置为 0 的通道在每次调用时是随机的.
	
	通常输入数据来自 Conv2d 模块.

	在论文 `Efficient Object Localization Using Convolutional Networks`_ 中有如下
	描述: 如果特征映射中的邻接像素是强相关的（在早期的卷积层中很常见）, 那么独立同分布
	的 dropout 将不会正则化激活函数, 相反其会导致有效的学习率的下降.

	在这样的情况下, 应该使用函数函数 nn.Dropout2d , 它能够提升特征映射之间的独立性.
    
	Args: 
		p (float,optional): 元素被置0的概率
		inplace（bool, optional）: 如果被设为’True’, 置0操作将直接作用在输入元素上
   
	Shape: 
		- Input: math:(N, C, H, W)
		- Output: math:(N, C, H, W) （与输入相同）
    
	Examples:: 

	    >>> m = nn.Dropout2d(p=0.2)
        >>> input = autograd.Variable(torch.randn(20, 16, 32, 32))
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       http://arxiv.org/abs/1411.4280
    """

    def __init__(self, p=0.5, inplace=False):
        super(Dropout2d, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout2d(input, self.p, self.training, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) \
            + inplace_str + ')'


class Dropout3d(Module):
    r"""Dropout3d 将输入张量的所有通道随机地置为 0.被置为 0 的通道在每次调用时是随机的.
	
	通常输入数据来自 Conv3d 模块.

	在论文 `Efficient Object Localization Using Convolutional Networks`_ 中有如下
	描述: 如果特征映射中的邻接像素是强相关的（在早期的卷积层中很常见）, 那么独立同分布
	的 dropout 将不会正则化激活函数, 相反其会导致有效的学习率的下降.

	在这样的情况下, 应该使用函数函数 nn.Dropout3d , 它能够促进特征映射之间的独立性.
    
	Args: 
		p (float,optional): 元素被置0的概率
		inplace（bool, optional）: 如果被设为 True , 置0操作将直接作用在输入元素上
    
	Shape: 
		- Input: math:(N, C, H, W)
		- Output: math:(N, C, H, W) （与输入相同）
    
	Examples:: 

	    >>> m = nn.Dropout3d(p=0.2)
        >>> input = autograd.Variable(torch.randn(20, 16, 4, 32, 32))
        >>> output = m(input)
    
    .. _Efficient Object Localization Using Convolutional Networks:
       http://arxiv.org/abs/1411.4280
    """

    def __init__(self, p=0.5, inplace=False):
        super(Dropout3d, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout3d(input, self.p, self.training, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) \
            + inplace_str + ')'


class AlphaDropout(Module):
    r"""在输入上应用 Alpha Dropout.

	Alpha Dropout 是一种维持自正交性质的 Dropout . 对于一个均值为 0 和标准差为 1 的输入
	来说, Alpha Dropout 能保持原始数据的均值和标准差.Alpha Dropout 和 SELU 激活函数
	携手同行, 后者也保证了输出拥有与输入相同的均值和标准差.
    
	Alpha Dropout 在训练期间, 按照伯努利概率分布, 以概率 p 随机地将输入张量中的部分元素
	置进行掩盖, 在每次调用中, 被掩盖的元素是随机的, 并且对输出会进行缩放、变换等操作
	以保持均值为 0、标准差为 1.

	在求值期间, 模块简单的计算一个归一化的函数.

	更多信息请参考论文: Self-Normalizing Neural Networks

	Args: 
		p（float）: 元素被掩盖的概率, 默认值: 0.5
    
	Shape: 
		- Input:  any.输入数据可以是任何大小
		- Output:  Same.输出数据大小与输入相同

	Examples:: 

	    >>> m = nn.AlphaDropout(p=0.2)
        >>> input = autograd.Variable(torch.randn(20, 16))
        >>> output = m(input)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """

    def __init__(self, p=0.5):
        super(AlphaDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def forward(self, input):
        return F.alpha_dropout(input, self.p, self.training)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) + ')'
