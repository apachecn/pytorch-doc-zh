from .module import Module
from .. import functional as F


class Dropout(Module):
    r"""Dropout��ѵ���ڼ䣬���ղ�Ŭ�����ʷֲ����Ը���p����ؽ����������еĲ���Ԫ��
	��Ϊ0����ÿ�ε����У�����Ϊ0��Ԫ��������ġ�

	Dropout�ѱ�֤�������򻯵�һ����֮��Ч�ļ����������ڷ�ֹ��Ԫ֮�以��Ӧ������
	Ҳ׿�г�Ч������Ԫ����Ӧ����������ġ�Improving neural networks by preventing 
	co-adaptation of feature detectors����

	���ң�Dropout���������*1/(1-p)*�ı���ϵ����������ˣ���֤����ֵʱ�����Ĺ�һ����
    
	������
		p��Ԫ�ر���Ϊ0�ĸ��ʣ�Ĭ��ֵ��0.5
		inplace�����Ϊ��True������0������ֱ�ӷ����ڴ����Ԫ���ϡ�Ĭ��ֵ����false��
    
	���ݴ�С��
		- Input��'any'���������ݿ������κδ�С
		- Output��'Same'��������ݴ�С��������ͬ
    
	ʾ����

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
    r"""Dropout2d����������������ͨ���������Ϊ0������Ϊ0��ͨ����ÿ�ε���ʱ
	������ء�
	
	ͨ��������������Conv2dģ�顣

	�����ġ�Efficient Object Localization Using Convolutional Networks`_ ����������
	�������������ӳ���е��ڽ�������ǿ��صģ������ڵľ�����кܳ���������ô����ͬ�ֲ�
	��dropout���������򻯼�������෴��ᵼ����Ч��ѧϰ�ʵ��½���

	������������£�Ӧ��ʹ�ú�������'nn.Dropout2d'�����ܹ��ٽ�����ӳ��֮��Ķ����ԡ�
    
	������
		p (float,optional)��Ԫ�ر���0�ĸ���
		inplace��bool��optional�����������Ϊ��True������0������ֱ������������Ԫ����
   
	���ݴ�С��
		- Input��math:(N, C, H, W)
		- Output��math:(N, C, H, W) ����������ͬ��
    
	ʾ����

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
    r"""Dropout2d����������������ͨ���������Ϊ0������Ϊ0��ͨ����ÿ�ε���ʱ
	������ء�
	
	ͨ��������������Conv3dģ�顣

	�����ġ�Efficient Object Localization Using Convolutional Networks`_ ����������
	�������������ӳ���е��ڽ�������ǿ��صģ������ڵľ�����кܳ���������ô����ͬ�ֲ�
	��dropout���������򻯼�������෴��ᵼ����Ч��ѧϰ�ʵ��½���

	������������£�Ӧ��ʹ�ú�������'nn.Dropout3d'�����ܹ��ٽ�����ӳ��֮��Ķ����ԡ�
    
	������
		p (float,optional)��Ԫ�ر���0�ĸ���
		inplace��bool��optional�����������Ϊ��True������0������ֱ������������Ԫ����
    
	���ݴ�С��
		- Input��math:(N, C, H, W)
		- Output��math:(N, C, H, W) ����������ͬ��
    
	ʾ����

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
    r"""��������Ӧ��Alpha Dropout��

	Alpha Dropout��һ��ά�����������ʵ�Dropout������һ����ֵΪ0�ͱ�׼��Ϊ1������
	��˵��Alpha Dropout�ܱ���ԭʼ���ݵľ�ֵ�ͱ�׼�Alpha Dropout��SELU�����
	Я��ͬ�У�����Ҳ��֤�����ӵ����������ͬ�ľ�ֵ�ͱ�׼�
    
	Alpha Dropout��ѵ���ڼ䣬���ղ�Ŭ�����ʷֲ����Ը���p����ؽ����������еĲ���Ԫ��
	�ý����ڸǣ���ÿ�ε����У����ڸǵ�Ԫ��������ģ����Ҷ������������š��任�Ȳ���
	�Ա��־�ֵΪ0����׼��Ϊ1.

	����ֵ�ڼ䣬ģ��򵥵ļ���һ����һ���ĺ�����

	������Ϣ��ο����ģ�Self-Normalizing Neural Networks

	������
		p��float����Ԫ�ر��ڸǵĸ��ʣ�Ĭ��ֵ��0.5
    
	���ݴ�С��
		- Input��'any'���������ݿ������κδ�С
		- Output��'Same'��������ݴ�С��������ͬ

	ʾ����

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
