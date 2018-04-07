from torch.autograd import Variable


class Parameter(Variable):
    r"""Variable 的一种, 常被用于 module parameter（模块参数）.

    Parameters 是 :class:`~torch.autograd.Variable` 的子类, 当它和 :class:`Module`
    一起使用的时候会有一些特殊的属性 - 当它们被赋值给 Module 属性时,
    它会自动的被加到 Module 的参数列表中, 并且会出现在 :meth:`~Module.parameters` iterator 迭代器方法中.
    将 Varibale 赋值给 Module 属性则不会有这样的影响.
    这样做的原因是: 我们有时候会需要缓存一些临时的 state（状态）,
    例如: 模型 RNN 中的最后一个隐藏状态.
    如果没有 :class:`Parameter` 这个类的话,
    那么这些临时表也会注册为模型变量.

    Variable 与 Parameter 的另一个不同之处在于,
    Parameter 不能被 volatile (即: 无法设置 volatile=True) 而且默认 requires_grad=True.
    Variable 默认 requires_grad=False.

    Arguments:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): 如果参数需要梯度. 更多细节请参阅 :ref:`excluding-subgraphs`.
    """
    def __new__(cls, data=None, requires_grad=True):
        return super(Parameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()
