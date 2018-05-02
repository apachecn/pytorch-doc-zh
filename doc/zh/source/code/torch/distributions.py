r"""
该 ``distributions`` 统计分布包中含有可自定义参数的概率分布和采样函数.

当概率密度函数对其参数可微时, 可以使用
:meth:`~torch.distributions.Distribution.log_prob` 方法来实施梯度方法 Policy Gradient.
它的一个基本方法是REINFORCE规则:

.. math::

    \Delta\theta  = \alpha r \frac{\partial\log p(a|\pi^\theta(s))}{\partial\theta}

这其中 :math:`\theta` 是参数, :math:`\alpha` 是学习率, :math:`r` 是奖惩,  :math:`p(a|\pi^\theta(s))`
是在策略 :math:`\pi^\theta` 中从 :math:`s` 状态下采取 :math:`a` 行动的概率. 

在实践中, 我们要从神经网络的输出中采样选出一个行动, 在某个环境中应用该行动, 然后
使用 ``log_prob`` 函数来构造一个等价的损失函数. 请注意, 这里我们使用了负号, 因为优化器使用
是是梯度下降法, 然而上面的REINFORCE规则是假设了梯度上升情形. 如下所示是在多项式分布下
实现REINFORCE的代码::

    probs = policy_network(state)
    # NOTE: 等同于多项式分布
    m = Categorical(probs)
    action = m.sample()
    next_state, reward = env.step(action)
    loss = -m.log_prob(action) * reward
    loss.backward()
"""
import math
from numbers import Number
import torch


__all__ = ['Distribution', 'Bernoulli', 'Categorical', 'Normal']


class Distribution(object):
    r"""
    Distribution是概率分布的抽象基类.
    """

    def sample(self):
        """
        生成一个样本, 如果分布参数有多个, 就生成一批样本.
        """
        raise NotImplementedError

    def sample_n(self, n):
        """
        生成n个样本, 如果分布参数有多个, 就生成n批样本.
        """
        raise NotImplementedError

    def log_prob(self, value):
        """
        返回在 `value` 处的概率密度函数的对数.

        Args:
            value (Tensor or Variable):（基类的参数,没有实际用处）
        """
        raise NotImplementedError


class Bernoulli(Distribution):
    r"""
    创建以 `probs` 为参数的伯努利分布.

    样本是二进制的 (0或1). 他们以p的概率取值为1, 以 (1 - p) 的概率取值为0.
    
    例:

        >>> m = Bernoulli(torch.Tensor([0.3]))
        >>> m.sample()  # 30% chance 1; 70% chance 0
         0.0
        [torch.FloatTensor of size 1]

    Args:
        probs (Tensor or Variable): 采样到 `1` 的概率
    """

    def __init__(self, probs):
        self.probs = probs

    def sample(self):
        return torch.bernoulli(self.probs)

    def sample_n(self, n):
        return torch.bernoulli(self.probs.expand(n, *self.probs.size()))

    def log_prob(self, value):
        # compute the log probabilities for 0 and 1
        log_pmf = (torch.stack([1 - self.probs, self.probs])).log()

        # evaluate using the values
        return log_pmf.gather(0, value.unsqueeze(0).long()).squeeze(0)


class Categorical(Distribution):
    r"""
    创建以 `probs` 为参数的类别分布.

    .. note::
    它和 ``multinomial()`` 采样的分布是一样的.

    样本是来自 "0 ... K-1" 的整数,其中 "K" 是probs.size(-1).

    如果 `probs` 是长度为 `K` 的一维列表,则每个元素是对该索引处的类进行抽样的相对概率.

    如果 `probs` 是二维的,它被视为一批概率向量.

    另见: :func:`torch.multinomial`

    例::

        >>> m = Categorical(torch.Tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
         3
        [torch.LongTensor of size 1]

    Args:
        probs (Tensor or Variable): 事件概率
    """

    def __init__(self, probs):
        if probs.dim() != 1 and probs.dim() != 2:
            # TODO: treat higher dimensions as part of the batch
            raise ValueError("probs must be 1D or 2D")
        self.probs = probs

    def sample(self):
        return torch.multinomial(self.probs, 1, True).squeeze(-1)

    def sample_n(self, n):
        if n == 1:
            return self.sample().expand(1, 1)
        else:
            return torch.multinomial(self.probs, n, True).t()

    def log_prob(self, value):
        p = self.probs / self.probs.sum(-1, keepdim=True)
        if value.dim() == 1 and self.probs.dim() == 1:
            # special handling until we have 0-dim tensor support
            return p.gather(-1, value).log()

        return p.gather(-1, value.unsqueeze(-1)).squeeze(-1).log()


class Normal(Distribution):
    r"""

    创建以 `mean` 和 `std` 为参数的正态分布（也称为高斯分布）.

    例::

        >>> m = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # normally distributed with mean=0 and stddev=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        mean (float or Tensor or Variable): 分布的均值
        std (float or Tensor or Variable): 分布的标准差
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        return torch.normal(self.mean, self.std)

    def sample_n(self, n):
        # cleanly expand float or Tensor or Variable parameters
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())
        return torch.normal(expand(self.mean), expand(self.std))

    def log_prob(self, value):
        # compute the variance
        var = (self.std ** 2)
        log_std = math.log(self.std) if isinstance(self.std, Number) else self.std.log()
        return -((value - self.mean) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))
