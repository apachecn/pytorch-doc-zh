import torch
from .optimizer import Optimizer, required


class SGD(Optimizer):
    r"""实现随机梯度下降算法（ momentum 可选）.

    Nesterov 动量基于 `On the importance of initialization and momentum in deep learning`__ 中的公式.

    Args:
    *    params (iterable): 待优化的迭代参数或者是定义了参数组的 dict
    *    lr (float): 学习率
    *    momentum (float, optional): 动量因子 (默认值: 0)
    *    weight_decay (float, optional): 权重衰减 (L2 正则化) (默认值: 0)
    *    dampening (float, optional): 动量的抑制因子 (默认值: 0)
    *    nesterov (bool, optional): 使用 Nesterov 动量 (默认值: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        带有动量 /Nesterov 的 SGD 的实现稍微不同于 Sutskever 等人以及其他框架中的实现.
        考虑动量的具体情况, 更新可以写成

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        其中 p, g, v 和 :math:`\rho` 分别是参数、梯度、速度和动量.

        这跟 Sutskever 等人以及其他框架的实现是相反的, 它们采用这样的更新.

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        Nesterov 的版本也相应的被修改了.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """进行单步优化.

        Args:
            closure (callable, optional): 一个重新评价模型并返回 loss 的闭包, 对于大多数参数来说是可选的.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
