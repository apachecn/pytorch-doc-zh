import math
import torch
from .optimizer import Optimizer


class ASGD(Optimizer):
    """实现平均随机梯度下降算法.

    它在 `Acceleration of stochastic approximation by averaging`_ 中被提出

    Args:
    *    params (iterable): 迭代的优化参数或者以字典的形式定义参数组
    *    lr (float, optional): 学习率 (默认值: 1e-2)
    *    lambd (float, optional): 衰减期 (默认值: 1e-4)
    *    alpha (float, optional): eta 更新的权重 (默认值: 0.75)
    *    t0 (float, optional): 指明在哪一次开始平均化 (默认值: 1e6)
    *    weight_decay (float, optional): 权重衰减 (L2 正则化) (默认值: 0)

    .. _Acceleration of stochastic approximation by averaging:
        http://dl.acm.org/citation.cfm?id=131098
    """

    def __init__(self, params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0):
        defaults = dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0,
                        weight_decay=weight_decay)
        super(ASGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """进行单步优化. 

        Args:
            closure (callable, optional): 一个重新评价模型并返回误差的闭包.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ASGD does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['eta'] = group['lr']
                    state['mu'] = 1
                    state['ax'] = torch.zeros_like(p.data)

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # decay term
                p.data.mul_(1 - group['lambd'] * state['eta'])

                # update parameter
                p.data.add_(-state['eta'], grad)

                # averaging
                if state['mu'] != 1:
                    state['ax'].add_(p.data.sub(state['ax']).mul(state['mu']))
                else:
                    state['ax'].copy_(p.data)

                # update eta and mu
                state['eta'] = (group['lr'] /
                                math.pow((1 + group['lambd'] * group['lr'] * state['step']), group['alpha']))
                state['mu'] = 1 / max(1, state['step'] - group['t0'])

        return loss
