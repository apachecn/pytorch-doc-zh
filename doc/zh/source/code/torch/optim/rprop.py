import math
import torch
from .optimizer import Optimizer


class Rprop(Optimizer):
    """实现弹性反向传播算法.

    Args:
    *    params (iterable): 待优化的迭代参数或者是定义了参数组的 dict
    *    lr (float, optional): 学习率 (默认值: 1e-2)
    *    etas (Tuple[float, float], optional): 一对 (etaminus, etaplis), t它们分别是乘法
        的增加和减小的因子 (默认值: (0.5, 1.2))
    *    step_sizes (Tuple[float, float], optional): 允许的一对最小和最大的步长 (默认值: (1e-6, 50))
    """

    def __init__(self, params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50)):
        defaults = dict(lr=lr, etas=etas, step_sizes=step_sizes)
        super(Rprop, self).__init__(params, defaults)

    def step(self, closure=None):
        """进行单步优化.

        Args:
            closure (callable, optional): 一个重新评价模型并返回 loss 的闭包, 对于大多数参数来说是可选的.
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
                    raise RuntimeError('Rprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['prev'] = torch.zeros_like(p.data)
                    state['step_size'] = grad.new().resize_as_(grad).fill_(group['lr'])

                etaminus, etaplus = group['etas']
                step_size_min, step_size_max = group['step_sizes']
                step_size = state['step_size']

                state['step'] += 1

                sign = grad.mul(state['prev']).sign()
                sign[sign.gt(0)] = etaplus
                sign[sign.lt(0)] = etaminus
                sign[sign.eq(0)] = 1

                # update stepsizes with step size updates
                step_size.mul_(sign).clamp_(step_size_min, step_size_max)

                # for dir<0, dfdx=0
                # for dir>=0 dfdx=dfdx
                grad = grad.clone()
                grad[sign.eq(etaminus)] = 0

                # update parameters
                p.data.addcmul_(-1, grad.sign(), step_size)

                state['prev'].copy_(grad)

        return loss
