import torch

from .optimizer import Optimizer


class Adadelta(Optimizer):
    """实施 Adadelta 算法.

    它在 `ADADELTA: 一种可调节学习率的方法`__ 中提出

    Args:
    *    params (iterable): 通过参数迭代去优化或者字典的形式定义参数组.
    *    rho (float, optional): 用来计算平均平方梯度的系数(默认值: 0.9)
    *    eps (float, optional): 增加分母来确保数值稳定性(默认值: 1e-6)
    *    lr (float, optional): 在将 delta 应用于参数之前对其进行系数的缩放(默认值: 1.0)
    *    weight_decay (float, optional): 权重衰减 (L2正则化) (默认值: 0)

    __ https://arxiv.org/abs/1212.5701
    """

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super(Adadelta, self).__init__(params, defaults)

    def step(self, closure=None):
        """实行单步优化. 

        Args:
            closure (callable, optional): 重新评估模型并返回误差损失的闭包.
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
                    raise RuntimeError('Adadelta does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    state['acc_delta'] = torch.zeros_like(p.data)

                square_avg, acc_delta = state['square_avg'], state['acc_delta']
                rho, eps = group['rho'], group['eps']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(rho).addcmul_(1 - rho, grad, grad)
                std = square_avg.add(eps).sqrt_()
                delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
                p.data.add_(-group['lr'], delta)
                acc_delta.mul_(rho).addcmul_(1 - rho, delta, delta)

        return loss
