import torch
from .optimizer import Optimizer


class Adamax(Optimizer):
    """实现 Adamax 算法 ( Adam 的一种基于无穷范数的变种).

    它在 `Adam: A Method for Stochastic Optimization`__ 中被提出.

    Args:
    *    params (iterable): 迭代的优化参数或者以字典的形式定义参数组.
    *    lr (float, optional): 学习率 (默认值: 2e-3)
    *    betas (Tuple[float, float], optional): 用来计算梯度和平方梯度的系数
    *    eps (float, optional): 增加分母来确保数值稳定性 (默认值: 1e-8)
    *    weight_decay (float, optional): 权重衰减 (L2 正则化) (默认值: 0)

    __ https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adamax, self).__init__(params, defaults)

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
                    raise RuntimeError('Adamax does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_inf'] = torch.zeros_like(p.data)

                exp_avg, exp_inf = state['exp_avg'], state['exp_inf']
                beta1, beta2 = group['betas']
                eps = group['eps']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # Update the exponentially weighted infinity norm.
                norm_buf = torch.cat([
                    exp_inf.mul_(beta2).unsqueeze(0),
                    grad.abs().add_(eps).unsqueeze_(0)
                ], 0)
                torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))

                bias_correction = 1 - beta1 ** state['step']
                clr = group['lr'] / bias_correction

                p.data.addcdiv_(-clr, exp_avg, exp_inf)

        return loss
