import torch
from .optimizer import Optimizer


class RMSprop(Optimizer):
    """实现 RMSprop 算法.

    由 G. Hinton 在此提出
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    中心版本首次出现在 `Generating Sequences With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    算法:
    *    params (iterable): 待优化的迭代参数或者是定义了参数组的 dict
    *    lr (float, optional): 学习率 (默认值: 1e-2)
    *    momentum (float, optional): 动量因子 (默认值: 0)
    *    alpha (float, optional): 平滑常量 (default: 0.99)
    *    eps (float, optional): 为了增加数值计算的稳定性而加到分母里的项 (默认值: 1e-8)
    *    centered (bool, optional) : 如果为 ``True``, 计算 RMSProp 的中值, 并且用它的方差预测值对梯度进行归一化
    *    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
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
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.data.add_(-group['lr'], buf)
                else:
                    p.data.addcdiv_(-group['lr'], grad, avg)

        return loss
