from bisect import bisect_right
from .optimizer import Optimizer


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class LambdaLR(_LRScheduler):
    """将每个参数组的学习速率设置为给定函数的初始LR. 当 last_epoch=-1, 设置出事的 lr 作为 lr.

    Args:
    *    optimizer (Optimizer): 封装好的优化器.
    *    lr_lambda (function or list): 计算给定整数参数历元的乘法因子的函数, 或者一系列的此类函数, 每组的一个都在 optimizer.param_groups 中.
    *    last_epoch (int): 最后一个 epoch 的索引. 默认值: -1.

    Example:
        >>> # Assuming optimizer has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.optimizer = optimizer
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        self.last_epoch = last_epoch
        super(LambdaLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


class StepLR(_LRScheduler):
    """通过 gamma 在每一个 epoch 里面的 step_size 设置每个参数组的初始学习率衰减变量. 当 last_epoch=-1, 设置初始 lr 为 lr.

    Args:
    *    optimizer (Optimizer): 封装好的优化器.
    *    step_size (int): 学习率衰减周期.
    *    gamma (float): 学习率衰减的乘法因子. 默认值: 0.1.
    *    last_epoch (int): 最后一个 epoch 的索引. 默认值: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.5 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]


class MultiStepLR(_LRScheduler):
    """一旦 epoch 的数量达到了一个临界点通过 gamma 在每一个 epoch 里面的 step_size 设置每个参数
    组的初始学习率衰减变量.当 last_epoch=-1, 设置初始 lr 作为 lr.

    Args:
    *    optimizer (Optimizer): 封装好的优化器.
    *    milestones (list): epoch 索引列表. 必须为递增的.
    *    gamma (float): 学习率衰减的乘法因子.
            默认值: 0.1.
    *    last_epoch (int): 最后一个 epoch 的索引. 默认值: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.5 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """通过 gamma 在每一个 epoch 里面的 step_size 设置每个参数组的初始学习率衰减变量 . 
    当 last_epoch=-1, 设置初始 lr 作为 lr.
    Args:
        optimizer (Optimizer): 封装好的优化器.
        gamma (float): 学习率衰减的乘法因子.
        last_epoch (int): 最后一个 epoch 的索引. 默认值: -1.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** self.last_epoch
                for base_lr in self.base_lrs]


class ReduceLROnPlateau(object):
    """当一个指标已经停止提升时减少学习率.模型通常受益于通过一次2-10的学习停止因素减少学习率
     这个调度程序读取一个指标质量 以及看到 'patience' 的数量在一个 epoch 里面如果没有提升, 
     这时学习率已经减小.

    Args:
    *    optimizer (Optimizer): 封装好的优化器.
        mode (str): `min`, `max` 其中一个. 在 `min` 模块下,当质量监测已经
        停止下降时 lr 将被减少; 在 `max` 模块下 当质量监测已经停止上升时 lr 将
        被减少. 默认值: 'min'.
    *    factor (float): 哪个学习率将会被减少的影响因子 . 
    *    new_lr = lr * factor. 默认值: 0.1.
    *    patience (int): epoch 中没有改善的次数, 学习率将会降低. . 默认值: 10.
    *    verbose (bool): 若为 ``True``, 每次更新打印信息到控制台输出. 默认值: ``False``.
    *    threshold (float): 测量新的最佳阈值, 只关注有重大意义的改变. 默认值: 1e-4.
    *    threshold_mode (str):  `rel`, `abs` 中的一个. 在 `rel` 模式下,
            dynamic_threshold = best * ( 1 + threshold ) 在 'max'
            模式下或者在 `min` 模式下 best * ( 1 - threshold ) .
            在 `abs` 模式下, dynamic_threshold = best + threshold 在
            `max` 模式下或者在 `min` 模式下 best - threshold . 默认值: 'rel'.
    *    cooldown (int): lr 已经减少之后去等待最佳的正常操作之前的 epoch 数目. 默认值: 0.
    *    min_lr (float or list): 一个列表的标量.所有参数组或每个组的学习率下限. 默认值: 0.
    *    eps (float): lr 最小的衰减值适应于. 如果新 lr 和旧 lr 之间的差异小于 eps,更新可以忽略. 默认值: 1e-8.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + mode + ' is unknown!')
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            self.is_better = lambda a, best: a < best * rel_epsilon
            self.mode_worse = float('Inf')
        elif mode == 'min' and threshold_mode == 'abs':
            self.is_better = lambda a, best: a < best - threshold
            self.mode_worse = float('Inf')
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            self.is_better = lambda a, best: a > best * rel_epsilon
            self.mode_worse = -float('Inf')
        else:  # mode == 'max' and epsilon_mode == 'abs':
            self.is_better = lambda a, best: a > best + threshold
            self.mode_worse = -float('Inf')
