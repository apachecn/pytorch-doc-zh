from collections import defaultdict

import torch
from copy import deepcopy
from itertools import chain
from torch.autograd import Variable

required = object()


class Optimizer(object):
    """优化器的基类.

    Args:
    *    params (iterable): :class:`Variable` 或 :class:`dict` 的迭代, 指定了应该优化哪些参数.
    *    defaults: (dict): 包含了优化选项默认值的字典(一个参数组没有指定的参数选项将会使用默认值).
    """

    def __init__(self, params, defaults):
        self.defaults = defaults

        if isinstance(params, Variable) or torch.is_tensor(params):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Variables or dicts, but got " +
                            torch.typename(params))

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __getstate__(self):
        return {
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def state_dict(self):
        """以 :class:`dict` 的形式返回优化器的状态.

        它包含两部分内容:

        * state - 一个包含当前优化状态的字典（dict）, 字典里的内容因优化器的不同而变换.
        * param_groups - 一个包含所有参数组的字典（dict）.
        """
        # Save ids instead of Variables
        def pack_group(group):
            packed = {k: v for k, v in group.items() if k != 'params'}
            packed['params'] = [id(p) for p in group['params']]
            return packed
        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use ids as keys
        packed_state = {(id(k) if isinstance(k, Variable) else k): v
                        for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        """加载优化器状态.

        Args:
            state_dict (dict): 优化器状态. 是调用 :meth:`state_dict` 时所返回的对象.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain(*(g['params'] for g in saved_groups)),
                      chain(*(g['params'] for g in groups)))}
        state = defaultdict(
            dict, {id_map.get(k, k): v for k, v in state_dict['state'].items()})

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if p.grad.volatile:
                        p.grad.data.zero_()
                    else:
                        data = p.grad.data
                        p.grad = Variable(data.new().resize_as_(data).zero_())

    def step(self, closure):
        """进行单次优化(参数更新).

        Args:
            closure (callable): 一个重新评价模型并返回 loss 的闭包大多数优化器可选择.
        """
        raise NotImplementedError

    def add_param_group(self, param_group):
        """增加一组参数到 :class:`Optimizer` 的 `param_groups` 里面.

        当微调一个预训练好的网络作为冻结层时是有用的, 它能够使用可训练的和可增加的参数到 :class:`Optimizer` 作为一个训练预处理.

        Args:
            param_group (dict): 指定这一组中具有特殊优化选项的那些 Variables 能够被优化.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, Variable):
            param_group['params'] = [params]
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, Variable):
                raise TypeError("optimizer can only optimize Variables, "
                                "but one of the params is " + torch.typename(param))
            if not param.requires_grad:
                raise ValueError("optimizing a parameter that doesn't require gradients")
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Variable")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)
