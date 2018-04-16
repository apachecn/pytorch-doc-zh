"""
Weight Normalization from https://arxiv.org/abs/1602.07868
"""
from torch.nn.parameter import Parameter


def _norm(p, dim):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return _norm(p.transpose(0, dim), 0).transpose(0, dim)


class WeightNorm(object):
    def __init__(self, name, dim):
        self.name = name
        self.dim = dim

    def compute_weight(self, module):
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        return v * (g / _norm(v, self.dim))

    @staticmethod
    def apply(module, name, dim):
        fn = WeightNorm(name, dim)

        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(name + '_g', Parameter(_norm(weight, dim).data))
        module.register_parameter(name + '_v', Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_g']
        del module._parameters[self.name + '_v']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def weight_norm(module, name='weight', dim=0):
    r"""将权重归一化应用于给定模块中的指定参数. .

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    权重归一化是将权重张量的大小和方向分离的再参数化. 该函数会用两个参数代替 `name` (e.g. "weight")所指定的参数. 在新的参数中, 一个指定参数的大小 (e.g. "weight_g"), 一个指定参数的方向. 权重归一化是通过一个钩子实现的, 该钩子会在 `~Module.forward` 的每次调用之前根据大小和方向(两个新参数)重新计算权重张量.

    默认情况下, `dim=0`, 范数会在每一个输出的 channel/plane 上分别计算. 若要对整个权重张量计算范数, 使用 `dim=None`.

    参见 https://arxiv.org/abs/1602.07868

    Args:
        module (nn.Module): 给定的 module
        name (str, optional): 权重参数的 name
        dim (int, optional): 进行范数计算的维度

    Returns:
        添加了权重归一化钩子的原 module

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        Linear (20 -> 40)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    WeightNorm.apply(module, name, dim)
    return module


def remove_weight_norm(module, name='weight'):
    r"""从模块中移除权重归一化/再参数化.

    Args:
        module (nn.Module): 给定的 module
        name (str, optional): 权重参数的 name

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))
