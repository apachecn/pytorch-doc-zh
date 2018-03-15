
def clip_grad_norm(parameters, max_norm, norm_type=2):
    r"""接收一个包含 Variable 的可迭代对象, 对 Variable 的梯度按范数进行裁剪.

    范数是对所有梯度进行计算的, 等价于把所有输入变量的梯度连接成一个向量, 然后对这个向量按范数进行裁剪. 梯度将会被原地修改.

    Arguments:
        parameters (Iterable[Variable]): 一个可迭代对象, 其包含将要进行梯度正规化的 Variable
        max_norm (float or int): 梯度的最大范数
        norm_type (float or int): p 范数(指定 p ). 用 ``'inf'`` 表示无穷范数

    Returns:
        梯度的范数 (视为单个向量的).
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm
