import torch
import math
irange = range


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """制作一个图形网格.

    Args:
        tensor (Tensor 或者 list): 给定 4D mini-batch Tensor 形状为 (B x C x H x W)
            或者一个同样形状的 list of images.
        nrow (int, optional): 网格每一行显示的image数量.
            最后网格的形状是 (B / nrow, nrow). 默认是 8.
        padding (int, optional): 填充的数量. 默认为 2.
        normalize (bool, optional):如果值为True,通过减去最小像素值并除以最大的像素值的方法, 
                把图像的范围变为 (0, 1),此过程为归一化处理.
        range (tuple, optional): tuple (min, max) 这里 min 和 max 都是数字,
            这些数字是用来规范 image的. 默认情况下, min 和 max
            是从 tensor 里计算出来的.
        scale_each (bool, optional): 如果值为True, 每个image独立规范化, 而不是根据所有image的像素最大最小值来归一化.
        pad_value (float, optional): 填充像素的值.

    Example:
        请参阅 `这里 <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_ 的手册

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # 如果tensors是一个列表, 把它转换成一个 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # 选出 image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # 选出 image
        if tensor.size(0) == 1:  # 如果是 单通道, 把它转换成 三通道
            tensor = torch.cat((tensor, tensor, tensor), 0)
        return tensor
    if tensor.dim() == 4 and tensor.size(1) == 1:  # 单通道 images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # 避免修改 tensor 原状态
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, t.min(), t.max())

        if scale_each is True:
            for t in tensor:  # 遍历 mini-batch 的纬度
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    # 把 image 的 mini-batch 放入到网格中
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """将一个给定的 Tensor 保存为 image（图像）文件.

    Args:
        tensor (Tensor or list): 被保存的图片. 如果给定的是 mini-batch tensor,
            通过调用 ``make_grid`` 将 tensor 保存为网格图像.
        **kwargs: 其它参数文档在 ``make_grid`` 中.
    """
    from PIL import Image
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)
