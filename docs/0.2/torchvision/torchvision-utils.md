# torchvision.utils

## torchvision.utils.make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False)
猜测，用来做 `雪碧图的`(`sprite image`）。

给定 `4D mini-batch Tensor`， 形状为 `(B x C x H x W)`,或者一个`a list of image`，做成一个`size`为`(B / nrow, nrow)`的雪碧图。

- normalize=True ，会将图片的像素值归一化处理

- 如果 range=(min, max)， min和max是数字，那么`min`，`max`用来规范化`image`

- scale_each=True ，每个图片独立规范化，而不是根据所有图片的像素最大最小值来规范化

[Example usage is given in this notebook](https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91)

## torchvision.utils.save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False)

将给定的`Tensor`保存成image文件。如果给定的是`mini-batch tensor`，那就用`make-grid`做成雪碧图，再保存。
