# torchvision.utils

> 译者：[@那伊抹微笑](https://github.com/wangyangting)、@dawenzi123、[@LeeGeong](https://github.com/LeeGeong)、@liandongze
> 
> 校对者：[@咸鱼](https://github.com/Watermelon233)

```py
torchvision.utils.make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
```

制作一个图形网格.

参数：

*   `tensor (Tensor 或 list)` – 给定 4D mini-batch Tensor 形状为 (B x C x H x W) 或者一个同样形状的 list of images.
*   `nrow (int, 可选)` – 网格每一行显示的image数量. 最后网格的形状是 (B / nrow, nrow). 默认是 8.
*   `padding (int, 可选)` – 填充的数量. 默认为 2.
*   `normalize (bool, 可选)` – 如果值为True,通过减去最小像素值并除以最大的像素值的方法, 把图像的范围变为 (0, 1),此过程为归一化处理.
*   `range (tuple, 可选)` – tuple (min, max) 这里 min 和 max 都是数字, 这些数字是用来规范 image的. 默认情况下, min 和 max 是从 tensor 里计算出来的.
*   `scale_each (bool, 可选)` – 如果值为True, 每个image独立规范化, 而不是根据所有image的像素最大最小值来归一化.
*   `pad_value (float, 可选)` – 填充像素的值.



示例：

请参阅 [这里](https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91) 的手册

```py
torchvision.utils.save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
```

将一个给定的 Tensor 保存为 image(图像）文件.

参数：

*   `tensor (Tensor 或 list)` – 被保存的图片. 如果给定的是 mini-batch tensor, 通过调用 `make_grid` 将 tensor 保存为网格图像.
*   `**kwargs` – 其它参数文档在 `make_grid` 中.

