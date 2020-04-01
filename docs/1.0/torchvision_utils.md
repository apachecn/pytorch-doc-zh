

# torchvision.utils

> 译者：[BXuan694](https://github.com/BXuan694)

```py
torchvision.utils.make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
```

把图片排列成网格形状。

 
参数： 

*   **tensor**([_Tensor_](../tensors.html#torch.Tensor "torch.Tensor") _或_ [_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")）– 四维批(batch）Tensor或列表。如果是Tensor，其形状应是(B x C x H x W）；如果是列表，元素应为相同大小的图片。
*   **nrow**([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选_）– 最终展示的图片网格中每行摆放的图片数量。网格的长宽应该是(B / nrow, nrow）。默认是8。
*   **padding**([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _可选_）– 扩展填充的像素宽度。默认是2。
*   **normalize**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_）– 如果设置为True，通过减去最小像素值然后除以最大像素值，把图片移到(0，1）的范围内。
*   **range**([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _可选_）– 元组(min, max），min和max用于对图片进行标准化处理。默认的，min和max由输入的张量计算得到。
*   **scale_each**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_）– 如果设置为True，将批中的每张图片按照各自的最值分别缩放，否则使用当前批中所有图片的最值(min, max)进行统一缩放。
*   **pad_value**([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _可选_）– 扩展填充的像素值。



示例：

请看 [这里](https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91)

```py
torchvision.utils.save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
```

用于把指定的Tensor保存成图片文件。

 
参数：

*   **tensor**([_Tensor_](../tensors.html#torch.Tensor "torch.Tensor") _或_ [_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")）– 需要保存成图片的Tensor。如果Tensor以批的形式给出，则会调用`make_grid`将这些图片保存成网格的形式。
*   ****kwargs** – 其他参数同`make_grid`。

