# torch.utils.dlpack

> 译者：[kunwuz](https://github.com/kunwuz)

```py
torch.utils.dlpack.from_dlpack(dlpack) → Tensor
```

将DLPack解码成Tensor张量。

| 参数: | **dlpack** – 一个有着dltensor张量的PyCapsule对象 |
| --- | --- |

这个张量会与dlpack对象共享存储空间。注意每个dlpack对象只能使用一次。

```py
torch.utils.dlpack.to_dlpack(tensor) → PyCapsule
```

返回一个表示张量的DLPack。

| 参数: | **tensor** –一个用来输出的tensor张量 |
| --- | --- |

这个张量会与dlpack对象共享存储空间。注意每个dlpack对象只能使用一次。

