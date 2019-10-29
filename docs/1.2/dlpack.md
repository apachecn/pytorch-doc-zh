# torch.utils.dlpack

`torch.utils.dlpack.from_dlpack`( _dlpack_ ) → Tensor

解码DLPack到张量。

    Parameters
        **dlpack** - 与dltensor一个PyCapsule对象

张量将与dlpack表示的对象共享存储器。请注意，每个dlpack只能使用一次消耗。


`torch.utils.dlpack.to_dlpack`( _tensor_ ) → PyCapsule

返回表示张量DLPack。

    Parameters

        **tensor** - 要导出的张量

该dlpack共享内存的张量。请注意，每个dlpack只能使用一次消耗。
