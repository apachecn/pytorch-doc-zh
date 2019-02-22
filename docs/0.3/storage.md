# torch.Storage

> 译者：@FanXing
> 
> 校对者：[@Timor](https://github.com/timors)

一个 `torch.Storage` 是一个单一数据类型的连续一维数组 .

每个 [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor") 都有一个对应的相同数据类型的存储 .

```py
class torch.FloatStorage
```

```py
byte()
```

将此存储转换为 byte 类型

```py
char()
```

将此存储转换为 char 类型

```py
clone()
```

返回此存储的一个副本

```py
copy_()
```

```py
cpu()
```

如果当前此存储不在CPU上 , 则返回一个它的CPU副本 .

```py
cuda(device=None, async=False)
```

返回此对象在 CUDA 内存中的一个副本 .

如果此对象已经在 CUDA 内存中并且在正确的设备上 , 那么不会执行复制操作 , 直接返回原对象 .

参数：

*   `device (int)` – 目标 GPU 的 id . 默认值是当前设备 .
*   `async (bool)` – 如果为 `True` 并且源位于锁定内存中 , 则副本相对于主机是异步的 . 否则此参数不起效果 .



```py
data_ptr()
```

```py
double()
```

将此存储转换为 double 类型

```py
element_size()
```

```py
fill_()
```

```py
float()
```

将此存储转换为 float 类型

```py
from_buffer()
```

```py
from_file(filename, shared=False, size=0) → Storage
```

如果 shared 为 True , 那么内存将会在所有进程间共享 . 所有的更改都会被写入文件 . 如果 shared 为 False , 那么对于内存的修改 , 则不会影响到文件 .

size 是存储中所包含的元素个数 . 如果 shared 为 False 则文件必须包含至少 `size * sizeof(Type)` 字节 ( `Type` 是所存储的类型) . 如果 shared 为 True , 文件会在需要的时候被创建 .

参数：

*   `filename (str)` – 要映射到的文件名
*   `shared (bool)` – 是否共享内存
*   `size (int)` – 存储中包含元素的个数



```py
half()
```

将此存储转换为 half 类型

```py
int()
```

将此存储转换为 int 类型

`is_cuda` _= False_

```py
is_pinned()
```

```py
is_shared()
```

`is_sparse` _= False_

```py
long()
```

将此存储转换为 long 类型

```py
new()
```

```py
pin_memory()
```

如果此存储当前未被锁定 , 则将它复制到锁定内存中 .

```py
resize_()
```

```py
share_memory_()
```

将存储移动到共享内存中 .

这对于已经存在于共享内存中的存储或者 CUDA 存储无效 , 它们不需要移动就能在进程间共享 . 共享内存中的存储不能调整大小 .

返回值：`self`

```py
short()
```

将此存储转换为 short 类型

```py
size()
```

```py
tolist()
```

返回一个包含此存储中的元素的列表

```py
type(new_type=None, async=False)
```

如果没有指定 `new_type` 则返回该类型 , 否则将此对象转换为指定类型 .

如果已经是正确的类型 , 则不执行复制并直接返回原对象 .

参数：

*   `new_type (type 或 string)` – 期望的类型
*   `async (bool)` – 如果为 `True` , 并且源在锁定内存中而目标在GPU中 , 则副本将与主机异步执行 , 反之亦然 . 否则此参数不起效果 .

