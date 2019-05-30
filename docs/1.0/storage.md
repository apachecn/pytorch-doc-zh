# torch.Storage

> 译者：[yuange250](https://github.com/yuange250)

`torch.Storage` 跟绝大部分基于连续存储的数据结构类似，本质上是一个单一数据类型的一维连续数组(array)。

每一个 [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor") 都有一个与之相对应的`torch.Storage`对象，两者存储数据的数据类型(data type)保持一致。

下面以数据类型为float的` torch.FloatStorage ` 为例介绍一下`torch.Storage`的成员函数。
```py
class torch.FloatStorage
```

```py
byte()
```

byte()函数可以将此storage对象的数据类型转换为byte


```py
char()
```
char()函数可以将此storage对象的数据类型转换为char


```py
clone()
```

clone()函数可以返回一个此storage对象的复制 

```py
copy_()
```

```py
cpu()
```

如果此storage对象一开始不在cpu设备上，调用cpu()函数返回此storage对象的一个cpu上的复制

```py
cuda(device=None, non_blocking=False, **kwargs)
```

cuda()函数返回一个存储在CUDA内存中的复制，其中device可以指定cuda设备。
但如果此storage对象早已在CUDA内存中存储，并且其所在的设备编号与cuda()函数传入的device参数一致，则不会发生复制操作，返回原对象。

cuda()函数的参数信息: 

*   **device** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 指定的GPU设备id. 默认为当前设备，即 [`torch.cuda.current_device()`](cuda.html#torch.cuda.current_device "torch.cuda.current_device")的返回值。
*   **non_blocking** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果此参数被设置为True, 并且此对象的资源存储在固定内存上(pinned memory)，那么此cuda()函数产生的复制将与host端的原storage对象保持同步。否则此参数不起作用。
*   ****kwargs** – 为了保证兼容性，也支持async参数，此参数的作用与no_blocking参数的作用完全相同，旧版本的遗留问题之一。



```py
data_ptr()
```

```py
double()
```

double()函数可以将此storage对象的数据类型转换为double
```py
element_size()
```

```py
fill_()
```

```py
float()
```

float()函数可以将此storage对象的数据类型转换为float

```py
static from_buffer()
```

```py
static from_file(filename, shared=False, size=0) → Storage
```

对于from_file()函数，如果`shared`参数被设置为`True`， 那么此部分内存可以在进程间共享，任何对storage对象的更改都会被写入存储文件。 如果 `shared` 被置为 `False`, 那么在内存中对storage对象的更改则不会影响到储存文件中的数据。

`size` 参数是此storage对象中的元素个数。 如果`shared`被置为`False`, 那么此存储文件必须要包含`size * sizeof(Type)`字节大小的数据 (`Type`是此storage对象的数据类型)。 如果 `shared` 被置为 `True`，那么此存储文件只有在需要的时候才会被创建。

from_file()函数的参数： 

*   **filename** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")) – 对应的存储文件名
*   **shared** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 是否共享内存
*   **size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 此storage对象中的元素个数



```py
half()
```

half()函数可以将此storage对象的数据类型转换为half


```py
int()
```

int()函数可以将此storage对象的数据类型转换为int

```py
is_cuda = False
```

```py
is_pinned()
```

```py
is_shared()
```

```py
is_sparse = False
```

```py
long()
```

long()函数可以将此storage对象的数据类型转换为long

```py
new()
```

```py
pin_memory()
```

如果此storage对象还没有被存储在固定内存中，则pin_memory()函数可以将此storage对象存储到固定内存中

```py
resize_()
```

```py
share_memory_()
```

share_memory_()函数可以将此storage对象转移到共享内存中。

对于早已在共享内存中的storage对象，这个操作无效；对于存储在CUDA设备上的storage对象，无需移动即可实现此类对象在进程间的共享，所以此操作对于它们来说也无效。

在共享内存中存储的storage对象无法被更改大小。

share_memory_()函数返回值: self

```py
short()
```

short()函数可以将此storage对象的数据类型转换为short

```py
size()
```

```py
tolist()
```

tolist()函数可以返回一个包含此storage对象所有元素的列表
```py
type(dtype=None, non_blocking=False, **kwargs)
```

如果函数调用时没有提供`dtype`参数，则type()函数的调用结果是返回此storage对象的数据类型。如果提供了此参数，则将此storage对象转化为此参数指定的数据类型。如果所提供参数所指定的数据类型与当前storage对象的数据类型一致，则不会进行复制操作，将原对象返回。

type()函数的参数信息: 

*   **dtype** ([_type_](https://docs.python.org/3/library/functions.html#type "(in Python v3.7)") _or_ _string_) – 想要转化为的数据类型
*   **non_blocking** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果此参数被设置为True, 并且此对象的资源存储在固定内存上(pinned memory)，那么此cuda()函数产生的复制将与host端的原storage对象保持同步。否则此参数不起作用。
*   ****kwargs** – 为了保证兼容性，也支持async参数，此参数的作用与no_blocking参数的作用完全相同，旧版本的遗留问题之一 (已经被deprecated)。



