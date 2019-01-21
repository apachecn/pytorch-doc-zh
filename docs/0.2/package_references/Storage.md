# torch.Storage

一个`torch.Storage`是一个单一数据类型的连续一维数组。

每个`torch.Tensor`都有一个对应的、相同数据类型的存储。

```python
class torch.FloatStorage
```

#### byte()  
将此存储转为byte类型

#### char()  
将此存储转为char类型

#### clone()  
返回此存储的一个副本

#### copy_()  

#### cpu()  
如果当前此存储不在CPU上，则返回一个它的CPU副本

#### cuda(*device=None, async=False*)
返回此对象在CUDA内存中的一个副本。  
如果此对象已在CUDA内存中且在正确的设备上，那么不会执行复制操作，直接返回原对象。
  
**参数：**

- **device** (*[int]()*) - 目标GPU的id。默认值是当前设备。
- **async** (*[bool]()*) -如果值为True，且源在锁定内存中，则副本相对于宿主是异步的。否则此参数不起效果。

#### data_ptr()

#### double()
将此存储转为double类型

#### element_size()

#### fill_()

#### float()
将此存储转为float类型

#### from_buffer()

#### half()
将此存储转为half类型

#### int()
将此存储转为int类型

#### is_cuda = *False* 

#### is_pinned()

#### is_shared()

#### is_sparse = *False*

#### long()
将此存储转为long类型

#### new()  

#### pin_memory()  
如果此存储当前未被锁定，则将它复制到锁定内存中。

#### resize_()

#### share_memory_()
将此存储移动到共享内存中。  
对于已经在共享内存中的存储或者CUDA存储，这是一条空指令，它们不需要移动就能在进程间共享。共享内存中的存储不能改变大小。  
返回：self

#### short()
将此存储转为short类型  

#### size()

#### tolist()  
返回一个包含此存储中元素的列表

#### type(*new_type=None, async=False*)  
将此对象转为指定类型。  
如果已经是正确类型，不会执行复制操作，直接返回原对象。  

**参数：**  

- **new_type** (*[type]() or [string]()*) -需要转成的类型
- **async** (*[bool]()*)  -如果值为True，且源在锁定内存中而目标在GPU中——或正好相反，则复制操作相对于宿主异步执行。否则此参数不起效果。