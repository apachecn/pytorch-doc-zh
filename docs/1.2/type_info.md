

# 数据类型信息 

> 译者：[冯宝宝](https://github.com/PEGASUS1993)  

可以通过[`torch.finfo`](#torch.torch.finfo "torch.torch.finfo") 或 [`torch.iinfo`](#torch.torch.iinfo "torch.torch.iinfo")访问[`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")的数字属性。  

## torch.finfo  

```py
class torch.finfo
``` 

 [`torch.finfo`](#torch.torch.finfo "torch.torch.finfo") 是一个用来表示浮点[`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")的数字属性的对象(即`torch.float32`，`torch.float64`和`torch.float16`）。 这类似于 [numpy.finfo](https://docs.scipy.org/doc/numpy/reference/generated/numpy.finfo.html)。  

[`torch.finfo`](#torch.torch.finfo "torch.torch.finfo") 提供以下属性:  

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| bits | 整型　int |数据类型占用的位数 |
| eps | 浮点型float | 可表示的最小数字，使得1.0 + eps！= 1.0|
| max | 浮点型float | 可表示的最大数字|
| tiny | 浮点型float |可表示的最小正数 |  

注意  

在使用pytorch默认dtype创建类(由`torch.get_default_dtype(）`返回）的情况下，构造的 [`torch.finfo`](#torch.torch.finfo "torch.torch.finfo") 函数可以不带参数被调用。  

##  torch.iinfo  

```py
class torch.iinfo
```  

 [`torch.iinfo`](#torch.torch.iinfo "torch.torch.iinfo")是一个用来表示整数[`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") 的数字属性的对象，(即`torch.uint8`，`torch.int8`，`torch.int16`，`torch.int32`和`torch.int64`）。 这与[numpy.iinfo](https://docs.scipy.org/doc/numpy/reference/generated/numpy.iinfo.html)类似。  

[`torch.iinfo`](#torch.torch.iinfo "torch.torch.iinfo") 提供以下属性：   

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| bits | 整型| 数据类型占用的位数 |
| max | 整型 | 可表示的最大数字 |
 

