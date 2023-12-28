# 类型信息 [¶](#type-in​​fo "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/type_info>
>
> 原始地址：<https://pytorch.org/docs/stable/type_info.html>


 [`torch.dtype`](tensor_attributes.html#torch.dtype "torch.dtype") 的数值属性可以通过 [`torch.finfo`](#torch.torch.finfo "torch.torch.finfo") 或 [`torch.iinfo`](#torch.torch.iinfo "torch.torch.iinfo") 。


## torch.finfo [¶](#torch-finfo "此标题的永久链接")


> *Class* `torch.finfo` [¶](#torch.torch.finfo "此定义的永久链接")


 [`torch.finfo`](#torch.torch.finfo "torch.torch.finfo") 是一个表示浮点数值属性的对象 [`torch.dtype`](tensor_attributes.html#torch.dtype "torch.dtype") ，(即 `torch.float32` 、 `torch.float64` 、 `torch.float16` 和 `torch.bfloat16` )。这类似于 [numpy.finfo](https://docs.scipy.org/doc/numpy/reference/generated/numpy.finfo.html) 。


 [`torch.finfo`](#torch.torch.finfo "torch.torch.finfo") 提供以下属性：


| 	 Name	  | 	 Type	  | 	 Description	  |
| --- | --- | --- |
| 	 bits	  | 	 int	  | 	 The number of bits occupied by the type.	  |
| 	 eps	  | 	 float	  | 	 The smallest representable number such that	 `1.0	 		 +	 		 eps	 		 !=	 		 1.0`	.	  |
| 	 max	  | 	 float	  | 	 The largest representable number.	  |
| 	 min	  | 	 float	  | 	 The smallest representable number (typically	 `-max`	 ).	  |
| 	 tiny	  | 	 float	  | 	 The smallest positive normal number. Equivalent to	 `smallest_normal`	.	  |
| 	 smallest_normal	  | 	 float	  | 	 The smallest positive normal number. See notes.	  |
| 	 resolution	  | 	 float	  | 	 The approximate decimal resolution of this type, i.e.,	 `10**-precision`	.	  |




!!! note "笔记"

    [`torch.finfo`](#torch.torch.finfo "torch.torch.finfo") 的构造函数可以在不带参数的情况下调用，在这种情况下，该类是为 pytorch 默认 dtype（由 [torch.get_default_dtype()](https://pytorch.org/docs/stable/generated/torch.get_default_dtype.html#torch.get_default_dtype "torch.get_default_dtype") 返回）创建的。

!!! note "笔记"

    smallest_normal 返回最小的正规数，但还有更小的次正规数。 请参阅 <https://en.wikipedia.org/wiki/Denormal_number> 了解更多信息。


## torch.iinfo [¶](#torch-iinfo "此标题的永久链接")


> *CLASS* `torch.iinfo` [¶](#torch.torch.iinfo "此定义的永久链接")


 [`torch.iinfo`](#torch.torch.iinfo "torch.torch.iinfo") 是一个表示整数数值属性的对象 [`torch.dtype`](tensor_attributes.html#torch.dtype " torch.dtype”)(即 `torch.uint8` 、 `torch.int8` 、 `torch.int16` 、 `torch.int32` 和 `torch.int64` )。这类似于 [numpy.iinfo](https://docs.scipy.org/doc/numpy/reference/generated/numpy.iinfo.html) 。


 [`torch.iinfo`](#torch.torch.iinfo "torch.torch.iinfo") 提供以下属性：


| 	 Name	  | 	 Type	  | 	 Description	  |
| --- | --- | --- |
| 	 bits	  | 	 int	  | 	 The number of bits occupied by the type.	  |
| 	 max	  | 	 int	  | 	 The largest representable number.	  |
| 	 min	  | 	 int	  | 	 The smallest representable number.	  |