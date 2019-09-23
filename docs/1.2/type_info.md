# 键入信息

的数值性质的[ `torch.dtype`](tensor_attributes.html#torch.torch.dtype
"torch.torch.dtype")可通过访问任一 `torch.finfo`或 `torch.iinfo`。

## torch.finfo

_class_`torch.``finfo`

    

A`torch.finfo`是表示一个浮点[ `torch.dtype 的数值性质[对象HTG10]
`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")（即`
torch.float32`，`torch.float64`和`torch.float16`）。这类似于[ numpy.finfo
](https://docs.scipy.org/doc/numpy/reference/generated/numpy.finfo.html)。

A`torch.finfo`提供以下属性：

名称

|

类型

|

描述  
  
---|---|---  
  
位

|

INT

|

由类型所占用的位数。  
  
EPS

|

浮动

|

可表示的最小数量，使得`1.0  +  EPS  ！=  1.0`。  
  
最大

|

float

|

最大可表示数。  
  
分

|

float

|

可表示的最小数目（典型`-max`）。  
  
小

|

float

|

最小的正表示数。  
  
注意

的构造 `torch.finfo`可以被称为无参数，在这种情况下，对于缺省pytorch D类创建的类（由[ [作为返回HTG7]
torch.get_default_dtype（） ](torch.html#torch.get_default_dtype
"torch.get_default_dtype")）。

## torch.iinfo

_class_`torch.``iinfo`

    

A`torch.iinfo`是表示一个整数[ `torch.dtype  [HTG10的数值属性的对象]
`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype")（即`
torch.uint8`，`torch.int8`，`torch.int16`，`torch.int32`和`torch.int64
`）。这类似于[ numpy.iinfo
](https://docs.scipy.org/doc/numpy/reference/generated/numpy.iinfo.html)。

A`torch.iinfo`提供以下属性：

Name

|

Type

|

Description  
  
---|---|---  
  
bits

|

int

|

The number of bits occupied by the type.  
  
max

|

int

|

The largest representable number.  
  
min

|

int

|

最小可表示数。  
  
[Next ![](_static/images/chevron-right-orange.svg)](sparse.html
"torch.sparse") [![](_static/images/chevron-right-orange.svg)
Previous](tensor_attributes.html "Tensor Attributes")

* * *

©版权所有2019年，Torch 贡献者。