# torch.utils.model_zoo

移动到 torch.hub 。

`torch.utils.model_zoo.``load_url`( _url_ , _model_dir=None_ ,
_map_location=None_ , _progress=True_ )

    

加载在给定的URLTorch 序列化对象。

如果对象已存在于 model_dir ，它的反序列化和返回。的URL的文件名部分应遵循命名惯例`的文件名 - & LT。; SHA256 & GT ;
EXT`其中`& LT ; SHA256 & GT ;
`是该文件的内容的散列SHA256的前八个或多个数字。哈希用于确保唯一的名称，并验证该文件的内容。

的 model_dir 默认值是`$ TORCH_HOME /检查点 `其中环境变量`$ TORCH_HOME`默认为`$
XDG_CACHE_HOME /Torch  [HTG13。 `$ XDG_CACHE_HOME`遵循了Linux
filesytem布局的X设计组规范，带有默认值HTG18] 〜/ .cache`如果没有设置。

Parameters

    

  * **URL** （ _串_ ） - 对象的URL下载

  * **model_dir** （ _串_ _，_ _可选_ ） - 目录中保存对象

  * **map_location** （ _可选_ ） - 一个功能或一个字典指定如何重新映射的存储位置（参见torch.load）

  * **进展** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 是否要显​​示进度条到stderr

例

    
    
    >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
    

[Next ![](_static/images/chevron-right-orange.svg)](tensorboard.html
"torch.utils.tensorboard") [![](_static/images/chevron-right-orange.svg)
Previous](dlpack.html "torch.utils.dlpack")

* * *

©版权所有2019年，Torch 贡献者。