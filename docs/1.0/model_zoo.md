

# torch.utils.model_zoo

> 译者：[BXuan694](https://github.com/BXuan694)

```py
torch.utils.model_zoo.load_url(url, model_dir=None, map_location=None, progress=True)
```

由给定URL加载Torch序列化对象。

如果该对象已经存在于`model_dir`中，将被反序列化并返回。URL的文件名部分应该遵循约定`filename-<sha256>.ext`，其中`<sha256>`是文件内容的SHA256哈希的前八位或更多位数。(哈希用于确保唯一的名称并验证文件的内容）

`model_dir`默认为`$TORCH_HOME/models`，其中`$TORCH_HOME`默认是`~/.torch`。如果不需要默认目录，可以通过环境变量`$TORCH_MODEL_ZOO`指定其它的目录。

参数：

*   **url**(_string_）– 要下载的对象的URL链接
*   **model_dir**(_string_ _,_ _可选_）– 保存下载对象的目录
*   **map_location**(_可选_）– 函数或字典，指定如何重新映射存储位置(见torch.load）
*   **progress**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_）– 是否向标准输出展示进度条



示例

```py
>>> state_dict = torch.utils.model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
```

