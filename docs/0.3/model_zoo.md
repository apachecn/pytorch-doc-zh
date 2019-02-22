# torch.utils.model_zoo

> 译者：[@之茗](https://github.com/mayuanucas)
> 
> 校对者：[@aleczhang](http://community.apachecn.org/?/people/aleczhang)

```py
torch.utils.model_zoo.load_url(url, model_dir=None, map_location=None)
```

从给定的 URL 处加载 Torch 序列化对象. 如果该对象已经存在于 `model_dir` 中, 则将被反序列化并返回. URL 的文件名部分应该遵循命名约定 `filename-&lt;sha256&gt;.ext` 其中 `&lt;sha256&gt;` 是文件内容的 SHA256 哈希的前八位或更多位数. 哈希用于确保唯一的名称并验证文件的内容.

`model_dir` 的默认值为 `$TORCH_HOME/models` 其中 `$TORCH_HOME` 默认值为 `~/.torch`. 可以使用 `$TORCH_MODEL_ZOO` 环境变量来覆盖默认目录.

参数：

*   `url (string)` – 需要下载对象的 URL
*   `model_dir (string, 可选)` – 保存对象的目录
*   `map_location (optional)` – 一个函数或者一个字典,指定如何重新映射存储位置 (详情查阅 torch.load)



示例：

```py
>>> state_dict = torch.utils.model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

```