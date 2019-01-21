# torch.utils.model_zoo

```python
torch.utils.model_zoo.load_url(url, model_dir=None)
```

在给定URL上加载Torch序列化对象。

如果对象已经存在于 *model_dir* 中，则将被反序列化并返回。URL的文件名部分应遵循命名约定`filename-<sha256>.ext`，其中`<sha256>`是文件内容的SHA256哈希的前八位或更多位数字。哈希用于确保唯一的名称并验证文件的内容。

*model_dir* 的默认值为`$TORCH_HOME/models`，其中`$TORCH_HOME`默认为`~/.torch`。可以使用`$TORCH_MODEL_ZOO`环境变量来覆盖默认目录。

**参数：**

- **url** (*string*) - 要下载对象的URL
- **model_dir** (*string*, optional) - 保存对象的目录

**例子：**
```python
>>> state_dict = torch.utils.model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
```
