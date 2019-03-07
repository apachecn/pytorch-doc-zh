# torch.hub

> 译者：[kunwuz](https://github.com/kunwuz)

```py
torch.hub.load(github, model, force_reload=False, *args, **kwargs)
```

从github上加载一个带有预训练权重的模型。

参数: 

*   **github** – 必需，一个字符串对象，格式为“repo_owner/repo_name[:tag_name]”，可选 tag/branch。如果未做指定，默认的 branch 是 `master` 。比方说: ‘pytorch/vision[:hub]’
*   **model** – 必须，一个字符串对象，名字在hubconf.py中定义。
*   **force_reload** – 可选， 是否丢弃现有缓存并强制重新下载。默认是：`False`。
*   ***args** – 可选， 可调用的`model`的相关args参数。
*   ****kwargs** – 可选， 可调用的`model`的相关kwargs参数。


| 返回: | 一个有相关预训练权重的单一模型。 |
| --- | --- |

```py
torch.hub.set_dir(d)
```

也可以将`hub_dir`设置为本地目录来保存中间模型和检查点文件。

如果未设置此参数,环境变量<cite>TORCH_HUB_DIR</cite> 会被首先搜寻，<cite>~/.torch/hub</cite> 将被创建并用作后备。
