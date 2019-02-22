# torch.utils.ffi

> 译者：[@之茗](https://github.com/mayuanucas)
> 
> 校对者：[@aleczhang](http://community.apachecn.org/?/people/aleczhang)

```py
torch.utils.ffi.create_extension(name, headers, sources, verbose=True, with_cuda=False, package=False, relative_to='.', **kwargs)
```

创建并配置一个 cffi.FFI 对象, 用于构建 PyTorch 的扩展.

参数：

*   `name (str)` – 包名. 可以是嵌套模块, 例如. `.ext.my_lib`.
*   `headers (str 或 List[str])` – 只包含导出函数的头文件列表.
*   `sources (List[str])` – 用于编译的sources列表.
*   `verbose (bool, 可选)` – 如果设置为 `False`, 则不会打印输出 (默认值: True).
*   `with_cuda (bool, 可选)` – 设置为 `True` 以使用 CUDA 头文件进行编译 (默认值: False)
*   `package (bool, 可选)` – 设置为 `True` 以在包模式下构建 (对于要作为 pip 程序包安装的模块) (默认值: False).
*   `relative_to (str, 可选)` – 构建文件的路径. 当 `package 为 True` 时需要. 最好使用 `__file__` 作为参数.
*   `kwargs` – 传递给 ffi 以声明扩展的附件参数. 参考 [Extension API reference](https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension) 查阅更详细内容.

