# torch.utils.ffi
```python
torch.utils.ffi.create_extension(name, headers, sources, verbose=True, with_cuda=False, package=False, relative_to='.', **kwargs)
```
创建并配置一个cffi.FFI对象,用于PyTorch的扩展。

**参数：**

- **name** (*str*) – 包名。可以是嵌套模块，例如 `.ext.my_lib`。
- **headers** (*str* or List[*str*]) – 只包含导出函数的头文件列表
- **sources** (List[*str*]) – 用于编译的sources列表
- **verbose** (*bool*, optional) – 如果设置为False，则不会打印输出(默认值：`True`）。
- **with_cuda** (*bool*, optional) – 设置为True以使用CUDA头文件进行编译(默认值：`False`）。
- **package** (*bool*, optional) – 设置为True以在程序包模式下构建(对于要作为pip程序包安装的模块）(默认值：`False`）。
- **relative_to** (*str*, optional) –构建文件的路径。`package`为`True`时需要。最好使用`__file__`作为参数。
- **kwargs** – 传递给ffi以声明扩展的附加参数。有关详细信息，请参阅[Extension API reference](https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension)。
