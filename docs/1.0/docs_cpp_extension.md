

# torch.utils.cpp_extension

```py
torch.utils.cpp_extension.CppExtension(name, sources, *args, **kwargs)
```

Creates a `setuptools.Extension` for C++.

Convenience method that creates a `setuptools.Extension` with the bare minimum (but often sufficient) arguments to build a C++ extension.

All arguments are forwarded to the `setuptools.Extension` constructor.

Example

```py
>>> from setuptools import setup
>>> from torch.utils.cpp_extension import BuildExtension, CppExtension
>>> setup(
 name='extension',
 ext_modules=[
 CppExtension(
 name='extension',
 sources=['extension.cpp'],
 extra_compile_args=['-g'])),
 ],
 cmdclass={
 'build_ext': BuildExtension
 })

```

```py
torch.utils.cpp_extension.CUDAExtension(name, sources, *args, **kwargs)
```

Creates a `setuptools.Extension` for CUDA/C++.

Convenience method that creates a `setuptools.Extension` with the bare minimum (but often sufficient) arguments to build a CUDA/C++ extension. This includes the CUDA include path, library path and runtime library.

All arguments are forwarded to the `setuptools.Extension` constructor.

Example

```py
>>> from setuptools import setup
>>> from torch.utils.cpp_extension import BuildExtension, CUDAExtension
>>> setup(
 name='cuda_extension',
 ext_modules=[
 CUDAExtension(
 name='cuda_extension',
 sources=['extension.cpp', 'extension_kernel.cu'],
 extra_compile_args={'cxx': ['-g'],
 'nvcc': ['-O2']})
 ],
 cmdclass={
 'build_ext': BuildExtension
 })

```

```py
torch.utils.cpp_extension.BuildExtension(*args, **kwargs)
```

A custom `setuptools` build extension .

This `setuptools.build_ext` subclass takes care of passing the minimum required compiler flags (e.g. `-std=c++11`) as well as mixed C++/CUDA compilation (and support for CUDA files in general).

When using [`BuildExtension`](#torch.utils.cpp_extension.BuildExtension "torch.utils.cpp_extension.BuildExtension"), it is allowed to supply a dictionary for `extra_compile_args` (rather than the usual list) that maps from languages (`cxx` or `cuda`) to a list of additional compiler flags to supply to the compiler. This makes it possible to supply different flags to the C++ and CUDA compiler during mixed compilation.

```py
torch.utils.cpp_extension.load(name, sources, extra_cflags=None, extra_cuda_cflags=None, extra_ldflags=None, extra_include_paths=None, build_directory=None, verbose=False, with_cuda=None, is_python_module=True)
```

Loads a PyTorch C++ extension just-in-time (JIT).

To load an extension, a Ninja build file is emitted, which is used to compile the given sources into a dynamic library. This library is subsequently loaded into the current Python process as a module and returned from this function, ready for use.

By default, the directory to which the build file is emitted and the resulting library compiled to is `&lt;tmp&gt;/torch_extensions/&lt;name&gt;`, where `&lt;tmp&gt;` is the temporary folder on the current platform and `&lt;name&gt;` the name of the extension. This location can be overridden in two ways. First, if the `TORCH_EXTENSIONS_DIR` environment variable is set, it replaces `&lt;tmp&gt;/torch_extensions` and all extensions will be compiled into subfolders of this directory. Second, if the `build_directory` argument to this function is supplied, it overrides the entire path, i.e. the library will be compiled into that folder directly.

To compile the sources, the default system compiler (`c++`) is used, which can be overridden by setting the `CXX` environment variable. To pass additional arguments to the compilation process, `extra_cflags` or `extra_ldflags` can be provided. For example, to compile your extension with optimizations, pass `extra_cflags=['-O3']`. You can also use `extra_cflags` to pass further include directories.

CUDA support with mixed compilation is provided. Simply pass CUDA source files (`.cu` or `.cuh`) along with other sources. Such files will be detected and compiled with nvcc rather than the C++ compiler. This includes passing the CUDA lib64 directory as a library directory, and linking `cudart`. You can pass additional flags to nvcc via `extra_cuda_cflags`, just like with `extra_cflags` for C++. Various heuristics for finding the CUDA install directory are used, which usually work fine. If not, setting the `CUDA_HOME` environment variable is the safest option.

Parameters: 

*   **name** – The name of the extension to build. This MUST be the same as the name of the pybind11 module!
*   **sources** – A list of relative or absolute paths to C++ source files.
*   **extra_cflags** – optional list of compiler flags to forward to the build.
*   **extra_cuda_cflags** – optional list of compiler flags to forward to nvcc when building CUDA sources.
*   **extra_ldflags** – optional list of linker flags to forward to the build.
*   **extra_include_paths** – optional list of include directories to forward to the build.
*   **build_directory** – optional path to use as build workspace.
*   **verbose** – If `True`, turns on verbose logging of load steps.
*   **with_cuda** – Determines whether CUDA headers and libraries are added to the build. If set to `None` (default), this value is automatically determined based on the existence of `.cu` or `.cuh` in `sources`. Set it to `True`` to force CUDA headers and libraries to be included.
*   **is_python_module** – If `True` (default), imports the produced shared library as a Python module. If `False`, loads it into the process as a plain dynamic library.


| Returns: | If `is_python_module` is `True`, returns the loaded PyTorch extension as a Python module. If `is_python_module` is `False` returns nothing (the shared library is loaded into the process as a side effect). |
| --- | --- |

Example

```py
>>> from torch.utils.cpp_extension import load
>>> module = load(
 name='extension',
 sources=['extension.cpp', 'extension_kernel.cu'],
 extra_cflags=['-O2'],
 verbose=True)

```

```py
torch.utils.cpp_extension.load_inline(name, cpp_sources, cuda_sources=None, functions=None, extra_cflags=None, extra_cuda_cflags=None, extra_ldflags=None, extra_include_paths=None, build_directory=None, verbose=False, with_cuda=None, is_python_module=True)
```

Loads a PyTorch C++ extension just-in-time (JIT) from string sources.

This function behaves exactly like [`load()`](#torch.utils.cpp_extension.load "torch.utils.cpp_extension.load"), but takes its sources as strings rather than filenames. These strings are stored to files in the build directory, after which the behavior of [`load_inline()`](#torch.utils.cpp_extension.load_inline "torch.utils.cpp_extension.load_inline") is identical to [`load()`](#torch.utils.cpp_extension.load "torch.utils.cpp_extension.load").

See [the tests](https://github.com/pytorch/pytorch/blob/master/test/test_cpp_extensions.py) for good examples of using this function.

Sources may omit two required parts of a typical non-inline C++ extension: the necessary header includes, as well as the (pybind11) binding code. More precisely, strings passed to `cpp_sources` are first concatenated into a single `.cpp` file. This file is then prepended with `#include &lt;torch/extension.h&gt;`.

Furthermore, if the `functions` argument is supplied, bindings will be automatically generated for each function specified. `functions` can either be a list of function names, or a dictionary mapping from function names to docstrings. If a list is given, the name of each function is used as its docstring.

The sources in `cuda_sources` are concatenated into a separate `.cu` file and prepended with `torch/types.h`, `cuda.h` and `cuda_runtime.h` includes. The `.cpp` and `.cu` files are compiled separately, but ultimately linked into a single library. Note that no bindings are generated for functions in `cuda_sources` per se. To bind to a CUDA kernel, you must create a C++ function that calls it, and either declare or define this C++ function in one of the `cpp_sources` (and include its name in `functions`).

See [`load()`](#torch.utils.cpp_extension.load "torch.utils.cpp_extension.load") for a description of arguments omitted below.

Parameters: 

*   **cpp_sources** – A string, or list of strings, containing C++ source code.
*   **cuda_sources** – A string, or list of strings, containing CUDA source code.
*   **functions** – A list of function names for which to generate function bindings. If a dictionary is given, it should map function names to docstrings (which are otherwise just the function names).
*   **with_cuda** – Determines whether CUDA headers and libraries are added to the build. If set to `None` (default), this value is automatically determined based on whether `cuda_sources` is provided. Set it to `True`` to force CUDA headers and libraries to be included.



Example

```py
>>> from torch.utils.cpp_extension import load_inline
>>> source = '''
at::Tensor sin_add(at::Tensor x, at::Tensor y) {
 return x.sin() + y.sin();
}
'''
>>> module = load_inline(name='inline_extension',
 cpp_sources=[source],
 functions=['sin_add'])

```

```py
torch.utils.cpp_extension.include_paths(cuda=False)
```

Get the include paths required to build a C++ or CUDA extension.

| Parameters: | **cuda** – If `True`, includes CUDA-specific include paths. |
| --- | --- |
| Returns: | A list of include path strings. |
| --- | --- |

```py
torch.utils.cpp_extension.check_compiler_abi_compatibility(compiler)
```

Verifies that the given compiler is ABI-compatible with PyTorch.

| Parameters: | **compiler** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")) – The compiler executable name to check (e.g. `g++`). Must be executable in a shell process. |
| --- | --- |
| Returns: | False if the compiler is (likely) ABI-incompatible with PyTorch, else True. |
| --- | --- |

```py
torch.utils.cpp_extension.verify_ninja_availability()
```

Returns `True` if the [ninja](https://ninja-build.org/) build system is available on the system.

