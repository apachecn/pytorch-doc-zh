## Serialization

```py
torch.save(obj, f, pickle_module=<module 'pickle' from '/scratch/rzou/pt/release-env/lib/python3.7/pickle.py'>, pickle_protocol=2)
```

Saves an object to a disk file.

See also: [Recommended approach for saving a model](notes/serialization.html#recommend-saving-models)

Parameters: 

*   **obj** – saved object
*   **f** – a file-like object (has to implement write and flush) or a string containing a file name
*   **pickle_module** – module used for pickling metadata and objects
*   **pickle_protocol** – can be specified to override the default protocol



Warning

If you are using Python 2, torch.save does NOT support StringIO.StringIO as a valid file-like object. This is because the write method should return the number of bytes written; StringIO.write() does not do this.

Please use something like io.BytesIO instead.

Example

```py
>>> # Save to file
>>> x = torch.tensor([0, 1, 2, 3, 4])
>>> torch.save(x, 'tensor.pt')
>>> # Save to io.BytesIO buffer
>>> buffer = io.BytesIO()
>>> torch.save(x, buffer)

```

```py
torch.load(f, map_location=None, pickle_module=<module 'pickle' from '/scratch/rzou/pt/release-env/lib/python3.7/pickle.py'>)
```

Loads an object saved with [`torch.save()`](#torch.save "torch.save") from a file.

[`torch.load()`](#torch.load "torch.load") uses Python’s unpickling facilities but treats storages, which underlie tensors, specially. They are first deserialized on the CPU and are then moved to the device they were saved from. If this fails (e.g. because the run time system doesn’t have certain devices), an exception is raised. However, storages can be dynamically remapped to an alternative set of devices using the `map_location` argument.

If `map_location` is a callable, it will be called once for each serialized storage with two arguments: storage and location. The storage argument will be the initial deserialization of the storage, residing on the CPU. Each serialized storage has a location tag associated with it which identifies the device it was saved from, and this tag is the second argument passed to map_location. The builtin location tags are `‘cpu’` for CPU tensors and `‘cuda:device_id’` (e.g. `‘cuda:2’`) for CUDA tensors. `map_location` should return either None or a storage. If `map_location` returns a storage, it will be used as the final deserialized object, already moved to the right device. Otherwise, ![](img/b69f1ef0735e18ff4ee132790112ce0d.jpg) will fall back to the default behavior, as if `map_location` wasn’t specified.

If `map_location` is a string, it should be a device tag, where all tensors should be loaded.

Otherwise, if `map_location` is a dict, it will be used to remap location tags appearing in the file (keys), to ones that specify where to put the storages (values).

User extensions can register their own location tags and tagging and deserialization methods using `register_package`.

Parameters: 

*   **f** – a file-like object (has to implement read, readline, tell, and seek), or a string containing a file name
*   **map_location** – a function, torch.device, string or a dict specifying how to remap storage locations
*   **pickle_module** – module used for unpickling metadata and objects (has to match the pickle_module used to serialize file)



Note

When you call [`torch.load()`](#torch.load "torch.load") on a file which contains GPU tensors, those tensors will be loaded to GPU by default. You can call `torch.load(.., map_location=’cpu’)` and then `load_state_dict()` to avoid GPU RAM surge when loading a model checkpoint.

Example

```py
>>> torch.load('tensors.pt')
# Load all tensors onto the CPU
>>> torch.load('tensors.pt', map_location=torch.device('cpu'))
# Load all tensors onto the CPU, using a function
>>> torch.load('tensors.pt', map_location=lambda storage, loc: storage)
# Load all tensors onto GPU 1
>>> torch.load('tensors.pt', map_location=lambda storage, loc: storage.cuda(1))
# Map tensors from GPU 1 to GPU 0
>>> torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})
# Load tensor from io.BytesIO object
>>> with open('tensor.pt') as f:
 buffer = io.BytesIO(f.read())
>>> torch.load(buffer)

```

## Parallelism

```py
torch.get_num_threads() → int
```

Gets the number of OpenMP threads used for parallelizing CPU operations

```py
torch.set_num_threads(int)
```

Sets the number of OpenMP threads used for parallelizing CPU operations

## Locally disabling gradient computation

The context managers `torch.no_grad()`, `torch.enable_grad()`, and `torch.set_grad_enabled()` are helpful for locally disabling and enabling gradient computation. See [Locally disabling gradient computation](autograd.html#locally-disable-grad) for more details on their usage.

Examples:

```py
>>> x = torch.zeros(1, requires_grad=True)
>>> with torch.no_grad():
...     y = x * 2
>>> y.requires_grad
False

>>> is_train = False
>>> with torch.set_grad_enabled(is_train):
...     y = x * 2
>>> y.requires_grad
False

>>> torch.set_grad_enabled(True)  # this can also be used as a function
>>> y = x * 2
>>> y.requires_grad
True

>>> torch.set_grad_enabled(False)
>>> y = x * 2
>>> y.requires_grad
False

```

## Utilities

```py
torch.compiled_with_cxx11_abi()
```

Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1
