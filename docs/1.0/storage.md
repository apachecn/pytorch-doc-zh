

# torch.Storage

A `torch.Storage` is a contiguous, one-dimensional array of a single data type.

Every [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor") has a corresponding storage of the same data type.

```py
class torch.FloatStorage
```

```py
byte()
```

Casts this storage to byte type

```py
char()
```

Casts this storage to char type

```py
clone()
```

Returns a copy of this storage

```py
copy_()
```

```py
cpu()
```

Returns a CPU copy of this storage if it’s not already on the CPU

```py
cuda(device=None, non_blocking=False, **kwargs)
```

Returns a copy of this object in CUDA memory.

If this object is already in CUDA memory and on the correct device, then no copy is performed and the original object is returned.

| Parameters: | 

*   **device** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – The destination GPU id. Defaults to the current device.
*   **non_blocking** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If `True` and the source is in pinned memory, the copy will be asynchronous with respect to the host. Otherwise, the argument has no effect.
*   ****kwargs** – For compatibility, may contain the key `async` in place of the `non_blocking` argument.

 |
| --- | --- |

```py
data_ptr()
```

```py
double()
```

Casts this storage to double type

```py
element_size()
```

```py
fill_()
```

```py
float()
```

Casts this storage to float type

```py
static from_buffer()
```

```py
static from_file(filename, shared=False, size=0) → Storage
```

If `shared` is `True`, then memory is shared between all processes. All changes are written to the file. If `shared` is `False`, then the changes on the storage do not affect the file.

`size` is the number of elements in the storage. If `shared` is `False`, then the file must contain at least `size * sizeof(Type)` bytes (`Type` is the type of storage). If `shared` is `True` the file will be created if needed.

| Parameters: | 

*   **filename** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")) – file name to map
*   **shared** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – whether to share memory
*   **size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – number of elements in the storage

 |
| --- | --- |

```py
half()
```

Casts this storage to half type

```py
int()
```

Casts this storage to int type

```py
is_cuda = False
```

```py
is_pinned()
```

```py
is_shared()
```

```py
is_sparse = False
```

```py
long()
```

Casts this storage to long type

```py
new()
```

```py
pin_memory()
```

Copies the storage to pinned memory, if it’s not already pinned.

```py
resize_()
```

```py
share_memory_()
```

Moves the storage to shared memory.

This is a no-op for storages already in shared memory and for CUDA storages, which do not need to be moved for sharing across processes. Storages in shared memory cannot be resized.

Returns: self

```py
short()
```

Casts this storage to short type

```py
size()
```

```py
tolist()
```

Returns a list containing the elements of this storage

```py
type(dtype=None, non_blocking=False, **kwargs)
```

Returns the type if `dtype` is not provided, else casts this object to the specified type.

If this is already of the correct type, no copy is performed and the original object is returned.

| Parameters: | 

*   **dtype** ([_type_](https://docs.python.org/3/library/functions.html#type "(in Python v3.7)") _or_ _string_) – The desired type
*   **non_blocking** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If `True`, and the source is in pinned memory and destination is on the GPU or vice versa, the copy is performed asynchronously with respect to the host. Otherwise, the argument has no effect.
*   ****kwargs** – For compatibility, may contain the key `async` in place of the `non_blocking` argument. The `async` arg is deprecated.

 |
| --- | --- |

