

# torch.utils.dlpack

```py
torch.utils.dlpack.from_dlpack(dlpack) → Tensor
```

Decodes a DLPack to a tensor.

| Parameters: | **dlpack** – a PyCapsule object with the dltensor |
| --- | --- |

The tensor will share the memory with the object represented in the dlpack. Note that each dlpack can only be consumed once.

```py
torch.utils.dlpack.to_dlpack(tensor) → PyCapsule
```

Returns a DLPack representing the tensor.

| Parameters: | **tensor** – a tensor to be exported |
| --- | --- |

The dlpack shares the tensors memory. Note that each dlpack can only be consumed once.

