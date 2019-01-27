

# Type Info

The numerical properties of a [`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") can be accessed through either the [`torch.finfo`](#torch.torch.finfo "torch.torch.finfo") or the [`torch.iinfo`](#torch.torch.iinfo "torch.torch.iinfo").

## torch.finfo

```py
class torch.finfo
```

A [`torch.finfo`](#torch.torch.finfo "torch.torch.finfo") is an object that represents the numerical properties of a floating point [`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype"), (i.e. &lt;cite&gt;torch.float32&lt;/cite&gt;, &lt;cite&gt;torch.float64&lt;/cite&gt;, and &lt;cite&gt;torch.float16&lt;/cite&gt;). This is similar to [numpy.finfo](https://docs.scipy.org/doc/numpy/reference/generated/numpy.finfo.html).

A [`torch.finfo`](#torch.torch.finfo "torch.torch.finfo") provides the following attributes:

| Name | Type | Description |
| --- | --- | --- |
| bits | int | The number of bits occupied by the type. |
| eps | float | The smallest representable number such that 1.0 + eps != 1.0. |
| max | float | The largest representable number. |
| tiny | float | The smallest positive representable number. |

Note

The constructor of [`torch.finfo`](#torch.torch.finfo "torch.torch.finfo") can be called without argument, in which case the class is created for the pytorch default dtype (as returned by `torch.get_default_dtype()`).

## torch.iinfo

```py
class torch.iinfo
```

A [`torch.iinfo`](#torch.torch.iinfo "torch.torch.iinfo") is an object that represents the numerical properties of a integer [`torch.dtype`](tensor_attributes.html#torch.torch.dtype "torch.torch.dtype") (i.e. &lt;cite&gt;torch.uint8&lt;/cite&gt;, &lt;cite&gt;torch.int8&lt;/cite&gt;, &lt;cite&gt;torch.int16&lt;/cite&gt;, &lt;cite&gt;torch.int32&lt;/cite&gt;, and &lt;cite&gt;torch.int64&lt;/cite&gt;). This is similar to [numpy.iinfo](https://docs.scipy.org/doc/numpy/reference/generated/numpy.iinfo.html).

A [`torch.iinfo`](#torch.torch.iinfo "torch.torch.iinfo") provides the following attributes:

| Name | Type | Description |
| --- | --- | --- |
| bits | int | The number of bits occupied by the type. |
| max | int | The largest representable number. |

