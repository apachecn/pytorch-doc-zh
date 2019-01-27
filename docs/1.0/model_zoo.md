

# torch.utils.model_zoo

```py
torch.utils.model_zoo.load_url(url, model_dir=None, map_location=None, progress=True)
```

Loads the Torch serialized object at the given URL.

If the object is already present in &lt;cite&gt;model_dir&lt;/cite&gt;, it’s deserialized and returned. The filename part of the URL should follow the naming convention `filename-&lt;sha256&gt;.ext` where `&lt;sha256&gt;` is the first eight or more digits of the SHA256 hash of the contents of the file. The hash is used to ensure unique names and to verify the contents of the file.

The default value of &lt;cite&gt;model_dir&lt;/cite&gt; is `$TORCH_HOME/models` where `$TORCH_HOME` defaults to `~/.torch`. The default directory can be overridden with the `$TORCH_MODEL_ZOO` environment variable.

| Parameters: | 

*   **url** (_string_) – URL of the object to download
*   **model_dir** (_string__,_ _optional_) – directory in which to save the object
*   **map_location** (_optional_) – a function or a dict specifying how to remap storage locations (see torch.load)
*   **progress** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – whether or not to display a progress bar to stderr

 |
| --- | --- |

Example

```py
>>> state_dict = torch.utils.model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

```

