

# torch.utils.checkpoint

Note

Checkpointing is implemented by rerunning a forward-pass segment for each checkpointed segment during backward. This can cause persistent states like the RNG state to be advanced than they would without checkpointing. By default, checkpointing includes logic to juggle the RNG state such that checkpointed passes making use of RNG (through dropout for example) have deterministic output as compared to non-checkpointed passes. The logic to stash and restore RNG states can incur a moderate performance hit depending on the runtime of checkpointed operations. If deterministic output compared to non-checkpointed passes is not required, set the global flag `torch.utils.checkpoint.preserve_rng_state=False` to omit stashing and restoring the RNG state during each checkpoint.

```py
torch.utils.checkpoint.checkpoint(function, *args)
```

Checkpoint a model or part of the model

Checkpointing works by trading compute for memory. Rather than storing all intermediate activations of the entire computation graph for computing backward, the checkpointed part does **not** save intermediate activations, and instead recomputes them in backward pass. It can be applied on any part of a model.

Specifically, in the forward pass, `function` will run in `torch.no_grad()` manner, i.e., not storing the intermediate activations. Instead, the forward pass saves the inputs tuple and the `function` parameter. In the backwards pass, the saved inputs and `function` is retreived, and the forward pass is computed on `function` again, now tracking the intermediate activations, and then the gradients are calculated using these activation values.

Warning

Checkpointing doesn’t work with [`torch.autograd.grad()`](autograd.html#torch.autograd.grad "torch.autograd.grad"), but only with [`torch.autograd.backward()`](autograd.html#torch.autograd.backward "torch.autograd.backward").

Warning

If `function` invocation during backward does anything different than the one during forward, e.g., due to some global variable, the checkpointed version won’t be equivalent, and unfortunately it can’t be detected.

Parameters: 

*   **function** – describes what to run in the forward pass of the model or part of the model. It should also know how to handle the inputs passed as the tuple. For example, in LSTM, if user passes `(activation, hidden)`, `function` should correctly use the first input as `activation` and the second input as `hidden`
*   **args** – tuple containing inputs to the `function`


| Returns: | Output of running `function` on `*args` |
| --- | --- |

```py
torch.utils.checkpoint.checkpoint_sequential(functions, segments, *inputs)
```

A helper function for checkpointing sequential models.

Sequential models execute a list of modules/functions in order (sequentially). Therefore, we can divide such a model in various segments and checkpoint each segment. All segments except the last will run in `torch.no_grad()` manner, i.e., not storing the intermediate activations. The inputs of each checkpointed segment will be saved for re-running the segment in the backward pass.

See [`checkpoint()`](#torch.utils.checkpoint.checkpoint "torch.utils.checkpoint.checkpoint") on how checkpointing works.

Warning

Checkpointing doesn’t work with [`torch.autograd.grad()`](autograd.html#torch.autograd.grad "torch.autograd.grad"), but only with [`torch.autograd.backward()`](autograd.html#torch.autograd.backward "torch.autograd.backward").

Parameters: 

*   **functions** – A [`torch.nn.Sequential`](nn.html#torch.nn.Sequential "torch.nn.Sequential") or the list of modules or functions (comprising the model) to run sequentially.
*   **segments** – Number of chunks to create in the model
*   **inputs** – tuple of Tensors that are inputs to `functions`


| Returns: | Output of running `functions` sequentially on `*inputs` |
| --- | --- |

Example

```py
>>> model = nn.Sequential(...)
>>> input_var = checkpoint_sequential(model, chunks, input_var)

```

