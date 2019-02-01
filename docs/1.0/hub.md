

# torch.hub

```py
torch.hub.load(github, model, force_reload=False, *args, **kwargs)
```

Load a model from a github repo, with pretrained weights.

Parameters: 

*   **github** – Required, a string with format “repo_owner/repo_name[:tag_name]” with an optional tag/branch. The default branch is `master` if not specified. Example: ‘pytorch/vision[:hub]’
*   **model** – Required, a string of callable name defined in repo’s hubconf.py
*   **force_reload** – Optional, whether to discard the existing cache and force a fresh download. Default is `False`.
*   ***args** – Optional, the corresponding args for callable `model`.
*   ****kwargs** – Optional, the corresponding kwargs for callable `model`.


| Returns: | a single model with corresponding pretrained weights. |
| --- | --- |

```py
torch.hub.set_dir(d)
```

```py
Optionally set hub_dir to a local dir to save the intermediate model & checkpoint files.
```

If this argument is not set, env variable <cite>TORCH_HUB_DIR</cite> will be searched first, <cite>~/.torch/hub</cite> will be created and used as fallback. 