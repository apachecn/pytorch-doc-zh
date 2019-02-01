

# torchvision.utils

```py
torchvision.utils.make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
```

Make a grid of images.

 
Parameters: 

*   **tensor** ([_Tensor_](../tensors.html#torch.Tensor "torch.Tensor") _or_ [_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")) – 4D mini-batch Tensor of shape (B x C x H x W) or a list of images all of the same size.
*   **nrow** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – Number of images displayed in each row of the grid. The Final grid size is (B / nrow, nrow). Default is 8.
*   **padding** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – amount of padding. Default is 2.
*   **normalize** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If True, shift the image to the range (0, 1), by subtracting the minimum and dividing by the maximum pixel value.
*   **range** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _optional_) – tuple (min, max) where min and max are numbers, then these numbers are used to normalize the image. By default, min and max are computed from the tensor.
*   **scale_each** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If True, scale each image in the batch of images separately rather than the (min, max) over all images.
*   **pad_value** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _optional_) – Value for the padded pixels.



Example

See this notebook [here](https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91)

```py
torchvision.utils.save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
```

Save a given Tensor into an image file.

 
Parameters: 

*   **tensor** ([_Tensor_](../tensors.html#torch.Tensor "torch.Tensor") _or_ [_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")) – Image to be saved. If given a mini-batch tensor, saves the tensor as a grid of images by calling `make_grid`.
*   ****kwargs** – Other arguments are documented in `make_grid`.

