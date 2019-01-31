

# torchvision.transforms

Transforms are common image transformations. They can be chained together using [`Compose`](#torchvision.transforms.Compose "torchvision.transforms.Compose"). Additionally, there is the [`torchvision.transforms.functional`](#module-torchvision.transforms.functional "torchvision.transforms.functional") module. Functional transforms give fine-grained control over the transformations. This is useful if you have to build a more complex transformation pipeline (e.g. in the case of segmentation tasks).

```py
class torchvision.transforms.Compose(transforms)¶
```

Composes several transforms together.

 
| Parameters: | **transforms** (list of `Transform` objects) – list of transforms to compose. |
| --- | --- |

Example

```py
&gt;&gt;&gt; transforms.Compose([
&gt;&gt;&gt;     transforms.CenterCrop(10),
&gt;&gt;&gt;     transforms.ToTensor(),
&gt;&gt;&gt; ])

```

## Transforms on PIL Image

```py
class torchvision.transforms.CenterCrop(size)¶
```

Crops the given PIL Image at the center.

 
| Parameters: | **size** (_sequence_ _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Desired output size of the crop. If size is an int instead of sequence like (h, w), a square crop (size, size) is made. |
| --- | --- |

```py
class torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)¶
```

Randomly change the brightness, contrast and saturation of an image.

 
| Parameters: | 

*   **brightness** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – How much to jitter brightness. brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
*   **contrast** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – How much to jitter contrast. contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
*   **saturation** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – How much to jitter saturation. saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
*   **hue** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue]. Should be &gt;=0 and &lt;= 0.5.

 |
| --- | --- |

```py
class torchvision.transforms.FiveCrop(size)¶
```

Crop the given PIL Image into four corners and the central crop

Note

This transform returns a tuple of images and there may be a mismatch in the number of inputs and targets your Dataset returns. See below for an example of how to deal with this.

 
| Parameters: | **size** (_sequence_ _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Desired output size of the crop. If size is an `int` instead of sequence like (h, w), a square crop of size (size, size) is made. |
| --- | --- |

Example

```py
&gt;&gt;&gt; transform = Compose([
&gt;&gt;&gt;    FiveCrop(size), # this is a list of PIL Images
&gt;&gt;&gt;    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
&gt;&gt;&gt; ])
&gt;&gt;&gt; #In your test loop you can do the following:
&gt;&gt;&gt; input, target = batch # input is a 5d tensor, target is 2d
&gt;&gt;&gt; bs, ncrops, c, h, w = input.size()
&gt;&gt;&gt; result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
&gt;&gt;&gt; result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops

```

```py
class torchvision.transforms.Grayscale(num_output_channels=1)¶
```

Convert image to grayscale.

 
| Parameters: | **num_output_channels** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – (1 or 3) number of channels desired for output image |
| --- | --- |
| Returns: | Grayscale version of the input. - If num_output_channels == 1 : returned image is single channel - If num_output_channels == 3 : returned image is 3 channel with r == g == b |
| --- | --- |
| Return type: | PIL Image |
| --- | --- |

```py
class torchvision.transforms.LinearTransformation(transformation_matrix)¶
```

Transform a tensor image with a square transformation matrix computed offline.

Given transformation_matrix, will flatten the torch.*Tensor, compute the dot product with the transformation matrix and reshape the tensor to its original shape.

Applications: - whitening: zero-center the data, compute the data covariance matrix

&gt; [D x D] with np.dot(X.T, X), perform SVD on this matrix and pass it as transformation_matrix.

 
| Parameters: | **transformation_matrix** ([_Tensor_](../tensors.html#torch.Tensor "torch.Tensor")) – tensor [D x D], D = C x H x W |
| --- | --- |

```py
class torchvision.transforms.Pad(padding, fill=0, padding_mode='constant')¶
```

Pad the given PIL Image on all sides with the given “pad” value.

 
| Parameters: | 

*   **padding** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _or_ [_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")) – Padding on each border. If a single int is provided this is used to pad all borders. If tuple of length 2 is provided this is the padding on left/right and top/bottom respectively. If a tuple of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.
*   **fill** – Pixel fill value for constant fill. Default is 0\. If a tuple of length 3, it is used to fill R, G, B channels respectively. This value is only used when the padding_mode is constant
*   **padding_mode** –

    Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant. constant: pads with a constant value, this value is specified with fill edge: pads with the last value at the edge of the image reflect: pads with reflection of image (without repeating the last value on the edge)

    &gt; padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode will result in [3, 2, 1, 2, 3, 4, 3, 2]

    ```py
    symmetric: pads with reflection of image (repeating the last value on the edge)
    ```

    padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode will result in [2, 1, 1, 2, 3, 4, 4, 3]

 |
| --- | --- |

```py
class torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0)¶
```

Random affine transformation of the image keeping center invariant

 
| Parameters: | 

*   **degrees** (_sequence_ _or_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees). Set to 0 to desactivate rotations.
*   **translate** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _optional_) – tuple of maximum absolute fraction for horizontal and vertical translations. For example translate=(a, b), then horizontal shift is randomly sampled in the range -img_width * a &lt; dx &lt; img_width * a and vertical shift is randomly sampled in the range -img_height * b &lt; dy &lt; img_height * b. Will not translate by default.
*   **scale** ([_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")_,_ _optional_) – scaling factor interval, e.g (a, b), then scale is randomly sampled from the range a &lt;= scale &lt;= b. Will keep original scale by default.
*   **shear** (_sequence_ _or_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees). Will not apply shear by default
*   **resample** (_{PIL.Image.NEAREST__,_ _PIL.Image.BILINEAR__,_ _PIL.Image.BICUBIC}__,_ _optional_) – An optional resampling filter. See [http://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters](http://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters) If omitted, or if the image has mode “1” or “P”, it is set to PIL.Image.NEAREST.
*   **fillcolor** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Optional fill color for the area outside the transform in the output image. (Pillow&gt;=5.0.0)

 |
| --- | --- |

```py
class torchvision.transforms.RandomApply(transforms, p=0.5)¶
```

Apply randomly a list of transformations with a given probability

 
| Parameters: | 

*   **transforms** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)") _or_ [_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")) – list of transformations
*   **p** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – probability

 |
| --- | --- |

```py
class torchvision.transforms.RandomChoice(transforms)¶
```

Apply single transformation randomly picked from a list

```py
class torchvision.transforms.RandomCrop(size, padding=0, pad_if_needed=False)¶
```

Crop the given PIL Image at a random location.

 
| Parameters: | 

*   **size** (_sequence_ _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Desired output size of the crop. If size is an int instead of sequence like (h, w), a square crop (size, size) is made.
*   **padding** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _or_ _sequence__,_ _optional_) – Optional padding on each border of the image. Default is 0, i.e no padding. If a sequence of length 4 is provided, it is used to pad left, top, right, bottom borders respectively.
*   **pad_if_needed** (_boolean_) – It will pad the image if smaller than the desired size to avoid raising an exception.

 |
| --- | --- |

```py
class torchvision.transforms.RandomGrayscale(p=0.1)¶
```

Randomly convert image to grayscale with a probability of p (default 0.1).

 
| Parameters: | **p** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – probability that image should be converted to grayscale. |
| --- | --- |
| Returns: | Grayscale version of the input image with probability p and unchanged with probability (1-p). - If input image is 1 channel: grayscale version is 1 channel - If input image is 3 channel: grayscale version is 3 channel with r == g == b |
| --- | --- |
| Return type: | PIL Image |
| --- | --- |

```py
class torchvision.transforms.RandomHorizontalFlip(p=0.5)¶
```

Horizontally flip the given PIL Image randomly with a given probability.

 
| Parameters: | **p** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – probability of the image being flipped. Default value is 0.5 |
| --- | --- |

```py
class torchvision.transforms.RandomOrder(transforms)¶
```

Apply a list of transformations in a random order

```py
class torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)¶
```

Crop the given PIL Image to random size and aspect ratio.

A crop of random size (default: of 0.08 to 1.0) of the original size and a random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop is finally resized to given size. This is popularly used to train the Inception networks.

 
| Parameters: | 

*   **size** – expected output size of each edge
*   **scale** – range of size of the origin size cropped
*   **ratio** – range of aspect ratio of the origin aspect ratio cropped
*   **interpolation** – Default: PIL.Image.BILINEAR

 |
| --- | --- |

```py
class torchvision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None)¶
```

Rotate the image by angle.

 
| Parameters: | 

*   **degrees** (_sequence_ _or_ [_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)") _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).
*   **resample** (_{PIL.Image.NEAREST__,_ _PIL.Image.BILINEAR__,_ _PIL.Image.BICUBIC}__,_ _optional_) – An optional resampling filter. See [http://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters](http://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters) If omitted, or if the image has mode “1” or “P”, it is set to PIL.Image.NEAREST.
*   **expand** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – Optional expansion flag. If true, expands the output to make it large enough to hold the entire rotated image. If false or omitted, make the output image the same size as the input image. Note that the expand flag assumes rotation around the center and no translation.
*   **center** (_2-tuple__,_ _optional_) – Optional center of rotation. Origin is the upper left corner. Default is the center of the image.

 |
| --- | --- |

```py
class torchvision.transforms.RandomSizedCrop(*args, **kwargs)¶
```

Note: This transform is deprecated in favor of RandomResizedCrop.

```py
class torchvision.transforms.RandomVerticalFlip(p=0.5)¶
```

Vertically flip the given PIL Image randomly with a given probability.

 
| Parameters: | **p** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – probability of the image being flipped. Default value is 0.5 |
| --- | --- |

```py
class torchvision.transforms.Resize(size, interpolation=2)¶
```

Resize the input PIL Image to the given size.

 
| Parameters: | 

*   **size** (_sequence_ _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Desired output size. If size is a sequence like (h, w), output size will be matched to this. If size is an int, smaller edge of the image will be matched to this number. i.e, if height &gt; width, then image will be rescaled to (size * height / width, size)
*   **interpolation** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – Desired interpolation. Default is `PIL.Image.BILINEAR`

 |
| --- | --- |

```py
class torchvision.transforms.Scale(*args, **kwargs)¶
```

Note: This transform is deprecated in favor of Resize.

```py
class torchvision.transforms.TenCrop(size, vertical_flip=False)¶
```

Crop the given PIL Image into four corners and the central crop plus the flipped version of these (horizontal flipping is used by default)

Note

This transform returns a tuple of images and there may be a mismatch in the number of inputs and targets your Dataset returns. See below for an example of how to deal with this.

 
| Parameters: | 

*   **size** (_sequence_ _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Desired output size of the crop. If size is an int instead of sequence like (h, w), a square crop (size, size) is made.
*   **vertical_flip** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – Use vertical flipping instead of horizontal

 |
| --- | --- |

Example

```py
&gt;&gt;&gt; transform = Compose([
&gt;&gt;&gt;    TenCrop(size), # this is a list of PIL Images
&gt;&gt;&gt;    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
&gt;&gt;&gt; ])
&gt;&gt;&gt; #In your test loop you can do the following:
&gt;&gt;&gt; input, target = batch # input is a 5d tensor, target is 2d
&gt;&gt;&gt; bs, ncrops, c, h, w = input.size()
&gt;&gt;&gt; result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
&gt;&gt;&gt; result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops

```

## Transforms on torch.*Tensor

```py
class torchvision.transforms.Normalize(mean, std)¶
```

Normalize a tensor image with mean and standard deviation. Given mean: `(M1,...,Mn)` and std: `(S1,..,Sn)` for `n` channels, this transform will normalize each channel of the input `torch.*Tensor` i.e. `input[channel] = (input[channel] - mean[channel]) / std[channel]`

 
| Parameters: | 

*   **mean** (_sequence_) – Sequence of means for each channel.
*   **std** (_sequence_) – Sequence of standard deviations for each channel.

 |
| --- | --- |

```py
__call__(tensor)¶
```

 
| Parameters: | **tensor** ([_Tensor_](../tensors.html#torch.Tensor "torch.Tensor")) – Tensor image of size (C, H, W) to be normalized. |
| --- | --- |
| Returns: | Normalized Tensor image. |
| --- | --- |
| Return type: | [Tensor](../tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

## Conversion Transforms

```py
class torchvision.transforms.ToPILImage(mode=None)¶
```

Convert a tensor or an ndarray to PIL Image.

Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape H x W x C to a PIL Image while preserving the value range.

 
| Parameters: | **mode** ([PIL.Image mode](http://pillow.readthedocs.io/en/latest/handbook/concepts.html#modes)) – color space and pixel depth of input data (optional). If `mode` is `None` (default) there are some assumptions made about the input data: 1\. If the input has 3 channels, the `mode` is assumed to be `RGB`. 2\. If the input has 4 channels, the `mode` is assumed to be `RGBA`. 3\. If the input has 1 channel, the `mode` is determined by the data type (i,e, `int`, `float`, `short`). |
| --- | --- |

```py
__call__(pic)¶
```

 
| Parameters: | **pic** ([_Tensor_](../tensors.html#torch.Tensor "torch.Tensor") _or_ [_numpy.ndarray_](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v1.15)")) – Image to be converted to PIL Image. |
| --- | --- |
| Returns: | Image converted to PIL Image. |
| --- | --- |
| Return type: | PIL Image |
| --- | --- |

```py
class torchvision.transforms.ToTensor¶
```

Convert a `PIL Image` or `numpy.ndarray` to tensor.

Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

```py
__call__(pic)¶
```

 
| Parameters: | **pic** (_PIL Image_ _or_ [_numpy.ndarray_](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v1.15)")) – Image to be converted to tensor. |
| --- | --- |
| Returns: | Converted image. |
| --- | --- |
| Return type: | [Tensor](../tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

## Generic Transforms

```py
class torchvision.transforms.Lambda(lambd)¶
```

Apply a user-defined lambda as a transform.

 
| Parameters: | **lambd** (_function_) – Lambda/function to be used for transform. |
| --- | --- |

## Functional Transforms

Functional transforms give you fine-grained control of the transformation pipeline. As opposed to the transformations above, functional transforms don’t contain a random number generator for their parameters. That means you have to specify/generate all parameters, but you can reuse the functional transform. For example, you can apply a functional transform to multiple images like this:

```py
import torchvision.transforms.functional as TF
import random

def my_segmentation_transforms(image, segmentation):
    if random.random() > 5:
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle)
        segmentation = TF.rotate(segmentation, angle)
    # more transforms ...
    return image, segmentation

```

```py
torchvision.transforms.functional.adjust_brightness(img, brightness_factor)¶
```

Adjust brightness of an Image.

 
| Parameters: | 

*   **img** (_PIL Image_) – PIL Image to be adjusted.
*   **brightness_factor** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – How much to adjust the brightness. Can be any non negative number. 0 gives a black image, 1 gives the original image while 2 increases the brightness by a factor of 2.

 |
| --- | --- |
| Returns: | Brightness adjusted image. |
| --- | --- |
| Return type: | PIL Image |
| --- | --- |

```py
torchvision.transforms.functional.adjust_contrast(img, contrast_factor)¶
```

Adjust contrast of an Image.

 
| Parameters: | 

*   **img** (_PIL Image_) – PIL Image to be adjusted.
*   **contrast_factor** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – How much to adjust the contrast. Can be any non negative number. 0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by a factor of 2.

 |
| --- | --- |
| Returns: | Contrast adjusted image. |
| --- | --- |
| Return type: | PIL Image |
| --- | --- |

```py
torchvision.transforms.functional.adjust_gamma(img, gamma, gain=1)¶
```

Perform gamma correction on an image.

Also known as Power Law Transform. Intensities in RGB mode are adjusted based on the following equation:

&gt; I_out = 255 * gain * ((I_in / 255) ** gamma)

See [https://en.wikipedia.org/wiki/Gamma_correction](https://en.wikipedia.org/wiki/Gamma_correction) for more details.

 
| Parameters: | 

*   **img** (_PIL Image_) – PIL Image to be adjusted.
*   **gamma** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – Non negative real number. gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
*   **gain** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – The constant multiplier.

 |
| --- | --- |

```py
torchvision.transforms.functional.adjust_hue(img, hue_factor)¶
```

Adjust hue of an image.

The image hue is adjusted by converting the image to HSV and cyclically shifting the intensities in the hue channel (H). The image is then converted back to original image mode.

`hue_factor` is the amount of shift in H channel and must be in the interval `[-0.5, 0.5]`.

See [https://en.wikipedia.org/wiki/Hue](https://en.wikipedia.org/wiki/Hue) for more details on Hue.

 
| Parameters: | 

*   **img** (_PIL Image_) – PIL Image to be adjusted.
*   **hue_factor** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – How much to shift the hue channel. Should be in [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in HSV space in positive and negative direction respectively. 0 means no shift. Therefore, both -0.5 and 0.5 will give an image with complementary colors while 0 gives the original image.

 |
| --- | --- |
| Returns: | Hue adjusted image. |
| --- | --- |
| Return type: | PIL Image |
| --- | --- |

```py
torchvision.transforms.functional.adjust_saturation(img, saturation_factor)¶
```

Adjust color saturation of an image.

 
| Parameters: | 

*   **img** (_PIL Image_) – PIL Image to be adjusted.
*   **saturation_factor** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – How much to adjust the saturation. 0 will give a black and white image, 1 will give the original image while 2 will enhance the saturation by a factor of 2.

 |
| --- | --- |
| Returns: | Saturation adjusted image. |
| --- | --- |
| Return type: | PIL Image |
| --- | --- |

```py
torchvision.transforms.functional.affine(img, angle, translate, scale, shear, resample=0, fillcolor=None)¶
```

Apply affine transformation on the image keeping image center invariant

 
| Parameters: | 

*   **img** (_PIL Image_) – PIL Image to be rotated.
*   **angle** (_{python:float__,_ _int}_) – rotation angle in degrees between -180 and 180, clockwise direction.
*   **translate** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)") _or_ _tuple of python:integers_) – horizontal and vertical translations (post-rotation translation)
*   **scale** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – overall scale
*   **shear** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")) – shear angle value in degrees between -180 to 180, clockwise direction.
*   **resample** (_{PIL.Image.NEAREST__,_ _PIL.Image.BILINEAR__,_ _PIL.Image.BICUBIC}__,_ _optional_) – An optional resampling filter. See [http://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters](http://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters) If omitted, or if the image has mode “1” or “P”, it is set to PIL.Image.NEAREST.
*   **fillcolor** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Optional fill color for the area outside the transform in the output image. (Pillow&gt;=5.0.0)

 |
| --- | --- |

```py
torchvision.transforms.functional.crop(img, i, j, h, w)¶
```

Crop the given PIL Image.

 
| Parameters: | 

*   **img** (_PIL Image_) – Image to be cropped.
*   **i** – Upper pixel coordinate.
*   **j** – Left pixel coordinate.
*   **h** – Height of the cropped image.
*   **w** – Width of the cropped image.

 |
| --- | --- |
| Returns: | Cropped image. |
| --- | --- |
| Return type: | PIL Image |
| --- | --- |

```py
torchvision.transforms.functional.five_crop(img, size)¶
```

Crop the given PIL Image into four corners and the central crop.

Note

This transform returns a tuple of images and there may be a mismatch in the number of inputs and targets your `Dataset` returns.

 
| Parameters: | **size** (_sequence_ _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Desired output size of the crop. If size is an int instead of sequence like (h, w), a square crop (size, size) is made. |
| --- | --- |
| Returns: | 

```py
tuple (tl, tr, bl, br, center) corresponding top left,
```

top right, bottom left, bottom right and center crop.

 |
| --- | --- |
| Return type: | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

```py
torchvision.transforms.functional.hflip(img)¶
```

Horizontally flip the given PIL Image.

 
| Parameters: | **img** (_PIL Image_) – Image to be flipped. |
| --- | --- |
| Returns: | Horizontall flipped image. |
| --- | --- |
| Return type: | PIL Image |
| --- | --- |

```py
torchvision.transforms.functional.normalize(tensor, mean, std)¶
```

Normalize a tensor image with mean and standard deviation.

See `Normalize` for more details.

 
| Parameters: | 

*   **tensor** ([_Tensor_](../tensors.html#torch.Tensor "torch.Tensor")) – Tensor image of size (C, H, W) to be normalized.
*   **mean** (_sequence_) – Sequence of means for each channel.
*   **std** (_sequence_) – Sequence of standard deviations for each channely.

 |
| --- | --- |
| Returns: | Normalized Tensor image. |
| --- | --- |
| Return type: | [Tensor](../tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

```py
torchvision.transforms.functional.pad(img, padding, fill=0, padding_mode='constant')¶
```

Pad the given PIL Image on all sides with speficified padding mode and fill value.

 
| Parameters: | 

*   **img** (_PIL Image_) – Image to be padded.
*   **padding** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)") _or_ [_tuple_](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)")) – Padding on each border. If a single int is provided this is used to pad all borders. If tuple of length 2 is provided this is the padding on left/right and top/bottom respectively. If a tuple of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.
*   **fill** – Pixel fill value for constant fill. Default is 0\. If a tuple of length 3, it is used to fill R, G, B channels respectively. This value is only used when the padding_mode is constant
*   **padding_mode** –

    Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant. constant: pads with a constant value, this value is specified with fill edge: pads with the last value on the edge of the image reflect: pads with reflection of image (without repeating the last value on the edge)

    &gt; padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode will result in [3, 2, 1, 2, 3, 4, 3, 2]

    ```py
    symmetric: pads with reflection of image (repeating the last value on the edge)
    ```

    padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode will result in [2, 1, 1, 2, 3, 4, 4, 3]

 |
| --- | --- |
| Returns: | Padded image. |
| --- | --- |
| Return type: | PIL Image |
| --- | --- |

```py
torchvision.transforms.functional.resize(img, size, interpolation=2)¶
```

Resize the input PIL Image to the given size.

 
| Parameters: | 

*   **img** (_PIL Image_) – Image to be resized.
*   **size** (_sequence_ _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Desired output size. If size is a sequence like (h, w), the output size will be matched to this. If size is an int, the smaller edge of the image will be matched to this number maintaing the aspect ratio. i.e, if height &gt; width, then image will be rescaled to (size * height / width, size)
*   **interpolation** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – Desired interpolation. Default is `PIL.Image.BILINEAR`

 |
| --- | --- |
| Returns: | Resized image. |
| --- | --- |
| Return type: | PIL Image |
| --- | --- |

```py
torchvision.transforms.functional.resized_crop(img, i, j, h, w, size, interpolation=2)¶
```

Crop the given PIL Image and resize it to desired size.

Notably used in RandomResizedCrop.

 
| Parameters: | 

*   **img** (_PIL Image_) – Image to be cropped.
*   **i** – Upper pixel coordinate.
*   **j** – Left pixel coordinate.
*   **h** – Height of the cropped image.
*   **w** – Width of the cropped image.
*   **size** (_sequence_ _or_ [_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Desired output size. Same semantics as `scale`.
*   **interpolation** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – Desired interpolation. Default is `PIL.Image.BILINEAR`.

 |
| --- | --- |
| Returns: | Cropped image. |
| --- | --- |
| Return type: | PIL Image |
| --- | --- |

```py
torchvision.transforms.functional.rotate(img, angle, resample=False, expand=False, center=None)¶
```

Rotate the image by angle.

 
| Parameters: | 

*   **img** (_PIL Image_) – PIL Image to be rotated.
*   **angle** (_{python:float__,_ _int}_) – In degrees degrees counter clockwise order.
*   **resample** (_{PIL.Image.NEAREST__,_ _PIL.Image.BILINEAR__,_ _PIL.Image.BICUBIC}__,_ _optional_) – An optional resampling filter. See [http://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters](http://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters) If omitted, or if the image has mode “1” or “P”, it is set to PIL.Image.NEAREST.
*   **expand** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – Optional expansion flag. If true, expands the output image to make it large enough to hold the entire rotated image. If false or omitted, make the output image the same size as the input image. Note that the expand flag assumes rotation around the center and no translation.
*   **center** (_2-tuple__,_ _optional_) – Optional center of rotation. Origin is the upper left corner. Default is the center of the image.

 |
| --- | --- |

```py
torchvision.transforms.functional.ten_crop(img, size, vertical_flip=False)¶
```

```py
Crop the given PIL Image into four corners and the central crop plus the
```

flipped version of these (horizontal flipping is used by default).

Note

&gt; This transform returns a tuple of images and there may be a mismatch in the number of inputs and targets your `Dataset` returns.

```py
Args:
```

&gt; ```py
&gt; size (sequence or int): Desired output size of the crop. If size is an
&gt; ```
&gt; 
&gt; int instead of sequence like (h, w), a square crop (size, size) is made.
&gt; 
&gt; vertical_flip (bool): Use vertical flipping instead of horizontal

```py
Returns:
```

```py
tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip,
```

br_flip, center_flip) corresponding top left, top right, bottom left, bottom right and center crop and same for the flipped image.

```py
torchvision.transforms.functional.to_grayscale(img, num_output_channels=1)¶
```

Convert image to grayscale version of image.

 
| Parameters: | **img** (_PIL Image_) – Image to be converted to grayscale. |
| --- | --- |
| Returns: | 

```py
Grayscale version of the image.
```

if num_output_channels == 1 : returned image is single channel if num_output_channels == 3 : returned image is 3 channel with r == g == b

 |
| --- | --- |
| Return type: | PIL Image |
| --- | --- |

```py
torchvision.transforms.functional.to_pil_image(pic, mode=None)¶
```

Convert a tensor or an ndarray to PIL Image.

See `ToPIlImage` for more details.

 
| Parameters: | 

*   **pic** ([_Tensor_](../tensors.html#torch.Tensor "torch.Tensor") _or_ [_numpy.ndarray_](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v1.15)")) – Image to be converted to PIL Image.
*   **mode** ([PIL.Image mode](http://pillow.readthedocs.io/en/latest/handbook/concepts.html#modes)) – color space and pixel depth of input data (optional).

 |
| --- | --- |

 
| Returns: | Image converted to PIL Image. |
| --- | --- |
| Return type: | PIL Image |
| --- | --- |

```py
torchvision.transforms.functional.to_tensor(pic)¶
```

Convert a `PIL Image` or `numpy.ndarray` to tensor.

See `ToTensor` for more details.

 
| Parameters: | **pic** (_PIL Image_ _or_ [_numpy.ndarray_](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v1.15)")) – Image to be converted to tensor. |
| --- | --- |
| Returns: | Converted image. |
| --- | --- |
| Return type: | [Tensor](../tensors.html#torch.Tensor "torch.Tensor") |
| --- | --- |

```py
torchvision.transforms.functional.vflip(img)¶
```

Vertically flip the given PIL Image.

 
| Parameters: | **img** (_PIL Image_) – Image to be flipped. |
| --- | --- |
| Returns: | Vertically flipped image. |
| --- | --- |
| Return type: | PIL Image |
| --- | --- |

