from PIL import ImageOps

import numpy as np

__all__ = ['rescale', 'normalize_global', 'normalize_channel', 'normalize_posterize']


def rescale(image, scale):
    """Rescale apply to image.
    
    new pixel = image * scale
    Args:
        image: a PIL instance.
        scale: if int or float, value multiply with image.
               if tuple or list, randomly picked in the interval
               `[central_rate[0], central_rate[1])`, value multiply with image.
    Returns:
        a PIL instance.
    Raises:
        scale type error.
    """
    if isinstance(scale, (tuple, list)):
        scale = np.random.uniform(scale[0], scale[1])
    elif not isinstance(scale, (int, float)):
        raise ValueError('scale type should be one of int, float, tuple, list.')
    return image.point(lambda i: i*scale)


def normalize_global(image, mean=None, std=None):
    """Normalize scales `image` to have mean and variance.
    
    This op computes `(x - mean) / std`.
    Args:
        image: a numpy array.
            shape is `[height, width, channels]` or `[height, width]`.
        mean: if None, computes image mean.
              if int or float, customize image all channels mean.
        std: if None, computes image std.
             if int or float, customize image all channels std.
      Returns:
        a numpy array. The standardized image with same shape as `image`.
      Raises:
        ValueError: if the shape of 'image' is incompatible with this function.
      """
    if mean is None:
        mean = image.mean()
    elif not isinstance(mean, (int, float)):
        raise ValueError('`mean` must be int or float.')
    if std is None:
        std = image.std()
    elif not isinstance(std, (int, float)):
        raise ValueError('`std` must be int or float.')
    return (image-mean)/std


def normalize_channel(image, mean=None, std=None):
    """Normalize scales `image` to have mean and variance.
    
        This op computes `(x - mean) / std`.
        Args:
            image: a numpy array.
                shape is `[height, width, channels]` or `[height, width]`.
            mean: if None, computes image mean.
                  if tuple or list, customize image each channels mean,
                  shape should 3 dims.
            std: if None, computes image std.
                 if tuple or list, customize image each channels std,
                 shape should 3 dims.
      Returns:
        a numpy array. The standardized image with same shape as `image`.
      Raises:
        ValueError: if the shape of 'image' is incompatible with this function.
      """
    if mean is None:
        mean = [image[:,:,i].mean() for i in range(image.shape[2])]
    elif not isinstance(mean, (list, tuple)):
        raise ValueError('`mean` must be list or tuple.')
    if std is None:
        std = [image[:,:,i].std() for i in range(image.shape[2])]
    elif not isinstance(std, (list, tuple)):
        raise ValueError('`std` must be list or tuple.')
    return (image-mean)/std


def normalize_posterize(image, bits):
    """Reduce the number of bits for each color channel.
    
    There are up to 2**bits types of pixel values per channel.
    Args:
        image: a PIL instance.
        bits: int or tuple or list, The number of bits to keep for each channel (1-8).
              if list or tuple, randomly picked in the interval `[bits[0], bits[1])` value.
    Returns:
        A PIL instance.
    """
    if isinstance(bits, (list, tuple)):
        bits = np.random.randint(bits[0], bits[1])
    return ImageOps.posterize(image, bits)