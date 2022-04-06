import numpy as np

__all__ = ['normalize_global', 'normalize_channel', 'rescale']

def normalize_global(image, mean=None, std=None):
    """Normalize scales `image` to have mean and variance.
    
    This op computes `(x - mean) / std`.
    Args:
        image: An n-D Tensor where the last 3 dimensions are
               `[height, width, channels]`.
        mean: if None, computes image mean.
              if int or float, customize image all channels mean.
        std: if None, computes image std.
             if int or float, customize image all channels std.
      Returns:
        The standardized image with same shape as `image`.
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
            image: An n-D Tensor where the last 3 dimensions are
                   `[height, width, channels]`.
            mean: if None, computes image mean.
                  if tuple or list, customize image each channels mean,
                  shape should 3 dims.
            std: if None, computes image std.
                 if tuple or list, customize image each channels std,
                 shape should 3 dims.
      Returns:
        The standardized image with same shape as `image`.
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

def rescale(image, scale):
    """Rescale apply to image.
    
    new pixel = image * scale
    Args:
        image: a Image instance.
        scale: if int float, value multiply with image.
               if tuple list, randomly picked in the interval
               `[central_rate[0], central_rate[1])`, value multiply with image.
    Returns:
        a Image instance.
    Raises:
        scale type error.
    """
    if isinstance(scale, (tuple, list)):
        scale = np.random.uniform(scale[0], scale[1])
    elif not isinstance(scale, (int, float)):
        raise ValueError('scale type should be one of int, float, tuple, list.')
    return image.point(lambda i: i*scale)
