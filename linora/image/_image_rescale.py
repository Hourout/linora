import numpy as np
from PIL import ImageOps

__all__ = ['rescale', 'normalize_global', 'normalize_channel', 'normalize_posterize']


def rescale(image, scale, p=1):
    """Rescale apply to image.
    
    new pixel = int(image * scale)
    Args:
        image: a PIL instance.
        scale: if int or float, value multiply with image.
               if tuple or list, randomly picked in the interval
               `[central_rate[0], central_rate[1])`, value multiply with image.
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(scale, (tuple, list)):
        scale = np.random.uniform(scale[0], scale[1])
    return image.point(lambda i: i*scale)


def normalize_global(image, mean=None, std=None, p=1):
    """Normalize scales `image` to have mean and variance.
    
    This op computes `(x - mean) / std`.
    Args:
        image: a numpy array. shape is `[height, width, channels]` or `[height, width]`.
        mean: if None, computes image mean.
              if int or float, customize image all channels mean.
              if tuple or list, randomly picked in the interval `[mean[0], mean[1])`
        std: if None, computes image std.
             if int or float, customize image all channels std.
             if tuple or list, randomly picked in the interval `[std[0], std[1])`
        p: probability that the image does this. Default value is 1.
    Returns:
        a numpy array.
    """
    if np.random.uniform()>p:
        return image
    if mean is None:
        mean = image.mean()
    elif isinstance(mean, (tuple, list)):
        mean = np.random.uniform(mean[0], mean[1])
    if std is None:
        std = image.std()
    elif isinstance(std, (tuple, list)):
        std = np.random.uniform(std[0], std[1])
    return (image-mean)/std


def normalize_channel(image, mean=None, std=None, p=1):
    """Normalize scales `image` to have mean and variance.
    
    This op computes `(x - mean) / std`.
    Args:
        image: a numpy array. shape is `[height, width, channels]` or `[height, width]`.
        mean: if None, computes image mean.
              if tuple or list, customize image each channels mean, shape should 3 dims.
        std: if None, computes image std.
             if tuple or list, customize image each channels std, shape should 3 dims.
        p: probability that the image does this. Default value is 1.
    Returns:
        a numpy array. The standardized image with same shape as `image`.
    """
    if np.random.uniform()>p:
        return image
    if mean is None:
        mean = [image[:,:,i].mean() for i in range(image.shape[2])]
    elif isinstance(mean, (int, float)):
        mean = [mean for i in range(image.shape[2])]
    if std is None:
        std = [image[:,:,i].std() for i in range(image.shape[2])]
    elif isinstance(std, (int, float)):
        std = [std for i in range(image.shape[2])]
    return (image-mean)/std


def normalize_posterize(image, bits, p=1):
    """Reduce the number of bits for each color channel.
    
    There are up to 2**bits types of pixel values per channel.
    Args:
        image: a PIL instance.
        bits: int or tuple or list, The number of bits to keep for each channel (1-8).
              if list or tuple, randomly picked in the interval `[bits[0], bits[1])` value.
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(bits, (list, tuple)):
        bits = np.random.randint(bits[0], bits[1])
    return ImageOps.posterize(image, bits)