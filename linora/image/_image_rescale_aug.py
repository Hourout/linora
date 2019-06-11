import tensorflow as tf
from linora.image._image_rescale import *

class ImageRescaleAug(object):
    def __init__(self, image=None):
        self._image = image
    
    def Normalize(self, mean=None, std=None):
        """Normalize scales `image` to have mean and variance.
    
        This op computes `(x - mean) / std`.
        Args:
            mean: if None, computes image mean.
                  if int float, customize image all channels mean.
                  if tuple list, customize image each channels mean,
                  shape should 3 dims.
            std: if None, computes image std.
                 if int float, customize image all channels std.
                 if tuple list, customize image each channels std,
                 shape should 3 dims.
        Returns:
            The standardized image with same shape as `image`.
        Raises:
            ValueError: if the shape of 'image' is incompatible with this function.
        """
        self._image = Normalize(self._image, mean, std, _=True)
        return self
    
    def RandomRescale(self, scale, seed=None):
        """Rescale apply to image.
    
        new pixel = image * scale
        Args:
            scale: if int float, value multiply with image.
                   if tuple list, randomly picked in the interval
                   `[central_rate[0], central_rate[1])`, value multiply with image.
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            3-D / 4-D float Tensor, as per the input.
        Raises:
            scale type error.
        """
        self._image = RandomRescale(self._image, scale, seed, _=True)
        return self
