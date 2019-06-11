import numpy as np
import tensorflow as tf
from linora.image._image_noise import *

class ImageNoiseAug(object):
    def __init__(self, image=None):
        self._image = image
    
    def RandomNoiseGaussian(self, scale=1, mean=0.0, std=1.0, seed=None):
        """Gaussian noise apply to image.
    
        new pixel = image + gaussian_noise * scale
        Args:
            scale: if int or float, value multiply with poisson_noise.
                   if tuple or list, randomly picked in the interval
                   `[scale[0], scale[1])`, value multiply with poisson_noise.
            mean: if int or float, value is gaussian distribution mean.
                  if tuple or list, randomly picked in the interval
                  `[mean[0], mean[1])`, value is gaussian distribution mean.
            std: if int or float, value is gaussian distribution std.
                 if tuple or list, randomly picked in the interval
                 `[std[0], std[1])`, value is gaussian distribution std.
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            3-D / 4-D float Tensor, as per the input.
        Raises:
            scale or lam type error.
        """
        self._image = RandomNoiseGaussian(self._image, scale, mean, std, seed, _=True)
        return self
    
    def RandomNoisePoisson(self, scale=1, lam=1.0, seed=None):
        """Poisson noise apply to image.
    
        new pixel = image + poisson_noise * scale
        Args:
            scale: if int or float, value multiply with poisson_noise.
                   if tuple or list, randomly picked in the interval
                   `[scale[0], scale[1])`, value multiply with poisson_noise.
            lam: if int or float, value is poisson distribution lambda.
                 if tuple or list, randomly picked in the interval
                 `[lam[0], lam[1])`, value is poisson distribution lambda.
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            3-D / 4-D float Tensor, as per the input.
        Raises:
            scale or lam type error.
        """
        self._image = RandomNoisePoisson(self._image, scale, lam, seed, _=True)
        return self
    
    def RandomNoiseMask(self, keep_prob=0.95, seed=None):
        """Mask noise apply to image.
    
        With probability `keep_prob`, outputs the input element scaled up by
        `1`, otherwise outputs `0`. 
        Tips:
            1 mean pixel have no change.
            a suitable interval is [0.9, 1].
        Args:
            image: Either a 3-D float Tensor of shape [height, width, depth], or a 4-D
                   Tensor of shape [batch_size, height, width, depth].
            keep_prob: should be in the interval (0, 1].
                       if int or float, the probability that each element is kept.
                       if tuple or list, randomly picked in the interval
                       `[keep_prob[0], keep_prob[1])`, the probability that each element is kept.
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            3-D / 4-D float Tensor, as per the input.
        Raises:
            ValueError: If `keep_prob` is not in `(0, 1]`.
        """
        self._image = RandomNoiseMask(self._image, keep_prob, seed, _=True)
        return self
    
    def RandomNoiseSaltPepper(self, keep_prob=0.95, seed=None):
        """Salt-Pepper noise apply to image.
    
        The salt-pepper noise is based on the signal-to-noise ratio of the image,
        randomly generating the pixel positions in some images all channel,
        and randomly assigning these pixels to 0 or 255.
        Tips:
            1 mean pixel have no change.
            a suitable interval is [0.9, 1].
        Args:
            image: Either a 3-D float Tensor of shape [height, width, depth], or a 4-D
                   Tensor of shape [batch_size, height, width, depth].
            keep_prob: should be in the interval (0, 1].
                       if int or float, the probability that each element is kept.
                       if tuple or list, randomly picked in the interval
                       `[keep_prob[0], keep_prob[1])`, the probability that each element is kept.
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            3-D / 4-D float Tensor, as per the input.
        Raises:
            ValueError: If `keep_prob` is not in `(0, 1]`.
        """
        self._image = RandomNoiseSaltPepper(self._image, keep_prob, seed, _=True)
        return self
    
    def RandomNoiseRainbow(self, keep_prob=0.95, seed=None):
        """Rainbowr noise apply to image.
    
        The rainbow noise is based on the signal-to-noise ratio of the image,
        randomly generating the pixel positions in some images,
        and randomly assigning these pixels to 0 or 255.
        Tips:
            1 mean pixel have no change.
            a suitable interval is [0.9, 1].
        Args:
            image: Either a 3-D float Tensor of shape [height, width, depth], or a 4-D
                   Tensor of shape [batch_size, height, width, depth].
            keep_prob: should be in the interval (0, 1].
                       if int or float, the probability that each element is kept.
                       if tuple or list, randomly picked in the interval
                       `[keep_prob[0], keep_prob[1])`, the probability that each element is kept.
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            3-D / 4-D float Tensor, as per the input.
        Raises:
            ValueError: If `keep_prob` is not in `(0, 1]`.
        """
        self._image = RandomNoiseRainbow(self._image, keep_prob, seed, _=True)
        return self
