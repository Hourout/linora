import numpy as np
import tensorflow as tf
from linora.image._image_position import *

class ImagePositionAug(object):
    def __init__(self, image=None):
        self._image = image
    
    def RandomFlipLeftRight(self, random=True, seed=None):
        """Randomly flip an image horizontally (left to right).
    
        With a 1 in 2 chance, outputs the contents of `image` flipped along the
        second dimension, which is `width`.  Otherwise output the image as-is.
        Args:
            random: bool, default True.
                    if True, random flip left or rignt image.
                    if False, flip left or right image.
            seed: A Python integer. Used to create a random seed. See
                  `tf.set_random_seed` for behavior.
        Returns:
            A tensor of the same type and shape as `image`.
        Raises:
            ValueError: if the shape of `image` not supported or `random` dtype not bool.
        """
        self._image = RandomFlipLeftRight(self._image, random, seed, _=True)
        return self
    
    def RandomFlipTopBottom(self, random=True, seed=None):
        """Randomly flips an image vertically (upside down).
    
        With a 1 in 2 chance, outputs the contents of `image` flipped along the first
        dimension, which is `height`.  Otherwise output the image as-is.
        Args:
            random: bool, default True.
                    if True, random flip top or bottom image.
                    if False, flip top or bottom image.
            seed: A Python integer. Used to create a random seed. See
                  `tf.set_random_seed` for behavior.
        Returns:
            A tensor of the same type and shape as `image`.
        Raises:
            ValueError: if the shape of `image` not supported or `random` dtype not bool.
        """
        self._image = RandomFlipTopBottom(self._image, random, seed, _=True)
        return self
    
    def RandomTranspose(self, random=True, seed=None):
        """Transpose image(s) by swapping the height and width dimension.
    
        Args:
            random: bool, default True.
                    if True, random transpose image.
                    if False, transpose image.
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            If `image` was 4-D, a 4-D float Tensor of shape `[batch, width, height, channels]`.
            If `image` was 3-D, a 3-D float Tensor of shape `[width, height, channels]`.
        Raises:
            ValueError: if the shape of `image` not supported or `random` dtype not bool.
        """
        self._image = RandomTranspose(self._image, random, seed, _=True)
        return self
    
    def RandomRotation(self, k=[0, 1, 2, 3], seed=None):
        """Rotate image(s) counter-clockwise by 90 degrees.
    
        Tips:
            k should be int one of [1, 2, 3] or sublist in the [0, 1, 2, 3].
        Args:
            k: if k is list, random select t form k, rotation image by 90 degrees * t.
               if k is int, rotation image by 90 degrees * k.
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
          A rotated tensor of the same type and shape as `image`.
        Raises:
          ValueError: if the shape of `image` not supported or `k` dtype not int or list.
        """
        self._image = RandomRotation(self._image, k, seed, _=True)
        return self
