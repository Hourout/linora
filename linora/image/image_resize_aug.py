import tensorflow as tf
from linora.image._image_resize import *

class ImageResizeAug(object):
    def __init__(self, image=None):
        self._image = image
        
    def RandomCropCentralResize(self, central_rate, size, method=0, seed=None):
        """Crop the central region of the image(s) and resize specify shape.
    
        Remove the outer parts of an image but retain the central region of the image
        along each dimension. If we specify central_fraction = 0.5, this function
        returns the region marked with "X" in the below diagram.
           --------
          |        |
          |  XXXX  |
          |  XXXX  |
          |        |   where "X" is the central 50% of the image.
           --------
        This function works on either a single image (`image` is a 3-D Tensor), or a
        batch of images (`image` is a 4-D Tensor).
        Tips:
            method should be one of [0, 1, 2, 3], "0:bilinear", "1:nearest_neighbor", "2:bicubic", "3:area".
        Args:
            central_rate: if int float, should be in the interval (0, 1], fraction of size to crop.
                          if tuple list, randomly picked in the interval
                          `[central_rate[0], central_rate[1])`, value is fraction of size to crop.
            size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.
                  The new size for the images.
            method: int, default 0. resize image shape method.
                    should be one of "0:bilinear", "1:nearest_neighbor", "2:bicubic", "3:area"
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            3-D / 4-D float Tensor, as per the input.
        Raises:
            ValueError: if central_crop_fraction is not within (0, 1].
        """
        self._image = RandomCropCentralResize(self._image, central_rate, size, method, seed, _=True)
        return self
    
    def RandomCropPointResize(self, height_rate, width_rate, size, method=0, seed=None):
        """Crop the any region of the image(s) and resize specify shape.
    
        Crop region area = height_rate * width_rate *image_height * image_width
        This function works on either a single image (`image` is a 3-D Tensor), or a
        batch of images (`image` is a 4-D Tensor).
        Tips:
            method should be one of [0, 1, 2, 3], "0:bilinear", "1:nearest_neighbor", "2:bicubic", "3:area".
        Args:
            height_rate: flaot, in the interval (0, 1].
            width_rate: flaot, in the interval (0, 1].
            size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.
                  The new size for the images.
            method: int, default 0. resize image shape method.
                    should be one of "0:bilinear", "1:nearest_neighbor", "2:bicubic", "3:area"
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            3-D / 4-D float Tensor, as per the input.
        Raises:
            ValueError: if central_crop_fraction is not within (0, 1].
        """
        self._image = RandomCropPointResize(self._image, height_rate, width_rate, size, method, seed, _=True)
        return self
