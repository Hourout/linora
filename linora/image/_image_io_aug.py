import numpy as np
import tensorflow as tf
from linora.image._image_io import *

class ImageIoAug(object):
    def __init__(self, image=None):
        self._image = image
    
    def read_image(self, filename, channel=0, image_format='mix'):
        """Convenience function for read image type one of `bmp`, `gif`, `jpeg`, `jpg`, and `png`.
    
        Detects whether an image is a BMP, GIF, JPEG, JPG, or PNG, and performs the
        appropriate operation to convert the input bytes `string` into a `Tensor`
        of type `dtype`.
        Note: `gif` returns a 4-D array `[num_frames, height, width, 3]`, as
        opposed to `bmp`, `jpeg`, `jpg` and `png`, which return 3-D
        arrays `[height, width, num_channels]`. Make sure to take this into account
        when constructing your graph if you are intermixing GIF files with BMP, JPEG, JPG,
        and/or PNG files.
        Args:
            filename: 0-D `string`. image absolute path.
            channels: An optional `int`. Defaults to `0`. Number of color channels for
                      the decoded image. 1 for `grayscale` and 3 for `rgb`.
            image_format: 0-D `string`. image format type one of `bmp`, `gif`, `jpeg`,
                          `jpg`, `png` and `mix`. `mix` mean contains many types image format.
        Returns:
            `Tensor` with type uint8 and shape `[height, width, num_channels]` for
            BMP, JPEG, and PNG images and shape `[num_frames, height, width, 3]` for
            GIF images.
        Raises:
            ValueError: On incorrect number of channels.
        """
        self._image = read_image(filename, channel, image_format, _=True)
        return self
    
    def save_image(self, filename):
        """Writes image to the file at input filename. 
    
        Args:
            image:    A Tensor of type string. scalar. The content to be written to the output file.
            filename: A string. scalar. The name of the file to which we write the contents.
        Raises:
            ValueError: If `filename` is not in `[`jpg`, `jpeg`, `png`]`.
        """
        return save_image(self._image, filename)
    
    def run(self):
        """return numpy array image."""
        return self._image if type(self._image)==np.ndarray else self._image.numpy()
    
    def show(self):
        """plot numpy array image."""
        return tf.keras.preprocessing.image.array_to_img((self._image if type(self._image)==np.ndarray else self._image.numpy()).astype('uint8'))
