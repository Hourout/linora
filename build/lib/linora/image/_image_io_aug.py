import numpy as np
from linora.image._image_io import *
from linora.image._image_util import *

class ImageIoAug(object):
    def __init__(self, image=None):
        self.image = image
    
    def read_image(self, filename):
        """
        Args:
            filename: str, image absolute path.
        Returns:
            a Image instance.
        """
        self.image = read_image(filename)
        return self
    
    def save_image(self, filename, file_format=None, **kwargs):
        """Saves an image stored as a Numpy array to a path or file object.
        Args
            filename: Path or file object.
            image: A PIL Image instance.
            file_format: Optional file format override. If omitted, the
                format to use is determined from the filename extension.
                If a file object was used instead of a filename, this
                parameter should always be used.
            **kwargs: Additional keyword arguments passed to `PIL.Image.save()`.
        """
        return save_image(filename, self.image, file_format=file_format, **kwargs)
    
    def color_convert(self, color_mode='RGB'):
        """Transform image color mode

        Args
            img: PIL Image instance.
            color_mode: Image color mode, more see la.image.ColorMode
        Returns
            PIL Image instance.
        Raises
            ValueError: color_mode error.
        """
        self.image = color_convert(self.image, color_mode=color_mode)
        return self
    
    def image_to_array(self, data_format='channels_last', dtype='float32'):
        """Converts a PIL Image instance to a Numpy array.
        
        Args
            img: PIL Image instance.
            data_format: Image data format,
                either "channels_first" or "channels_last".
            dtype: Dtype to use for the returned array.
        Returns
            A 3D Numpy array.
        Raises
            ValueError: if invalid `img` or `data_format` is passed.
        """
        self.image = image_to_array(self.image, data_format, dtype)
        return self
        
    def array_to_image(self, data_format='channels_last'):
        """Converts a 3D Numpy array to a PIL Image instance.
        Args
            x: Input Numpy array.
            data_format: Image data format, either "channels_first" or "channels_last".
                Default: "channels_last".
        Returns
            A PIL Image instance.
        Raises
            ValueError: if invalid `x` or `data_format` is passed.
        """
        self.image = array_to_image(self.image, data_format)
        return self
