import numpy as np
from PIL import Image

__all__ = ['crop', 'crop_central', 'crop_point']


def crop(image, box):
    """
    Returns a rectangular region from this image. The box is a
    4-tuple defining the left, upper, right, and lower pixel coordinate.

    Args:
        image: a Image instance.
        box: The crop rectangle, as a (left, upper, right, lower)-tuple.
    returns: 
        a Image instance.
    """
    return image.crop(box)

def crop_central(image, central_rate):
    """Crop the central region of the image.
    
    Remove the outer parts of an image but retain the central region of the image
    along each dimension. If we specify central_fraction = 0.5, this function
    returns the region marked with "X" in the below diagram.
       --------
      |        |
      |  XXXX  |
      |  XXXX  |
      |        |   where "X" is the central 50% of the image.
       --------
    
    Args:
        image: a Image instance.
        central_rate: if int float, should be in the interval (0, 1], fraction of size to crop.
                      if tuple list, randomly picked in the interval
                      `[central_rate[0], central_rate[1])`, value is fraction of size to crop.
    Returns:
        a Image instance.
    Raises:
        ValueError: if central_crop_fraction is not within (0, 1].
    """
    if isinstance(central_rate, (int, float)):
        assert 0<central_rate<=1, 'if central_rate type one of int or float, must be in the interval (0, 1].'
    elif isinstance(central_rate, (tuple, list)):
        assert 0<central_rate[0]<central_rate[1]<=1, 'central_rate should be 1 >= central_rate[1] > central_rate[0] > 0.'
        central_rate = np.random.uniform(central_rate[0], central_rate[1])
    else:
        raise ValueError('central_rate should be one of int, float, tuple, list.')
    left = int(im.size[0]*(0.5-central_rate/2))
    upper = int(im.size[1]*(0.5-central_rate/2))
    right = int(im.size[0]*(0.5+central_rate/2))
    lower = int(im.size[1]*(0.5+central_rate/2))
    return image.crop((left, upper, right, lower))

def crop_point(image, height_rate, width_rate):
    """Crop the any region of the image.
    
    Crop region area = height_rate * width_rate *image_height * image_width
    
    Args:
        image: a Image instance.
        height_rate: flaot, in the interval (0, 1].
        width_rate: flaot, in the interval (0, 1].
    Returns:
        a Image instance.
    Raises:
        ValueError: if central_crop_fraction is not within (0, 1].
    """
    assert 0<height_rate<=1 and 0<width_rate<=1, 'height_rate and width_rate should be in the interval (0, 1].'
    left = image.size[0]*np.random.uniform(0, 1-width_rate)
    upper = image.size[1]*np.random.uniform(0, 1-height_rate)
    right = left+image.size[0]*width_rate
    lower = upper+image.size[1]*height_rate
    return image.crop((left, upper, right, lower))
