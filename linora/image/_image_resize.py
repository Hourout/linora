import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import _ImageDimensions

__all__ = ['RandomCropCentralResize', 'RandomCropPointResize']


def resize_method(mode):
    mode_dict = {0:tf.image.ResizeMethod.BILINEAR, 1:tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                 2:tf.image.ResizeMethod.BICUBIC, 3:tf.image.ResizeMethod.AREA,
                 4:tf.image.ResizeMethod.LANCZOS3, 5:tf.image.ResizeMethod.LANCZOS5,
                 6:tf.image.ResizeMethod.GAUSSIAN, 7:tf.image.ResizeMethod.MITCHELLCUBIC}
    return mode_dict[mode]

def RandomCropCentralResize(image, central_rate, size, method=0, seed=None, **kwarg):
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
        method should be one of [0, 1, 2, 3, 4, 5, 6, 7], "0:bilinear", "1:nearest", "2:bicubic",
        "3:area", "4:lanczos3", "5:lanczos5", "6:gaussian", "7:mitchellcubic".
    Args:
        image: Either a 3-D float Tensor of shape [height, width, depth], or a 4-D
               Tensor of shape [batch_size, height, width, depth].
        central_rate: if int float, should be in the interval (0, 1], fraction of size to crop.
                      if tuple list, randomly picked in the interval
                      `[central_rate[0], central_rate[1])`, value is fraction of size to crop.
        size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.
              The new size for the images.
        method: int, default 0. resize image shape method.
                should be one of "0:bilinear", "1:nearest", "2:bicubic", "3:area", "4:lanczos3",
                "5:lanczos5", "6:gaussian", "7:mitchellcubic".
        seed: A Python integer. Used to create a random seed.
              See `tf.set_random_seed` for behavior.
    Returns:
        3-D / 4-D float Tensor, as per the input.
    Raises:
        ValueError: if central_crop_fraction is not within (0, 1].
    """
    assert isinstance(size, (tuple, list)), 'size should be one of tuple, list.'
    assert method in [0, 1, 2, 3, 4, 5, 6, 7], """method should be one of "0:bilinear", "1:nearest", "2:bicubic",
    "3:area", "4:lanczos3", "5:lanczos5", "6:gaussian", "7:mitchellcubic" """
    if isinstance(central_rate, (int, float)):
        assert 0<central_rate<=1, 'if central_rate type one of int or float, must be in the interval (0, 1].'
        image = tf.image.central_crop(image, central_fraction=central_rate)
    elif isinstance(central_rate, (tuple, list)):
        assert 0<central_rate[0]<central_rate[1]<=1, 'central_rate should be 1 >= central_rate[1] > central_rate[0] > 0.'
        random_central_rate = tf.random.uniform([], central_rate[0], central_rate[1], seed=seed)
        image = tf.image.central_crop(image, central_fraction=random_central_rate)
    else:
        raise ValueError('central_rate should be one of int, float, tuple, list.')
    image = tf.image.resize(image, size=size, method=resize_method(method))
    return image if kwarg else image.numpy()

def RandomCropPointResize(image, height_rate, width_rate, size, method=0, seed=None, **kwarg):
    """Crop the any region of the image(s) and resize specify shape.
    
    Crop region area = height_rate * width_rate *image_height * image_width
    
    This function works on either a single image (`image` is a 3-D Tensor), or a
    batch of images (`image` is a 4-D Tensor).
    
    Tips:
        method should be one of [0, 1, 2, 3, 4, 5, 6, 7], "0:bilinear", "1:nearest", "2:bicubic",
        "3:area", "4:lanczos3", "5:lanczos5", "6:gaussian", "7:mitchellcubic".
    Args:
        image: Either a 3-D float Tensor of shape [height, width, depth], or a 4-D
               Tensor of shape [batch_size, height, width, depth].
        height_rate: flaot, in the interval (0, 1].
        width_rate: flaot, in the interval (0, 1].
        size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.
              The new size for the images.
        method: int, default 0. resize image shape method.
                should be one of "0:bilinear", "1:nearest", "2:bicubic", "3:area", "4:lanczos3",
                "5:lanczos5", "6:gaussian", "7:mitchellcubic".
        seed: A Python integer. Used to create a random seed.
              See `tf.set_random_seed` for behavior.
    Returns:
        3-D / 4-D float Tensor, as per the input.
    Raises:
        ValueError: if central_crop_fraction is not within (0, 1].
    """
    assert isinstance(height_rate, (int, float)), 'height_rate should be one of int, float.'
    assert isinstance(width_rate, (int, float)), 'width_rate should be one of int, float.'
    assert 0<height_rate<=1 and 0<width_rate<=1, 'height_rate and width_rate should be in the interval (0, 1].'
    assert isinstance(size, (tuple, list)), 'size should be one of tuple, list.'
    assert method in [0, 1, 2, 3, 4, 5, 6, 7], """method should be one of "0:bilinear", "1:nearest", "2:bicubic",
    "3:area", "4:lanczos3", "5:lanczos5", "6:gaussian", "7:mitchellcubic" """
    image = tf.cast(image, dtype=tf.float32)
    shape = tf.cast(_ImageDimensions(image, image.get_shape().ndims), dtype=tf.float32)
    offset_height = tf.cast(tf.math.multiply(tf.random.uniform([], 0, 1-height_rate, seed=seed), shape[-3]), dtype=tf.int32)
    offset_width = tf.cast(tf.math.multiply(tf.random.uniform([], 0, 1-width_rate, seed=seed), shape[-2]), dtype=tf.int32)
    target_height = tf.cast(tf.math.multiply(height_rate, shape[-3]), dtype=tf.int32)
    target_width = tf.cast(tf.math.multiply(width_rate, shape[-2]), dtype=tf.int32)
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
    image = tf.image.resize(image, size=size, method=resize_method(method))
    return image if kwarg else image.numpy()
