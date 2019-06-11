import tensorflow as tf

__all__ = ['Normalize', 'RandomRescale']
           
def Normalize(image, mean=None, std=None, **kwarg):
    """Normalize scales `image` to have mean and variance.
    
    This op computes `(x - mean) / std`.
    Args:
        image: An n-D Tensor where the last 3 dimensions are
               `[height, width, channels]`.
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
    image = tf.cast(image, dtype=tf.float32)
    assert image.get_shape().ndims==3, 'image ndims must be 3.'
    if mean is None and std is None:
        image = tf.image.per_image_standardization(image)
    else:
        assert isinstance(mean, (int, float, tuple, list)), 'mean type one of int, float, tuple, list.'
        assert isinstance(std, (int, float, tuple, list)), 'std type one of int, float, tuple, list.'
        image = tf.math.divide(tf.math.subtract(image, mean), std)
    return image if kwarg else image.numpy()

def RandomRescale(image, scale, seed=None, **kwarg):
    """Rescale apply to image.
    
    new pixel = image * scale
    Args:
        image: Either a 3-D float Tensor of shape [height, width, depth], or a 4-D
               Tensor of shape [batch_size, height, width, depth].
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
    image = tf.cast(image, dtype=tf.float32)
    if isinstance(scale, (int, float)):
        image = tf.math.multiply(image, scale)
    elif isinstance(scale, (tuple, list)):
        random_scale = tf.random.uniform([], scale[0], scale[1], seed=seed)
        image = tf.math.multiply(image, random_scale)
    else:
        raise ValueError('scale type should be one of int, float, tuple, list.')
    return image if kwarg else image.numpy()
