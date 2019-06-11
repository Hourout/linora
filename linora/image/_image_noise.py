import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import _ImageDimensions

__all__ = ['RandomNoiseGaussian', 'RandomNoisePoisson', 'RandomNoiseMask',
           'RandomNoiseSaltPepper', 'RandomNoiseRainbow']

def RandomNoiseGaussian(image, scale=1, mean=0.0, std=1.0, seed=None, **kwarg):
    """Gaussian noise apply to image.
    
    new pixel = image + gaussian_noise * scale
    Args:
        image: Either a 3-D float Tensor of shape [height, width, depth], or a 4-D
               Tensor of shape [batch_size, height, width, depth].
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
    if isinstance(scale, (int, float)):
        new_scale = scale
    else:
        assert isinstance(scale, (tuple, list)), 'scale type should be one of int, float, tuple, list.'
        new_scale = tf.random.uniform([], scale[0], scale[1], seed=seed)
    image = tf.cast(image, dtype=tf.float32)
    image_shape = tf.cast(_ImageDimensions(image, image.get_shape().ndims), dtype=tf.int32)
    if isinstance(mean, (int, float)):
        if isinstance(std, (int, float)):
            image = tf.math.add(tf.math.multiply(tf.random.normal(image_shape, mean, std, seed=seed), new_scale), image)
        elif isinstance(std, (tuple, list)):
            random_std = tf.random.uniform([], std[0], std[1])
            image = tf.math.add(tf.math.multiply(tf.random.normal(image_shape, mean, random_std, seed=seed), new_scale), image)
        else:
            raise ValueError('std type should be one of int, float, tuple, list.')
    elif isinstance(mean, (tuple, list)):
        if isinstance(std, (int, float)):
            random_mean = tf.random.uniform([], mean[0], mean[1])
            image = tf.math.add(tf.math.multiply(tf.random.normal(image_shape, random_mean, std, seed=seed), new_scale), image)
        elif isinstance(std, (tuple, list)):
            random_mean = tf.random.uniform([], mean[0], mean[1])
            random_std = tf.random.uniform([], std[0], std[1])
            image = tf.math.add(tf.math.multiply(tf.random.normal(image_shape, random_mean, random_std, seed=seed), new_scale), image)
        else:
            raise ValueError('std type should be one of int, float, tuple, list.')
    else:
        raise ValueError('mean type should be one of int, float, tuple, list.')
    return image if kwarg else image.numpy()

def RandomNoisePoisson(image, scale=1, lam=1.0, seed=None, **kwarg):
    """Poisson noise apply to image.
    
    new pixel = image + poisson_noise * scale
    Args:
        image: Either a 3-D float Tensor of shape [height, width, depth], or a 4-D
               Tensor of shape [batch_size, height, width, depth].
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
    if isinstance(scale, (int, float)):
        new_scale = scale
    else:
        assert isinstance(scale, (tuple, list)), 'scale type should be one of int, float, tuple, list.'
        new_scale = tf.random.uniform([], scale[0], scale[1], seed=seed)
    image = tf.cast(image, dtype=tf.float32)
    image_shape = tf.cast(_ImageDimensions(image, image.get_shape().ndims), dtype=tf.int32)
    if isinstance(lam, (int, float)):
        image = tf.math.add(tf.math.multiply(tf.random.poisson(image_shape, lam, seed=seed), new_scale), image)
    elif isinstance(lam, (tuple, list)):
        random_lam = tf.random.uniform([], lam[0], lam[1])
        image = tf.math.add(tf.math.multiply(tf.random.poisson(image_shape, random_lam, seed=seed), new_scale), image)
    else:
        raise ValueError('lam type should be one of int, float, tuple, list.')
    return image if kwarg else image.numpy()

def RandomNoiseMask(image, keep_prob=0.95, seed=None, **kwarg):
    """Mask noise apply to image.
    
    With probability `drop_prob`, outputs the input element scaled up by
    `1`, otherwise outputs `0`. 
    
    Tips:
        1 mean pixel have no change.
        a suitable interval is (0., 0.1].
    Args:
        image: Either a 3-D float Tensor of shape [height, width, depth], or a 4-D
               Tensor of shape [batch_size, height, width, depth].
        keep_prob: should be in the interval (0, 1.].
                   if float, the probability that each element is drop.
                   if tuple or list, randomly picked in the interval
                   `[keep_prob[0], keep_prob[1])`, the probability that each element is drop.
        seed: A Python integer. Used to create a random seed.
              See `tf.set_random_seed` for behavior.
    Returns:
        3-D / 4-D float Tensor, as per the input.
    Raises:
        ValueError: If `keep_prob` is not in `(0, 1.]`.
    """
    image = tf.cast(image, dtype=tf.float32)
    image_shape = tf.cast(_ImageDimensions(image, image.get_shape().ndims), dtype=tf.int32)
    if isinstance(keep_prob, float):
        mask = tf.clip_by_value(tf.nn.dropout(tf.random.uniform(image_shape, 1., 2.), keep_prob), 0., 1.)
        image = tf.math.multiply(mask, image)
    elif isinstance(keep_prob, (tuple, list)):
        random_keep_prob = tf.random.uniform([], keep_prob[0], keep_prob[1], seed=seed)
        mask = tf.clip_by_value(tf.nn.dropout(tf.random.uniform(image_shape, 1., 2.), random_keep_prob), 0., 1.)
        image = tf.math.multiply(mask, image)
    else:
        raise ValueError('keep_prob type should be one of float, tuple, list.')
    return image if kwarg else image.numpy()

def RandomNoiseSaltPepper(image, keep_prob=0.95, seed=None, **kwarg):
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
    image = tf.cast(image, dtype=tf.float32)
    if image.get_shape().ndims==4:
        shape = tf.cast(_ImageDimensions(image, 4), dtype=tf.int32)
        image_shape = [shape[0], shape[1], shape[2], 1]
    else:
        shape = tf.cast(_ImageDimensions(image, image.get_shape().ndims), dtype=tf.int32)
        image_shape = [shape[0], shape[1], 1]
    if isinstance(keep_prob, (int, float)):
        noise = tf.clip_by_value(tf.math.floor(tf.random.uniform(image_shape, 0.5-0.5/keep_prob, 0.5+0.5/keep_prob)), -1, 1)
        image = tf.clip_by_value(tf.math.add(tf.math.multiply(noise, 255.), image), 0, 255)
    elif isinstance(keep_prob, (tuple, list)):
        random_keep_prob = tf.random.uniform([], keep_prob[0], keep_prob[1], seed=seed)
        noise = tf.clip_by_value(tf.math.floor(tf.random.uniform(image_shape, 0.5-0.5/random_keep_prob, 0.5+0.5/random_keep_prob)), -1, 1)
        image = tf.clip_by_value(tf.math.add(tf.math.multiply(noise, 255.), image), 0, 255)
    else:
        raise ValueError('keep_prob type should be one of int, float, tuple, list.')
    return image if kwarg else image.numpy()

def RandomNoiseRainbow(image, keep_prob=0.95, seed=None, **kwarg):
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
    image = tf.cast(image, dtype=tf.float32)
    image_shape = tf.cast(_ImageDimensions(image, image.get_shape().ndims), dtype=tf.int32)
    if isinstance(keep_prob, (int, float)):
        noise = tf.clip_by_value(tf.math.floor(tf.random.uniform(image_shape, 0.5-0.5/keep_prob, 0.5+0.5/keep_prob)), -1, 1)
        image = tf.clip_by_value(tf.math.add(tf.math.multiply(noise, 255.), image), 0, 255)
    elif isinstance(keep_prob, (tuple, list)):
        random_keep_prob = tf.random.uniform([], keep_prob[0], keep_prob[1], seed=seed)
        noise = tf.clip_by_value(tf.math.floor(tf.random.uniform(image_shape, 0.5-0.5/random_keep_prob, 0.5+0.5/random_keep_prob)), -1, 1)
        image = tf.clip_by_value(tf.math.add(tf.math.multiply(noise, 255.), image), 0, 255)
    else:
        raise ValueError('keep_prob type should be one of int, float, tuple, list.')
    return image if kwarg else image.numpy()
