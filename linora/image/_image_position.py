import tensorflow as tf

__all__ = ['RandomFlipLeftRight', 'RandomFlipTopBottom', 'RandomTranspose', 'RandomRotation']

def RandomFlipLeftRight(image, random=True, seed=None, **kwarg):
    """Randomly flip an image horizontally (left to right).
    
    With a 1 in 2 chance, outputs the contents of `image` flipped along the
    second dimension, which is `width`.  Otherwise output the image as-is.
    Args:
        image: 4-D Tensor of shape `[batch, height, width, channels]` or
               3-D Tensor of shape `[height, width, channels]`.
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
    assert isinstance(random, bool), 'random should be bool type.'
    if random:
        image = tf.image.random_flip_left_right(image, seed=seed)
    else:
        image = tf.image.flip_left_right(image)
    return image if kwarg else image.numpy()

def RandomFlipTopBottom(image, random=True, seed=None, **kwarg):
    """Randomly flips an image vertically (upside down).
    
    With a 1 in 2 chance, outputs the contents of `image` flipped along the first
    dimension, which is `height`.  Otherwise output the image as-is.
    Args:
        image: 4-D Tensor of shape `[batch, height, width, channels]` or
               3-D Tensor of shape `[height, width, channels]`.
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
    assert isinstance(random, bool), 'random should be bool type.'
    if random:
        image = tf.image.random_flip_up_down(image, seed=seed)
    else:
        image = tf.image.flip_up_down(image)
    return image if kwarg else image.numpy()

def RandomTranspose(image, random=True, seed=None, **kwarg):
    """Transpose image(s) by swapping the height and width dimension.
    
    Args:
        image: 4-D Tensor of shape `[batch, height, width, channels]` or
               3-D Tensor of shape `[height, width, channels]`.
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
    assert isinstance(random, bool), 'random should be bool type.'
    if random:
        r = tf.random.uniform([2], 0, 1, seed=seed)
        image = tf.case([(tf.less(r[0], r[1]), lambda: tf.image.transpose(image))],
                        default=lambda: image)
    else:
        image = tf.image.transpose(image)
    return image if kwarg else image.numpy()

def RandomRotation(image, k=[0, 1, 2, 3], seed=None, **kwarg):
    """Rotate image(s) counter-clockwise by 90 degrees.
    
    Tips:
        k should be int one of [1, 2, 3] or sublist in the [0, 1, 2, 3].
    Args:
        image: 4-D Tensor of shape `[batch, height, width, channels]` or
               3-D Tensor of shape `[height, width, channels]`.
        k: if k is list, random select t form k, rotation image by 90 degrees * t.
           if k is int, rotation image by 90 degrees * k.
        seed: A Python integer. Used to create a random seed.
              See `tf.set_random_seed` for behavior.
    Returns:
    A rotated tensor of the same type and shape as `image`.
  Raises:
    ValueError: if the shape of `image` not supported or `k` dtype not int or list.
  """
    if isinstance(k, list):
        k_value = tf.convert_to_tensor(k)
        index = tf.argmax(tf.random.uniform([tf.shape(k_value)[0]], 0, 1))
        image = tf.image.rot90(image, k=k_value[index])
    elif k in [1, 2, 3]:
        image = tf.image.rot90(image, k)
    else:
        raise ValueError('k should be int one of [1, 2, 3] or sublist in the [0, 1, 2, 3].')
    return image if kwarg else image.numpy()
