import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import _ImageDimensions

__all__ = ['read_image', 'save_image', 'RandomBrightness', 'RandomContrast', 'RandomHue',
           'RandomSaturation', 'RandomGamma', 'RandomFlipLeftRight', 'RandomFlipTopBottom',
           'RandomTranspose', 'RandomRotation', 'RandomCropCentralResize', 'RandomCropPointResize',
           'Normalize', 'RandomRescale', 'RandomNoiseGaussian', 'RandomNoisePoisson',
           'RandomNoiseMask', 'RandomNoiseSaltPepper', 'RandomNoiseRainbow']

def read_image(filename, channel=0, image_format='mix', **kwarg):
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
    assert channel in [0, 1, 3], 'channel should be one of [0, 1, 3].'
    image = tf.io.read_file(filename)
    if image_format=='png':
        image = tf.io.decode_png(image, channel)
    elif image_format=='bmp':
        image = tf.io.decode_bmp(image, channel)
    elif image_format=='gif':
        image = tf.io.decode_gif(image)
    elif image_format in ["jpg", "jpeg"]:
        image = tf.io.decode_jpeg(image, channel)
    elif image_format=='mix':
        image = tf.io.decode_image(image)
    else:
        raise ValueError('image_format should be one of "mix", "jpg", "jpeg", "png", "gif", "bmp".')
    return image if kwarg else image.numpy()

def save_image(image, filename):
    """Writes image to the file at input filename. 
    
    Args:
        image:    A Tensor of type string. scalar. The content to be written to the output file.
        filename: A string. scalar. The name of the file to which we write the contents.
    Raises:
        ValueError: If `filename` is not in `[`jpg`, `jpeg`, `png`]`.
    """
    if filename.split('.')[-1] in ['jpg', 'jpeg']:
        tf.io.write_file(filename, tf.io.encode_jpeg(image))
    elif filename.split('.')[-1]=='png':
        tf.io.write_file(filename, tf.image.encode_png(image))
    else:
        raise ValueError('filename should be one of [`jpg`, `jpeg`, `png`].')

def RandomBrightness(image, delta, seed=None, **kwarg):
    """Adjust the brightness of RGB or Grayscale images.
    
    Tips:
        delta extreme value in the interval [-1, 1], >1 to white, <-1 to black.
        a suitable interval is [-0.5, 0.5].
        0 means pixel value no change.
    Args:
        image: Tensor or array. An image.
        delta: if int, float, Amount to add to the pixel values.
               if list, tuple, randomly picked in the interval
               `[delta[0], delta[1])` to add to the pixel values.
        seed: A Python integer. Used to create a random seed. See
             `tf.set_random_seed` for behavior.
    Returns:
        A brightness-adjusted tensor of the same shape and type as `image`.
    Raises:
        ValueError: if `delta` type is error.
    """
    if isinstance(delta, (int, float)):
        assert -1<=delta<=1, 'delta should be in the interval [-1, 1].'
        image = tf.image.adjust_brightness(image, delta)
    elif isinstance(delta, (list, tuple)):
        assert -1<=delta[0]<delta[1]<=1, 'delta should be 1 >= delta[1] > delta[0] >= -1.'
        random_delta = tf.random.uniform([], delta[0], delta[1], seed=seed)
        image = tf.image.adjust_brightness(image, random_delta)
    else:
        raise ValueError('delta should be one of int, float, list, tuple.')
    return image if kwarg else image.numpy()

def RandomContrast(image, delta, seed=None, **kwarg):
    """Adjust contrast of RGB or grayscale images.
    
    `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
    interpreted as `[height, width, channels]`.  The other dimensions only
    represent a collection of images, such as `[batch, height, width, channels].`
  
    Contrast is adjusted independently for each channel of each image.
    
    For each channel, this Ops computes the mean of the image pixels in the
    channel and then adjusts each component `x` of each pixel to
    `(x - mean) * delta + mean`.
    
    Tips:
        1 means pixel value no change.
        0 means all pixel equal. 
        a suitable interval is (0, 4].
    Args:
        images: Tensor or array. An image. At least 3-D.
        delta: if int, float, a float multiplier for adjusting contrast.
               if list, tuple, randomly picked in the interval
               `[delta[0], delta[1])` , value is float multiplier for adjusting contrast.
        seed: A Python integer. Used to create a random seed. See
             `tf.set_random_seed` for behavior.
    Returns:
        The contrast-adjusted image or images tensor of the same shape and type as `image`.
    Raises:
        ValueError: if `delta` type is error.
    """
    if isinstance(delta, (int, float)):
        image = tf.image.adjust_contrast(image, delta)
    elif isinstance(delta, (list, tuple)):
        assert delta[0]<delta[1], 'delta should be delta[1] > delta[0].'
        random_delta = tf.random.uniform([], delta[0], delta[1], seed=seed)
        image = tf.image.adjust_contrast(image, random_delta)
    else:
        raise ValueError('delta should be one of int, float, list, tuple.')
    return image if kwarg else image.numpy()

def RandomHue(image, delta, seed=None, **kwarg):
    """Adjust hue of an RGB image.
    
    `image` is an RGB image.  The image hue is adjusted by converting the
    image to HSV and rotating the hue channel (H) by `delta`.
    The image is then converted back to RGB.
    
    Tips:
        `delta` should be in the interval `[-1, 1]`, but any value is allowed.
        a suitable interval is [-0.5, 0.5].
        int value means pixel value no change.
    Args:
        image: Tensor or array. RGB image or images. Size of the last dimension must be 3.
        delta: if float, How much to add to the hue channel.
               if list, tuple, randomly picked in the interval
               `[delta[0], delta[1])` , value is how much to add to the hue channel.
        seed: A Python integer. Used to create a random seed. See
             `tf.set_random_seed` for behavior.
    Returns:
        The hue-adjusted image or images tensor of the same shape and type as `image`.
    Raises:
        ValueError: if `delta` type is error.
    """
    if isinstance(delta, (int, float)):
        image = tf.image.adjust_hue(image, delta)
    elif isinstance(delta, (list, tuple)):
        assert delta[0]<delta[1], 'delta should be delta[1] > delta[0].'
        random_delta = tf.random.uniform([], delta[0], delta[1], seed=seed)
        image = tf.image.adjust_hue(image, random_delta)
    else:
        raise ValueError('delta should be one of int, float, list, tuple.')
    return image if kwarg else image.numpy()

def RandomSaturation(image, delta, seed=None, **kwarg):
    """Adjust saturation of an RGB image.
    
    `image` is an RGB image.  The image saturation is adjusted by converting the
    image to HSV and multiplying the saturation (S) channel by `delta` and clipping.
    The image is then converted back to RGB.
    
    Tips:
        if delta <= 0, image channels value are equal, image color is gray.
        a suitable interval is delta >0
    Args:
        image: RGB image or images. Size of the last dimension must be 3.
        delta: if int, float, Factor to multiply the saturation by.
               if list, tuple, randomly picked in the interval
               `[delta[0], delta[1])` , value is factor to multiply the saturation by.
        seed: A Python integer. Used to create a random seed. See
             `tf.set_random_seed` for behavior.
    Returns:
        The saturation-adjusted image or images tensor of the same shape and type as `image`.
    Raises:
        ValueError: if `delta` type is error.
    """
    if isinstance(delta, (int, float)):
        image = tf.image.adjust_saturation(image, delta)
    elif isinstance(delta, (list, tuple)):
        assert delta[0]<delta[1], 'delta should be delta[1] > delta[0].'
        image = tf.image.random_saturation(image, delta[0], delta[1], seed=seed)
    else:
        raise ValueError('delta should be one of int, float, list, tuple.')
    return image if kwarg else image.numpy()

def RandomGamma(image, gamma, seed=None, **kwarg):
    """Performs Gamma Correction on the input image.
    
    Also known as Power Law Transform. This function transforms the
    input image pixelwise according to the equation `Out = In**gamma`
    after scaling each pixel to the range 0 to 1.
    
    Tips:
        For gamma greater than 1, the histogram will shift towards left and
        the output image will be darker than the input image.
        For gamma less than 1, the histogram will shift towards right and
        the output image will be brighter than the input image.
        if gamma is 1, image pixel value no change.
    Args:
        image : A Tensor.
        gamma : if int, float, Non negative real number.
                if list, tuple, randomly picked in the interval
                `[delta[0], delta[1])` , value is Non negative real number.
        seed: A Python integer. Used to create a random seed. See
              `tf.set_random_seed` for behavior.
    Returns:
        A float Tensor. Gamma corrected output image.
    Raises:
        ValueError: If gamma is negative.
    References:
        [1] http://en.wikipedia.org/wiki/Gamma_correction
    """
    image = tf.cast(image, dtype=tf.float32)
    if isinstance(gamma, (int, float)):
        assert 0<gamma, 'gamma should be > 0.'
        image = tf.image.adjust_gamma(image, gamma, gain=1)
    elif isinstance(gamma, (list, tuple)):
        assert 0<gamma[0]<gamma[1], 'gamma should be gamma[1] > gamma[0] > 0.'
        random_gamma = tf.random.uniform([], gamma[0], gamma[1], seed=seed)
        image = tf.image.adjust_gamma(image, random_gamma, gain=1)
    else:
        raise ValueError('gamma should be one of int, float, list, tuple.')
    return image if kwarg else image.numpy()

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
