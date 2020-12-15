import numpy as np
from PIL import ImageEnhance

__all__ = ['enhance_color', 'enhance_contrast', 'enhance_brightness', 'enhance_sharpness']


def enhance_color(image, delta):
    """Adjust image color balance.

    This class can be used to adjust the colour balance of an image, 
    in a manner similar to the controls on a colour TV set. 
    An enhancement factor of 0.0 gives a black and white image. 
    A factor of 1.0 gives the original image.
    
    Args:
    image: a Image instance.
    delta: A floating point value controlling the enhancement. 
           delta 1.0 always returns a copy of the original image, 
           lower factors mean less color, 
           and higher values more. There are no restrictions on this value.
           if list, tuple, randomly picked in the interval
               `[delta[0], delta[1])` , value is float multiplier for adjusting color.
    Returns:
        a Image instance.
    """
    if isinstance(delta, (list, tuple)):
        assert delta[0]<delta[1], 'delta should be delta[1] > delta[0].'
        delta = np.random.uniform(delta[0], delta[1])
    return ImageEnhance.Color(image).enhance(delta)

def enhance_contrast(image, delta):
    """Adjust contrast of RGB or grayscale images.
  
    Contrast is adjusted independently for each channel of each image.
    
    For each channel, this Ops computes the mean of the image pixels in the
    channel and then adjusts each component `x` of each pixel to
    `(x - mean) * delta + mean`.
    
    Tips:
        1 means pixel value no change.
        0 means all pixel equal. 
        a suitable interval is (0, 4].
    Args:
        images: a Image instance.
        delta: if int, float, a float multiplier for adjusting contrast.
               if list, tuple, randomly picked in the interval
               `[delta[0], delta[1])` , value is float multiplier for adjusting contrast.
    Returns:
        a Image instance.
    """
    if isinstance(delta, (list, tuple)):
        assert delta[0]<delta[1], 'delta should be delta[1] > delta[0].'
        delta = np.random.uniform(delta[0], delta[1])
    return ImageEnhance.Contrast(image).enhance(delta)

def enhance_brightness(image, delta):
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
    Returns:
        a Image instance.
    """
    if isinstance(delta, (list, tuple)):
        assert delta[0]<delta[1], 'delta should be delta[1] > delta[0]'
        delta = np.random.uniform(delta[0], delta[1])
    return ImageEnhance.Brightness(image).enhance(delta)

def enhance_sharpness(image, delta):
    """Adjust image sharpness.

    This class can be used to adjust the sharpness of an image. 
    An enhancement factor of 0.0 gives a blurred image, 
    a factor of 1.0 gives the original image, 
    and a factor of 2.0 gives a sharpened image.
    
    Args:
    image: a Image instance.
    delta: A floating point value controlling the enhancement. 
           delta 1.0 always returns a copy of the original image, 
           lower factors mean less sharpness, 
           and higher values more. There are no restrictions on this value.
           if list, tuple, randomly picked in the interval
               `[delta[0], delta[1])` , value is float multiplier for adjusting color.
    Returns:
        a Image instance.
    """
    if isinstance(delta, (list, tuple)):
        assert delta[0]<delta[1], 'delta should be delta[1] > delta[0]'
        delta = np.random.uniform(delta[0], delta[1])
    return ImageEnhance.Sharpness(image).enhance(delta)


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

def RandomPencilSketch(image, delta=0.1, seed=None, **kwarg):
    """Adjust the pencil sketch of RGB.
    
    Args:
        image: Tensor or array. An image.
        delta: if int, float, Amount to add to the pixel values.
               if list, tuple, randomly picked in the interval
               `[delta[0], delta[1])` to add to the pixel values.
               a suitable interval is [0.1, 0.5].
        seed: A Python integer. Used to create a random seed. See
             `tf.set_random_seed` for behavior.
    Returns:
        A pencil_sketch-adjusted tensor of the same shape as `image`.
    """
    if isinstance(delta, (list, tuple)):
        random_delta = tf.random.uniform([], delta[0], delta[1], seed=seed)
    else:
        random_delta = delta
        
    temp = tf.cast(image, tf.float32)
    t = temp[:, 1:]-temp[:, :-1]
    grad_y = tf.concat([t[:,0:1], (t[:, 1:]+t[:, :-1])/2, t[:, -1:]], 1)*random_delta
    t = temp[1:, :]-temp[:-1, :]
    grad_x = tf.concat([t[0:1,:], (t[1:, :]+t[:-1, :])/2, t[-1:, :]], 0)*random_delta

    A = tf.math.sqrt(tf.math.square(grad_x)+tf.math.square(grad_y)+1.)
    uni_x = tf.math.divide(grad_x, A)
    uni_y = tf.math.divide(grad_y, A)
    uni_z = tf.math.divide(1., A)

    dx = tf.math.cos(3.141592653589793/2.2)*tf.math.cos(3.141592653589793/4)
    dy = tf.math.cos(3.141592653589793/2.2)*tf.math.sin(3.141592653589793/4)
    dz = tf.math.sin(3.141592653589793/2.2)

    b = tf.clip_by_value(255.*(dx*uni_x + dy*uni_y + dz*uni_z), 0., 255.)
    return b if kwarg else b.numpy()
