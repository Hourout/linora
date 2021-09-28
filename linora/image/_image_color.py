import colorsys
from PIL import ImageEnhance, ImageOps

import numpy as np


__all__ = ['enhance_color', 'enhance_contrast', 'enhance_brightness', 'enhance_sharpness',
           'hls_to_rgb', 'rgb_to_hls', 'hsv_to_rgb', 'rgb_to_hsv', 'rgb_to_yiq', 'yiq_to_rgb',
           'color_invert', 'equalize'
          ]


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
        image: a Image instance.
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

def rgb_to_hsv(image):
    """
    Convert RGB image to HSV image.

    Args:
        image (numpy.ndarray): NumPy RGB image array of shape (H, W, C) to be converted.

    Returns:
        numpy.ndarray, NumPy HSV image with same type of image.
    """
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    h, s, v = to_hsv(r, g, b)
    return np.stack((h, s, v), axis=2)

def hsv_to_rgb(image):
    """
    Convert HSV img to RGB img.

    Args:
        image (numpy.ndarray): NumPy HSV image array of shape (H, W, C) to be converted.

    Returns:
        numpy.ndarray, NumPy HSV image with same shape of image.
    """
    h, s, v = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    to_rgb = np.vectorize(colorsys.hsv_to_rgb)
    r, g, b = to_rgb(h, s, v)
    return np.stack((r, g, b), axis=2)

def rgb_to_hls(image):
    """
    Convert RGB img to HLS img.

    Args:
        image (numpy.ndarray): NumPy RGB image array of shape (H, W, C) to be converted.

    Returns:
        numpy.ndarray, NumPy HLS image with same type of image.
    """
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    to_hls = np.vectorize(colorsys.rgb_to_hls)
    h, l, s = to_hls(r, g, b)
    return np.stack((h, l, s), axis=2)

def hls_to_rgb(image):
    """
    Convert HLS img to RGB img.

    Args:
        image (numpy.ndarray): NumPy HLS image array of shape (H, W, C) to be converted.

    Returns:
        numpy.ndarray, NumPy HLS image with same shape of image.
    """
    h, l, s = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    to_rgb = np.vectorize(colorsys.hls_to_rgb)
    r, g, b = to_rgb(h, l, s)
    return np.stack((r, g, b), axis=2)

def rgb_to_yiq(image):
    """
    Convert RGB img to YIQ img.

    Args:
        image (numpy.ndarray): NumPy RGB image array of shape (H, W, C) to be converted.

    Returns:
        numpy.ndarray, NumPy YIQ image with same type of image.
    """
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    to_yiq = np.vectorize(colorsys.rgb_to_yiq)
    y, i, q = to_yiq(r, g, b)
    return np.stack((y, i, q), axis=2)

def yiq_to_rgb(image):
    """
    Convert YIQ img to RGB img.

    Args:
        image (numpy.ndarray): NumPy YIQ image array of shape (H, W, C) to be converted.

    Returns:
        numpy.ndarray, NumPy YIQ image with same shape of image.
    """
    y, i, q = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    to_rgb = np.vectorize(colorsys.yiq_to_rgb)
    r, g, b = to_rgb(y, i, q)
    return np.stack((r, g, b), axis=2)

def color_invert(image):
    """
    Invert colors of input PIL image.

    Args:
        image (PIL image): Image to be color inverted.

    Returns:
        image (PIL image), Color inverted image.

    """
    return ImageOps.invert(image)

def equalize(image):
    """
    Equalize the image histogram. This function applies a non-linear
    mapping to the input image, in order to create a uniform
    distribution of grayscale values in the output image.

    Args:
        image (PIL image): Image to be equalized

    Returns:
        image (PIL image), Equalized image.

    """
    return ImageOps.equalize(image)

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

