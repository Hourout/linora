import colorsys

import numpy as np
from PIL import ImageEnhance, ImageOps

__all__ = ['enhance_color', 'enhance_contrast', 'enhance_brightness', 'enhance_sharpness',
           'hls_to_rgb', 'rgb_to_hls', 'hsv_to_rgb', 'rgb_to_hsv', 'rgb_to_yiq', 'yiq_to_rgb',
           'color_invert', 'equalize', 'rgb_hex'
          ]


def enhance_color(image, delta):
    """Adjust image color balance.

    This class can be used to adjust the colour balance of an image, 
    in a manner similar to the controls on a colour TV set. 
    An enhancement factor of 0.0 gives a black and white image. 
    A factor of 1.0 gives the original image.
    
    Args:
        image: a PIL instance.
        delta: A floating point value controlling the enhancement. 
               delta 1.0 always returns a copy of the original image, 
               lower factors mean less color, 
               and higher values more. There are no restrictions on this value.
               if list, tuple, randomly picked in the interval
                   `[delta[0], delta[1])` , value is float multiplier for adjusting color.
    Returns:
        a PIL instance.
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
        images: a PIL instance.
        delta: if int, float, a float multiplier for adjusting contrast.
               if list, tuple, randomly picked in the interval
               `[delta[0], delta[1])` , value is float multiplier for adjusting contrast.
    Returns:
        a PIL instance.
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
        image: a PIL instance.
        delta: if int, float, Amount to add to the pixel values.
               if list, tuple, randomly picked in the interval
               `[delta[0], delta[1])` to add to the pixel values.
    Returns:
        a PIL instance.
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
        image: a PIL instance.
        delta: A floating point value controlling the enhancement. 
               delta 1.0 always returns a copy of the original image, 
               lower factors mean less sharpness, 
               and higher values more. There are no restrictions on this value.
               if list, tuple, randomly picked in the interval
                   `[delta[0], delta[1])` , value is float multiplier for adjusting color.
    Returns:
        a PIL instance.
    """
    if isinstance(delta, (list, tuple)):
        assert delta[0]<delta[1], 'delta should be delta[1] > delta[0]'
        delta = np.random.uniform(delta[0], delta[1])
    return ImageEnhance.Sharpness(image).enhance(delta)


def color_invert(image, threshold=0):
    """Invert colors of input PIL image.
    
    Pixels with values less than threshold are not inverted.

    Args:
        image: a PIL instance.
        threshold: int or list or tuple, [0, 255], All pixels above this greyscale level are inverted.
                if list or tuple, randomly picked in the interval `[threshold[0], threshold[1])`
    Returns:
        a PIL instance.
    """
    if isinstance(threshold, (list, tuple)):
        threshold = np.random.randint(threshold[0], threshold[1])
    return ImageOps.solarize(image, threshold=threshold)


def equalize(image):
    """
    Equalize the image histogram. This function applies a non-linear
    mapping to the input image, in order to create a uniform
    distribution of grayscale values in the output image.

    Args:
        image: a PIL instance.
    Returns:
        a PIL instance.
    """
    return ImageOps.equalize(image)


def rgb_to_hsv(image):
    """Convert RGB image to HSV(Hue,Saturation,Value(Brightness)) image.

    Args:
        image: NumPy RGB image array of shape (H, W, C) to be converted.
    Returns:
        a numpy array, NumPy HSV image with same type of image.
    """
    if image.max()>1:
        image = image/255.
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    h, s, v = to_hsv(r, g, b)
    return np.stack((h, s, v), axis=2)


def hsv_to_rgb(image, normalize=False, dtype='float32'):
    """Convert HSV(Hue,Saturation,Value(Brightness)) image to RGB image.

    Args:
        image: NumPy HSV image array of shape (H, W, C) to be converted.
        normalize: if True, rgb numpy array is [0,255], if False, rgb numpy array is [0,1]
        dtype: rgb numpy array dtype.
    Returns:
        a numpy array, NumPy HSV image with same shape of image.
    """
    norm = 255. if normalize else 1.
    h, s, v = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    to_rgb = np.vectorize(colorsys.hsv_to_rgb)
    r, g, b = to_rgb(h, s, v)
    return (np.stack((r, g, b), axis=2)*norm).astype(dtype)


def rgb_to_hls(image):
    """Convert RGB image to HLS(Hue,Lightness,Saturation) image.

    Args:
        image: NumPy RGB image array of shape (H, W, C) to be converted.
    Returns:
        a numpy array, NumPy HLS image with same type of image.
    """
    if image.max()>1:
        image = image/255.
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    to_hls = np.vectorize(colorsys.rgb_to_hls)
    h, l, s = to_hls(r, g, b)
    return np.stack((h, l, s), axis=2)


def hls_to_rgb(image, normalize=False, dtype='float32'):
    """Convert HLS(Hue,Lightness,Saturation) image to RGB image.

    Args:
        image (numpy.ndarray): NumPy HLS image array of shape (H, W, C) to be converted.
        normalize: if True, rgb numpy array is [0,255], if False, rgb numpy array is [0,1]
        dtype: rgb numpy array dtype.
    Returns:
        a numpy array, NumPy HLS image with same shape of image.
    """
    norm = 255. if normalize else 1.
    h, l, s = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    to_rgb = np.vectorize(colorsys.hls_to_rgb)
    r, g, b = to_rgb(h, l, s)
    return (np.stack((r, g, b), axis=2)*norm).astype(dtype)


def rgb_to_yiq(image):
    """Convert RGB image to YIQ(Luminance,Chrominance) image.

    Args:
        image (numpy.ndarray): NumPy RGB image array of shape (H, W, C) to be converted.
    Returns:
        a numpy array, NumPy YIQ image with same type of image.
    """
    if image.max()>1:
        image = image/255.
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    to_yiq = np.vectorize(colorsys.rgb_to_yiq)
    y, i, q = to_yiq(r, g, b)
    return np.stack((y, i, q), axis=2)


def yiq_to_rgb(image, normalize=False, dtype='float32'):
    """Convert YIQ(Luminance,Chrominance) image to RGB image.

    Args:
        image: NumPy YIQ image array of shape (H, W, C) to be converted.
        normalize: if True, rgb numpy array is [0,255], if False, rgb numpy array is [0,1]
        dtype: rgb numpy array dtype.
    Returns:
        a numpy array, NumPy YIQ image with same shape of image.
    """
    norm = 255. if normalize else 1.
    y, i, q = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    to_rgb = np.vectorize(colorsys.yiq_to_rgb)
    r, g, b = to_rgb(y, i, q)
    return (np.stack((r, g, b), axis=2)*norm).astype(dtype)


def rgb_hex(color):
    """rgb to hexadecimal or hexadecimal to rgb.
    
    Args:
        color: str or list or tuple, if str, is hexadecimal; if list or tuple, is rgb value.
    return:
        if str, return rgb value; if list or tuple, return hexadecimal.
    """
    if isinstance(color, str):
        if len(color)==7 and color[0]=='#':
            return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
    if isinstance(color, (list, tuple)):
        return '#'+''.join([str(hex(i))[-2:].replace('x', '0').upper() for i in color])
    raise ValueError('`color` value error, should be str or list or tuple.')