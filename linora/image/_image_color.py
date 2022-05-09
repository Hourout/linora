import colorsys

import numpy as np
from PIL import ImageEnhance, ImageOps

__all__ = ['enhance_saturation', 'enhance_contrast', 'enhance_brightness', 'enhance_sharpness',
           'enhance_hue', 'gamma', 'color_invert', 'color_clip', 'equalize', 'rgb_hex',
           'hls_to_rgb', 'rgb_to_hls', 'hsv_to_rgb', 'rgb_to_hsv', 'rgb_to_yiq', 'yiq_to_rgb',
          ]


def enhance_saturation(image, delta, p=1):
    """Adjust image color balance.

    This class can be used to adjust the colour balance of an image, 
    in a manner similar to the controls on a colour TV set. 
    An enhancement factor of 0.0 gives a black and white image. 
    A factor of 1.0 gives the original image.
    
    Args:
        image: a PIL instance.
        delta: A floating point value controlling the enhancement. 
               delta 1.0 always returns a copy of the original image, 
               lower factors mean less color, and higher values more. 
               There are no restrictions on this value.
               if list, tuple, randomly picked in the interval `[delta[0], delta[1])` , 
               value is float multiplier for adjusting color.
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(delta, (list, tuple)):
        delta = np.random.uniform(delta[0], delta[1])
    return ImageEnhance.Color(image).enhance(delta)


def enhance_contrast(image, delta, p=1):
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
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(delta, (list, tuple)):
        delta = np.random.uniform(delta[0], delta[1])
    return ImageEnhance.Contrast(image).enhance(delta)


def enhance_brightness(image, delta, p=1):
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
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(delta, (list, tuple)):
        delta = np.random.uniform(delta[0], delta[1])
    return ImageEnhance.Brightness(image).enhance(delta)


def enhance_sharpness(image, delta, p=1):
    """Adjust image sharpness.

    This class can be used to adjust the sharpness of an image. 
    An enhancement factor of 0.0 gives a blurred image, 
    a factor of 1.0 gives the original image, 
    and a factor of 2.0 gives a sharpened image.
    
    Args:
        image: a PIL instance.
        delta: A floating point value controlling the enhancement. 
               delta 1.0 always returns a copy of the original image, 
               lower factors mean less sharpness, and higher values more. 
               There are no restrictions on this value.
               if list, tuple, randomly picked in the interval
               `[delta[0], delta[1])` , value is float multiplier for adjusting color.
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(delta, (list, tuple)):
        assert delta[0]<delta[1], 'delta should be delta[1] > delta[0]'
        delta = np.random.uniform(delta[0], delta[1])
    return ImageEnhance.Sharpness(image).enhance(delta)


def enhance_hue(image, delta, p=1):
    """Adjust hue of an image.
    
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    
    `delta` is the amount of shift in H channel and must be in the interval `[-1, 1]`.

    Args:
        image: a PIL instance.
        delta: How much to shift the hue channel. Should be in [-1, 1]. 
               1 and -1 give complete reversal of hue channel in
               HSV space in positive and negative direction respectively.
               0 means no shift. Therefore, both -1 and 1 will give an image
               with complementary colors while 0 gives the original image.
               if list or tuple, randomly picked in the interval `[delta[0], delta[1])`.
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(delta, (list, tuple)):
        delta = np.random.randint(delta[0], delta[1])
    assert -1. <= delta <= 1., "`delta` is not in [-1, 1]."
    input_mode = image.mode
    if input_mode in {"L", "1", "I", "F"}:
        return image
    h, s, v = image.convert("HSV").split()
    h = h.point(lambda x:x+delta * 127.5)
    return Image.merge("HSV", (h, s, v)).convert(input_mode)


def gamma(image, gamma=1, gain=1.0, p=1):
    """Perform gamma correction on an image.
    
    For gamma greater than 1, the histogram will shift towards left and the output image will be darker than the input image. 
    For gamma less than 1, the histogram will shift towards right and the output image will be brighter than the input image.
    Args:
        image: a PIL instance.
        gamma: float, Non negative real number, gamma larger than 1 make the shadows darker, 
               while gamma smaller than 1 make dark regions lighter.
               if list or tuple, randomly picked in the interval `[gamma[0], gamma[1])`.
        gain: float, The constant multiplier.
              if list or tuple, randomly picked in the interval `[gain[0], gain[1])`.
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(gamma, (list, tuple)):
        gamma = np.random.randint(gamma[0], gamma[1])
    if isinstance(gain, (list, tuple)):
        gain = np.random.randint(gain[0], gain[1])
    assert gamma >= 0, "Gamma should be a non-negative real number."
    input_mode = image.mode
    if input_mode!="RGB":
        image = image.convert("RGB")
    gamma_map = [int((255 + 1 - 1e-3) * gain * pow(ele / 255.0, gamma)) for ele in range(256)] * 3
    image = image.point(gamma_map)
    if input_mode!="RGB":
        image = image.convert(input_mode)
    return image


def color_invert(image, lower=None, upper=None, p=1):
    """Invert colors of input PIL image.

    Args:
        image: a PIL instance.
        lower: int or list or tuple, [0, 255], All pixels below this greyscale level are inverted.
               if list or tuple, randomly picked in the interval `[lower[0], lower[1])`.
        upper: int or list or tuple, [0, 255], All pixels above this greyscale level are inverted.
               if list or tuple, randomly picked in the interval `[upper[0], upper[1])`.
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(lower, (list, tuple)):
        lower = np.random.randint(lower[0], lower[1])
    if isinstance(upper, (list, tuple)):
        upper = np.random.randint(upper[0], upper[1])
    if lower is None and upper is None:
        lower = upper = 128
    elif lower is None:
        lower = 0
    elif upper is None:
        upper = 255
    return image.point(lambda x: 255-x if x<=lower or x>=upper else x)


def color_clip(image, lower=None, upper=None, p=1):
    """Clip colors of input PIL image.
    
    Args:
        image: a PIL instance.
        lower: int or list or tuple, [0, 255], All pixels below this greyscale level are clipped.
               if list or tuple, randomly picked in the interval `[lower[0], lower[1])`.
        upper: int or list or tuple, [0, 255], All pixels above this greyscale level are clipped.
               if list or tuple, randomly picked in the interval `[upper[0], upper[1])`.
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(lower, (list, tuple)):
        lower = np.random.randint(lower[0], lower[1])
    if isinstance(upper, (list, tuple)):
        upper = np.random.randint(upper[0], upper[1])
    if lower is None and upper is None:
        raise ValueError('`lower` and `upper` at least one value.')
    if lower is not None:
        image = image.point(lambda x: lower if x<=lower else x)
    if upper is not None:
        image = image.point(lambda x: upper if x>=upper else x)
    return image


def equalize(image, p=1):
    """
    Equalize the image histogram. This function applies a non-linear
    mapping to the input image, in order to create a uniform
    distribution of grayscale values in the output image.

    Args:
        image: a PIL instance.
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
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