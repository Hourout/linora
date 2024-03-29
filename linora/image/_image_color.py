import colorsys

import numpy as np
from PIL import ImageEnhance, ImageOps, Image

__all__ = ['enhance_saturation', 'enhance_brightness', 'enhance_sharpness', 'enhance_contrast_sigmoid',
           'enhance_contrast_log', 'enhance_contrast_linear', 'enhance_contrast_gamma',
           'enhance_hue', 'color_invert', 'color_clip', 'equalize', 'rgb_hex',
           'hls_to_rgb', 'rgb_to_hls', 'hsv_to_rgb', 'rgb_to_hsv', 'rgb_to_yiq', 'yiq_to_rgb',
           'rgb_to_yuv', 'yuv_to_rgb', 'dropout', 'ydbdr_to_rgb', 'rgb_to_ydbdr', 'ycbcr_to_rgb', 
           'rgb_to_ycbcr', 'rgb_to_ypbpr', 'ypbpr_to_rgb', 'bgr_to_rgb', 'rgb_to_bgr',
           'rgb_to_grayscale', 'lab_to_rgb', 'rgb_to_lab', 'rgb_to_xyz', 'xyz_to_rgb'
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


def enhance_contrast_linear(image, delta, p=1):
    """Adjust image contrast.
  
    Contrast is adjusted independently for each channel of each image.
    pixel = (x - mean) * delta + mean
    
    Tips:
        1 means pixel value no change.
        0 means all pixel equal, pixel is image mean.
    Args:
        images: a PIL instance.
        delta: if int, float, a float multiplier for adjusting contrast.
               if list, tuple, randomly picked in the interval `[delta[0], delta[1])`.
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(delta, (list, tuple)):
        delta = np.random.uniform(delta[0], delta[1])
    return ImageEnhance.Contrast(image).enhance(delta)


def enhance_contrast_gamma(image, gamma=1, gain=1.0, p=1):
    """Perform gamma correction on an image.
    
    For gamma greater than 1, the histogram will shift towards left and the output image will be darker than the input image. 
    For gamma less than 1, the histogram will shift towards right and the output image will be brighter than the input image.
    
    pixel = 255*gain *((pixel/255)**gamma)
    Values in the range gamma=(0.5, 2.0) seem to be sensible.
    
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
        gain = np.random.uniform(gain[0], gain[1])
    return image.point(lambda x:int((255 + 1 - 1e-3) * gain * pow(x / 255.0, gamma)))


def enhance_contrast_sigmoid(image, cutoff=0.5, gain=10, p=1):
    """Perform sigmoid correction on an image.
    
    pixel = 255*1/(1+exp(gain*(cutoff-pixel/255)))
    Values in the range gain=(5, 20) and cutoff=(0.25, 0.75) seem to be sensible.
    
    Args:
        image: a PIL instance.
        cutoff: float, Cutoff that shifts the sigmoid function in horizontal direction. 
                Higher values mean that the switch from dark to light pixels happens later. 
                if list or tuple, randomly picked in the interval `[cutoff[0], cutoff[1])`.
        gain: float, The constant multiplier.
              if list or tuple, randomly picked in the interval `[gain[0], gain[1])`.
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(cutoff, (list, tuple)):
        cutoff = np.random.uniform(cutoff[0], cutoff[1])
    if isinstance(gain, (list, tuple)):
        gain = np.random.randint(gain[0], gain[1])
    return image.point(lambda x:int(255*1/(1+np.exp(gain*(cutoff-x / 255.0)))))


def enhance_contrast_log(image, gain=1, p=1):
    """Perform log correction on an image.
    
    pixel = 255*gain*log_2(1+pixel/255)
    Values in the range gain=[0.6, 1.4] seem to be sensible.
    
    Args:
        image: a PIL instance.
        gain: float, The constant multiplier.
              if list or tuple, randomly picked in the interval `[gain[0], gain[1])`.
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(gain, (list, tuple)):
        gain = np.random.randint(gain[0], gain[1])
    return image.point(lambda x:int(255*gain*np.log2(1+x / 255.0)))


def enhance_brightness(image, delta, p=1):
    """Adjust image brightness.
    
    Tips:
        delta >1 to white, <1 to black.
        a suitable interval is [0.5, 1.5].
        0 means black.
        1 means pixel value no change.
    Args:
        image: a PIL instance.
        delta: if int, float, Amount to add to the pixel values.
               if list, tuple, randomly picked in the interval `[delta[0], delta[1])`.
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
    """Adjust image hue.
    
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


def color_invert(image, lower=None, upper=None, wise='pixel', prob=1, p=1):
    """Invert colors of input PIL image.

    Args:
        image: a PIL instance.
        lower: int or list or tuple, [0, 255], All pixels below this greyscale level are inverted.
               if list or tuple, randomly picked in the interval `[lower[0], lower[1])`.
        upper: int or list or tuple, [0, 255], All pixels above this greyscale level are inverted.
               if list or tuple, randomly picked in the interval `[upper[0], upper[1])`.
        wise: 'pixel' or 'channel' or list of channel, method of applying operate.
        prob: probability of every pixel or channel being changed.
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
    if wise=='pixel':
        return image.point(lambda x: 255-x if (x<=lower or x>=upper) and np.random.uniform()<prob else x)
    elif wise=='channel':
        split = list(image.split())
        for i in range(len(split)):
            if np.random.uniform()<prob:
                split[i] = split[i].point(lambda x: 255-x if x<=lower or x>=upper else x)
        return Image.merge(image.mode, split)
    elif isinstance(wise, (list, tuple)):
        split = list(image.split())
        for i in wise:
            if np.random.uniform()<prob:
                split[i] = split[i].point(lambda x: 255-x if x<=lower or x>=upper else x)
        return Image.merge(image.mode, split)
    else:
        raise ValueError("`wise` should be 'pixel' or 'channel' or list of channel.")


def color_clip(image, lower=None, upper=None, wise='pixel', prob=1, p=1):
    """Clip colors of input PIL image.
    
    Args:
        image: a PIL instance.
        lower: int or list or tuple, [0, 255], All pixels below this greyscale level are clipped.
               if list or tuple, randomly picked in the interval `[lower[0], lower[1])`.
        upper: int or list or tuple, [0, 255], All pixels above this greyscale level are clipped.
               if list or tuple, randomly picked in the interval `[upper[0], upper[1])`.
        wise: 'pixel' or 'channel' or list of channel, method of applying operate.
        prob: probability of every pixel or channel being changed.
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
    if wise=='pixel':
        if lower is not None:
            image = image.point(lambda x: lower if x<=lower and np.random.uniform()<prob else x)
        if upper is not None:
            image = image.point(lambda x: upper if x>=upper and np.random.uniform()<prob else x)
        return image
    elif wise=='channel':
        split = list(image.split())
        for i in range(len(split)):
            if np.random.uniform()<prob:
                if lower is not None:
                    split[i] = split[i].point(lambda x: lower if x<=lower else x)
                if upper is not None:
                    split[i] = split[i].point(lambda x: upper if x>=upper else x)
        return Image.merge(image.mode, split)
    elif isinstance(wise, (list, tuple)):
        split = list(image.split())
        for i in wise:
            if np.random.uniform()<prob:
                if lower is not None:
                    split[i] = split[i].point(lambda x: lower if x<=lower else x)
                if upper is not None:
                    split[i] = split[i].point(lambda x: upper if x>=upper else x)
        return Image.merge(image.mode, split)
    else:
        raise ValueError("`wise` should be 'pixel' or 'channel' or list of channel.")


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


def rgb_to_yuv(image):
    """Converts image from RGB to YUV.

    Args:
        image: NumPy RGB image array of shape (H, W, C) to be converted.
    Returns:
        a numpy array.
    """
    if image.max()>1:
        image = image/255.
    kernel = np.array([[0.299, -0.14714119, 0.61497538],
                       [0.587, -0.28886916, -0.51496512],
                       [0.114, 0.43601035, -0.10001026]])
    return np.dot(image, kernel)


def yuv_to_rgb(image, normalize=False, dtype='float32'):
    """Converts image from YUV to RGB.

    Args:
        image: NumPy YUV image array of shape (H, W, C) to be converted.
        normalize: if True, rgb numpy array is [0,255], if False, rgb numpy array is [0,1]
        dtype: rgb numpy array dtype.
    Returns:
        a numpy array.
    """
    norm = 255. if normalize else 1.
    kernel = np.array([[1, 1, 1], 
                       [0, -0.394642334, 2.03206185],
                       [1.13988303, -0.58062185, 0]])
    return (np.dot(image, kernel)*norm).astype(dtype)


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
    

def dropout(image, value=0, wise='pixel', prob=0.1, p=1):
    """Drop random channels from images.

    For image data, dropped channels will be filled with value.
    
    Args:
        image: a PIL instance.
        value: int or list, dropped channels will be filled with value.
        wise: 'pixel' or 'channel' or list of channel, method of applying operate.
        prob: probability of every pixel or channel being changed.
        p: probability that the image does this. Default value is 1.
    returns: 
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(value, (int, float)):
        value = [int(value)]
    if wise=='pixel':
        return image.point(lambda x:np.random.choice(value) if np.random.uniform()<prob else x)
    elif wise=='channel':
        split = list(image.split())
        index = np.random.randint(0, len(split))
        for i in range(len(split)):
            if np.random.uniform()<prob and i!=index:
                split[i] = split[i].point(lambda x:np.random.choice(value))
        return Image.merge(image.mode, split)
    elif isinstance(wise, (list, tuple)):
        split = list(image.split())
        index = np.random.randint(0, len(split))
        if index in wise:
            if len(split)==len(wise):
                index = 0
            else:
                index = -1
        for i in wise:
            if np.random.uniform()<prob and i!=index:
                split[i] = split[i].point(lambda x:np.random.choice(value))
        return Image.merge(image.mode, split)
    else:
        raise ValueError("`wise` value error.")


def rgb_to_bgr(image):
    """Convert RGB image to BGR image.

    Args:
        image: NumPy RGB image array of shape (H, W, C) to be converted.
    Returns:
        a numpy array, NumPy BGR image with same type of image.
    """
    return np.stack((image[:,:,2], image[:,:,1], image[:,:,0]), axis=2)


def bgr_to_rgb(image):
    """Convert BGR image to RGB image.

    Args:
        image: NumPy BGR image array of shape (H, W, C) to be converted.
    Returns:
        a numpy array, NumPy BGR image with same shape of image.
    """
    return rgb_to_bgr(image)


def rgb_to_ypbpr(image):
    """Convert BGR image to YPbPr image.

    Args:
        image: NumPy RGB image array of shape (H, W, C) to be converted.
    Returns:
        a numpy array, NumPy YPbPr image with same shape of image.
    """
    kernel = np.array([[0.299, 0.587, 0.114],
                       [-0.168736, -0.331264, 0.5],
                       [0.5, -0.418688, -0.081312]])
    return np.tensordot(image, kernel.transpose(), axes=((-1,), (0,)))


def ypbpr_to_rgb(image, normalize=False, dtype='float32'):
    """Convert a YPbPr image to RGB image.
    
    Args:
        image: NumPy YPbPr image array of shape (H, W, C) to be converted.
        normalize: if True, rgb numpy array is [0,255], if False, rgb numpy array is [0,1]
        dtype: rgb numpy array dtype.
    Returns:
        a numpy array, NumPy RGB image with same shape of image.
    """
    norm = 255. if normalize else 1.
    kernel = np.array([[1.00000000e00, -1.21889419e-06, 1.40199959e00],
                       [1.00000000e00, -3.44135678e-01, -7.14136156e-01],
                       [1.00000000e00, 1.77200007e00, 4.06298063e-07]])
    return (np.tensordot(image, kernel.transpose(), axes=((-1,), (0,)))*norm).astype(dtype)


def rgb_to_ycbcr(image):
    """Convert a RGB image to YCbCr image.
    
    Args:
        image: NumPy RGB image array of shape (H, W, C) to be converted.
    Returns:
        a numpy array, NumPy YCbCr image with same shape of image.
    """
    if image.max()>1:
        image = image/255.
    image = rgb_to_ypbpr(image)
    return image*np.array([219, 224, 224], dtype=image.dtype)+np.array([16, 128, 128], dtype=image.dtype)


def ycbcr_to_rgb(image, normalize=False, dtype='float32'):
    """Convert a YCbCr image to RGB image.
    
    Args:
        image: NumPy YCbCr image array of shape (H, W, C) to be converted.
        normalize: if True, rgb numpy array is [0,255], if False, rgb numpy array is [0,1]
        dtype: rgb numpy array dtype.
    Returns:
        a numpy array, NumPy RGB image with same shape of image.
    """
    norm = 255. if normalize else 1.
    image = image-np.array([16, 128, 128], dtype=image.dtype)
    image = image/np.array([219, 224, 224], dtype=image.dtype)
    return (ypbpr_to_rgb(image)*norm).astype(dtype)


def rgb_to_ydbdr(image):
    """Convert BGR image to YDbDr image.

    Args:
        image: NumPy RGB image array of shape (H, W, C) to be converted.
    Returns:
        a numpy array, NumPy YDbDr image with same shape of image.
    """
    kernel = np.array([[0.299, 0.587, 0.114], [-0.45, -0.883, 1.333], [-1.333, 1.116, 0.217]])
    return np.tensordot(image, kernel.transpose(), axes=((-1,), (0,)))


def ydbdr_to_rgb(image, normalize=False, dtype='float32'):
    """Convert a YDbDr image to RGB image.
    
    Args:
        image: NumPy YDbDr image array of shape (H, W, C) to be converted.
        normalize: if True, rgb numpy array is [0,255], if False, rgb numpy array is [0,1]
        dtype: rgb numpy array dtype.
    Returns:
        a numpy array, NumPy RGB image with same shape of image.
    """
    kernel = np.array([[1.00000000e00, 9.23037161e-05, -5.25912631e-01],
                       [1.00000000e00, -1.29132899e-01, 2.67899328e-01],
                       [1.00000000e00, 6.64679060e-01, -7.92025435e-05]])
    return np.tensordot(image, kernel.transpose(), axes=((-1,), (0,)))


def rgb_to_xyz(image):
    """Convert a RGB image to CIE XYZ image.
    
    Args:
        image: NumPy RGB image array of shape (H, W, C) to be converted.
    Returns:
        a numpy array, NumPy CIE XYZ image with same shape of image.
    """
    kernel = np.array([[0.412453, 0.357580, 0.180423],
                       [0.212671, 0.715160, 0.072169],
                       [0.019334, 0.119193, 0.950227]])
    image = np.where(np.greater(image, 0.04045), np.power((image+0.055)/1.055, 2.4), image/12.92)
    return np.tensordot(image, kernel.transpose(), axes=((-1,), (0,)))


def xyz_to_rgb(image, normalize=False, dtype='float32'):
    """Convert a CIE XYZ image to RGB image.
    
    Args:
        image: NumPy CIE XYZ image array of shape (H, W, C) to be converted.
        normalize: if True, rgb numpy array is [0,255], if False, rgb numpy array is [0,1]
        dtype: rgb numpy array dtype.
    Returns:
        a numpy array, NumPy RGB image with same shape of image.
    """
    norm = 255. if normalize else 1.
    kernel = np.array([[3.24048134, -1.53715152, -0.49853633],
                       [-0.96925495, 1.87599, 0.04155593],
                       [0.05564664, -0.20404134, 1.05731107]])
    image = np.tensordot(image, kernel.transpose(), axes=((-1,), (0,)))
    image = np.where(np.greater(image, 0.0031308), np.power(image, 1.0/2.4)*1.055-0.055, image*12.92)
    return (np.clip(image, 0, 1)*norm).astype(dtype)


def rgb_to_lab(image, illuminant="D65", observer="2"):
    """Convert a RGB image to CIE LAB image.
    
    Args:
        image: NumPy RGB image array of shape (H, W, C) to be converted.
        illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, the name of the illuminant.
        observer : {"2", "10"}, the aperture angle of the observer.
    Returns:
        a numpy array, NumPy CIE LAB image with same shape of image.
    """
    illuminants = {
        "A": {
            "2": (1.098466069456375, 1, 0.3558228003436005),
            "10": (1.111420406956693, 1, 0.3519978321919493),
        },
        "D50": {
            "2": (0.9642119944211994, 1, 0.8251882845188288),
            "10": (0.9672062750333777, 1, 0.8142801513128616),
        },
        "D55": {
            "2": (0.956797052643698, 1, 0.9214805860173273),
            "10": (0.9579665682254781, 1, 0.9092525159847462),
        },
        "D65": {
            "2": (0.95047, 1.0, 1.08883),
            "10": (0.94809667673716, 1, 1.0730513595166162),
        },
        "D75": {
            "2": (0.9497220898840717, 1, 1.226393520724154),
            "10": (0.9441713925645873, 1, 1.2064272211720228),
        },
        "E": {"2": (1.0, 1.0, 1.0), "10": (1.0, 1.0, 1.0)},
    }
    coords = np.array(illuminants[illuminant.upper()][observer])
    xyz = rgb_to_xyz(image)
    xyz = xyz / coords
    xyz = np.where(np.greater(xyz, 0.008856), np.power(xyz, 1.0 / 3.0), xyz * 7.787 + 16.0 / 116.0)
    x, y, z = xyz[:,:,0], xyz[:,:,1], xyz[:,:,2]
    l = (y * 116.0) - 16.0
    a = (x - y) * 500.0
    b = (y - z) * 200.0
    return np.stack([l, a, b], axis=2)


def lab_to_rgb(image, illuminant="D65", observer="2", normalize=False, dtype='float32'):
    """Convert a CIE LAB image to RGB image.
    
    Args:
        image: NumPy CIE LAB image array of shape (H, W, C) to be converted.
        illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, the name of the illuminant.
        observer : {"2", "10"}, the aperture angle of the observer.
        normalize: if True, rgb numpy array is [0,255], if False, rgb numpy array is [0,1]
        dtype: rgb numpy array dtype.
    Returns:
        a numpy array, NumPy RGB image with same shape of image.
    """
    norm = 255. if normalize else 1.
    l, a, b = image[:,:,0], image[:,:,1], image[:,:,2]

    y = (l + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)

    z = np.maximum(z, 0)

    xyz = np.stack([x, y, z], axis=2)
    xyz = np.where(np.greater(xyz, 0.2068966), np.power(xyz, 3.0), (xyz - 16.0 / 116.0) / 7.787)

    illuminants = {
        "A": {
            "2": (1.098466069456375, 1, 0.3558228003436005),
            "10": (1.111420406956693, 1, 0.3519978321919493),
        },
        "D50": {
            "2": (0.9642119944211994, 1, 0.8251882845188288),
            "10": (0.9672062750333777, 1, 0.8142801513128616),
        },
        "D55": {
            "2": (0.956797052643698, 1, 0.9214805860173273),
            "10": (0.9579665682254781, 1, 0.9092525159847462),
        },
        "D65": {
            "2": (0.95047, 1.0, 1.08883),
            "10": (0.94809667673716, 1, 1.0730513595166162),
        },
        "D75": {
            "2": (0.9497220898840717, 1, 1.226393520724154),
            "10": (0.9441713925645873, 1, 1.2064272211720228),
        },
        "E": {"2": (1.0, 1.0, 1.0), "10": (1.0, 1.0, 1.0)},
    }
    coords = np.array(illuminants[illuminant.upper()][observer])
    return (xyz_to_rgb(xyz * coords)*norm).astype(dtype)


def rgb_to_grayscale(image):
    """Convert a RGB image to grayscale image.
    
    Args:
        image: NumPy RGB image array of shape (H, W, C) to be converted.
    Returns:
        a numpy array, NumPy grayscale image with same shape of image.
    """
    return np.expand_dims(np.tensordot(image, np.array([0.2125, 0.7154, 0.0721]), (-1, -1)), -1)