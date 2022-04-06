import os

import numpy as np
from PIL import Image

__all__ = ['list_images', 'ColorMode', 'color_convert', 'image_to_array', 'array_to_image']

def list_images(directory, file_format=('jpg', 'jpeg', 'bmp', 'png', 'ppm', 'tif', 'tiff')):
    """Lists all pictures in a directory, including all subdirectories.
    
    Args:
        directory: string, absolute path to the directory.
        file_format: tuple of strings or single string, extensions of the pictures.
    Returns:
        a list of image paths.
    """
    file_format = tuple('.%s' % e for e in ((file_format,) if isinstance(file_format, str) else file_format))
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if f.lower().endswith(file_format)]

class color_mode:
    """image color type."""
    O     = {'mode':'1',      'description':'1-bit pixels, black and white, stored with one pixel per byte'}
    L     = {'mode':'L',      'description':'8-bit pixels, black and white'}
    LA    = {'mode':'LA',     'description':'L with alpha'}
    La    = {'mode':'La',     'description':'L with premultiplied alpha'}
    I     = {'mode':'I',      'description':'32-bit signed integer pixels'}
    F     = {'mode':'F',      'description':'32-bit floating point pixels'}
    P     = {'mode':'P',      'description':'8-bit pixels, mapped to any other mode using a color palette'}
    PA    = {'mode':'PA',     'description':'P with alpha'}
    RGB   = {'mode':'RGB',    'description':'3x8-bit pixels, true color'}
    RGBA  = {'mode':'RGBA',   'description':'4x8-bit pixels, true color with transparency mask'}
    RGBX  = {'mode':'RGBX',   'description':'true color with padding'}
    RGBa  = {'mode':'RGBa',   'description':'true color with premultiplied alpha'}
    CMYK  = {'mode':'CMYK',   'description':'4x8-bit pixels, color separation'}
    YCbCr = {'mode':'YCbCr',  'description':'3x8-bit pixels, color video format'}
    LAB   = {'mode':'LAB',    'description':'3x8-bit pixels, the L*a*b color space'}
    HSV   = {'mode':'HSV',    'description':'3x8-bit pixels, Hue, Saturation, Value color space'}
    I16   = {'mode':'I;16',   'description':'16-bit unsigned integer pixels'}
    I16L  = {'mode':'I;16L',  'description':'16-bit little endian unsigned integer pixels'}
    I16B  = {'mode':'I;16B',  'description':'16-bit big endian unsigned integer pixels'}
    I16N  = {'mode':'I;16N',  'description':'16-bit native endian unsigned integer pixels'}
    BGR15 = {'mode':'BGR;15', 'description':'15-bit reversed true colour'}
    BGR16 = {'mode':'BGR;16', 'description':'16-bit reversed true colour'}
    BGR24 = {'mode':'BGR;24', 'description':'24-bit reversed true colour'}
    BGR32 = {'mode':'BGR;32', 'description':'32-bit reversed true colour'}

ColorMode = color_mode()

def color_convert(image, color_mode=ColorMode.RGB):
    """Transform image color mode.
    
    Args:
        image: PIL Image instance.
        color_mode: Image color mode, more see api "la.image.ColorMode".
    Returns:
        PIL Image instance.
    Raises:
        ValueError: color_mode error.
    """
    if color_mode == 'grayscale':
        if image.mode not in ('L', 'I;16', 'I'):
            image = image.convert('L')
    elif isinstance(color_mode, dict):
        image = image.convert(color_mode['mode'])
    elif isinstance(color_mode, str):
        image = image.convert(color_mode)
    else:
        raise ValueError('color_mode error.')
    return image

def image_to_array(image, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.
    
    Args:
        image: PIL Image instance.
        data_format: Image data format, either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.
    Returns:
        A 3D Numpy array.
    Raises:
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    x = np.asarray(image, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x

def array_to_image(x, data_format='channels_last'):
    """Converts a 3D Numpy array to a PIL Image instance.
    
    Args:
        x: Input Numpy array.
        data_format: Image data format, either "channels_first" or "channels_last".
    Returns:
        A PIL Image instance.
    Raises:
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape: %s' % (x.shape,))

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format: %s' % data_format)

    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)

    if x.shape[2] == 4:
        return Image.fromarray(x.astype('uint8'), 'RGBA')
    elif x.shape[2] == 3:
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        if np.max(x) > 255:
            return Image.fromarray(x[:, :, 0].astype('int32'), 'I')
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: %s' % (x.shape[2],))
