from random import randint

import numpy as np
from PIL import Image, ImageChops

__all__ = ['flip_up_left', 'flip_up_right', 'flip_left_right', 'flip_up_down', 
           'rotate', 'translate', 'offset', 'pad', 'shuffle_channel']


def flip_up_left(image, random=False):
    """Randomly flip an image (up to left).
    
    Args:
        image: a PIL instance.
        random: bool, default False.
                if True, random flip up and left image.
                if False, flip up and left image.
    Returns:
        A PIL instance. of the same type and shape as `image`.
    """
    if random:
        random = np.random.choice([True, False])
    return image.transpose(Image.Transpose.TRANSPOSE) if not random else image


def flip_up_right(image, random=False):
    """Randomly flip an image (up to right).

    Args:
        image: a PIL instance.
        random: bool, default False.
                if True, random flip up and right image.
                if False, flip up and right image.
    Returns:
        A PIL instance. of the same type and shape as `image`.
    """
    if random:
        random = np.random.choice([True, False])
    return image.transpose(Image.Transpose.TRANSVERSE) if not random else image


def flip_left_right(image, random=False):
    """Randomly flip an image (left to right).

    Args:
    image: a PIL instance.
    random: bool, default False.
            if True, random flip left and rignt image.
            if False, flip left and right image.
    Returns:
            A PIL instance. of the same type and shape as `image`.
    """
    if random:
        random = np.random.choice([True, False])
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if not random else image


def flip_up_down(image, random=False):
    """Randomly flip an image (up to down).

    Args:
    image: a PIL instance.
    random: bool, default False.
            if True, random flip up and down image.
            if False, flip up and down image.
    Returns:
            A PIL instance. of the same type and shape as `image`.
    """
    if random:
        random = np.random.choice([True, False])
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM) if not random else image


def rotate(image, angle, expand=True, center=None, translate=None, fillcolor=None):
    """Returns a rotated copy of this image. This method returns a copy of this image, 
    rotated the given number of degrees counter clockwise around its centre.
    
    Args:
        image: a PIL instance.
        angle: In degrees counter clockwise.
               if int or float, rotation angle.
               if list or tuple, randomly picked in the interval `[angle[0], angle[1])` value.
        expand: Optional expansion flag. If true, expands the output image to make it large 
                enough to hold the entire rotated image. If false or omitted, 
                make the output image the same size as the input image. 
                Note that the expand flag assumes rotation around the center and no translation.
                if value is 'random', then the function is random.
        center: Optional center of rotation (a 2-tuple). Origin is the upper left corner. 
                Default is the center of the image.
                if value is 'random', then the function is random.
        translate: An optional post-rotate translation (a 2-tuple).
                if value is 'random', then the function is random.
        fillcolor: An optional color for area outside the rotated image.
                if value is 'random', fillcolor is one of ['green', 'red', 'white', 'black'].
                you can also pass in a list of colors or la.image.RGBMode.
    
    Returns:
        A PIL instance. of the same type and shape as `image`.
    """
    if isinstance(angle, (list, tuple)):
        assert angle[0]<angle[1], '`angle` must be angle[0]<angle[1].'
        angle = np.random.uniform(angle[0], angle[1])
    if expand=='random':
        expand = np.random.choice([True, False])
    if center=='random':
        center = (np.random.randint(0, image.size[0]), np.random.randint(0, image.size[1]))
    if translate=='random':
        translate = (np.random.randint(0, image.size[0]*0.8), np.random.randint(0, image.size[1]*0.8))
    if fillcolor=='random':
        fillcolor = np.random.choice(['green', 'red', 'white', 'black'])
    elif isinstance(fillcolor, dict):
        fillcolor = fillcolor['rgb']
    elif isinstance(fillcolor, list):
        fillcolor = np.random.choice(fillcolor)
    return image.rotate(angle, resample=Image.Resampling.NEAREST, expand=expand, center=center, translate=translate, fillcolor=fillcolor)


def translate(image, translate='random', fillcolor=None):
    """Returns a translate copy of this image. 
    
    Args:
        image: a PIL instance.
        translate: An optional post-rotate translation (a 2-tuple).
                if value is 'random', then the function is random.
        fillcolor: An optional color for area outside the rotated image.
                if value is 'random', fillcolor is one of ['green', 'red', 'white', 'black'].
                you can also pass in a list of colors or la.image.RGBMode.
    
    Returns:
        A PIL instance. of the same type and shape as `image`.
    """
    if translate=='random':
        translate = (np.random.randint(0, image.size[0]*0.8), np.random.randint(0, image.size[1]*0.8))
    if fillcolor=='random':
        fillcolor = np.random.choice(['green', 'red', 'white', 'black'])
    elif isinstance(fillcolor, dict):
        fillcolor = fillcolor['rgb']
    elif isinstance(fillcolor, list):
        fillcolor = np.random.choice(fillcolor)
    return rotate(image, angle=0, expand=1, center=None, translate=translate, fillcolor=fillcolor)


def offset(image, xoffset, yoffset=None):
    """Returns a copy of the image where data has been offset by the given
    distances. Data wraps around the edges. If ``yoffset`` is omitted, it
    is assumed to be equal to ``xoffset``.

    Args:
        image: a PIL instance.
        xoffset: int or list ot tuple, The horizontal distance.
                 if tuple or list, randomly picked in the interval `[xoffset[0], xoffset[1])`
        yoffset: int or list ot tuple, The vertical distance. 
                 If omitted, both distances are set to the same value.
                 if tuple or list, randomly picked in the interval `[yoffset[0], yoffset[1])`
    Return:
        a PIL instance.
    """
    if isinstance(xoffset, (list, tuple)):
        xoffset = np.random.randint(xoffset[0], xoffset[1])
    if yoffset is None:
        yoffset = xoffset
    elif isinstance(yoffset, (list, tuple)):
        yoffset = np.random.randint(yoffset[0], yoffset[1])
    return ImageChops.offset(image, xoffset, yoffset)


def pad(image, pad_value, pad_color=None):
    """Add border to the image
    
    Args:
        image: a PIL instance.
        pad_value: int or list or tuple, if int, pad same value with border, 
                   if list or tuple, len(pad_value)==2, left, top = right, bottom = pad_value
                   if list or tuple, len(pad_value)==4, left, top, right, bottom = pad_value
        pad_color: str or tuple or list or la.image.RGBMode, fill RGB color value, 
                   if str, hexadecimal color;
                   if len(pad_color) == 2, left_color, top_color = right_color, bottom_color = pad_color
                   if len(pad_color) == 3, left_color = top_color = right_color = bottom_color = pad_color
                   if len(pad_color) == 4, left_color, top_color, right_color, bottom_color = pad_color
    Returns:
        A PIL instance. 
    """
    if isinstance(pad_value, (tuple, list)):
        if len(pad_value) == 2:
            left, top = right, bottom = pad_value
        elif len(pad_value) == 4:
            left, top, right, bottom = pad_value
        else:
            raise ValueError('`pad_value` value error.')
    else:
        left = top = right = bottom = pad_value
    
    if pad_color is None:
        left_color = (randint(0, 255), randint(0, 255), randint(0, 255))
        top_color = (randint(0, 255), randint(0, 255), randint(0, 255))
        right_color = (randint(0, 255), randint(0, 255), randint(0, 255))
        bottom_color = (randint(0, 255), randint(0, 255), randint(0, 255))
    elif isinstance(pad_color, dict):
        left_color = top_color = right_color = bottom_color = pad_color['rgb']
    elif isinstance(pad_color, (tuple, list)):
        pad_color = tuple([i['rgb'] if isinstance(i, dict) else i for i in pad_color])
        if len(pad_color) == 2:
            left_color, top_color = right_color, bottom_color = pad_color
        elif len(pad_color) == 3:
            left_color = top_color = right_color = bottom_color = pad_color
        elif len(pad_color) == 4:
            left_color, top_color, right_color, bottom_color = pad_color
        else:
            raise ValueError('`pad_value` value error.')
    else:
        left_color = top_color = right_color = bottom_color = pad_color

    width = left + image.size[0] + right
    height = top + image.size[1] + bottom

    out = Image.new(image.mode, (width, height), left_color)
    out.paste(image, (left, top))
    out.paste(Image.new(image.mode, (image.size[0] + right, top), top_color), (left, 0))
    out.paste(Image.new(image.mode, (right, top + image.size[1] + bottom), right_color), (left + image.size[0], top))
    out.paste(Image.new(image.mode, (left + image.size[0], bottom), bottom_color), (0, top + image.size[1]))
    return out


def shuffle_channel(image):
    """Random shuffle image channel.
    
    Args:
        image: a PIL instance.
    returns: 
        a PIL instance.
    """
    assert image.mode=='RGB', 'image mode should be RGB.'
    t = image.split()
    return Image.merge("RGB", tuple(t[i] for i in np.random.choice([0,1,2], 3, replace=False)))