import math

import numpy as np
from PIL import Image, ImageChops

__all__ = ['flip_up_left', 'flip_up_right', 'flip_left_right', 'flip_up_down', 
           'rotate', 'translate', 'offset', 'pad', 'shuffle_channel',
           'transform_perspective', 'transform_affine'
          ]


def flip_up_left(image, p=1):
    """Randomly flip an image (up to left).
    
    Args:
        image: a PIL instance.
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance.
    """
    if np.random.uniform()>p:
        return image
    return image.transpose(Image.Transpose.TRANSPOSE)


def flip_up_right(image, p=1):
    """Randomly flip an image (up to right).

    Args:
        image: a PIL instance.
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance.
    """
    if np.random.uniform()>p:
        return image
    return image.transpose(Image.Transpose.TRANSVERSE)


def flip_left_right(image, p=1):
    """Randomly flip an image (left to right).

    Args:
        image: a PIL instance.
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance.
    """
    if np.random.uniform()>p:
        return image
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def flip_up_down(image, p=1):
    """Randomly flip an image (up to down).

    Args:
        image: a PIL instance.
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance.
    """
    if np.random.uniform()>p:
        return image
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def rotate(image, angle, expand=True, center=None, translate=None, fillcolor=None, p=1):
    """Returns a rotated copy of this image. 
    
    This method returns a copy of this image, rotated the given number of degrees counter clockwise around its centre.
    
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
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance.
    """
    if np.random.uniform()>p:
        return image
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
    return image.rotate(angle, resample=Image.Resampling.NEAREST, expand=expand, 
                        center=center, translate=translate, fillcolor=fillcolor)


def translate(image, translate='random', fillcolor=None, p=1):
    """Returns a translate copy of this image. 
    
    Args:
        image: a PIL instance.
        translate: An optional post-rotate translation (a 2-tuple).
                   if value is 'random', then the function is random.
        fillcolor: An optional color for area outside the rotated image.
                   if value is 'random', fillcolor is one of ['green', 'red', 'white', 'black'].
                   you can also pass in a list of colors or la.image.RGBMode.
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if translate=='random':
        translate = (np.random.randint(0, image.size[0]*0.8), np.random.randint(0, image.size[1]*0.8))
    if fillcolor=='random':
        fillcolor = np.random.choice(['green', 'red', 'white', 'black'])
    elif isinstance(fillcolor, dict):
        fillcolor = fillcolor['rgb']
    elif isinstance(fillcolor, list):
        fillcolor = np.random.choice(fillcolor)
    return rotate(image, angle=0, expand=1, center=None, translate=translate, fillcolor=fillcolor)


def offset(image, xoffset, yoffset=None, p=1):
    """Returns a copy of the image where data has been offset by the given distances.
    
    Data wraps around the edges. If ``yoffset`` is omitted, it is assumed to be equal to ``xoffset``.

    Args:
        image: a PIL instance.
        xoffset: int or list ot tuple, The horizontal distance.
                 if tuple or list, randomly picked in the interval `[xoffset[0], xoffset[1])`
        yoffset: int or list ot tuple, The vertical distance. 
                 If omitted, both distances are set to the same value.
                 if tuple or list, randomly picked in the interval `[yoffset[0], yoffset[1])`
        p: probability that the image does this. Default value is 1.
    Return:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(xoffset, (list, tuple)):
        xoffset = np.random.randint(xoffset[0], xoffset[1])
    if yoffset is None:
        yoffset = xoffset
    elif isinstance(yoffset, (list, tuple)):
        yoffset = np.random.randint(yoffset[0], yoffset[1])
    return ImageChops.offset(image, xoffset, yoffset)


def pad(image, pad_value, pad_color=None, p=1):
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
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance. 
    """
    if np.random.uniform()>p:
        return image
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
        left_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        top_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        right_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        bottom_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
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


def shuffle_channel(image, p=1):
    """Random shuffle image channel.
    
    Args:
        image: a PIL instance.
        p: probability that the image does this. Default value is 1.
    returns: 
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    assert image.mode=='RGB', 'image mode should be RGB.'
    t = image.split()
    return Image.merge("RGB", tuple(t[i] for i in np.random.choice([0,1,2], 3, replace=False)))


def transform_perspective(image, distortion_scale, fill_color=None, p=1):
    """Performs a random perspective transformation of the given image with a given probability. 

    Args:
        image: a PIL instance.
        distortion_scale: float, argument to control the degree of distortion and ranges from 0 to 1.
        fill_color: int or str or tuple or la.image.RGBMode, rgb color.
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    (width, height) = image.size
    half_height = height // 2
    half_width = width // 2
    topleft = [np.random.randint(0, distortion_scale*half_width+1), 
               np.random.randint(0, distortion_scale*half_height+1)]
    topright = [np.random.randint(width - distortion_scale * half_width- 1, width),
                np.random.randint(0, distortion_scale * half_height + 1)]
    botright = [np.random.randint(width - distortion_scale * half_width - 1, width),
                np.random.randint(height - distortion_scale * half_height - 1, height)]
    botleft = [np.random.randint(0, distortion_scale * half_width + 1),
               np.random.randint(height - distortion_scale * half_height - 1, height)]
    
    startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    endpoints = [topleft, topright, botright, botleft]
    
    a_matrix = np.zeros([2 * len(startpoints), 8])
    for i, (p1, p2) in enumerate(zip(endpoints, startpoints)):
        a_matrix[2 * i, :] = np.array([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        a_matrix[2 * i + 1, :] = np.array([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    b_matrix = np.array(startpoints, dtype=np.float32).flatten()
    coeffs = np.linalg.lstsq(a_matrix, b_matrix, rcond=None)[0].tolist()
    if fill_color is None:
        fill_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    elif isinstance(fill_color, dict):
        fill_color = fill_color['mode']
    return image.transform(image.size, Image.Transform.PERSPECTIVE, coeffs, 
                           Image.Resampling.BILINEAR, fillcolor=fill_color)


def transform_affine(image, angle=(-180, 180), translate=(0, 0), scale=1., shear=(0,0), center=None, fill_color=None, p=1):
    """Apply affine transformation on the image keeping image center invariant.
    
    Args:
        image: a PIL instance.
        angle: int or float, rotation angle in degrees between -180 and 180. Set to 0 to deactivate rotations.
               if list or tuple, randomly picked in the interval `[angle[0], angle[1])`.
        translate: list or tuple, tuple of maximum absolute fraction for horizontal and vertical translations. 
                   For example translate=(a, b), then horizontal shift is randomly sampled 
                   in the range -img_width * a < dx < img_width * a 
                   and vertical shift is randomly sampled in the range -img_height * b < dy < img_height * b. 
                   Will not translate by default.
        scale: float, scaling factor interval, if list or tuple, randomly picked in the interval `[scale[0], scale[1])`.
        shear: Range of degrees to select from. 
               Else if shear is a sequence of 2 values a shear parallel to the x axis in the range (shear[0], shear[1]) will be applied. 
               Else if shear is a sequence of 4 values, a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied. 
        fill_color: int or str or tuple or la.image.RGBMode, rgb color. Pixel fill value for the area outside the transformed image.
        center: Optional center of rotation. Origin is the upper left corner. Default is the center of the image.
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(angle, (list, tuple)):
        angle = np.random.uniform(angle[0], angle[1])

    width, height = image.size
    max_dx = translate[0] * width
    max_dy = translate[1] * height
    tx = np.random.randint(-max_dx, max_dx+1)
    ty = np.random.randint(-max_dy, max_dy+1)
        
    if isinstance(scale, (list, tuple)):
        scale = np.random.uniform(scale[0], scale[1])
    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    shear_x = shear_y = 0.0
    if isinstance(shear, (int, float)):
        shear_x = shear_y = shear
    elif isinstance(shear, (list, tuple)):
        shear_x = np.random.uniform(shear[0], shear[1])
        if len(shear) == 4:
            shear_y = np.random.uniform(shear[2], shear[3])
    shear = (shear_x, shear_y)

    if center is None:
        cx, cy = [width * 0.5, height * 0.5]
    elif isinstance(center, (int, float)):
        cx, cy = center
    
    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])

    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    matrix = [d, -b, 0.0, -c, a, 0.0]
    matrix = [x / scale for x in matrix]
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
    matrix[2] += cx
    matrix[5] += cy
    if fill_color is None:
        fill_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    elif isinstance(fill_color, dict):
        fill_color = fill_color['mode']
    return image.transform(image.size, Image.Transform.AFFINE, matrix, 
                           Image.Resampling.BILINEAR, fillcolor=fill_color)