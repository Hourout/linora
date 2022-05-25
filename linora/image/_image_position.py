import math

import numpy as np
from PIL import Image, ImageChops

from linora.image._image_rgb import _fill_color

__all__ = ['flip_up_left', 'flip_up_right', 'flip_left_right', 'flip_up_down', 
           'rotate', 'translate', 'offset', 'pad', 'channel_shuffle',
           'perspective', 'affine', 'shear', 'rescale', 'jigsaw'
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


def rotate(image, angle, expand=False, center=(0.5,0.5), translate=(0,0), fill_color=None, p=1):
    """Returns a rotated copy of this image. 
    
    This method returns a copy of this image, rotated the given number of degrees counter clockwise around its centre.
    
    Args:
        image: a PIL instance.
        angle: In degrees counter clockwise, angle in [-180, 180].
               if int or float, rotation angle.
               if list or tuple, randomly picked in the interval `[angle[0], angle[1])` value.
        expand: if true, expands the output image to make it large enough to hold the entire rotated image. 
                if false, make the output image the same size as the input image. 
                Note that the expand flag assumes rotation around the center and no translation.
                if value is None, then the function is random.
        center: center of rotation, xaxis and yaxis in [0,1], default is the center of the image.
                if int or float, xaxis=yaxis,
                if 2-tuple, (xaxis, yaxis), 
                if 4-tuple, xaxis in (center[0], center[1]) and yaxis in (center[2], center[3]).
        translate: post-rotate translation, xoffset and yoffset in [-1,1], see la.image.translate method.
                   if int or float, xoffset=yoffset,
                   if 2-tuple, (xoffset, yoffset), 
                   if 4-tuple, xoffset in (translate[0], translate[1]) and  yoffset in (translate[2], translate[3]).
        fill_color: color for area outside, int or str or tuple or la.image.RGBMode, rgb color.
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(angle, (list, tuple)):
        angle = np.random.uniform(angle[0], angle[1])
    if expand is None:
        expand = np.random.choice([True, False])
        
    if isinstance(center, (int, float)):
        center = (int(center*image.size[0]), int(center*image.size[1]))
    elif isinstance(center, (list, tuple)):
        if len(center)==2:
            center = (int(center[0]*image.size[0]), int(center[1]*image.size[1]))
        elif len(center)==4:
            center = (int(np.random.uniform(center[0], center[1])*image.size[0]), 
                      int(np.random.uniform(center[2], center[3])*image.size[1]))
        else:
            raise ValueError('`center` value format error.')
    else:
        raise ValueError('`center` value format error.')
    
    if isinstance(translate, (int, float)):
        translate = (int(translate*image.size[0]), int(translate*image.size[1]))
    elif isinstance(translate, (list, tuple)):
        if len(translate)==2:
            translate = (int(translate[0]*image.size[0]), int(translate[1]*image.size[1]))
        elif len(translate)==4:
            translate = (int(np.random.uniform(translate[0], translate[1])*image.size[0]), 
                         int(np.random.uniform(translate[2], translate[3])*image.size[1]))
        else:
            raise ValueError('`translate` value format error.')
    else:
        raise ValueError('`translate` value format error.')
    
    fill_color = _fill_color(image, fill_color)
    return image.rotate(angle, resample=Image.Resampling.NEAREST, expand=expand, 
                        center=center, translate=translate, fillcolor=fill_color)


def translate(image, xoffset=(-0.5,0.5), yoffset=None, fill_color=None, p=1):
    """Returns a translate copy of this image. 
    
    Args:
        image: a PIL instance.
        xoffset: [-1, 1], int or float, width offset.
                 if list or tuple, randomly picked in the interval `[xoffset[0], xoffset[1])`.
        yoffset: [-1, 1], int or float, height offset.
                 if list or tuple, randomly picked in the interval `[yoffset[0], yoffset[1])`.
        fill_color: int or str or tuple or la.image.RGBMode, rgb color.
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(xoffset, (list, tuple)):
        xoffset = np.random.uniform(xoffset[0], xoffset[1])
    if yoffset is None:
        yoffset = xoffset
    elif isinstance(yoffset, (list, tuple)):
        yoffset = np.random.uniform(yoffset[0], yoffset[1])
    xoffset = int(image.size[0]*xoffset)
    yoffset = int(image.size[1]*yoffset)
    fill_color = _fill_color(image, fill_color)
    return rotate(image, angle=0, expand=0, center=None, translate=(xoffset, yoffset), fillcolor=fill_color)


def offset(image, xoffset, yoffset=None, p=1):
    """Returns a copy of the image where data has been offset by the given distances.
    
    Data wraps around the edges. If ``yoffset`` is omitted, it is assumed to be equal to ``xoffset``.

    Args:
        image: a PIL instance.
        xoffset: int or list ot tuple, The horizontal distance.
                 if tuple or list, randomly picked in the interval `[xoffset[0], xoffset[1])`.
        yoffset: int or list ot tuple, The vertical distance. 
                 If omitted, both distances are set to the same value.
                 if tuple or list, randomly picked in the interval `[yoffset[0], yoffset[1])`.
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


def channel_shuffle(image, p=1):
    """Random shuffle image channel.
    
    Args:
        image: a PIL instance.
        p: probability that the image does this. Default value is 1.
    returns: 
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    t = image.split()
    return Image.merge(image.mode, [t[i] for i in np.random.choice(list(range(len(t))), len(t), replace=False)])
        

def perspective(image, distortion_scale, fill_color=None, p=1):
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
    fill_color = _fill_color(image, fill_color)
    return image.transform(image.size, Image.Transform.PERSPECTIVE, coeffs, 
                           Image.Resampling.NEAREST, fillcolor=fill_color)


def affine(image, angle=(-180, 180), center=(0.5,0.5), translate=(0, 0), scale=1., shear=(0,0), fill_color=None, p=1):
    """Apply affine transformation on the image.
    
    Args:
        image: a PIL instance.
        angle: int or float, rotation angle in degrees between -180 and 180. Set to 0 to deactivate rotations.
               if list or tuple, randomly picked in the interval `[angle[0], angle[1])`.
        center: center of rotation, xaxis and yaxis in [0,1], default is the center of the image.
                if int or float, xaxis=yaxis,
                if 2-tuple, (xaxis, yaxis), 
                if 4-tuple, xaxis in (center[0], center[1]) and yaxis in (center[2], center[3]).
        translate: post-rotate translation, xoffset and yoffset in [-1,1], see la.image.translate method.
                   if int or float, xoffset=yoffset,
                   if 2-tuple, (xoffset, yoffset), 
                   if 4-tuple, xoffset in (translate[0], translate[1]) and  yoffset in (translate[2], translate[3]).
        scale: float, scaling factor interval, should be positive.
               if list or tuple, randomly picked in the interval `[scale[0], scale[1])`.
        shear: Range of degrees to select from, xoffset and yoffset in [-360,360]. 
               if int or float, xoffset=yoffset,
               if 2-tuple, (xoffset, yoffset), 
               if 4-tuple, xoffset in (shear[0], shear[1]) and  yoffset in (shear[2], shear[3]).
        fill_color: int or str or tuple or la.image.RGBMode, rgb color. 
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(angle, (list, tuple)):
        angle = np.random.uniform(angle[0], angle[1])
        
    if isinstance(center, (int, float)):
        cx, cy = (int(center*image.size[0]), int(center*image.size[1]))
    elif isinstance(center, (list, tuple)):
        if len(center)==2:
            cx, cy = (int(center[0]*image.size[0]), int(center[1]*image.size[1]))
        elif len(center)==4:
            cx = int(np.random.uniform(center[0], center[1])*image.size[0])
            cy = int(np.random.uniform(center[2], center[3])*image.size[1])
        else:
            raise ValueError('`center` value format error.')
    else:
        raise ValueError('`center` value format error.')
    
    if isinstance(translate, (int, float)):
        tx, ty = int(translate*image.size[0]), int(translate*image.size[1])
    elif isinstance(translate, (list, tuple)):
        if len(translate)==2:
            tx, ty = int(translate[0]*image.size[0]), int(translate[1]*image.size[1])
        elif len(translate)==4:
            tx = int(np.random.uniform(translate[0], translate[1])*image.size[0])
            ty = int(np.random.uniform(translate[2], translate[3])*image.size[1])
        else:
            raise ValueError('`translate` value format error.')
    else:
        raise ValueError('`translate` value format error.')
        
    if isinstance(scale, (list, tuple)):
        scale = np.random.uniform(scale[0], scale[1])
    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    if isinstance(shear, (int, float)):
        shear = (shear, shear)
    elif isinstance(shear, (list, tuple)):
        if len(shear)==4:
            shear = (np.random.uniform(shear[0], shear[1]), np.random.uniform(shear[2], shear[3]))
        elif len(shear)!=2:
            raise ValueError('`translate` value format error.')
    else:
        raise ValueError('`translate` value format error.')
    
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
    fill_color = _fill_color(image, fill_color)
    return image.transform(image.size, Image.Transform.AFFINE, matrix, 
                           Image.Resampling.NEAREST, fillcolor=fill_color)


def shear(image, xoffset=(-90, 90), yoffset=None, fill_color=None, p=1):
    """Apply affine shear on the image.
    
    Args:
        image: a PIL instance.
        xoffset: int or list ot tuple, The horizontal degrees, xoffset in [-360,360].
                 if tuple or list, randomly picked in the interval `[xoffset[0], xoffset[1])`
        yoffset: int or list ot tuple, The vertical degrees, yoffset in [-360,360]. 
                 If omitted, both distances are set to the same value.
                 if tuple or list, randomly picked in the interval `[yoffset[0], yoffset[1])`
        fill_color: int or str or tuple or la.image.RGBMode, rgb color. 
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(xoffset, (list, tuple)):
        xoffset = np.random.uniform(xoffset[0], xoffset[1])
    if yoffset is None:
        yoffset = xoffset
    elif isinstance(yoffset, (list, tuple)):
        yoffset = np.random.uniform(yoffset[0], yoffset[1])
    fill_color = _fill_color(image, fill_color)
    return affine(image, angle=(0,0), center=(0.5,0.5), translate=(0, 0), scale=1., 
                  shear=(xoffset, yoffset), fill_color=fill_color, p=1)


def rescale(image, xscale=(0.5,1.5), yscale=(0.5,1.5), fill_color=None, p=1):
    """Apply scaling on the x or y axis to input data.
    
    Args:
        image: a PIL instance.
        xscale: if int or float, width expansion and contraction, xscale should be positive.
                if tuple or list, randomly picked in the interval `[xscale[0], xscale[1])`.
        yscale: if int or float, height expansion and contraction, yscale should be positive.
                if tuple or list, randomly picked in the interval `[yscale[0], yscale[1])`.
        fill_color: int or str or tuple or la.image.RGBMode, rgb color.
        p: probability that the image does this. Default value is 1.
    Return:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(xscale, (list, tuple)):
        xscale = np.random.uniform(xscale[0], xscale[1])
    if isinstance(yscale, (list, tuple)):
        yscale = np.random.uniform(yscale[0], yscale[1])
    fill_color = _fill_color(image, fill_color)
    size = [int(image.size[0]*xscale), int(image.size[1]*yscale)]
    size1 = [min(image.size[0], size[0]), min(image.size[1], size[1])]
    box1 = [int(size[0]/2-size1[0]/2), int(size[1]/2-size1[1]/2)]
    box1 = box1 + [box1[0]+size1[0], box1[1]+size1[1]]
    box2 = [int(image.size[0]/2-size1[0]/2), int(image.size[1]/2-size1[1]/2)]
    box2 = box2 + [box2[0]+size1[0], box2[1]+size1[1]]
    image2 = Image.new(image.mode, image.size, fill_color)
    image2.paste(image.resize(size).crop(box1), box2)
    return image2


def jigsaw(image, size=(10,10), prob=0.1, p=1):
    """Move cells within images similar to jigsaw patterns.
    
    Args:
        image: a PIL instance.
        size: if int or float, xsize=ysize, numbers of jigsaw.
              if 2-tuple, (xsize, ysize), 
              if 4-tuple, xsize in (size[0], size[1]) and  ysize in (size[2], size[3]).
        prob: probability of every jigsaw being changed.
        p: probability that the image does this. Default value is 1.
    Return:
        a PIL instance.
    """
    if isinstance(size, (int, float)):
        size = (size, size)
    elif isinstance(size, (list, tuple)):
        if len(size)==4:
            size = (np.random.uniform(size[0], size[1]), np.random.uniform(size[2], size[3]))
        elif len(size)!=2:
            raise ValueError('`size` value format error.')
    else:
        raise ValueError('`size` value format error.')
    size = (max(2, int(size[0])), max(2, int(size[1])))
    
    w = image.size[0]//size[0]
    h = image.size[1]//size[1]
    axis = [[i*w, j*h, i*w+w, j*h+h] for i in range(size[0]) for j in range(size[1])]
    image2 = image.copy()
    s = np.random.choice(range(size[0]*size[1]), max(int(size[0]*size[1]*prob), 2), replace=False)
    s_index = np.random.choice(s, len(s), replace=False)
    for i,j in zip(s, s_index):
        image2.paste(image.crop(axis[i]), axis[j])
    return image2