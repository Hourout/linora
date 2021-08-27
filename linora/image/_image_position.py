import numpy as np
from PIL import Image

__all__ = ['flip_up_left', 'flip_up_right', 'flip_left_right', 'flip_up_down', 'rotate', 'translate']

def flip_up_left(image, random=False):
    """Randomly flip an image (up to left).
    
    With a 1 in 2 chance, outputs the contents of `image` flipped along the
    second dimension, which is `width`.  Otherwise output the image as-is.
    
    Args:
    image: a Image instance.
    random: bool, default False.
            if True, random flip up and left image.
            if False, flip up and left image.
    Returns:
            A Image instance. of the same type and shape as `image`.
    """
    if random:
        random = np.random.choice([True, False])
    return image.transpose(Image.TRANSPOSE) if not random else image

def flip_up_right(image, random=False):
    """Randomly flip an image (up to right).
    
    With a 1 in 2 chance, outputs the contents of `image` flipped along the
    second dimension, which is `width`.  Otherwise output the image as-is.
    
    Args:
    image: a Image instance.
    random: bool, default False.
            if True, random flip up and right image.
            if False, flip up and right image.
    Returns:
            A Image instance. of the same type and shape as `image`.
    """
    if random:
        random = np.random.choice([True, False])
    return image.transpose(Image.TRANSVERSE) if not random else image

def flip_left_right(image, random=False):
    """Randomly flip an image (left to right).
    
    With a 1 in 2 chance, outputs the contents of `image` flipped along the
    second dimension, which is `width`.  Otherwise output the image as-is.
    
    Args:
    image: a Image instance.
    random: bool, default False.
            if True, random flip left and rignt image.
            if False, flip left and right image.
    Returns:
            A Image instance. of the same type and shape as `image`.
    """
    if random:
        random = np.random.choice([True, False])
    return image.transpose(Image.FLIP_LEFT_RIGHT) if not random else image

def flip_up_down(image, random=False):
    """Randomly flip an image (up to down).
    
    With a 1 in 2 chance, outputs the contents of `image` flipped along the
    second dimension, which is `width`.  Otherwise output the image as-is.
    
    Args:
    image: a Image instance.
    random: bool, default False.
            if True, random flip up and down image.
            if False, flip up and down image.
    Returns:
            A Image instance. of the same type and shape as `image`.
    """
    if random:
        random = np.random.choice([True, False])
    return image.transpose(Image.FLIP_TOP_BOTTOM) if not random else image

def rotate(image, angle, expand=True, center=None, translate=None, fillcolor=None):
    """Returns a rotated copy of this image. This method returns a copy of this image, 
    rotated the given number of degrees counter clockwise around its centre.
    
    Args:
    image: a Image instance.
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
            you can also pass in a list of colors.
    
    Returns:
            A Image instance. of the same type and shape as `image`.
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
    elif isinstance(fillcolor, (list, tuple)):
        fillcolor = np.random.choice(fillcolor)
    return image.rotate(angle, expand, center, translate, fillcolor)

def translate(image, translate=None, fillcolor=None):
    """Returns a translate copy of this image. 
    
    Args:
    image: a Image instance.
    translate: An optional post-rotate translation (a 2-tuple).
            if value is 'random', then the function is random.
    fillcolor: An optional color for area outside the rotated image.
            if value is 'random', fillcolor is one of ['green', 'red', 'white', 'black'].
            you can also pass in a list of colors.
    
    Returns:
            A Image instance. of the same type and shape as `image`.
    """
    if translate=='random':
        translate = (np.random.randint(0, image.size[0]*0.8), np.random.randint(0, image.size[1]*0.8))
    if fillcolor=='random':
        fillcolor = np.random.choice(['green', 'red', 'white', 'black'])
    elif isinstance(fillcolor, (list, tuple)):
        fillcolor = np.random.choice(fillcolor)
    return rotate(image, angle=0, expand=1, center=None, translate=translate, fillcolor=fillcolor)

