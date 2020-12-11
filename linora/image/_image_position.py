import numpy as np
from PIL import Image

__all_ = ['filp_up_left', 'flip_up_right', 'flip_left_right', 'flip_up_down', 'rotate', 'translate']

def filp_up_left(image, random=False):
    if random:
        random = np.random.choice([True, False])
    return image.transpose(Image.TRANSPOSE) if not random else image

def flip_up_right(image, random=False):
    if random:
        random = np.random.choice([True, False])
    return image.transpose(Image.TRANSVERSE) if not random else image

def flip_left_right(image, random=False):
    if random:
        random = np.random.choice([True, False])
    return image.transpose(Image.FLIP_LEFT_RIGHT) if not random else image

def flip_up_down(image, random=False):
    if random:
        random = np.random.choice([True, False])
    return image.transpose(Image.FLIP_TOP_BOTTOM) if not random else image

def rotate(image, angle, expand=True, center=None, translate=None, fillcolor=None):
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
    if translate=='random':
        translate = (np.random.randint(0, image.size[0]*0.8), np.random.randint(0, image.size[1]*0.8))
    if fillcolor=='random':
        fillcolor = np.random.choice(['green', 'red', 'white', 'black'])
    elif isinstance(fillcolor, (list, tuple)):
        fillcolor = np.random.choice(fillcolor)
    return rotate(image, angle=0, expand=1, center=None, translate=translate, fillcolor=fillcolor)

