import itertools
from random import randint

from PIL import ImageDraw

__all__ = ['draw_box', 'draw_point', 'draw_mask']


def draw_point(image, points, size=0, color=None):
    """Draws a point.
    
    Args:
        image: a PIL instance.
        points: Sequence of either 2-tuples like [(x, y), (x, y), ...] or numeric values like [x, y, x, y, ...].
        size: point size.
        color: point color.
    returns: 
        a PIL instance.
    """
    image2 = image.copy()
    draw = ImageDraw.Draw(image2)
    point = list(itertools.chain.from_iterable(points)) if isinstance(points[0], (list, tuple)) else points
    for i in range(int(len(point)/2)):
        axis = list(itertools.product(range(int(point[i*2]-size), int(point[i*2]+size+1)), 
                                      range(int(point[i*2+1]-size), int(point[i*2+1]+size+1))))
        if color is None:
            draw.point(axis, (randint(0, 255), randint(0, 255), randint(0, 255)))
        else:
            draw.point(axis, color)
    return image2


def draw_mask(image, size, max_num, random=False, color=None):
    """Draws a mask.
    
    Args:
        image: a PIL instance.
        size: list or tuple, mask size, [height, width].
        max_num: int, max mask number.
        random: bool, whether the mask position is random.
        color: mask color.
    returns: 
        a PIL instance.
    """
    image2 = image.copy()
    draw = ImageDraw.Draw(image2)
    color = (randint(0, 255), randint(0, 255), randint(0, 255)) if color is None else color
    if random:
        for i in range(max_num):
            axis = (randint(0, image.width-size[1]), randint(0, image.height-size[0]))
            axis = [axis, (axis[0]+size[1], axis[1]+size[0])]
            draw.rectangle(axis, fill=color, width=0)
    else:
        width_num = min(int(image.width/size[1]*0.6), int(max_num**0.5))
        height_num = min(int(max_num/width_num), int(image.height/size[0]*0.6))
        width_pix = int((image.width-width_num*size[1])/(width_num+1))
        height_pix = int((image.height-height_num*size[0])/(height_num+1))
        for i in range(width_num):
            for j in range(height_num):
                axis = [width_pix*(i+1)+size[1]*i, height_pix*(j+1)+size[0]*j, 
                        width_pix*(i+1)+size[1]*(i+1), height_pix*(j+1)+size[0]*(j+1)]
                draw.rectangle(axis, fill=color, width=0)
    return image2


def draw_box(image, boxs, fill_color=None, line_color=None):
    """Draws a polygon.
    
    The polygon outline consists of straight lines between the given coordinates, 
    plus a straight line between the last and the first coordinate.
    
    Args:
        image: a PIL instance.
        boxs: If it is a single sample, like [(x1, y1), (x2, y2), ...] or like [x1, y1, x2, y2, ...].
              If there are multiple samples, like [[(x1, y1), (x2, y2), ...]] or like [[x1, y1, x2, y2, ...]].
              special, like [(x1, y1), (x2, y2)] or like [x1, y1, x2, y2], draw a rectangle from the top left to the bottom right.
              special, like [x,y], or like [[x,y]], Translate into [x, x, y, y], draw a rectangle from the top left to the bottom right.
        fill_color: str or tuple or la.image.RGBMode, rgb color, box fill color.
        line_color: str or tuple or la.image.RGBMode, rgb color, box line color.
    returns: 
        a PIL instance.
    """
    image2 = image.copy()
    draw = ImageDraw.Draw(image2)
    if isinstance(boxs[0], (int, float)):
        if len(boxs)==2:
            boxs = [[min(boxs), min(boxs), min(boxs), max(boxs), max(boxs), max(boxs), max(boxs), min(boxs)]]
        elif len(boxs)==4:
            boxs = [[min(boxs[0], boxs[2]), min(boxs[1], boxs[3]), max(boxs[0], boxs[2]), min(boxs[1], boxs[3]),
                     max(boxs[0], boxs[2]), max(boxs[1], boxs[3]), min(boxs[0], boxs[2]), max(boxs[1], boxs[3])]]
        elif len(boxs)%2 == 0:
            boxs = [boxs]
        else:
            raise ValueError('boxs axis error')
    elif isinstance(boxs[0], (list, tuple)):
        if len(boxs)==1 and len(boxs[0])==2 and isinstance(boxs[0][0], (int, float)):
            boxs = [[min(boxs[0]), min(boxs[0]), min(boxs[0]), max(boxs[0]), max(boxs[0]), max(boxs[0]), max(boxs[0]), min(boxs[0])]]
        elif len(boxs)==2 and len(boxs[0])==2:
            boxs = [boxs[0][0], boxs[0][1], boxs[1][0], boxs[1][1]]
            boxs = [[min(boxs[0], boxs[2]), min(boxs[1], boxs[3]), max(boxs[0], boxs[2]), min(boxs[1], boxs[3]),
                     max(boxs[0], boxs[2]), max(boxs[1], boxs[3]), min(boxs[0], boxs[2]), max(boxs[1], boxs[3])]]
        elif len(boxs)>2 and len(boxs[0])==2:
            boxs = [[j for i in boxs for j in i]]
        else:
            boxs = [list(itertools.chain.from_iterable(i)) if isinstance(i[0], (list, tuple)) else i for i in boxs]
            for i in boxs:
                if len(i)%2 != 0 or len(i)<6:
                    raise ValueError('boxs axis error')
    else:
            raise ValueError('boxs axis error')
    for i in boxs:
        if line_color is None:
            line_color = (randint(0, 255), randint(0, 255), randint(0, 255))
        elif isinstance(line_color, dict):
            line_color = line_color['mode']
        if isinstance(fill_color, dict):
            fill_color = fill_color['mode']
        draw.polygon(i, fill=fill_color, outline=line_color)
    return image2