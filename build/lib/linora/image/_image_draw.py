import itertools
from random import randint

from PIL import ImageDraw

__all__ = ['draw_box', 'draw_point', 'draw_mask']

def draw_box(image, boxs):
    """Draws a polygon.
    
    The polygon outline consists of straight lines between the given coordinates, 
    plus a straight line between the last and the first coordinate.
    
    Args:
        image: a Image instance.
        boxs: Sequence of either 2-tuples like [(x, y), (x, y), ...] or numeric values like [x, y, x, y, ...].
    returns: 
        a Image instance.
    """
    image2 = image.copy()
    draw = ImageDraw.Draw(image2)
    for i in boxs:
        box = list(itertools.chain.from_iterable(i)) if isinstance(i[0], (list, tuple)) else i
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        draw.polygon(box, outline=color)
    return image2

def draw_point(image, points, size=0, color=None):
    """Draws a point.
    
    Args:
        image: a Image instance.
        points: Sequence of either 2-tuples like [(x, y), (x, y), ...] or numeric values like [x, y, x, y, ...].
        size: point size.
        color: point color.
    returns: 
        a Image instance.
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
        image: a Image instance.
        size: mask size.
        max_num: max mask number.
        random: Whether the mask position is random.
        color: mask color.
    returns: 
        a Image instance.
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