import random
import itertools

from PIL import ImageDraw

__all__ = ['draw_box', 'draw_point']

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
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
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
        axis = list(itertools.product(range(int(point[i*2]-size), int(point[i*2]+size)), 
                                      range(int(point[i*2+1]-size), int(point[i*2+1]+size))))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) if color is None else color
        draw.point(axis, color)
    return image2