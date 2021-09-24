import random
import itertools

from PIL import ImageDraw

__all__ = ['draw_box']

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