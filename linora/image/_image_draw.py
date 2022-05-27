import itertools

import numpy as np
from PIL import ImageDraw, ImageChops

from linora.image._image_util import array_to_image
from linora.image._image_rgb import _fill_color

__all__ = ['draw_box', 'draw_point', 'mask', 'mask_box', 'draw_line', 'draw_keypoints', 'draw_segmentation_masks', 'stripe']


def draw_point(image, points, size=0, color=None):
    """Draws a point.
    
    Args:
        image: a PIL instance.
        points: Sequence of either 2-tuples like [(x, y), (x, y), ...] or numeric values like [x, y, x, y, ...].
        size: point size.
        color: str or tuple or la.image.RGBMode, rgb color, point color.
    returns: 
        a PIL instance.
    """
    image2 = image.copy()
    draw = ImageDraw.Draw(image2)
    point = list(itertools.chain.from_iterable(points)) if isinstance(points[0], (list, tuple)) else points
    for i in range(int(len(point)/2)):
        axis = list(itertools.product(range(int(point[i*2]-size), int(point[i*2]+size+1)), 
                                      range(int(point[i*2+1]-size), int(point[i*2+1]+size+1))))
        color1 = _fill_color(image, color)
        draw.point(axis, color1)
    return image2


def mask(image, size, max_num, random=True, color=None, p=1):
    """Draws a mask.
    
    Args:
        image: a PIL instance.
        size: list or tuple, mask size, [height, width]. if int, transform [size, size].
        max_num: int, max mask number.
                 if tuple or list, randomly picked in the interval `[max_num[0], max_num[1])`.
        random: bool, whether the mask position is random.
        color: str or tuple or la.image.RGBMode, rgb color, mask fill color.
        p: probability that the image does this. Default value is 1.
    returns: 
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(max_num, (list, tuple)):
        max_num = int(np.random.uniform(max_num[0], max_num[1]))
    if isinstance(size, (int, float)):
        size = [int(size), int(size)]
    image2 = image.copy()
    draw = ImageDraw.Draw(image2)
    if random:
        for i in range(max_num):
            axis = (np.random.randint(0, image.width-size[1]), np.random.randint(0, image.height-size[0]))
            axis = [axis, (axis[0]+size[1], axis[1]+size[0])]
            color1 = _fill_color(image, color)
            draw.rectangle(axis, fill=color1, width=0)
    else:
        width_num = min(int(image.width/size[1]*0.6), int(max_num**0.5))
        height_num = min(int(max_num/width_num), int(image.height/size[0]*0.6))
        width_pix = int((image.width-width_num*size[1])/(width_num+1))
        height_pix = int((image.height-height_num*size[0])/(height_num+1))
        for i in range(width_num):
            for j in range(height_num):
                axis = [width_pix*(i+1)+size[1]*i, height_pix*(j+1)+size[0]*j, 
                        width_pix*(i+1)+size[1]*(i+1), height_pix*(j+1)+size[0]*(j+1)]
                color1 = _fill_color(image, color)
                draw.rectangle(axis, fill=color1, width=0)
    return image2


def mask_box(image, boxes, fill='inner', color=None, p=1):
    """Fill color inside or outside a rectangular area.
    
    Args:
        image: a PIL  instance.
        boxes: box axis with same [[x1,y1,x2,y2],...]
        fill: 'inner' or 'outer', color fill position.
        color: str or tuple or la.image.RGBMode, rgb color, box line color.
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL  instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(boxes[0], int):
        boxes = [boxes]
    color = _fill_color(image, color)
    if fill=='inner':
        image1 = image.copy()
        for i in boxes:
            image1.paste(Image.new(image.mode, (i[2]-i[0], i[3]-i[1]), color=color), (i[0], i[1]))
    else:
        image1 = Image.new(image.mode, image.size, color=color)
        for i in boxes:
            image1.paste(image.crop(i), (i[0], i[1]))
    return image1


def stripe(image, width=4, mode=1, color=None, p=1):
    """vertically and horizontally line apply to image.
    
    Args:
        image: a numpy array.
        width: int, line spacing width.
        mode: 0 is vertically and 1 is horizontally.
        color: int or list, line color.
        p: probability that the image does this. Default value is 1.
    returns: 
        a numpy array.
    """
    if np.random.uniform()>p:
        return image
    image1 = image.copy()
    color = _fill_color(image1, color)
    if mode:
        image1[::width,:] = color
    else:
        image1[:,::width] = color
    return image1


def draw_box(image, boxs, fill_color=None, line_color=None, width=1):
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
        width: box line width.
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
        if isinstance(fill_color, dict):
            fill_color = fill_color['mode']
        line_color1 = _fill_color(image, line_color)
        draw.polygon(i, fill=fill_color, outline=line_color1, width=width)
    return image2


def draw_line(image, axis, width=1, color=None):
    """Draws a line.
    
    Args:
        image: a PIL instance.
        axis: Sequence of either 2-tuples like [(x, y), (x, y), ...] or numeric values like [x, y, x, y, ...].
        width: int, the line width, in pixels.
        color: str or tuple or la.image.RGBMode, rgb color, line color.
    returns: 
        a PIL instance.
    """
    image2 = image.copy()
    draw = ImageDraw.Draw(image2)
    axis = list(itertools.chain.from_iterable(axis)) if isinstance(axis[0], (list, tuple)) else axis
    for i in range(int(len(axis)/2-1)):
        line = [axis[i*2], axis[i*2+1], axis[i*2+2], axis[i*2+3]]
        color1 = _fill_color(image, color)
        draw.line(line, fill=color1, width=width)
    return image2


def draw_keypoints(image, keypoints, connectivity=None, point_width=3, line_width=2, point_color=None, line_color=None):
    """Draws Keypoints on given RGB image.
    
    Args:
        image: a PIL instance.
        keypoints: list of shape (num_instances, K, 2) the K keypoints location for each of the N instances,
                   in the format [x, y].
        connectivity: A List of tuple where, each tuple contains pair of keypoints to be connected.
        point_width: Integer denoting radius of keypoint.
        line_width: Integer denoting width of line connecting keypoints.
        point_color: str or tuple or la.image.RGBMode, rgb color, key points color.
        line_color: str or tuple or la.image.RGBMode, rgb color, line color.
    Returns:
        a PIL instance.
    """    
    image2 = image.copy()
    draw = ImageDraw.Draw(image2)
    if isinstance(keypoints[0][0], (int, float)):
        keypoints = [keypoints]
    for kpt_id, kpt_inst in enumerate(keypoints):
        for inst_id, kpt in enumerate(kpt_inst):
            x1 = kpt[0] - point_width
            x2 = kpt[0] + point_width
            y1 = kpt[1] - point_width
            y2 = kpt[1] + point_width
            point_color = _fill_color(image, point_color)
            draw.ellipse([x1, y1, x2, y2], fill=point_color, outline=None, width=0)
        if connectivity:
            for connection in connectivity:
                start_pt_x = kpt_inst[connection[0]][0]
                start_pt_y = kpt_inst[connection[0]][1]
                end_pt_x = kpt_inst[connection[1]][0]
                end_pt_y = kpt_inst[connection[1]][1]
                color = _fill_color(image, line_color)
                draw.line(((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)), fill=color, width=line_width)
    return image2


def draw_segmentation_masks(mask, image=None, alpha=0.8, color=None):
    """Draws segmentation masks on given RGB image.
    
    Args:
        mask: a numpy array, shape is (H, W).
        image: a PIL instance.
        alpha: float number between 0 and 1 denoting the transparency of the masks.
               0 means full transparency, 1 means no transparency.
        color: str or tuple or la.image.RGBMode, rgb color, point color.
               List containing the colors of the masks.
    returns: 
        a PIL instance.
    """
    canvas = np.zeros((mask.shape[0], mask.shape[1], 3))
    if color is not None:
        assert len(color)==mask.max()+1, 'color size not equal mask nunique size.'
    else:
        color = tuple([tuple([np.random.randint(0, 256) for j in range(3)]) for i in range(int(mask.max()+1))])
    for label in range(int(mask.max()+1)):
        if canvas[mask == label].size:
            canvas[mask == label] = color[label]
    canvas = array_to_image(canvas)
    if image is None:
        return canvas
    if image.size!=canvas.size:
        return ImageChops.blend(image.resize(canvas.size), canvas, alpha=alpha)
    return ImageChops.blend(image, canvas, alpha=alpha)