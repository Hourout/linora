import itertools

import numpy as np
from PIL import Image

from linora.image._image_draw import draw_box
from linora.image._image_feature import histogram

__all__ = ['box_area', 'box_convert', 'mask_to_box']


def box_area(box):
    """Computes the area of a set of bounding boxes.
    
    Args:
        box: boxes for which the area will be computed. eg.(x1, y1, x2, y2,...) or [[x1, y1], [x2, y2], ...]
    Returns:
        the area for box.
    """
    if isinstance(box[0], (list, tuple)):
        box = list(itertools.chain.from_iterable(box))
    image = Image.new('L', [max([box[i*2] for i in range(int(len(box)/2))]), max([box[i*2+1] for i in range(int(len(box)/2))])])
    image = draw_box(image, box, fill_color=255, line_color=255)
    return histogram(image, if_global=True)[-1]


def mask_to_box(mask):
    """Compute the bounding boxes around the provided masks.
    
    Args:
        mask: a numpy array, shape is (H, W).
    returns: 
        a list of bounding boxes, eg [[x1,y1,x2,y2],...].
    """
    axis = []
    for i in np.unique(mask):
        if i!=0:
            y, x = np.where(mask==i)
            axis.append([x.min(), y.min(), x.max(), y.max()])
    return axis


def box_convert(box, in_fmt, out_fmt):
    """Converts boxes from given in_fmt to out_fmt. 
    
    Supported in_fmt and out_fmt are:
    'xyxy': boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right. 
    'xywh': boxes are represented via corner, width and height, x1, y1 being top left, w, h being width and height.
    'cxcywh': boxes are represented via centre, width and height, cx, cy being center of box, w, h being width and height.
    'axis': all coordinates of the 4 points in the boxes.
    Args:
        box: 4 tuple or list, boxes which will be converted.
        in_fmt: str, input format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh', 'axis'].
        out_fmt: str, output format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh', 'axis'].
    Returns:
        a box, list, boxes which will be converted.
    """
    if in_fmt=='xyxy' and out_fmt=='xywh':
        return [box[0], box[1], box[2]-box[0], box[3]-box[1]]
    if in_fmt=='xyxy' and out_fmt=='cxcywh':
        return [(box[2]+box[0])/2, (box[3]+box[1])/2, box[2]-box[0], box[3]-box[1]]
    if in_fmt=='xyxy' and out_fmt=='axis':
        return [box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]]
    if in_fmt=='xywh' and out_fmt=='xyxy':
        return [box[0], box[1], box[2]+box[0], box[3]+box[1]]
    if in_fmt=='xywh' and out_fmt=='cxcywh':
        return [box[0]+box[2]/2, box[1]+box[3]/2, box[2], box[3]]
    if in_fmt=='xywh' and out_fmt=='axis':
        return [box[0], box[1], box[0]+box[2], box[1], box[0]+box[2], box[1]+box[3], box[0], box[1]+box[3]]
    if in_fmt=='cxcywh' and out_fmt=='xywh':
        return [box[0]-box[2]/2, box[1]-box[3]/2, box[2], box[3]]
    if in_fmt=='cxcywh' and out_fmt=='xyxy':
        return [box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2]
    if in_fmt=='cxcywh' and out_fmt=='axis':
        return [box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]-box[3]/2, 
                box[0]+box[2]/2, box[1]+box[3]/2, box[0]-box[2]/2, box[1]+box[3]/2]
    if in_fmt=='axis' and out_fmt=='xyxy':
        return [box[0], box[1], box[4], box[5]]
    if in_fmt=='axis' and out_fmt=='cxcywh':
        return [(box[0]+box[4])/2, (box[1]+box[5])/2, box[4]-box[0], box[5]-box[1]]
    if in_fmt=='axis' and out_fmt=='xywh':
        return [box[0], box[1], box[4]-box[0], box[5]-box[1]]