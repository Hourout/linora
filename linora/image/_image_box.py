import itertools

from PIL import Image

from linora.image._image_draw import draw_box
from linora.image._image_feature import histogram

__all__ = ['box_area', 'box_convert']


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


def box_convert(box, in_fmt, out_fmt):
    """Converts boxes from given in_fmt to out_fmt. 
    
    Supported in_fmt and out_fmt are:
    'xyxy': boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right. 
    'xywh': boxes are represented via corner, width and height, x1, y1 being top left, w, h being width and height.
    'cxcywh': boxes are represented via centre, width and height, cx, cy being center of box, w, h being width and height.
    Args:
        box: 4 tuple or list, boxes which will be converted.
        in_fmt: str, input format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh'].
        out_fmt: str, output format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh'].
    Returns:
        a box, list, boxes which will be converted.
    """
    if in_fmt=='xyxy' and out_fmt=='xywh':
        return [box[0], box[1], box[2]-box[0], box[3]-box[1]]
    if in_fmt=='xyxy' and out_fmt=='cxcywh':
        return [(box[2]+box[0])/2, (box[3]+box[1])/2, box[2]-box[0], box[3]-box[1]]
    if in_fmt=='xywh' and out_fmt=='xyxy':
        return [box[0], box[1], box[2]+box[0], box[3]+box[1]]
    if in_fmt=='xywh' and out_fmt=='cxcywh':
        return [box[0]+box[2]/2, box[1]+box[3]/2, box[2], box[3]]
    if in_fmt=='cxcywh' and out_fmt=='xywh':
        return [box[0]-box[2]/2, box[1]-box[3]/2, box[2], box[3]]
    if in_fmt=='cxcywh' and out_fmt=='xyxy':
        return [box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2]
    