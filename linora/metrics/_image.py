import numpy as np
from linora.metrics._classification import confusion_matrix

__all__ = ['iou_detection', 'iou_segmentation']

def iou_detection(box1, box2):
    """Computational object detection iou.
    Args:
        box1: [xmin1, ymin1, xmax1, ymax1]
        box2: [xmin2, ymin2, xmax2, ymax2]
    return:
        object detection iou values.
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)
 
    w = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
    h = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
    return w * h / (s1 + s2 - a1)

def iou_segmentation(y_true, y_pre):
    """Calculate object segmentation iou.
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pre: pd.Series or array or list, predicted labels.
    Returns:
        a dict, object segmentation iou values.
    """
    cm = confusion_matrix(np.array(y_true).flatten(), np.array(y_pre).flatten())
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    IoU = intersection / union 
    MIoU = round(np.mean(IoU), 4)
    return {'IoU_mean':MIoU, 'IoU_class':IoU.round(4).to_dict()}
