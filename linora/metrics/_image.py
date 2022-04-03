import numpy as np
from linora.metrics._classification import confusion_matrix

__all__ = ['iou_detection', 'iou_segmentation', 'psnr', 'ssim']

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
    return w * h / (s1 + s2 - w*h)

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

def psnr(image1, image2):
    """Peak Signal to Noise Ratio, PSNR.
    Args:
        image1: image array.
        image2: image array with same shape as image1.
    Returns:
        PSNR values.
    """
    mse = np.mean(np.square(image1.astype(np.float64) - image2.astype(np.float64)))
    return 10. * np.log10(np.square(255.) / mse)

def ssim(image1,image2):
    """structural similarity inde, SSIM.
    Args:
        image1: image array, grayscale image.
        image2: image array with same shape as image1.
    Returns:
        SSIM values.
    """
    assert len(image1.shape) == 2 and len(image2.shape) == 2 and image1.shape == image2.shape
    mu1 = image1.mean()
    mu2 = image2.mean()
    sigma1 = np.sqrt(((image1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((image2 - mu2) ** 2).mean())
    sigma12 = ((image1 - mu1) * (image2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2/2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim
