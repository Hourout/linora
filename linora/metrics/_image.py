import numpy as np

from linora.metrics._classification import confusion_matrix

__all__ = ['iou_detection', 'psnr', 'ssim', 'total_variation']


def iou_detection(box1, box2, mode=None):
    """Computational object detection iou.
    
    Args:
        box1: list, [xmin1, ymin1, xmax1, ymax1]
        box2: list, [xmin2, ymin2, xmax2, ymax2]
        mode: str, one of ['C', 'D', 'E', 'G'], mean CIoU, DIoU, EIoU, GIoU, if None, is IoU.
    return:
        object detection iou values.
    """
    if isinstance(box1[0], (list,tuple)):
        box1 = [box1[0][0], box1[0][1], box1[1][0], box1[1][1]]
    if isinstance(box2[0], (list,tuple)):
        box2 = [box2[0][0], box2[0][1], box2[1][0], box2[1][1]]
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)
 
    w = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
    h = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
    
    iou = w * h / (s1 + s2 - w*h)
    if mode is None:
        return iou
    if mode=='G':
        area_c = (max(xmax1, xmax2) - min(xmin1, xmin2)) * (max(ymax1, ymax2) - min(ymin1, ymin2))
        return  iou - (area_c - (s1 + s2 - w*h))/area_c
    
    center1 = ((xmin1 + xmax1) / 2, (ymin1 + ymax1) / 2)
    center2 = ((xmin2 + xmax2) / 2, (ymin2 + ymax2) / 2)
    d_center = (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2

    corner1 = (min(xmin1, xmax1, xmin2, xmax2), min(ymin1, ymax1, ymin2, ymax2))
    corner2 = (max(xmin1, xmax1, xmin2, xmax2), max(ymin1, ymax1, ymin2, ymax2))
    d_corner = (corner1[0] - corner2[0]) ** 2 + (corner1[1] + corner2[1]) ** 2
    if mode=='D':
        return  iou - d_center / d_corner
    
    w1, h1 = xmax1 - xmin1, ymax1 - ymin1
    w2, h2 = xmax2 - xmin2, ymax2 - ymin2
    if mode=='C':
        v = 4 * (np.arctan(w1 / h1) - np.arctan(w2 / h2)) ** 2 / (np.pi ** 2)
        alpha = v / (1 - iou + v)
        return iou - d_center / d_corner - alpha * v
    
    if mode=='E':
        w_center = (w1-w2)**2
        w_corner = (corner1[0] - corner2[0]) ** 2

        h_center = (h1-h2)**2
        h_corner = (corner1[1] + corner2[1]) ** 2
        return iou - d_center/d_corner - w_center/w_corner - h_center/h_corner
    raise ValueError("mode must be one of [None, 'C', 'D', 'E', 'G']")


def iou_segmentation(y_true, y_pred):
    """Calculate object segmentation iou.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels.
    Returns:
        a dict, object segmentation iou values.
    """
    cm = confusion_matrix(np.array(y_true).flatten(), np.array(y_pred).flatten())
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


def ssim(image1, image2):
    """structural similarity index Measure, SSIM.
    
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


def total_variation(image):
    """Calculate and return the total variation for image.
    
    The total variation is the sum of the absolute differences for neighboring
    pixel-values in the input images. This measures how much noise is in the image.
    Args:
        image: a NumPy array of image.
    Returns:
        a float.
    """
    if image.ndim==3:
        pixel_dif1 = image[1:, :, :] - image[:-1, :, :]
        pixel_dif2 = image[:, 1:, :] - image[:, :-1, :]
        return np.abs(pixel_dif1).sum()+np.abs(pixel_dif2).sum()
    elif image.ndim==4:
        pixel_dif1 = image[:, 1:, :, :] - image[:, :-1, :, :]
        pixel_dif2 = image[:, :, 1:, :] - image[:, :, :-1, :]
        return np.abs(pixel_dif1).sum()+np.abs(pixel_dif2).sum()
    else:
        raise ValueError('`image` must be either 3 or 4-dimensional.')