import os
from math import ceil
from itertools import product

import numpy as np

from linora.parallel import ProcessLoom
from linora.image._image_util import *
from linora.image._image_io import read_image

__all__ = ['mean_std', 'lego', 'pencil_sketch', 'histogram', 'flow_to_image']


def mean_std(image_file, mode=True):
    """Statistical image mean and standard deviation.
    
    Args:
        image_file: str, image path or image folder, image channel should be 3 or 4 channels.
        mode: int or bool, True is 3 channels and False is all channels.
    Returns:
        a dict about mean and std of `image_file`.
    """
    def image_map(path, mode):
        image = read_image(path)
        image = color_convert(image, ColorMode.RGB)
        image = image_to_array(image).reshape(-1, 3)
        return [image.mean(axis=0), image.std(axis=0)] if mode else [image.mean(), image.std()]
    if os.path.isdir(image_file):
        loom = ProcessLoom()
        for i in list_images(image_file):
            loom.add_function(image_map, [i, mode])
        t = loom.execute()
        if mode:
            mean = [[t[i]['output'][0][0] for i in t], [t[i]['output'][0][1] for i in t], [t[i]['output'][0][2] for i in t]]
            std = [[t[i]['output'][1][0] for i in t], [t[i]['output'][1][1] for i in t], [t[i]['output'][1][2] for i in t]]
            result = {'mean':[round(sum(i)/len(i),3) for i in mean], 'std':[round(sum(i)/len(i),3) for i in std]}
        else:
            mean = [t[i]['output'][0] for i in t]
            std = [t[i]['output'][1] for i in t]
            result = {'mean':round(sum(mean)/len(mean),3), 'std':round(sum(std)/len(std),3)}
    else:
        result = image_map(image_file, mode)
        if mode:
            result = {'mean':[round(i,3) for i in result[0].tolist()], 'std':[round(i,3) for i in result[1].tolist()]}
        else:
            result = {'mean':round(result[0],3), 'std':round(result[1],3)}
    return result


def lego(image, stride=15, overlay_ratio=0):
    """Generate a Lego picture.
    
    Args:
        image: image path or PIL instance.
        stride: image stride numbers.
        overlay_ratio: display the ratio of the original image in the newly generated image.
    Returns:
        a numpy array
    """
    lego = [122, 122, 122, 127, 127, 127, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 124, 124, 124, 124, 124, 124, 
            123, 123, 123, 122, 122, 122, 123, 123, 123, 123, 123, 123, 124, 124, 124, 123, 123, 123, 124, 124, 124, 123, 123, 123, 123, 123, 123, 
            124, 124, 124, 117, 117, 117, 131, 131, 131, 136, 136, 136, 134, 134, 134, 133, 133, 133, 135, 135, 135, 135, 135, 135, 135, 135, 135, 
            133, 133, 133, 131, 131, 131, 130, 130, 130, 130, 130, 130, 130, 130, 130, 131, 131, 131, 133, 133, 133, 133, 133, 133, 132, 132, 132, 
            131, 131, 131, 131, 131, 131, 132, 132, 132, 126, 126, 126, 129, 129, 129, 136, 136, 136, 133, 133, 133, 133, 133, 133, 135, 135, 135, 
            131, 131, 131, 132, 132, 132, 144, 144, 144, 156, 156, 156, 162, 162, 162, 163, 163, 163, 157, 157, 157, 142, 142, 142, 129, 129, 129, 
            131, 131, 131, 132, 132, 132, 131, 131, 131, 129, 129, 129, 132, 132, 132, 125, 125, 125, 129, 129, 129, 136, 136, 136, 135, 135, 135, 
            135, 135, 135, 132, 132, 132, 138, 138, 138, 162, 162, 162, 165, 165, 165, 155, 155, 155, 149, 149, 149, 150, 150, 150, 159, 159, 159, 
            170, 170, 170, 160, 160, 160, 135, 135, 135, 131, 131, 131, 132, 132, 132, 131, 131, 131, 132, 132, 132, 125, 125, 125, 129, 129, 129, 
            137, 137, 137, 135, 135, 135, 134, 134, 134, 139, 139, 139, 163, 163, 163, 145, 145, 145, 129, 129, 129, 129, 129, 129, 130, 130, 130, 
            130, 130, 130, 128, 128, 128, 131, 131, 131, 154, 154, 154, 162, 162, 162, 134, 134, 134, 131, 131, 131, 131, 131, 131, 132, 132, 132, 
            125, 125, 125, 129, 129, 129, 137, 137, 137, 135, 135, 135, 134, 134, 134, 150, 150, 150, 137, 137, 137, 128, 128, 128, 133, 133, 133, 
            130, 130, 130, 127, 127, 127, 127, 127, 127, 126, 126, 126, 126, 126, 126, 126, 126, 126, 145, 145, 145, 152, 152, 152, 129, 129, 129, 
            132, 132, 132, 133, 133, 133, 125, 125, 125, 130, 130, 130, 138, 138, 138, 132, 132, 132, 134, 134, 134, 138, 138, 138, 135, 135, 135, 
            136, 136, 136, 127, 127, 127, 137, 137, 137, 147, 147, 147, 129, 129, 129, 138, 138, 138, 136, 136, 136, 127, 127, 127, 139, 139, 139, 
            148, 148, 148, 128, 128, 128, 129, 129, 129, 134, 134, 134, 126, 126, 126, 130, 130, 130, 138, 138, 138, 126, 126, 126, 134, 134, 134, 
            133, 133, 133, 129, 129, 129, 137, 137, 137, 120, 120, 120, 127, 127, 127, 128, 128, 128, 126, 126, 126, 126, 126, 126, 134, 134, 134, 
            126, 126, 126, 131, 131, 131, 142, 142, 142, 127, 127, 127, 121, 121, 121, 134, 134, 134, 126, 126, 126, 129, 129, 129, 134, 134, 134, 
            118, 118, 118, 134, 134, 134, 131, 131, 131, 123, 123, 123, 129, 129, 129, 124, 124, 124, 128, 128, 128, 119, 119, 119, 126, 126, 126, 
            119, 119, 119, 111, 111, 111, 125, 125, 125, 115, 115, 115, 131, 131, 131, 129, 129, 129, 112, 112, 112, 131, 131, 131, 126, 126, 126, 
            128, 128, 128, 130, 130, 130, 112, 112, 112, 134, 134, 134, 128, 128, 128, 129, 129, 129, 124, 124, 124, 123, 123, 123, 148, 148, 148, 
            124, 124, 124, 125, 125, 125, 143, 143, 143, 130, 130, 130, 125, 125, 125, 125, 125, 125, 132, 132, 132, 126, 126, 126, 104, 104, 104, 
            127, 127, 127, 124, 124, 124, 126, 126, 126, 126, 126, 126, 104, 104, 104, 131, 131, 131, 126, 126, 126, 128, 128, 128, 124, 124, 124, 
            124, 124, 124, 116, 116, 116, 124, 124, 124, 124, 124, 124, 120, 120, 120, 131, 131, 131, 119, 119, 119, 128, 128, 128, 133, 133, 133, 
            119, 119, 119, 101, 101, 101, 127, 127, 127, 124, 124, 124, 126, 126, 126, 127, 127, 127, 101, 101, 101, 121, 121, 121, 128, 128, 128, 
            147, 147, 147, 132, 132, 132, 132, 132, 132, 148, 148, 148, 122, 122, 122, 142, 142, 142, 129, 129, 129, 123, 123, 123, 144, 144, 144, 
            126, 126, 126, 133, 133, 133, 107, 107, 107, 100, 100, 100, 128, 128, 128, 125, 125, 125, 128, 128, 128, 131, 131, 131, 103, 103, 103, 
            101, 101, 101, 127, 127, 127, 125, 125, 125, 124, 124, 124, 112, 112, 112, 127, 127, 127, 121, 121, 121, 115, 115, 115, 121, 121, 121, 
            113, 113, 113, 119, 119, 119, 125, 125, 125, 130, 130, 130, 88, 88, 88, 103, 103, 103, 128, 128, 128, 124, 124, 124, 127, 127, 127, 130, 
            130, 130, 109, 109, 109, 85, 85, 85, 120, 120, 120, 126, 126, 126, 124, 124, 124, 124, 124, 124, 121, 121, 121, 128, 128, 128, 123, 123, 123, 
            127, 127, 127, 129, 129, 129, 123, 123, 123, 136, 136, 136, 106, 106, 106, 80, 80, 80, 111, 111, 111, 128, 128, 128, 123, 123, 123, 127, 127, 
            127, 130, 130, 130, 118, 118, 118, 90, 90, 90, 80, 80, 80, 123, 123, 123, 131, 131, 131, 130, 130, 130, 129, 129, 129, 127, 127, 127, 129, 129, 
            129, 130, 130, 130, 130, 130, 130, 137, 137, 137, 120, 120, 120, 72, 72, 72, 89, 89, 89, 117, 117, 117, 128, 128, 128, 123, 123, 123, 127, 127, 
            127, 131, 131, 131, 124, 124, 124, 107, 107, 107, 75, 75, 75, 71, 71, 71, 110, 110, 110, 134, 134, 134, 136, 136, 136, 134, 134, 134, 134, 134, 
            134, 134, 134, 134, 137, 137, 137, 118, 118, 118, 65, 65, 65, 74, 74, 74, 106, 106, 106, 122, 122, 122, 129, 129, 129, 124, 124, 124, 127, 127, 
            127, 131, 131, 131, 127, 127, 127, 120, 120, 120, 98, 98, 98, 69, 69, 69, 59, 59, 59, 79, 79, 79, 102, 102, 102, 113, 113, 113, 112, 112, 112, 
            103, 103, 103, 80, 80, 80, 56, 56, 56, 68, 68, 68, 98, 98, 98, 119, 119, 119, 124, 124, 124, 128, 128, 128, 124, 124, 124, 127, 127, 127, 131, 
            131, 131, 127, 127, 127, 127, 127, 127, 117, 117, 117, 96, 96, 96, 70, 70, 70, 55, 55, 55, 55, 55, 55, 58, 58, 58, 57, 57, 57, 53, 53, 53, 52, 
            52, 52, 69, 69, 69, 96, 96, 96, 117, 117, 117, 124, 124, 124, 124, 124, 124, 128, 128, 128, 124, 124, 124, 129, 129, 129, 136, 136, 136, 131, 
            131, 131, 131, 131, 131, 130, 130, 130, 123, 123, 123, 108, 108, 108, 89, 89, 89, 74, 74, 74, 69, 69, 69, 68, 68, 68, 74, 74, 74, 90, 90, 90, 
            108, 108, 108, 123, 123, 123, 127, 127, 127, 128, 128, 128, 129, 129, 129, 132, 132, 132, 127, 127, 127, 123, 123, 123, 129, 129, 129, 125, 
            125, 125, 124, 124, 124, 124, 124, 124, 123, 123, 123, 121, 121, 121, 116, 116, 116, 108, 108, 108, 102, 102, 102, 102, 102, 102, 108, 108, 
            108, 116, 116, 116, 119, 119, 119, 121, 121, 121, 122, 122, 122, 123, 123, 123, 122, 122, 122, 125, 125, 125, 119, 119, 119]
    lego = array_to_image(np.array(lego).reshape(20,20,3)).resize([stride, stride])
    lego = image_to_array(lego).astype(np.int64)
    
    lego[lego < 33] = -100
    lego[(33 <= lego) & (lego <= 233)] -= 133
    lego[lego > 233] = 100

    if isinstance(image, str):
        image = read_image(image)
        image = color_convert(image, ColorMode.RGB)
    image = image.resize([ceil(image.size[0]/stride)*stride, ceil(image.size[1]/stride)*stride])
    if overlay_ratio:
        overlay = image.resize([int(image.size[0] * overlay_ratio), int(image.size[1] * overlay_ratio)])
    image = image_to_array(image).astype(np.int64)
    height, width, num_channels = image.shape
    
    for i, j in product(range(int(width / stride)), range(int(height / stride))):
        avg_color = image[j*stride:(j+1)*stride, i*stride:(i+1)*stride, :].mean(axis=0).mean(axis=0)
        image[j*stride: (j+1)*stride, i*stride:(i+1)*stride, :] = np.clip(avg_color+lego, 0, 255)
    if overlay_ratio:
        image[height - int(height * overlay_ratio):, width - int(width * overlay_ratio):, :] = overlay
    return image


def pencil_sketch(image, delta=0.1, seed=None):
    """Adjust the pencil sketch of RGB.
    
    Args:
        image:  a numpy array
        delta: if int, float, Amount to add to the pixel values.
               if list, tuple, randomly picked in the interval
               `[delta[0], delta[1])` to add to the pixel values.
               a suitable interval is [0.1, 0.5].
        seed: A Python integer. Used to create a random seed. 
    Returns:
        a numpy array.
    """
    if isinstance(delta, (list, tuple)):
        random_delta = np.random.uniform([], delta[0], delta[1], seed=seed)
    else:
        random_delta = delta
        
    t = image[:, 1:]-image[:, :-1]
    grad_y = np.concatenate([t[:,0:1], (t[:, 1:]+t[:, :-1])/2, t[:, -1:]], 1)*random_delta
    t = image[1:, :]-image[:-1, :]
    grad_x = np.concatenate([t[0:1,:], (t[1:, :]+t[:-1, :])/2, t[-1:, :]], 0)*random_delta

    A = np.sqrt(np.square(grad_x)+np.square(grad_y)+1.)
    uni_x = grad_x/A
    uni_y = grad_y/A
    uni_z = 1./A

    dx = np.cos(3.141592653589793/2.2)*np.cos(3.141592653589793/4)
    dy = np.cos(3.141592653589793/2.2)*np.sin(3.141592653589793/4)
    dz = np.sin(3.141592653589793/2.2)

    b = np.clip(255.*(dx*uni_x + dy*uni_y + dz*uni_z), 0., 255.)
    return b


def histogram(image, if_global=False, if_prob=False):
    """Returns a histogram for the image. 
    
    The histogram is returned as a list of pixel counts, one for each pixel value in the source image. 
    Counts are grouped into 256 bins for each band, even if the image has more than 8 bits per band. 
    If the image has more than one band, the histograms for all bands are concatenated.
    For example, the histogram for an “RGB” image contains 768 values.
    
    Args:
        image: a PIL instance.
        if_global: bool, Whether to merge all channels into statistics.
        if_prob: bool, Whether to convert frequency histogram to probability histogram.
    Returns:
        a list of histograms.
    """
    t = image.histogram()
    if len(t)>256:
        t = [t[i*256:(i+1)*256] for i in range(len(t)//256)]
        if if_global:
            t = [sum([j[i] for j in t]) for i in range(256)]
    if if_prob:
        if len(t)==256:
            s = sum(t)
            t = [i/s for i in t]
        else:
            s = [sum(i) for i in t]
            t = [[k/j for k in i] for i,j in zip(t, s)]
    return t


def flow_to_image(flow):
    """Converts a flow to an RGB image.

    Args:
        flow: a numpy array with shape (2, H, W) or (H, W, 2).
    Returns:
        a numpy array with shape (H, W, 3).
    """
    if flow.shape[0]==2:
        flow = np.transpose(flow, (1,2,0))

    max_norm = np.sqrt(np.sum(flow ** 2, axis=2)).max()
    epsilon = np.finfo((flow).dtype).eps
    normalized_flow = flow / (max_norm + epsilon)
    
    H, W, _ = normalized_flow.shape
    flow_image = np.zeros((H, W, 3), dtype=np.uint8)
    colorwheel = _make_colorwheel()
    num_cols = colorwheel.shape[0]
    norm = np.sqrt(np.sum(normalized_flow ** 2, axis=2))
    a = np.arctan2(-normalized_flow[:, :, 1], -normalized_flow[:, :, 0]) / np.pi
    fk = (a + 1) / 2 * (num_cols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == num_cols] = 0
    f = fk - k0

    for c in range(colorwheel.shape[1]):
        tmp = colorwheel[:, c]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        col = 1 - norm * (1 - col)
        flow_image[:, :, c] = np.floor(255 * col)
    return flow_image


def _make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
    Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
    URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf.

    Returns:
        colorwheel (Tensor[55, 3]): Colorwheel Tensor.
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel