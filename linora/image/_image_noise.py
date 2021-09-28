import random

import numpy as np

from linora.image._image_draw import draw_point

__all__ = ['noise_gaussian', 'noise_poisson', 'noise_color']

def noise_gaussian(image, scale=1, mean=0.0, std=1.0):
    """Gaussian noise apply to image.
    
    new pixel = image + gaussian_noise * scale
    Args:
        image: a Image instance.
        scale: if int or float, value multiply with gaussian noise.
               if tuple or list, randomly picked in the interval
               `[scale[0], scale[1])`, value multiply with gaussian noise.
        mean: if int or float, value is gaussian distribution mean.
              if tuple or list, randomly picked in the interval
              `[mean[0], mean[1])`, value is gaussian distribution mean.
        std: if int or float, value is gaussian distribution std.
             if tuple or list, randomly picked in the interval
             `[std[0], std[1])`, value is gaussian distribution std.
    Returns:
        a Image instance.
    """
    if isinstance(scale, (tuple, list)):
        scale = np.random.uniform(scale[0], scale[1])
    if isinstance(std, (tuple, list)):
        std = np.random.uniform(std[0], std[1])
    if isinstance(mean, (tuple, list)):
        mean = np.random.uniform(mean[0], mean[1])
    return image.point(lambda x:int((np.random.normal(mean, std, size=1)*scale+x).round()))

def noise_poisson(image, scale=1, lam=1.0):
    """Poisson noise apply to image.
    
    new pixel = image + poisson_noise * scale
    Args:
        image: a Image instance.
        scale: if int or float, value multiply with poisson noise.
               if tuple or list, randomly picked in the interval
               `[scale[0], scale[1])`, value multiply with poisson noise.
        lam: if int or float, value is poisson distribution lambda.
             if tuple or list, randomly picked in the interval
             `[lam[0], lam[1])`, value is poisson distribution lambda.
    Returns:
        a Image instance.
    """
    if isinstance(scale, (tuple, list)):
        scale = np.random.uniform(scale[0], scale[1])
    if isinstance(lam, (tuple, list)):
        lam = np.random.uniform(std[0], std[1])
    return image.point(lambda x:int((np.random.poisson(lam, size=1)*scale+x).round()))

def noise_color(image, white_prob=0.05, black_prob=0.05, rainbow_prob=0):
    """Mask noise apply to image with color.
    
    while: 
        white_prob = black_prob and rainbow_prob=0, is salt-pepper noise
        
    The salt-pepper noise is based on the signal-to-noise ratio of the image,
    randomly generating the pixel positions in some images all channel,
    and randomly assigning these pixels to 0 or 255.
    
    while: 
        white_prob = black_prob=0 and rainbow_prob>0, is rainbow noise
        
    The rainbow noise is based on the signal-to-noise ratio of the image,
    randomly generating the pixel positions in some images,
    and randomly assigning these pixels to 0 or 255.
    
    Args:
        image: a Image instance.
        white_prob: white pixel prob.
        black_prob: black pixel prob.
        rainbow_prob: rainbow color pixel prob.
    Returns:
        a Image instance.
    """
    img = image.copy()
    axis = [(i,j) for i in range(image.width) for j in range(image.height)]
    random.shuffle(axis)
    if isinstance(white_prob, (tuple, list)):
        white_prob = np.random.uniform(white_prob[0], white_prob[1])
    if isinstance(black_prob, (tuple, list)):
        black_prob = np.random.uniform(black_prob[0], black_prob[1])
    if isinstance(rainbow_prob, (tuple, list)):
        rainbow_prob = np.random.uniform(rainbow_prob[0], rainbow_prob[1])
    if white_prob>0:
        axis_white = axis[0:int(len(axis)*white_prob)]
        img = draw_point(img, axis_white, size=0, color=(255,255,255))
    if black_prob>0:
        axis_black = axis[int(len(axis)*white_prob):int(len(axis)*(white_prob+black_prob))]
        img = draw_point(img, axis_black, size=0, color=(0,0,0))
    if rainbow_prob>0:
        axis_rainbow = axis[-int(len(axis)*rainbow_prob):]
        img = draw_point(img, axis_rainbow, size=0)
    return img
