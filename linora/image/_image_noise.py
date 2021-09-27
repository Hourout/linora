import random

import numpy as np

from linora.image._image_draw import draw_point

__all__ = ['noise_gaussian', 'noise_poisson', 'noise_color']

def noise_gaussian(image, scale=1, mean=0.0, std=1.0):
    """Gaussian noise apply to image.
    
    new pixel = image + gaussian_noise * scale
    Args:
        image: Either a 3-D float Tensor of shape [height, width, depth].
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
        3-D float Tensor, as per the input.
    Raises:
        scale or lam type error.
    """
    if isinstance(scale, (tuple, list)):
        scale = np.random.uniform(scale[0], scale[1])
    if isinstance(std, (tuple, list)):
        std = np.random.uniform(std[0], std[1])
    if isinstance(mean, (tuple, list)):
        mean = np.random.uniform(mean[0], mean[1])
    return np.random.normal(mean, std, size=image.size)*scale+image

def noise_poisson(image, scale=1, lam=1.0):
    """Poisson noise apply to image.
    
    new pixel = image + poisson_noise * scale
    Args:
        image: Either a 3-D float Tensor of shape [height, width, depth].
        scale: if int or float, value multiply with poisson noise.
               if tuple or list, randomly picked in the interval
               `[scale[0], scale[1])`, value multiply with poisson noise.
        lam: if int or float, value is poisson distribution lambda.
             if tuple or list, randomly picked in the interval
             `[lam[0], lam[1])`, value is poisson distribution lambda.
    Returns:
        3-D float Tensor, as per the input.
    Raises:
        scale or lam type error.
    """
    if isinstance(scale, (tuple, list)):
        scale = np.random.uniform(scale[0], scale[1])
    if isinstance(lam, (tuple, list)):
        lam = np.random.uniform(std[0], std[1])
    return np.random.poisson(lam, size=image.size)*scale+image

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
    if white_prob>0:
        axis_white = axis[0:int(len(axis)*white_prob)]
        img = la.image.draw_point(img, axis_white, size=0, color=(255,255,255))
    if black_prob>0:
        axis_black = axis[int(len(axis)*white_prob):int(len(axis)*(white_prob+black_prob))]
        img = la.image.draw_point(img, axis_black, size=0, color=(0,0,0))
    if rainbow_prob>0:
        axis_rainbow = axis[-int(len(axis)*rainbow_prob):]
        img = la.image.draw_point(img, axis_rainbow, size=0)
    return img

def noise_mask(image, noise_prob=0.2):
    """Mask noise apply to image.
    
    With probability `drop_prob`, outputs the input element scaled up by
    `1`, otherwise outputs `0`. 
    
    Tips:
        1 mean pixel have no change.
        a suitable interval is (0., 0.1].
    Args:
        image: Either a 3-D float Tensor of shape [height, width, depth].
        noise_prob: should be in the interval (0, 1.].
                   if float, the probability that each element is drop.
                   if tuple or list, randomly picked in the interval
                   `[noise_prob[0], noise_prob[1])`, the probability that each element is drop.
    Returns:
        3-D float Tensor, as per the input.
    Raises:
        ValueError: If `keep_prob` is not in `(0, 1.]`.
    """
    if isinstance(noise_prob, (tuple, list)):
        noise_prob = random.uniform(noise_prob[0], noise_prob[1])
    return np.random.choice([0, 1], size=image.size, p=[noise_prob,1-noise_prob])*image

def noise_saltpepper(image, noise_prob=0.2):
    """Salt-Pepper noise apply to image.
    
    The salt-pepper noise is based on the signal-to-noise ratio of the image,
    randomly generating the pixel positions in some images all channel,
    and randomly assigning these pixels to 0 or 255.
    
    Args:
        image: Either a 3-D float Tensor of shape [height, width, depth].
        noise_prob: should be in the interval (0, 1].
                   if int or float, the probability that each element is kept.
                   if tuple or list, randomly picked in the interval
                   `[noise_prob[0], noise_prob[1])`, the probability that each element is drop.
    Returns:
        3-D float Tensor, as per the input.
    Raises:
        ValueError: If `keep_prob` is not in `(0, 1]`.
    """
    if isinstance(noise_prob, (tuple, list)):
        noise_prob = random.uniform(noise_prob[0], noise_prob[1])
    shape = [i for i in image.shape if i!=min(image.shape)]
    noise = np.random.choice([0, 1, 2], size=image.size, p=[noise_prob/2, 1-noise_prob, noise_prob/2])
    image[noise>1] = 255
    image[noise<1] = 0
    return image

def noise_rainbow(image, noise_prob=0.2):
    """Rainbowr noise apply to image.
    
    The rainbow noise is based on the signal-to-noise ratio of the image,
    randomly generating the pixel positions in some images,
    and randomly assigning these pixels to 0 or 255.
    
    Args:
        image: Either a 3-D float Tensor of shape [height, width, depth].
        noise_prob: should be in the interval (0, 1].
                   if int or float, the probability that each element is kept.
                   if tuple or list, randomly picked in the interval
                   `[keep_prob[0], keep_prob[1])`, the probability that each element is kept.
    Returns:
        3-D float Tensor, as per the input.
    Raises:
        ValueError: If `keep_prob` is not in `(0, 1]`.
    """
    if isinstance(noise_prob, (tuple, list)):
        noise_prob = random.uniform(noise_prob[0], noise_prob[1])
    noise = np.random.choice([0, 1, 2], size=image.shape, p=[noise_prob/2, 1-noise_prob, noise_prob/2])
    image[noise>1] = 255
    image[noise<1] = 0
    return image
