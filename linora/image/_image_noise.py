import random

import numpy as np
from PIL import Image

from linora.image._image_draw import draw_point

__all__ = ['noise_color', 'NoiseMode', 'noise']


class noise_mode:
    Gaussian   = 'gaussian'
    Laplace    = 'laplace'
    Poisson    = 'poisson'
    Uniform    = 'uniform'
    
NoiseMode = noise_mode()


def noise(image, mode=NoiseMode.Gaussian, wise='pixel', scale=1, prob=0.6, p=1, **kwargs):
    """noise apply to image.
    
    pixel = scale*noise+pixel
    
    la.image.NoiseMode.Gaussian
        Gaussian noise apply to image.
        
        you should append param `mean` and `std`.
        mean: if int or float, value is gaussian distribution mean.
              if tuple or list, randomly picked in the interval `[mean[0], mean[1])`.
        std: if int or float, value is gaussian distribution std.
             if tuple or list, randomly picked in the interval `[std[0], std[1])`.
        eg.
        la.image.noise(image, mode=la.image.NoiseMode.Gaussian, mean=0, std=30)
        
    la.image.NoiseMode.Laplace
        Laplace noise apply to image.
        
        you should append param `mean` and `lam`.
        mean: if int or float, value is laplace distribution mean.
              if tuple or list, randomly picked in the interval `[mean[0], mean[1])`.
        lam: if int or float, value is laplace distribution lam.
             if tuple or list, randomly picked in the interval `[lam[0], lam[1])`.
        eg.
        la.image.noise(image, mode=la.image.NoiseMode.Laplace, mean=0, lam=30)
        
    la.image.NoiseMode.Poisson
        Poisson noise apply to image.
        
        you should append param `lam`.
        lam: if int or float, value is poisson distribution lam.
             if tuple or list, randomly picked in the interval `[lam[0], lam[1])`.
        eg.
        la.image.noise(image, mode=la.image.NoiseMode.Poisson, lam=30)
        
    la.image.NoiseMode.Uniform
        Uniform noise apply to image.
        
        you should append param `lower` and `upper`.
        lower: if int or float, value is uniform distribution lower.
               if tuple or list, randomly picked in the interval `[lower[0], lower[1])`.
        upper: if int or float, value is uniform distribution upper.
               if tuple or list, randomly picked in the interval `[upper[0], upper[1])`.
        eg.
        la.image.noise(image, mode=la.image.NoiseMode.Uniform, lower=-50, upper=50)
    
    Args:
        image: A PIL instance.
        mode: la.image.NoiseMode
        wise: 'pixel' or 'channel', method of applying noise.
        scale: if int or float, value multiply with noise.
               if tuple or list, randomly picked in the interval `[scale[0], scale[1])`.
        prob: probability of every pixel or channel being changed.
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if isinstance(scale, (tuple, list)):
        scale = np.random.uniform(scale[0], scale[1])
    if 'mean' not in kwargs:
        mean = 0
    elif isinstance(kwargs['mean'], (tuple, list)):
        mean = np.random.uniform(kwargs['mean'][0], kwargs['mean'][1])
    else:
        mean = kwargs['mean']
    if 'std' not in kwargs:
        std = 30
    elif isinstance(kwargs['std'], (tuple, list)):
        std = np.random.uniform(kwargs['std'][0], kwargs['std'][1])
    else:
        std = kwargs['std']
    if 'lam' not in kwargs:
        lam = 30
    elif isinstance(kwargs['lam'], (tuple, list)):
        lam = np.random.uniform(kwargs['lam'][0], kwargs['lam'][1])
    else:
        lam = kwargs['lam']
    if 'lower' not in kwargs:
        lower = -50
    elif isinstance(kwargs['lower'], (tuple, list)):
        lower = np.random.uniform(kwargs['lower'][0], kwargs['lower'][1])
    else:
        lower = kwargs['lower']
    if 'upper' not in kwargs:
        upper = 50
    elif isinstance(kwargs['upper'], (tuple, list)):
        upper = np.random.uniform(kwargs['upper'][0], kwargs['upper'][1])
    else:
        upper = kwargs['upper']
    if mode=='gaussian':
        if wise=='pixel':
            return image.point(lambda x:np.random.normal(mean, std)*scale+x if np.random.uniform()<prob else x)
        else:
            split = list(image.split())
            for i in range(len(split)):
                if np.random.uniform()<prob:
                    split[i] = split[i].point(lambda x:np.random.normal(mean, std)*scale+x)
            return Image.merge(image.mode, split)
    elif mode=='laplace':
        if wise=='pixel':
            return image.point(lambda x:np.random.laplace(mean, lam)*scale+x if np.random.uniform()<prob else x)
        else:
            split = list(image.split())
            for i in range(len(split)):
                if np.random.uniform()<prob:
                    split[i] = split[i].point(lambda x:np.random.laplace(mean, lam)*scale+x)
            return Image.merge(image.mode, split)
    elif mode=='poisson':
        if wise=='pixel':
            return image.point(lambda x:np.random.poisson(lam)*scale+x if np.random.uniform()<prob else x)
        else:
            split = list(image.split())
            for i in range(len(split)):
                if np.random.uniform()<prob:
                    split[i] = split[i].point(lambda x:np.random.poisson(lam)*scale+x)
            return Image.merge(image.mode, split)
    elif mode=='uniform':
        if wise=='pixel':
            return image.point(lambda x:np.random.uniform(lower, upper)*scale+x if np.random.uniform()<prob else x)
        else:
            split = list(image.split())
            for i in range(len(split)):
                if np.random.uniform()<prob:
                    split[i] = split[i].point(lambda x:np.random.uniform(lower, upper)*scale+x)
            return Image.merge(image.mode, split)
    else:
        raise ValueError("mode must be la.image.NoiseMode param.")


def noise_color(image, white_prob=0.05, black_prob=0.05, rainbow_prob=0, p=1):
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
        image: a PIL instance.
        white_prob: white pixel prob.
        black_prob: black pixel prob.
        rainbow_prob: rainbow color pixel prob.
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
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
