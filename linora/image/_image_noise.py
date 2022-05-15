import random

import numpy as np
from PIL import Image

from linora.image._image_draw import draw_point

__all__ = ['mosaic', 'noise_color', 'NoiseMode', 'noise']


def mosaic(image, size=(80,80), block=0, axis=None, prob=0.3, p=1):
    """mosaic noise apply to image.
    
    Args:
        image: a PIL instance.
        size: if int or float, xsize=ysize, how many mosaic blocks the image is divided into.
              if 2-tuple, (xsize, ysize), 
              if 4-tuple, xsize in (size[0], size[1]) and  ysize in (size[2], size[3]).
        block: mosaic area block
        axis: list or tuple, one or more mosaic center axis point.
        prob: probability of numbers of mosaci.
        p: probability that the image does this. Default value is 1.
    Return:
        a PIL instance.
    """
    if isinstance(size, (int, float)):
        size = (size, size)
    elif isinstance(size, (list, tuple)):
        if len(size)==4:
            size = (np.random.uniform(size[0], size[1]), np.random.uniform(size[2], size[3]))
        elif len(size)!=2:
            raise ValueError('`size` value format error.')
    else:
        raise ValueError('`size` value format error.')
    size = (max(2, int(size[0])), max(2, int(size[1])))
    
    if axis is not None:
        if isinstance(axis[0], (list, tuple)):
            block = max(block, len(axis))
            axis_num = len(axis)
        else:
            block = max(block, 1)
            axis_num = 1
            axis = [axis]

    w = image.size[0]//size[0]
    h = image.size[1]//size[1]
    image2 = image.copy()
    if block==0:
        axis_list = [[i*w, j*h, i*w+w, j*h+h] for i in range(size[0]) for j in range(size[1])]
        s = np.random.choice(range(size[0]*size[1]), max(int(size[0]*size[1]*prob), 2), replace=False)
        for i in s:
            image2.paste(Image.new(image.mode, (w,h), tuple([int(j) for j in ImageStat.Stat(image.crop(axis_list[i])).mean])), axis_list[i])
    else:
        num = max(int(size[0]*size[1]*prob/block), 2)
        edge = np.sqrt(num)
        
        for b in range(int(block)):
            if axis is not None:
                if b<=axis_num-1:
                    axis_list = [axis[b][0]/w, axis[b][1]/h]
                else:
                    axis_list = [np.random.uniform(edge/2/size[0], 1-edge/2/size[0])*size[0], 
                                 np.random.uniform(edge/2/size[1], 1-edge/2/size[0])*size[1]]
            else:
                axis_list = [np.random.uniform(edge/2/size[0], 1-edge/2/size[0])*size[0], 
                             np.random.uniform(edge/2/size[1], 1-edge/2/size[0])*size[1]]
            print(axis_list)
            axis_prob = []
            axis_must = []
            for i in range(int(axis_list[0]-edge/2), int(axis_list[0]+edge/2)+1):
                for j in range(int(axis_list[1]-edge/2), int(axis_list[1]+edge/2)+1):
                    if i<int(axis_list[0]-edge/3) or i>int(axis_list[0]+edge/3) or j<int(axis_list[1]-edge/3) or j>int(axis_list[1]+edge/3):
                        axis_prob.append([i*w, j*h, i*w+w, j*h+h])
                    else:
                        axis_must.append([i*w, j*h, i*w+w, j*h+h])
            s = np.random.choice(range(len(axis_prob)), int(len(axis_prob)*0.8), replace=False)
            for i in s:
                image2.paste(Image.new(image.mode, (w,h), tuple([int(j) for j in ImageStat.Stat(image.crop(axis_prob[i])).mean])), axis_prob[i])
            for i in axis_must:
                image2.paste(Image.new(image.mode, (w,h), tuple([int(j) for j in ImageStat.Stat(image.crop(i)).mean])), i)
    return image2


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
            return image.point(lambda x:np.random.normal(mean, std)*scale if np.random.uniform()<prob else x)
        else:
            split = list(image.split())
            for i in range(len(split)):
                if np.random.uniform()<prob:
                    split[i] = split[i].point(lambda x:np.random.normal(mean, std)*scale)
            return Image.merge(image.mode, split)
    elif mode=='laplace':
        if wise=='pixel':
            return image.point(lambda x:np.random.laplace(mean, lam)*scale if np.random.uniform()<prob else x)
        else:
            split = list(image.split())
            for i in range(len(split)):
                if np.random.uniform()<prob:
                    split[i] = split[i].point(lambda x:np.random.laplace(mean, lam)*scale)
            return Image.merge(image.mode, split)
    elif mode=='poisson':
        if wise=='pixel':
            return image.point(lambda x:np.random.poisson(lam)*scale if np.random.uniform()<prob else x)
        else:
            split = list(image.split())
            for i in range(len(split)):
                if np.random.uniform()<prob:
                    split[i] = split[i].point(lambda x:np.random.poisson(lam)*scale)
            return Image.merge(image.mode, split)
    elif mode=='uniform':
        if wise=='pixel':
            return image.point(lambda x:np.random.uniform(lower, upper)*scale if np.random.uniform()<prob else x)
        else:
            split = list(image.split())
            for i in range(len(split)):
                if np.random.uniform()<prob:
                    split[i] = split[i].point(lambda x:np.random.uniform(lower, upper)*scale)
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
