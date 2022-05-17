import random

import numpy as np
from PIL import Image, ImageStat

from linora.image._image_draw import draw_point

__all__ = ['mosaic', 'noise_saltpepper', 'noise_gaussian', 'noise_laplace', 'noise_poisson', 
           'noise_uniform', 'noise_speckle', 'noise_impulse'
          ]


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


def noise_saltpepper(image, white_prob=0.025, black_prob=0.025, p=1):
    """Mask salt pepper noise apply to image.
    
    The salt-pepper noise is based on the signal-to-noise ratio of the image,
    randomly generating the pixel positions in some images all channel,
    and randomly assigning these pixels to 0 or 255.
    
    Args:
        image: a PIL instance.
        white_prob: white pixel prob.
        black_prob: black pixel prob.
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    img = image.copy()
    axis = [(i,j) for i in range(image.width) for j in range(image.height)]
    np.random.shuffle(axis)
    if isinstance(white_prob, (tuple, list)):
        white_prob = np.random.uniform(white_prob[0], white_prob[1])
    if isinstance(black_prob, (tuple, list)):
        black_prob = np.random.uniform(black_prob[0], black_prob[1])
    if white_prob>0:
        axis_white = axis[0:int(len(axis)*white_prob)]
        pixel = (255,)*len(image.getbands())
        for i in axis_white:
            img.putpixel(i, pixel)
    if black_prob>0:
        axis_black = axis[-int(len(axis)*(black_prob)):]
        pixel = (0,)*len(image.getbands())
        for i in axis_black:
            img.putpixel(i, pixel)
    return img


def _noise(image, function, wise='pixel', scale=1, prob=0.2, p=1, **kwargs):
    if np.random.uniform()>p:
        return image
    if wise=='pixel':
        return image.point(lambda x:function(x, **kwargs)*scale if np.random.uniform()<prob else x)
    elif wise=='channel':
        split = list(image.split())
        for i in range(len(split)):
            if np.random.uniform()<prob:
                split[i] = split[i].point(lambda x:function(x, **kwargs)*scale)
        return Image.merge(image.mode, split)
    elif wise=='pixelchannel':
        img = image.point(lambda x:function(x, **kwargs)*scale)
        axis = [(i,j) for i in range(image.width) for j in range(image.height)]
        np.random.shuffle(axis)
        if prob>0.5:
            axis = axis[:int(len(axis)*(1-prob))]
            for i in axis:
                img.putpixel(i, image.getpixel(i))
            return img
        else:
            axis = axis[:int(len(axis)*prob)]
            for i in axis:
                image.putpixel(i, img.getpixel(i))
            return image
    elif isinstance(wise, (list, tuple)):
        split = list(image.split())
        for i in wise:
            if np.random.uniform()<prob:
                split[i] = split[i].point(lambda x:function(x, **kwargs)*scale)
        return Image.merge(image.mode, split)
    else:
        raise ValueError("`wise` should be 'pixel' or 'channel' or 'pixelchannel' or list of channel.")


def noise_gaussian(image, mean=0, std=30, wise='pixel', scale=1, prob=0.2, p=1):
    """gaussian distribution noise apply to image.
    
    Args:
        image: A PIL instance.
        mean: if int or float, value is gaussian distribution mean.
              if tuple or list, randomly picked in the interval `[mean[0], mean[1])`.
        std: if int or float, value is gaussian distribution std.
             if tuple or list, randomly picked in the interval `[std[0], std[1])`.
        wise: 'pixel' or 'channel' or 'pixelchannel' or list of channel, method of applying noise.
        scale: if int or float, value multiply with noise.
               if tuple or list, randomly picked in the interval `[scale[0], scale[1])`.
        prob: probability of every pixel or channel being changed.
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance.
    """
    def func(x, mean, std):
        return np.random.normal(mean, std)
    if isinstance(mean, (tuple, list)):
        mean = np.random.uniform(mean[0], mean[1])
    if isinstance(std, (tuple, list)):
        std = np.random.uniform(std[0], std[1])
    if isinstance(scale, (tuple, list)):
        scale = np.random.uniform(scale[0], scale[1])
    return _noise(image, func, wise=wise, scale=scale, prob=prob, p=p, mean=mean, std=std)


def noise_laplace(image, mean=0, lam=30, wise='pixel', scale=1, prob=0.2, p=1):
    """laplace distribution noise apply to image.
    
    Args:
        image: A PIL instance.
        mean: if int or float, value is laplace distribution mean.
              if tuple or list, randomly picked in the interval `[mean[0], mean[1])`.
        lam: if int or float, value is laplace distribution lam.
             if tuple or list, randomly picked in the interval `[lam[0], lam[1])`.
        wise: 'pixel' or 'channel' or 'pixelchannel' or list of channel, method of applying noise.
        scale: if int or float, value multiply with noise.
               if tuple or list, randomly picked in the interval `[scale[0], scale[1])`.
        prob: probability of every pixel or channel being changed.
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance.
    """
    def func(x, mean, lam):
        return np.random.laplace(mean, lam)
    if isinstance(mean, (tuple, list)):
        mean = np.random.uniform(mean[0], mean[1])
    if isinstance(lam, (tuple, list)):
        lam = np.random.uniform(lam[0], lam[1])
    if isinstance(scale, (tuple, list)):
        scale = np.random.uniform(scale[0], scale[1])
    return _noise(image, func, wise=wise, scale=scale, prob=prob, p=p, mean=mean, lam=lam)


def noise_poisson(image, lam=30, wise='pixel', scale=1, prob=0.2, p=1):
    """poisson distribution noise apply to image.
    
    Args:
        image: A PIL instance.
        lam: if int or float, value is poisson distribution lam.
             if tuple or list, randomly picked in the interval `[lam[0], lam[1])`.
        wise: 'pixel' or 'channel' or 'pixelchannel' or list of channel, method of applying noise.
        scale: if int or float, value multiply with noise.
               if tuple or list, randomly picked in the interval `[scale[0], scale[1])`.
        prob: probability of every pixel or channel being changed.
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance.
    """
    def func(x, lam):
        return np.random.poisson(lam)
    if isinstance(lam, (tuple, list)):
        lam = np.random.uniform(lam[0], lam[1])
    if isinstance(scale, (tuple, list)):
        scale = np.random.uniform(scale[0], scale[1])
    return _noise(image, func, wise=wise, scale=scale, prob=prob, p=p, lam=lam)


def noise_uniform(image, lower=-50, upper=50, wise='pixel', scale=1, prob=0.2, p=1):
    """uniform distribution noise apply to image.
    
    Args:
        image: A PIL instance.
        lower: if int or float, value is uniform distribution lower.
               if tuple or list, randomly picked in the interval `[lower[0], lower[1])`.
        upper: if int or float, value is uniform distribution upper.
               if tuple or list, randomly picked in the interval `[upper[0], upper[1])`.
        wise: 'pixel' or 'channel' or 'pixelchannel' or list of channel, method of applying noise.
        scale: if int or float, value multiply with noise.
               if tuple or list, randomly picked in the interval `[scale[0], scale[1])`.
        prob: probability of every pixel or channel being changed.
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance.
    """
    def func(x, lower, upper):
        return np.random.uniform(lower, upper)
    if isinstance(lower, (tuple, list)):
        lower = np.random.uniform(lower[0], lower[1])
    if isinstance(upper, (tuple, list)):
        upper = np.random.uniform(upper[0], upper[1])
    if isinstance(scale, (tuple, list)):
        scale = np.random.uniform(scale[0], scale[1])
    return _noise(image, func, wise=wise, scale=scale, prob=prob, p=p, lower=lower, upper=upper)


def noise_speckle(image, wise='pixel', prob=0.2, p=1):
    """speckle noise apply to image.
    
    pixel = pixel+pixel*gaussian
    
    Args:
        image: A PIL instance.
        prob: probability of every pixel or channel being changed.
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance.
    """
    def func(x):
        return x*(1+np.random.normal())
    return _noise(image, func, wise=wise, scale=1, prob=prob, p=p)


def noise_impulse(image, binomial=0.5, wise='pixel', prob=0.2, p=1):
    """impulse noise apply to image.
        
    Args:
        image: A PIL instance.
        binomial: if int or float, value is binomial distribution prob.
                  if tuple or list, randomly picked in the interval `[binomial[0], binomial[1])`.
        wise: 'pixel' or 'channel' or 'pixelchannel' or list of channel, method of applying noise.
        prob: probability of every pixel or channel being changed.
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance.
    """
    def func(x, binomial):
        mask = np.random.binomial(n=1, p=binomial)
        return x*(1 - mask) + np.random.randint(256) * mask
    if isinstance(binomial, (tuple, list)):
        binomial = np.random.uniform(binomial[0], binomial[1])
    return _noise(image, func, wise=wise, scale=1, prob=prob, p=p, binomial=binomial)




