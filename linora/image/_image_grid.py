import numpy as np
from PIL import Image

__all__ = ['grid']


def grid(image, mode, **kwargs):
    if isinstance(image, tuple):
        image = list(image)
    elif not isinstance(image, list):
        image = [image]
    if mode==0:
        rate = kwargs['rate'] if 'rate' in kwargs else 0.25
        img = image[0].copy()
        size = (int(img.width*rate), int(img.height*rate))
        img.paste(image[-1].resize(size), (img.width-size[0], img.height-size[1]))
        return img
    if mode==1:
        rate = kwargs['rate'] if 'rate' in kwargs else 0.25
        img = image[0].copy()
        size = (int(img.width*rate), int(img.height*rate))
        img.paste(image[-1].resize(size), (img.width-size[0], 0))
        return img
    if mode==2:
        rate = kwargs['rate'] if 'rate' in kwargs else 0.25
        img = image[0].copy()
        size = (int(img.width*rate), int(img.height*rate))
        img.paste(image[-1].resize(size), (0, 0))
        return img
    if mode==3:
        rate = kwargs['rate'] if 'rate' in kwargs else 0.25
        img = image[0].copy()
        size = (int(img.width*rate), int(img.height*rate))
        img.paste(image[-1].resize(size), (0, img.height-size[1]))
        return img
    if mode==4:
        size = image[0].size
        for r, i in enumerate(image):
            if size!=i.size:
                image[r] = i.resize(size)
        xnums = kwargs['xnums'] if 'xnums' in kwargs else np.ceil(np.sqrt(len(image)))
        xspace = kwargs['xspace'] if 'xspace' in kwargs else 5
        yspace = kwargs['yspace'] if 'yspace' in kwargs else 5
        
        shape = [int(np.ceil(len(image)/xnums)), int(xnums)]
        img = Image.new(image[0].mode, [size[0]*shape[1]+(shape[1]-1)*xspace, size[1]*shape[0]+(shape[0]-1)*yspace], (255,255,255))
        for r, i in enumerate(image):
            img.paste(i, (r%shape[1]*(size[0]+xspace), r//shape[1]*(size[1]+yspace)))
        return img