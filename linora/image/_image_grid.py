import numpy as np
from PIL import Image, ImageFont, ImageDraw

__all__ = ['grid']


def grid(image, mode, size=None, **kwargs):
    """grid image.

    Args:
        image: a list or tuple of PIL instance.
        mode: grid type.
        size: frist image size.
        rate: if mode in [0,1,2,3], the ratio of the second image, default is 0.25.
        xnums: if mode in [4,5,6,7,8], how many columns to display.
        xspace: if mode in [4,5,6,7,8], width between images in each column.
        yspace: if mode in [4,5,6,7,8], width between images in each row.
        text: if mode in [4,5,6,7,8], each image text, len(text)==len(image)
    Returns:
        a PIL instance.
    """
    if isinstance(image, tuple):
        image = list(image)
    elif not isinstance(image, list):
        image = [image]
    size = image[0].size if size is None else tuple(size)
    for r, i in enumerate(image):
        if size!=i.size:
            image[r] = i.resize(size)
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
        
        shape = [int(xnums), int(np.ceil(len(image)/xnums))]
        img = Image.new(image[0].mode, [shape[0]*(size[0]+xspace)-xspace, shape[1]*(size[1]+yspace)-yspace], (255,255,255))
        for r, i in enumerate(image):
            img.paste(i, (r%shape[0]*(size[0]+xspace), r//shape[0]*(size[1]+yspace)))
        return img
    if mode==5:
        xnums = kwargs['xnums'] if 'xnums' in kwargs else np.ceil(np.sqrt(len(image)))
        text = kwargs['text']
        xspace = kwargs['xspace'] if 'xspace' in kwargs else 5
        yspace = kwargs['yspace'] if 'yspace' in kwargs else 60
        
        shape = [int(xnums), int(np.ceil(len(image)/xnums))]
        img = Image.new(image[0].mode, [shape[0]*(size[0]+xspace)-xspace, shape[1]*(size[1]+yspace)], (255,255,255))
        draw = ImageDraw.Draw(img)
        
        for r, i in enumerate(image):
            img.paste(i, (r%shape[0]*(size[0]+xspace), r//shape[0]*(size[1]+yspace)))
            textsize = draw.textsize(text[r])
            t = min(yspace/2/textsize[1], size[0]/min(textsize[0], textsize[1]*len(text[r])))
            xoffset = int((size[0]-t*min(textsize[0], textsize[1]*len(text[r])))/2)
            font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf",size=int(t*min(textsize)))
            draw.text((r%shape[0]*(size[0]+xspace)+xoffset, r//shape[0]*(size[1]+yspace)+size[1]+int(yspace/4)), 
                      text[r], fill=0, font=font)
        return img
    if mode==6:
        xnums = kwargs['xnums'] if 'xnums' in kwargs else np.ceil(np.sqrt(len(image)))
        text = kwargs['text']
        xspace = kwargs['xspace'] if 'xspace' in kwargs else 5
        yspace = kwargs['yspace'] if 'yspace' in kwargs else 60
        
        shape = [int(xnums), int(np.ceil(len(image)/xnums))]
        img = Image.new(image[0].mode, [shape[0]*(size[0]+xspace)-xspace, shape[1]*(size[1]+yspace)], (255,255,255))
        draw = ImageDraw.Draw(img)
        
        for r, i in enumerate(image):
            img.paste(i, (r%shape[0]*(size[0]+xspace), r//shape[0]*(size[1]+yspace)+yspace))
            textsize = draw.textsize(text[r])
            t = min(yspace/2/textsize[1], size[0]/min(textsize[0], textsize[1]*len(text[r])))
            xoffset = int((size[0]-t*min(textsize[0], textsize[1]*len(text[r])))/2)
            font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf",size=int(t*min(textsize)))
            draw.text((r%shape[0]*(size[0]+xspace)+xoffset, r//shape[0]*(size[1]+yspace)+int(yspace/4)), 
                      text[r], fill=0, font=font)
        return img
    if mode==7:
        xnums = kwargs['xnums'] if 'xnums' in kwargs else np.ceil(np.sqrt(len(image)))
        text = kwargs['text']
        xspace = kwargs['xspace'] if 'xspace' in kwargs else 60
        yspace = kwargs['yspace'] if 'yspace' in kwargs else 5
        
        shape = [int(xnums), int(np.ceil(len(image)/xnums))]
        img = Image.new(image[0].mode, [shape[0]*(size[0]+xspace), shape[1]*(size[1]+yspace)-yspace], (255,255,255))
        draw = ImageDraw.Draw(img)
        
        for r, i in enumerate(image):
            img.paste(i, (r%shape[0]*(size[0]+xspace), r//shape[0]*(size[1]+yspace)))
            textsize = draw.textsize(text[r])
            t = min(xspace/2/textsize[1], size[1]/textsize[1]/len(text[r]))
            yoffset = int((size[1]-t*textsize[1]*len(text[r]))/2)
            font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf",size=int(t*min(textsize)))
            draw.text((r%shape[0]*(size[0]+xspace)+size[0]+int(xspace/4), r//shape[0]*(size[1]+yspace)+yoffset), 
                      text[r], fill=0, font=font, direction='ttb')
        return img
    if mode==8:
        xnums = kwargs['xnums'] if 'xnums' in kwargs else np.ceil(np.sqrt(len(image)))
        text = kwargs['text']
        xspace = kwargs['xspace'] if 'xspace' in kwargs else 60
        yspace = kwargs['yspace'] if 'yspace' in kwargs else 5
        
        shape = [int(xnums), int(np.ceil(len(image)/xnums))]
        img = Image.new(image[0].mode, [size[0]*shape[0]+(shape[0]-1)*xspace, size[1]*shape[1]+shape[1]*yspace], (255,255,255))
        draw = ImageDraw.Draw(img)
        
        for r, i in enumerate(image):
            img.paste(i, (r%shape[0]*(size[0]+xspace)+xspace, r//shape[0]*(size[1]+yspace)))
            textsize = draw.textsize(text[r])
            t = min(xspace/2/textsize[1], size[1]/textsize[1]/len(text[r]))
            yoffset = int((size[1]-t*textsize[1]*len(text[r]))/2)
            font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf",size=int(t*min(textsize)))
            draw.text((r%shape[0]*(size[0]+xspace)+int(xspace/4), r//shape[0]*(size[1]+yspace)+yoffset), 
                      text[r], fill=0, font=font, direction='ttb')
        return img
    raise ValueError('`mode` value error.')