import numpy as np
from PIL import Image

__all__ = ['crop', 'crop_central', 'crop_point', 'concat', 'split', 'paste']


def crop(image, box, p=1):
    """Returns a rectangular region from this image. 
    
    The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.
    Args:
        image: a PIL instance.
        box: The crop rectangle, as a (left, upper, right, lower)-tuple.
             if [x,y], transform [x,x,y,y] 4-tuple.
        p: probability that the image does this. Default value is 1.
    returns: 
        a PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if len(box)==2:
        if isinstance(box[0], (list, tuple)):
            box = [box[0][0], box[0][1], box[1][0], box[1][1]]
        else:
            box = [box[0], box[0], box[1], box[1]]
    return image.crop(box)


def crop_central(image, central_rate, p=1):
    """Crop the central region of the image.
    
    Remove the outer parts of an image but retain the central region of the image
    along each dimension. If we specify central_fraction = 0.5, this function
    returns the region marked with "X" in the below diagram.
       --------
      |        |
      |  XXXX  |
      |  XXXX  |
      |        |   where "X" is the central 50% of the image.
       --------
    
    Args:
        image: a PIL instance.
        central_rate: if int float, should be in the interval (0, 1], fraction of size to crop.
                      if tuple list, randomly picked in the interval
                      `[central_rate[0], central_rate[1])`, value is fraction of size to crop.
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    Raises:
        ValueError: if central_crop_fraction is not within (0, 1].
    """
    if np.random.uniform()>p:
        return image
    if isinstance(central_rate, (int, float)):
        assert 0<central_rate<=1, 'if central_rate type one of int or float, must be in the interval (0, 1].'
    elif isinstance(central_rate, (tuple, list)):
        assert 0<central_rate[0]<central_rate[1]<=1, 'central_rate should be 1 >= central_rate[1] > central_rate[0] > 0.'
        central_rate = np.random.uniform(central_rate[0], central_rate[1])
    else:
        raise ValueError('central_rate should be one of int, float, tuple, list.')
    left = int(image.size[0]*(0.5-central_rate/2))
    upper = int(image.size[1]*(0.5-central_rate/2))
    right = int(image.size[0]*(0.5+central_rate/2))
    lower = int(image.size[1]*(0.5+central_rate/2))
    return image.crop((left, upper, right, lower))


def crop_point(image, height_rate, width_rate, p=1):
    """Crop the any region of the image.
    
    Crop region area = height_rate * width_rate *image_height * image_width
    
    Args:
        image: a PIL instance.
        height_rate: float, in the interval (0, 1].
        width_rate: float, in the interval (0, 1].
        p: probability that the image does this. Default value is 1.
    Returns:
        a PIL instance.
    Raises:
        ValueError: if central_crop_fraction is not within (0, 1].
    """
    if np.random.uniform()>p:
        return image
    assert 0<height_rate<=1 and 0<width_rate<=1, 'height_rate and width_rate should be in the interval (0, 1].'
    left = image.size[0]*np.random.uniform(0, 1-width_rate)
    upper = image.size[1]*np.random.uniform(0, 1-height_rate)
    right = left+image.size[0]*width_rate
    lower = upper+image.size[1]*height_rate
    return image.crop((left, upper, right, lower))


def paste(image1, image2, box=None):
    """Pastes another image into this image. 
    
    Args:
        image1: a PIL instance. The first image.
        image2: a PIL instance. The second image.
        box: An optional 4-tuple giving the region to paste into. 
             If a 2-tuple is used instead, it’s treated as the upper left corner. 
             If omitted or None, the source is pasted into the upper left corner.
    Return:
        a PIL instance.
    """
    image = image1.copy()
    if box is None:
        box = [0, 0, min(image2.size[0], image.size[0]), min(image2.size[1], image.size[1])]
    elif len(box)==4:
        box = [box[0], box[1], min(box[2], image.size[0]), min(box[3], image.size[1])]
    elif len(box)==2:
        box = [box[0], box[1], min(box[0]+image2.size[0], image.size[0]), min(box[1]+image2.size[1], image.size[1])]
    else:
        raise ValueError('`box` value error.')
    if (box[2]-box[0])!=image2.size[0] or (box[3]-box[1])!=image2.size[1]:
        image.paste(image2.resize([box[2]-box[0], box[3]-box[1]], resample=Image.Resampling.NEAREST), box)
    else:
        image.paste(image2, box)
    return image


def concat(image, col_nums=None):
    """Merge multiple pictures into one picture.
    
    Splice multiple pictures from left to right and from top to bottom to one picture.
    
    Args:
        image: list or tuple or dict, a list or tuple of PIL instance.
               if dict, should be la.image.split function output.
        col_nums: int, maximum number of columns spliced.
    returns:
        a PIL instance.
    """
    if isinstance(image, dict):
        if image['image_strides'][0]>image['image_size'][0] or image['image_strides'][1]>image['image_size'][1]:
            image = image['image_list']
        else:
            img = Image.new(image['image_list'][0].mode, image['image_shape'], (0,0,0))
            for r, i in enumerate(image['image_true']):
                img.paste(image['image_list'][r].crop(i[1]), i[0])
            return img
    if isinstance(image, (list, tuple)):
        image = list(image)
        if len(image)==1:
            return image[0]
        size = image[0].size
        for r, i in enumerate(image):
            if size!=i.size:
                image[r] = i.resize(size)
        if col_nums is None or col_nums<1:
            col_nums = np.ceil(np.sqrt(len(image)))
        shape = [int(np.ceil(len(image)/col_nums)), int(col_nums)]
        img = Image.new(image[0].mode, [size[0]*shape[1], size[1]*shape[0]], (0,0,0))
        for r, i in enumerate(image):
            img.paste(i, (r%shape[1]*size[0], r//shape[1]*size[1]))
        return img
    raise ValueError('`image` should be list or tuple or dict.')
    

def split(image, size, strides=None, keep_last=True):
    """Split one picture into multiple pictures.
    
    Args:
        image: a PIL instance.
        size: list or tuple, split picture size, [width, height].
        strides: int or tuple or list, eg. [width, height], specifying the crop strides along the height and width. 
        keep_last: bool, whether to preserve the clipped image when the right and bottom edges are not long enough.
    returns:
        a dict.
    """
    if strides is None:
        strides = size
    elif isinstance(strides, int):
        strides = [strides, strides]
    shape = image.size
    assert shape[0]>size[0] and shape[1]>size[1] and shape[0]>strides[0] and shape[1]>strides[1], '`size` or `strides` value error.'
    xaxis = []
    yaxis = []
    xtrue = []
    ytrue = []
    i = 0
    while 1:
        if i+size[0]==shape[0]:
            xaxis.append(i)
            xtrue.append([[i, i+size[0]], [0, size[0]]])
            break
        elif i+size[0]<shape[0]:
            xaxis.append(i)
            xtrue.append([[i, i+strides[0]], [0, strides[0]]])
            if i+size[0]+strides[0]>shape[0]:
                if keep_last:
                    xaxis.append(shape[0]-size[0])
                    xtrue.append([[i+strides[0], shape[0]], [size[0]-(shape[0]-i-strides[0]), size[0]]])
                break
        i += strides[0]
    i = 0
    while 1:
        if i+size[1]==shape[1]:
            yaxis.append(i)
            ytrue.append([[i, i+size[1]], [0, size[1]]])
            break
        elif i+size[1]<shape[1]:
            yaxis.append(i)
            ytrue.append([[i, i+strides[1]], [0, strides[1]]])
            if i+size[1]+strides[1]>shape[1]:
                if keep_last:
                    yaxis.append(shape[1]-size[1])
                    ytrue.append([[i+strides[1], shape[1]], [size[1]-(shape[1]-i-strides[1]), size[1]]])
                break
        i += strides[1]
    if not keep_last:
        shape = [xaxis[-1]+strides[0], yaxis[-1]+strides[1]]
    return {'image_list': [image.crop([j, i, j+size[0], i+size[1]]) for i in yaxis for j in xaxis],
             'image_size': size, 'image_strides': strides, 'image_block':[len(xaxis), len(yaxis)], 'image_shape':shape,
             'image_true': [[[j[0][0], i[0][0], j[0][1], i[0][1]], [j[1][0], i[1][0], j[1][1], i[1][1]]] for i in ytrue for j in xtrue]}