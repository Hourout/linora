import numpy as np
from PIL import ImageFilter

__all__ = ['FilterMode', 'filters', 'convolution']


def convolution(image, kernel, scale=None, offset=0):
    """Create a convolution kernel.  
    
    The current version only supports 3x3 and 5x5 integer and floating point kernels.
    In the current version, kernels can only be applied to "L" and "RGB" images.

    Args:
        image: a PIL instance.
        kernel: a sequence containing kernel weights.
        scale: each pixel is divided by this value, the default is the sum of the kernel weights.
        offset: this value is added to the result, after it has been divided by the scale factor.
    Return:
        a PIL instance.
    """
    kernel = np.array(kernel).flatten().tolist()
    if len(kernel)==9:
        size = (3,3)
    elif len(kernel)==25:
        size = (5,5)
    else:
        raise ValueError('Kernel size must be (3,3) or (5,5).')
    return image.filter(ImageFilter.Kernel(size, kernel, scale, offset))


class filter_mode:
    """image blur type."""
    Box               = 'box'
    Gaussian          = 'gaussian'
    Unsharpmask       = 'unsharpmask'
    Rank              = 'rank'
    Median            = 'median'
    Min               = 'min'
    Max               = 'max'
    Mode              = 'mode'
    Mean              = 'BLUR'
    CONTOUR           = 'CONTOUR'
    DETAIL            = 'DETAIL'
    EDGE_ENHANCE      = 'EDGE_ENHANCE'
    EDGE_ENHANCE_MORE = 'EDGE_ENHANCE_MORE'
    EMBOSS            = 'EMBOSS'
    FIND_EDGES        = 'FIND_EDGES'
    SHARPEN           = 'SHARPEN'
    SMOOTH            = 'SMOOTH'
    SMOOTH_MORE       = 'SMOOTH_MORE'
    
FilterMode = filter_mode()


def filters(image, mode=FilterMode.Mean, p=1, **kwarg):
    """
    la.image.FilterMode.Box:
        Blurs the image by setting each pixel to the average value of the pixels 
        in a square box extending radius pixels in each direction. 
        Supports float radius of arbitrary size. Uses an optimized implementation 
        which runs in linear time relative to the size of the image for any radius value.
        
        you should append param `radius`
        radius: Size of the box in one direction. Radius 0 does not blur, returns an identical image. 
                Radius 1 takes 1 pixel in each direction, i.e. 9 pixels in total.
        eg.
        la.image.filters(image, mode=la.image.FilterMode.Box, radius=2)
    
    la.image.FilterMode.Gaussian:
        Gaussian blur filter.
        
        you should append param `radius`
        radius: Blur radius.
        eg.
        la.image.filters(image, mode=la.image.FilterMode.Gaussian, radius=2)
        
    la.image.FilterMode.Unsharpmask:
        Unsharp mask filter.
        See Wikipedia’s entry on digital unsharp masking for an explanation of the parameters.
        
        you should append param `radius`,`percent`,`threshold`
        radius: Blur radius.
        percent: Unsharp strength, in percent
        threshold: Threshold controls the minimum brightness change that will be sharpened
        eg.
        la.image.filters(image, mode=la.image.FilterMode.Unsharpmask, radius=2, percent=150, threshold=3)
        
    la.image.FilterMode.Rank:
        Create a rank filter. The rank filter sorts all pixels in a window of the given size, and returns the rank’th value.
        
        you should append param `size`,`rank`
        size: The kernel size, in pixels.
        rank: What pixel value to pick. Use 0 for a min filter, 
              size * size / 2 for a median filter, size * size - 1 for a max filter, etc.
        eg.
        la.image.filters(image, mode=la.image.FilterMode.Rank, size, rank)
        
    la.image.FilterMode.Median:
        Create a median filter. Picks the median pixel value in a window with the given size.
        
        you should append param `size`
        size: The kernel size, in pixels.
        eg.
        la.image.filters(image, mode=la.image.FilterMode.Median, size=3)
        
    la.image.FilterMode.Min:
        Create a min filter. Picks the lowest pixel value in a window with the given size.
        
        you should append param `size`
        size: The kernel size, in pixels.
        eg.
        la.image.filters(image, mode=la.image.FilterMode.Min, size=3)
        
    la.image.FilterMode.Max:
        Create a max filter. Picks the largest pixel value in a window with the given size.
        
        you should append param `size`
        size: The kernel size, in pixels.
        eg.
        la.image.filters(image, mode=la.image.FilterMode.Max, size=3)
        
    la.image.FilterMode.Mode:
        Create a mode filter. Picks the most frequent pixel value in a box with the given size. 
        
        Pixel values that occur only once or twice are ignored; 
        if no pixel value occurs more than twice, the original pixel value is preserved.
        
        you should append param `size`
        size: The kernel size, in pixels.
        eg.
        la.image.filters(image, mode=la.image.FilterMode.Mode, size=3)
        
    la.image.FilterMode.Mean:
        Normal blur.
        eg.
        la.image.filters(image, mode=la.image.FilterMode.Mean)
        
    la.image.FilterMode.CONTOUR:
        contour blur.
        eg.
        la.image.filters(image, mode=la.image.FilterMode.CONTOUR)
        
    la.image.FilterMode.DETAIL:
        detail blur.
        eg.
        la.image.filters(image, mode=la.image.FilterMode.DETAIL)
        
    la.image.FilterMode.EDGE_ENHANCE:
        Edge enhancement blur.
        eg.
        la.image.filters(image, mode=la.image.FilterMode.EDGE_ENHANCE)
        
    la.image.FilterMode.EDGE_ENHANCE_MORE:
        Edge enhancement more blur.
        eg.
        la.image.filters(image, mode=la.image.FilterMode.EDGE_ENHANCE_MORE)
        
    la.image.FilterMode.EMBOSS:
        emboss blur.
        eg.
        la.image.filters(image, mode=la.image.FilterMode.EMBOSS)
        
    la.image.FilterMode.FIND_EDGES:
        Find the edge blur.
        eg.
        la.image.filters(image, mode=la.image.FilterMode.FIND_EDGES)
        
    la.image.FilterMode.SHARPEN:
        Sharpen blur.
        eg.
        la.image.filters(image, mode=la.image.FilterMode.SHARPEN)
    
    la.image.FilterMode.SMOOTH:
        Smooth blur.
        eg.
        la.image.filters(image, mode=la.image.FilterMode.SMOOTH)
        
    la.image.FilterMode.SMOOTH_MORE:
        Smooth threshold blur.
        eg.
        la.image.filters(image, mode=la.image.FilterMode.SMOOTH_MORE)

    Args:
        image: a PIL instance.
        mode:la.image.FilterMode.
        p: probability that the image does this. Default value is 1.
    Returns:
        A PIL instance.
    """
    if np.random.uniform()>p:
        return image
    if 'radius' not in kwarg:
        kwarg['radius'] = 2
    if 'size' not in kwarg:
        kwarg['size'] = 3
    if 'percent' not in kwarg:
        kwarg['percent'] = 150
    if 'threshold' not in kwarg:
        kwarg['threshold'] = 3
    if 'rank' not in kwarg:
        kwarg['rank'] = 1
    if mode=='box':
        return image.filter(ImageFilter.BoxBlur(radius=kwarg['radius']))
    elif mode=='gaussian':
        return image.filter(ImageFilter.GaussianBlur(radius=kwarg['radius']))
    elif mode=='unsharpmask':
        return image.filter(ImageFilter.UnsharpMask(radius=kwarg['radius'], percent=kwarg['percent'], threshold=kwarg['threshold']))
    elif mode=='rank':
        return image.filter(ImageFilter.RankFilter(size=kwarg['size'], rank=kwarg['rank']))
    elif mode=='median':
        return image.filter(ImageFilter.MedianFilter(size=kwarg['size']))
    elif mode=='min':
        return image.filter(ImageFilter.MinFilter(size=kwarg['size']))
    elif mode=='max':
        return image.filter(ImageFilter.MaxFilter(size=kwarg['size']))
    elif mode=='mode':
        return image.filter(ImageFilter.ModeFilter(size=kwarg['size']))
    elif mode=='BLUR':
        return image.filter(ImageFilter.BLUR)
    elif mode=='CONTOUR':
        return image.filter(ImageFilter.CONTOUR)
    elif mode=='DETAIL':
        return image.filter(ImageFilter.DETAIL)
    elif mode=='EDGE_ENHANCE':
        return image.filter(ImageFilter.EDGE_ENHANCE)
    elif mode=='EDGE_ENHANCE_MORE':
        return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    elif mode=='EMBOSS':
        return image.filter(ImageFilter.EMBOSS)
    elif mode=='FIND_EDGES':
        return image.filter(ImageFilter.FIND_EDGES)
    elif mode=='SHARPEN':
        return image.filter(ImageFilter.SHARPEN)
    elif mode=='SMOOTH':
        return image.filter(ImageFilter.SMOOTH)
    elif mode=='SMOOTH_MORE':
        return image.filter(ImageFilter.SMOOTH_MORE)
    else:
        raise ValueError("mode must be la.image.FilterMode param")

