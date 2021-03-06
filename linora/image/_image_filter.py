from PIL import ImageFilter

__all__ = ['filter_BoxBlur', 'filter_GaussianBlur', 'filter_UnsharpMask', 'filter_Rank', 'filter_Median', 'filter_Min',
           'filter_Max', 'filter_Mode', 'filter_BLUR', 'filter_CONTOUR', 'filter_DETAIL', 'filter_EDGE_ENHANCE', 
           'filter_EDGE_ENHANCE_MORE', 'filter_EMBOSS', 'filter_FIND_EDGES', 'filter_SHARPEN', 'filter_SMOOTH', 
           'filter_SMOOTH_MORE']

def filter_BoxBlur(image, radius=2):
    """
    Blurs the image by setting each pixel to the average value of the pixels 
    in a square box extending radius pixels in each direction. 
    Supports float radius of arbitrary size. Uses an optimized implementation 
    which runs in linear time relative to the size of the image for any radius value.

    Args:
    radius: Size of the box in one direction. Radius 0 does not blur, returns an identical image. 
            Radius 1 takes 1 pixel in each direction, i.e. 9 pixels in total.
            
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.BoxBlur(radius))

def filter_GaussianBlur(image, radius=2):
    """Gaussian blur filter.

    Args:
    radius: Blur radius.
    
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.GaussianBlur(radius))

def filter_UnsharpMask(image, radius=2, percent=150, threshold=3):
    """Unsharp mask filter.
    See Wikipedia’s entry on digital unsharp masking for an explanation of the parameters.

    Args:
    radius: Blur Radius
    percent: Unsharp strength, in percent
    threshold: Threshold controls the minimum brightness change that will be sharpened
    
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

def filter_Rank(image, size, rank):
    """Create a rank filter. 
    The rank filter sorts all pixels in a window of the given size, and returns the rank’th value.
    
    Args:
    size: The kernel size, in pixels.
    rank: What pixel value to pick. Use 0 for a min filter, 
          size * size / 2 for a median filter, size * size - 1 for a max filter, etc.
          
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.RankFilter(size, rank))

def filter_Median(image, size=3):
    """Create a median filter. Picks the median pixel value in a window with the given size.
    
    Args:
    size: The kernel size, in pixels.
          
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.MedianFilter(size))

def filter_Min(image, size=3):
    """Create a min filter. Picks the lowest pixel value in a window with the given size.
    
    Args:
    size: The kernel size, in pixels.
          
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.MinFilter(size))

def filter_Max(image, size=3):
    """Create a max filter. Picks the largest pixel value in a window with the given size.
    
    Args:
    size: The kernel size, in pixels.
          
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.MaxFilter(size))

def filter_Mode(image, size=3):
    """Create a mode filter. Picks the most frequent pixel value in a box with the given size. 
    
    Pixel values that occur only once or twice are ignored; 
    if no pixel value occurs more than twice, 
    the original pixel value is preserved.
    
    Args:
    size: The kernel size, in pixels.
          
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.ModeFilter(size))

def filter_BLUR(image):
    """Normal blur.
          
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.BLUR)

def filter_CONTOUR(image):
    """contour blur.
          
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.CONTOUR)

def filter_DETAIL(image):
    """detall blur.
          
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.DETAIL)

def filter_EDGE_ENHANCE(image):
    """Edge enhancement blur.
          
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.EDGE_ENHANCE)

def filter_EDGE_ENHANCE_MORE(image):
    """Edge enhancement threshold blur.
          
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.EDGE_ENHANCE_MORE)

def filter_EMBOSS(image):
    """emboss blur.
          
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.EMBOSS)

def filter_FIND_EDGES(image):
    """Find the edge blur.
          
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.FIND_EDGES)

def filter_SHARPEN(image):
    """Sharpen blur.
          
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.SHARPEN)

def filter_SMOOTH(image):
    """Smooth blur.
          
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.SMOOTH)

def filter_SMOOTH_MORE(image):
    """Smooth threshold blur.
          
    Returns:
            A Image instance. of the same type and shape as `image`. 
    """
    return image.filter(ImageFilter.SMOOTH_MORE)

