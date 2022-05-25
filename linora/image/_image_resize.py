from PIL import Image

__all__ = ['ResizeMode', 'resize']


class resize_method:
    """For geometry operations that may map multiple input pixels to a single output pixel, 
    the Python Imaging Library provides different resampling filters.
    
    NEAREST: Pick one nearest pixel from the input image. Ignore all other input pixels.
    BOX: Each pixel of source image contributes to one pixel of the destination 
         image with identical weights. 
    BILINEAR: For resize calculate the output pixel value using linear interpolation on all 
              pixels that may contribute to the output value. 
              For other transformations linear interpolation over a 2x2 environment 
              in the input image is used.
    HAMMING: Produces a sharper image than BILINEAR, 
             doesn’t have dislocations on local level like with BOX.
    BICUBIC: For resize calculate the output pixel value using cubic interpolation on all 
             pixels that may contribute to the output value. For other transformations cubic 
             interpolation over a 4x4 environment in the input image is used.
    LANCZOS: Calculate the output pixel value using a high-quality Lanczos filter (a truncated sinc) 
             on all pixels that may contribute to the output value.
    """
    NEAREST = Image.Resampling.NEAREST
    BOX = Image.Resampling.BOX
    BILINEAR = Image.Resampling.BILINEAR
    HAMMING = Image.Resampling.HAMMING
    BICUBIC = Image.Resampling.BICUBIC
    LANCZOS = Image.Resampling.LANCZOS
    
ResizeMode = resize_method()


def resize(image, size, method=ResizeMode.BILINEAR):
    """Returns a resized copy of this image.
    
    Args:
        image: a PIL instance.
        size: The requested size in pixels, as a 2-tuple: (width, height).
              if float, (0,1], size=[int(width*size), int(height*size)]
              if [0.2, 0.7], size=[int(width*0.2), int(height*0.7)]
        method: An optional resampling filter. see la.image.ResizeMode.
    Returns:
        a PIL instance.
    """
    if image.mode in ['1', 'P'] and method!=ResizeMode.NEAREST:
        method = ResizeMode.NEAREST
    if isinstance(size, float):
        size = (int(image.size[0]*size), int(image.size[1]*size))
    elif isinstance(size, (list, tuple)):
        if isinstance(size[0], float):
            size = (int(image.size[0]*size[0]), size[1])
        if isinstance(size[1], float):
            size = (size[0], int(image.size[1]*size[1]))
    return image.resize(size, resample=method)
