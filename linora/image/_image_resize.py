from PIL import Image

__all__ = ['ResizeMethod', 'resize']

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
    NEAREST = Image.NEAREST
    BOX = Image.BOX
    BILINEAR = Image.BILINEAR
    HAMMING = Image.HAMMING
    BICUBIC = Image.BICUBIC
    LANCZOS = Image.LANCZOS
    
ResizeMethod = resize_method()

def resize(image, size, method=ResizeMethod.BILINEAR):
    """Returns a resized copy of this image.
    
    Args:
    image: a Image instance.
    size: The requested size in pixels, as a 2-tuple: (width, height).
    method: An optional resampling filter. see la.image.ResizeMethod.
    
    Returns:
        a Image instance.
    """
    if image.mode in ['1', 'P'] and method!=ResizeMethod.NEAREST:
        method = ResizeMethod.NEAREST
    return image.resize(size, resample=method)
