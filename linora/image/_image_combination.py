from PIL import Image

__all__ = ['blend']

def blend(image1, image2, alpha):
    """Creates a new image by interpolating between two input images.
    
    using a constant alpha.
    out = image1 * (1.0 - alpha) + image2 * alpha
    
    If alpha is 0.0, a copy of the first image is returned. 
    If alpha is 1.0, a copy of the second image is returned. 
    There are no restrictions on the alpha value. 
    If necessary, the result is clipped to fit into the allowed output range.
    
    Args:
      image1: a Image instance. The first image.
      image2: a Image instance. The second image.  Must have the same mode and size as the first image.
      alpha: The interpolation alpha factor.
    Return:
      a Image instance.
   """
    return Image.blend(image1, image2, alpha)
