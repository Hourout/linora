from linora.image._image_resize import *

class ImageResizeAug(object):
    def __init__(self, image=None):
        self.image = image
        
    def resize(self, size, method=ResizeMode.BILINEAR):
        """Returns a resized copy of this image.
    
        Args:
            size: The requested size in pixels, as a 2-tuple: (width, height).
            method: An optional resampling filter. see la.image.ResizeMode.
        """
        self.image = resize(self.image, size, method)
        return self
    
    
