from linora.image._image_resize import *

class ImageResizeAug(object):
    def __init__(self, image=None):
        self.image = image
        
    def resize(self, size, method=ResizeMethod.BILINEAR):
        self.image = resize(self.image, size, method)
        return self
    
    
