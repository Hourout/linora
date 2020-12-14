from linora.image._image_color import *

class ImageColorAug(object):
    def __init__(self, image=None):
        self.image = image
    
    def contrast(self, factor):
        self.image = contrast(self.image, factor)
        return self
    
    def brightness(self, factor):
        self.image = brightness(self.image, factor)
        return self

    def sharpness(self, factor):
        self.image = sharpness(self.image, factor)
        return self
    
