from linora.image._image_crop import *

class ImageCropAug(object):
    def __init__(self, image=None):
        self.image = image
        
    def crop_central(self, central_rate):
        self.image = crop_central(self.image, central_rate)
        return self
    
    def crop_point(self, height_rate, width_rate):
        self.image = crop_point(self.image, height_rate, width_rate)
        return self
    
    
    
    
