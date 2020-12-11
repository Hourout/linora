from linora.image._image_position import *

class ImagePositionAug(object):
    def __init__(self, image=None):
        self.image = image
    
    def filp_up_left(self, random=False):
        self.image = filp_up_left(self.image, random)
        return self
    
    def flip_up_right(self, random=False):
        self.image = flip_up_right(self.image, random)
        return self
    
    def flip_left_right(self, random=False):
        self.image = flip_left_right(self.image, random)
        return self
    
    def flip_up_down(self, random=False):
        self.image = flip_up_down(self.image, random)
        return self
    
    def rotate(self, angle, expand=True, center=None, translate=None, fillcolor=None):
        self.image = rotate(self.image, angle, expand, center, translate, fillcolor)
        return self
    
    def translate(self, translate=None, fillcolor=None):
        self.image = translate(self.image, translate, fillcolor)
        return self
    
