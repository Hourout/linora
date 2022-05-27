from linora.image._image_draw import *

class ImageDrawAug(object):
    def __init__(self, image=None):
        self.image = image
    
    def mask(self, size, max_num, random=True, color=None, p=None):
        """Draws a mask.

        Args:
            size: list or tuple, mask size, [height, width]. if int, transform [size, size].
            max_num: int, max mask number.
                     if tuple or list, randomly picked in the interval `[max_num[0], max_num[1])`.
            random: bool, whether the mask position is random.
            color: str or tuple or la.image.RGBMode, rgb color, mask fill color.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = mask(self.image, size, max_num, random, color, p)
        return self
    
    def mask_box(self, boxes, fill='inner', color=None, p=None):
        """Fill color inside or outside a rectangular area.

        Args:
            boxes: box axis with same [[x1,y1,x2,y2],...]
            fill: 'inner' or 'outer', color fill position.
            color: str or tuple or la.image.RGBMode, rgb color, box line color.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = mask_box(self.image, boxes, fill, color, p)
        return self
    
    def stripe(self, width=4, mode=1, color=None, p=None):
        """vertically and horizontally line apply to image.

        Args:
            width: int, line spacing width.
            mode: 0 is vertically and 1 is horizontally.
            color: int or list, line color.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = stripe(self.image, width, mode, color, p)
        return self