from linora.image._image_position import *

class ImagePositionAug(object):
    def __init__(self, image=None):
        self.image = image
    
    def flip_up_left(self, p=None):
        """Randomly flip an image (up to left).
    
        Args:
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = flip_up_left(self.image, p)
        return self
    
    def flip_up_right(self, p=None):
        """Randomly flip an image (up to right).
    
        Args:
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = flip_up_right(self.image, p)
        return self
    
    def flip_left_right(self, p=None):
        """Randomly flip an image (left to right).
    
        Args:
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = flip_left_right(self.image, p)
        return self
    
    def flip_up_down(self, p=None):
        """Randomly flip an image (up to down).
    
        Args:
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = flip_up_down(self.image, p)
        return self
    
    def rotate(self, angle, expand=False, center=(0.5,0.5), translate=(0,0), fillcolor=None, p=None):
        """Returns a rotated copy of this image. 

        This method returns a copy of this image, rotated the given number of degrees counter clockwise around its centre.

        Args:
            angle: In degrees counter clockwise, angle in [-180, 180].
                   if int or float, rotation angle.
                   if list or tuple, randomly picked in the interval `[angle[0], angle[1])` value.
            expand: if true, expands the output image to make it large enough to hold the entire rotated image. 
                    if false, make the output image the same size as the input image. 
                    Note that the expand flag assumes rotation around the center and no translation.
                    if value is None, then the function is random.
            center: center of rotation, xaxis and yaxis in [0,1], default is the center of the image.
                    if int or float, xaxis=yaxis,
                    if 2-tuple, (xaxis, yaxis), 
                    if 4-tuple, xaxis in (center[0], center[1]) and yaxis in (center[2], center[3]).
            translate: post-rotate translation, xoffset and yoffset in [-1,1], see la.image.translate method.
                       if int or float, xoffset=yoffset,
                       if 2-tuple, (xoffset, yoffset), 
                       if 4-tuple, xoffset in (translate[0], translate[1]) and  yoffset in (translate[2], translate[3]).
            fill_color: color for area outside, int or str or tuple or la.image.RGBMode, rgb color.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = rotate(self.image, angle, expand, center, translate, fillcolor, p)
        return self
    
    def translate(self, xoffset=(-0.5,0.5), yoffset=None, fill_color=None, p=None):
        """Returns a translate copy of this image. 

        Args:
            xoffset: [-1, 1], int or float, width offset.
                     if list or tuple, randomly picked in the interval `[xoffset[0], xoffset[1])`.
            yoffset: [-1, 1], int or float, height offset.
                     if list or tuple, randomly picked in the interval `[yoffset[0], yoffset[1])`.
            fill_color: int or str or tuple or la.image.RGBMode, rgb color.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = translate(self.image, xoffset, yoffset, fillcolor, p)
        return self
    
    def offset(self, xoffset, yoffset=None, p=None):
        """Returns a copy of the image where data has been offset by the given distances.

        Data wraps around the edges. If ``yoffset`` is omitted, it is assumed to be equal to ``xoffset``.

        Args:
            xoffset: int or list ot tuple, The horizontal distance.
                     if tuple or list, randomly picked in the interval `[xoffset[0], xoffset[1])`
            yoffset: int or list ot tuple, The vertical distance. 
                     If omitted, both distances are set to the same value.
                     if tuple or list, randomly picked in the interval `[yoffset[0], yoffset[1])`
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = offset(self.image, xoffset, yoffset, p)
        return self
    
    def pad(self, pad_value, pad_color=None, p=None):
        """Add border to the image

        Args:
            pad_value: int or list or tuple, if int, pad same value with border, 
                       if list or tuple, len(pad_value)==2, left, top = right, bottom = pad_value
                       if list or tuple, len(pad_value)==4, left, top, right, bottom = pad_value
            pad_color: str or tuple or list or la.image.RGBMode, fill RGB color value, 
                       if str, hexadecimal color;
                       if len(pad_color) == 2, left_color, top_color = right_color, bottom_color = pad_color
                       if len(pad_color) == 3, left_color = top_color = right_color = bottom_color = pad_color
                       if len(pad_color) == 4, left_color, top_color, right_color, bottom_color = pad_color
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = pad(self.image, pad_value, pad_color, p)
        return self
    
    def channel_shuffle(self, p=None):
        """Random shuffle image channel.

        Args:
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = channel_shuffle(self.image, p)
        return self
    
    def perspective(self, distortion_scale, fill_color=None, p=None):
        """Performs a random perspective transformation of the given image with a given probability. 

        Args:
            distortion_scale: float, argument to control the degree of distortion and ranges from 0 to 1.
            fill_color: int or str or tuple or la.image.RGBMode, rgb color.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = perspective(self.image, distortion_scale, fill_color, p)
        return self
    
    def affine(self, angle=(-180, 180), center=(0.5,0.5), translate=(0, 0), scale=1., shear=(0,0), fill_color=None, p=None):
        """Apply affine transformation on the image keeping image center invariant.

        Args:
            angle: int or float, rotation angle in degrees between -180 and 180. Set to 0 to deactivate rotations.
                   if list or tuple, randomly picked in the interval `[angle[0], angle[1])`.
            center: center of rotation, xaxis and yaxis in [0,1], default is the center of the image.
                    if int or float, xaxis=yaxis,
                    if 2-tuple, (xaxis, yaxis), 
                    if 4-tuple, xaxis in (center[0], center[1]) and yaxis in (center[2], center[3]).
            translate: post-rotate translation, xoffset and yoffset in [-1,1], see la.image.translate method.
                       if int or float, xoffset=yoffset,
                       if 2-tuple, (xoffset, yoffset), 
                       if 4-tuple, xoffset in (translate[0], translate[1]) and  yoffset in (translate[2], translate[3]).
            scale: float, scaling factor interval, should be positive.
                   if list or tuple, randomly picked in the interval `[scale[0], scale[1])`.
            shear: Range of degrees to select from, xoffset and yoffset in [-360,360]. 
                   if int or float, xoffset=yoffset,
                   if 2-tuple, (xoffset, yoffset), 
                   if 4-tuple, xoffset in (shear[0], shear[1]) and  yoffset in (shear[2], shear[3]).
            fill_color: int or str or tuple or la.image.RGBMode, rgb color. 
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = affine(self.image, angle, center, translate, scale, shear, fill_color, p)
        return self
    
    def shear(self, xoffset=(-90, 90), yoffset=None, fill_color=None, p=None):
        """Apply affine shear on the image.

        Args:
            xoffset: int or list ot tuple, The horizontal degrees, xoffset in [-360,360].
                     if tuple or list, randomly picked in the interval `[xoffset[0], xoffset[1])`
            yoffset: int or list ot tuple, The vertical degrees, yoffset in [-360,360]. 
                     If omitted, both distances are set to the same value.
                     if tuple or list, randomly picked in the interval `[yoffset[0], yoffset[1])`
            fill_color: int or str or tuple or la.image.RGBMode, rgb color. 
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = shear(self.image, xoffset, yoffset, fill_color, p)
        return self
    
    def rescale(self, xscale=(0.5,1.5), yscale=(0.5,1.5), fill_color=None, p=None):
        """Apply scaling on the y-axis to input data.

        Args:
            xscale: if int or float, width expansion and contraction, xscale should be positive.
                    if tuple or list, randomly picked in the interval `[xscale[0], xscale[1])`.
            yscale: if int or float, height expansion and contraction, yscale should be positive.
                    if tuple or list, randomly picked in the interval `[yscale[0], yscale[1])`.
            fill_color: int or str or tuple or la.image.RGBMode, rgb color.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = rescale(self.image, xscale, yscale, fill_color, p=p)
        return self
    
    def jigsaw(self, size=(10,10), prob=0.1, p=None):
        """Move cells within images similar to jigsaw patterns.

        Args:
            size: if int or float, xsize=ysize, numbers of jigsaw.
                  if 2-tuple, (xsize, ysize), 
                  if 4-tuple, xsize in (size[0], size[1]) and  ysize in (size[2], size[3]).
            prob: probability of every jigsaw being changed.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = jigsaw(self.image, size, prob, p)
        return self