from linora.image._image_position import *

class ImagePositionAug(object):
    def __init__(self, image=None, p=1):
        self.image = image
        self._p = p
    
    def flip_up_left(self, p=None):
        """Randomly flip an image (up to left).
    
        Args:
            p: probability that the image does this. Default value is 1.
        """
        if p is None:
            p = self._p
        self.image = flip_up_left(self.image, p)
        return self
    
    def flip_up_right(self, p=None):
        """Randomly flip an image (up to right).
    
        Args:
            p: probability that the image does this. Default value is 1.
        """
        if p is None:
            p = self._p
        self.image = flip_up_right(self.image, p)
        return self
    
    def flip_left_right(self, p=None):
        """Randomly flip an image (left to right).
    
        Args:
            p: probability that the image does this. Default value is 1.
        """
        if p is None:
            p = self._p
        self.image = flip_left_right(self.image, p)
        return self
    
    def flip_up_down(self, p=None):
        """Randomly flip an image (up to down).
    
        Args:
            p: probability that the image does this. Default value is 1.
        """
        if p is None:
            p = self._p
        self.image = flip_up_down(self.image, p)
        return self
    
    def rotate(self, angle, expand=False, center=None, translate=None, fillcolor=None, p=None):
        """Returns a rotated copy of this image. 
        
        This method returns a copy of this image, rotated the given number of degrees counter clockwise around its centre.

        Args:
            angle: In degrees counter clockwise.
                   if int or float, rotation angle.
                   if list or tuple, randomly picked in the interval `[angle[0], angle[1])` value.
            expand: Optional expansion flag. If true, expands the output image to make it large 
                    enough to hold the entire rotated image. If false or omitted, 
                    make the output image the same size as the input image. 
                    Note that the expand flag assumes rotation around the center and no translation.
                    if value is 'random', then the function is random.
            center: Optional center of rotation (a 2-tuple). Origin is the upper left corner. 
                    Default is the center of the image.
                    if value is 'random', then the function is random.
            translate: An optional post-rotate translation (a 2-tuple).
                       if value is 'random', then the function is random.
            fillcolor: An optional color for area outside the rotated image.
                       if value is 'random', fillcolor is one of ['green', 'red', 'white', 'black'].
                       you can also pass in a list of colors.
            p: probability that the image does this. Default value is 1.
        """
        if p is None:
            p = self._p
        self.image = rotate(self.image, angle, expand, center, translate, fillcolor, p)
        return self
    
    def translate(self, translate='random', fillcolor=None, p=None):
        """Returns a translate copy of this image. 
    
        Args:
            translate: An optional post-rotate translation (a 2-tuple).
                      if value is 'random', then the function is random.
            fillcolor: An optional color for area outside the rotated image.
                       if value is 'random', fillcolor is one of ['green', 'red', 'white', 'black'].
                       you can also pass in a list of colors.
            p: probability that the image does this. Default value is 1.
        """
        if p is None:
            p = self._p
        self.image = translate(self.image, translate, fillcolor, p)
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
        if p is None:
            p = self._p
        self.image = pad(self.image, pad_value, pad_color, p)
        return self
    
    def channel_shuffle(self, p=None):
        """Random shuffle image channel.

        Args:
            p: probability that the image does this. Default value is 1.
        """
        if p is None:
            p = self._p
        self.image = shuffle_channel(self.image, p)
        return self
    
    def transform_perspective(self, distortion_scale, fill_color=None, p=None):
        """Performs a random perspective transformation of the given image with a given probability. 

        Args:
            distortion_scale: float, argument to control the degree of distortion and ranges from 0 to 1.
            fill_color: int or str or tuple or la.image.RGBMode, rgb color.
            p: probability that the image does this. Default value is 1.
        """
        if p is None:
            p = self._p
        self.image = transform_perspective(self.image, distortion_scale, fill_color, p)
        return self
    
    def transform_affine(self, angle=(-180, 180), translate=(0, 0), scale=1., shear=(0,0), center=None, fill_color=None, p=None):
        """Apply affine transformation on the image keeping image center invariant.

        Args:
            angle: int or float, rotation angle in degrees between -180 and 180. Set to 0 to deactivate rotations.
                   if list or tuple, randomly picked in the interval `[angle[0], angle[1])`.
            translate: list or tuple, tuple of maximum absolute fraction for horizontal and vertical translations. 
                       For example translate=(a, b), then horizontal shift is randomly sampled 
                       in the range -img_width * a < dx < img_width * a 
                       and vertical shift is randomly sampled in the range -img_height * b < dy < img_height * b. 
                       Will not translate by default.
            scale: float, scaling factor interval, if list or tuple, randomly picked in the interval `[scale[0], scale[1])`.
            shear: Range of degrees to select from. 
                   Else if shear is a sequence of 2 values a shear parallel to the x axis in the range (shear[0], shear[1]) will be applied. 
                   Else if shear is a sequence of 4 values, a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied. 
            fill_color: int or str or tuple or la.image.RGBMode, rgb color. Pixel fill value for the area outside the transformed image.
            center: Optional center of rotation. Origin is the upper left corner. Default is the center of the image.
            p: probability that the image does this. Default value is 1.
        """
        if p is None:
            p = self._p
        self.image = transform_affine(self.image, angle=(-180, 180), translate=(0, 0), scale=1., shear=(0,0), center=None, fill_color=None, p=p)
        return self