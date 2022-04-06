from linora.image._image_position import *

class ImagePositionAug(object):
    def __init__(self, image=None):
        self.image = image
    
    def flip_up_left(self, random=True):
        """Randomly flip an image (up to left).
    
        With a 1 in 2 chance, outputs the contents of `image` flipped along the
        second dimension, which is `width`.  Otherwise output the image as-is.

        Args:
        random: bool, default True.
                if True, random flip up or left image.
                if False, flip up or left image.
        Returns:
                A Image instance. of the same type and shape as `image`.
        """
        self.image = flip_up_left(self.image, random)
        return self
    
    def flip_up_right(self, random=True):
        """Randomly flip an image (up to right).
    
        With a 1 in 2 chance, outputs the contents of `image` flipped along the
        second dimension, which is `width`.  Otherwise output the image as-is.

        Args:
        random: bool, default True.
                if True, random flip up or right image.
                if False, flip up or right image.
        Returns:
                A Image instance. of the same type and shape as `image`.
        """
        self.image = flip_up_right(self.image, random)
        return self
    
    def flip_left_right(self, random=True):
        """Randomly flip an image (left to right).
    
        With a 1 in 2 chance, outputs the contents of `image` flipped along the
        second dimension, which is `width`.  Otherwise output the image as-is.

        Args:
        random: bool, default True.
                if True, random flip left or rignt image.
                if False, flip left or right image.
        Returns:
                A Image instance. of the same type and shape as `image`.
        """
        self.image = flip_left_right(self.image, random)
        return self
    
    def flip_up_down(self, random=True):
        """Randomly flip an image (up to down).
    
        With a 1 in 2 chance, outputs the contents of `image` flipped along the
        second dimension, which is `width`.  Otherwise output the image as-is.

        Args:
        random: bool, default True.
                if True, random flip up or down image.
                if False, flip up or down image.
        Returns:
                A Image instance. of the same type and shape as `image`.
        """
        self.image = flip_up_down(self.image, random)
        return self
    
    def rotate(self, angle, expand=True, center=None, translate=None, fillcolor=None):
        """Returns a rotated copy of this image. This method returns a copy of this image, 
        rotated the given number of degrees counter clockwise around its centre.

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

        Returns:
                A Image instance. of the same type and shape as `image`.
        """
        self.image = rotate(self.image, angle, expand, center, translate, fillcolor)
        return self
    
    def translate(self, translate=None, fillcolor=None):
        """Returns a translate copy of this image. 
    
        Args:
        translate: An optional post-rotate translation (a 2-tuple).
                if value is 'random', then the function is random.
        fillcolor: An optional color for area outside the rotated image.
                if value is 'random', fillcolor is one of ['green', 'red', 'white', 'black'].
                you can also pass in a list of colors.

        Returns:
                A Image instance. of the same type and shape as `image`.
        """
        self.image = translate(self.image, translate, fillcolor)
        return self
    
