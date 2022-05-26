from linora.image._image_crop import *

class ImageCropAug(object):
    def __init__(self, image=None):
        self.image = image
        
    def crop(self, box, p=None):
        """Returns a rectangular region from this image. 
        
        The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.
        Args:
            box: The crop rectangle, as a (left, upper, right, lower)-tuple.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = crop(self.image, box, p)
        return self

    def crop_central(self, central_rate, p=None):
        """Crop the central region of the image.
    
        Remove the outer parts of an image but retain the central region of the image
        along each dimension. If we specify central_fraction = 0.5, this function
        returns the region marked with "X" in the below diagram.
           --------
          |        |
          |  XXXX  |
          |  XXXX  |
          |        |   where "X" is the central 50% of the image.
           --------

        This function works on either a single image (`image` is a 3-D Tensor), or a
        batch of images (`image` is a 4-D Tensor).

        Args:
            central_rate: if int float, should be in the interval (0, 1], fraction of size to crop.
                          if tuple list, randomly picked in the interval
                          `[central_rate[0], central_rate[1])`, value is fraction of size to crop.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = crop_central(self.image, central_rate, p)
        return self
    
    def crop_point(self, height_rate, width_rate, p=None):
        """Crop the any region of the image(s) and resize specify shape.
    
        Crop region area = height_rate * width_rate *image_height * image_width

        This function works on either a single image (`image` is a 3-D Tensor), or a
        batch of images (`image` is a 4-D Tensor).

        Args:
            height_rate: float, in the interval (0, 1].
            width_rate: float, in the interval (0, 1].
            p: probability that the image does this. Default value is 1.
        Raises:
            ValueError: if central_crop_fraction is not within (0, 1].
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = crop_point(self.image, height_rate, width_rate, p)
        return self
