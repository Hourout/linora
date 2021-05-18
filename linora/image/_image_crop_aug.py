from linora.image._image_crop import *

class ImageCropAug(object):
    def __init__(self, image=None):
        self.image = image
        
    def crop(self, box):
        """
        Returns a rectangular region from this image. The box is a
        4-tuple defining the left, upper, right, and lower pixel coordinate.
        Args:
            box: The crop rectangle, as a (left, upper, right, lower)-tuple.
        returns: 
            a Image instance.
        """
        self.image = crop(self.image, box)
        return self

    def crop_central(self, central_rate):
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
            image: a Image instance.
            central_rate: if int float, should be in the interval (0, 1], fraction of size to crop.
                          if tuple list, randomly picked in the interval
                          `[central_rate[0], central_rate[1])`, value is fraction of size to crop.
        Returns:
            a Image instance.
        Raises:
            ValueError: if central_crop_fraction is not within (0, 1].
        """
        self.image = crop_central(self.image, central_rate)
        return self
    
    def crop_point(self, height_rate, width_rate):
        """Crop the any region of the image(s) and resize specify shape.
    
        Crop region area = height_rate * width_rate *image_height * image_width

        This function works on either a single image (`image` is a 3-D Tensor), or a
        batch of images (`image` is a 4-D Tensor).

        Args:
            image: a Image instance.
            height_rate: flaot, in the interval (0, 1].
            width_rate: flaot, in the interval (0, 1].
        Returns:
            a Image instance.
        Raises:
            ValueError: if central_crop_fraction is not within (0, 1].
        """
        self.image = crop_point(self.image, height_rate, width_rate)
        return self
    
    
    
    
