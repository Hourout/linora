from linora.image._image_rescale import *

class ImageRescaleAug(object):
    def __init__(self, image=None):
        self.image = image
    
    def normalize_global(self, mean=None, std=None):
        """Normalize scales `image` to have mean and variance.
    
        This op computes `(x - mean) / std`.
        Args:
            mean: if None, computes image mean.
                  if int or float, customize image all channels mean.
            std: if None, computes image std.
                 if int or float, customize image all channels std.
       Returns:
         The standardized image with same shape as `image`.
       Raises:
         ValueError: if the shape of 'image' is incompatible with this function.
       """
        if type(self.image)!=np.ndarray:
            self.image_to_array(self.image)
        self.image = normalize_global(self.image, mean, std)
        return self
    
    def normalize_channel(self, mean=None, std=None):
        """Normalize scales `image` to have mean and variance.
    
        This op computes `(x - mean) / std`.
        Args:
            mean: if None, computes image mean.
                  if tuple or list, customize image each channels mean,
                  shape should 3 dims.
            std: if None, computes image std.
                 if tuple or list, customize image each channels std,
                 shape should 3 dims.
      Returns:
        The standardized image with same shape as `image`.
      Raises:
        ValueError: if the shape of 'image' is incompatible with this function.
      """
        if type(self.image)!=np.ndarray:
            self.image_to_array(self.image)
        self.image = normalize_channel(self.image, mean, std)
        return self

    def rescale(self, scale):
        """Rescale apply to image.

        new pixel = image * scale
        Args:
            scale: if int float, value multiply with image.
                   if tuple list, randomly picked in the interval
                   `[central_rate[0], central_rate[1])`, value multiply with image.
        Returns:
            The image with same shape as `image`.
        Raises:
            scale type error.
        """
        if type(self.image)!=np.ndarray:
            self.image_to_array(self.image)
        self.image = rescale(self.image, scale)
        return self
    
    
    
