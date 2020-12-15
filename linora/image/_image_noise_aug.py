from linora.image._image_noise import *

class ImageNoiseAug(object):
    def __init__(self, image=None):
        self.image = image
    
    def noise_gaussian(self, scale=1, mean=0.0, std=1.0):
        """Gaussian noise apply to image.

        new pixel = image + gaussian_noise * scale
        Args:
            image: Either a 3-D float Tensor of shape [height, width, depth].
            scale: if int or float, value multiply with poisson_noise.
                   if tuple or list, randomly picked in the interval
                   `[scale[0], scale[1])`, value multiply with poisson_noise.
            mean: if int or float, value is gaussian distribution mean.
                  if tuple or list, randomly picked in the interval
                  `[mean[0], mean[1])`, value is gaussian distribution mean.
            std: if int or float, value is gaussian distribution std.
                 if tuple or list, randomly picked in the interval
                 `[std[0], std[1])`, value is gaussian distribution std.
        Returns:
            3-D float Tensor, as per the input.
        """
        if type(self.image)!=np.ndarray:
            self.image_to_array(self.image)
        self.image = noise_gaussian(self.image, scale, mean, std)
        return self
    
    def noise_poisson(self, scale=1, lam=1.0):
        """Poisson noise apply to image.

        new pixel = image + poisson_noise * scale
        Args:
            image: Either a 3-D float Tensor of shape [height, width, depth].
            scale: if int or float, value multiply with poisson_noise.
                   if tuple or list, randomly picked in the interval
                   `[scale[0], scale[1])`, value multiply with poisson_noise.
            lam: if int or float, value is poisson distribution lambda.
                 if tuple or list, randomly picked in the interval
                 `[lam[0], lam[1])`, value is poisson distribution lambda.
        Returns:
            3-D float Tensor, as per the input.
        """
        if type(self.image)!=np.ndarray:
            self.image_to_array(self.image)
        self.image = noise_poisson(self.image, scale, lam)
        return self
    
    def noise_mask(self, noise_prob=0.2):
        """Mask noise apply to image.

        With probability `drop_prob`, outputs the input element scaled up by
        `1`, otherwise outputs `0`. 

        Tips:
            1 mean pixel have no change.
            a suitable interval is (0., 0.1].
        Args:
            image: Either a 3-D float Tensor of shape [height, width, depth].
            noise_prob: should be in the interval (0, 1.].
                       if float, the probability that each element is drop.
                       if tuple or list, randomly picked in the interval
                       `[keep_prob[0], keep_prob[1])`, the probability that each element is drop.
        Returns:
            3-D float Tensor, as per the input.
        Raises:
            ValueError: If `keep_prob` is not in `(0, 1.]`.
        """
        if type(self.image)!=np.ndarray:
            self.image_to_array(self.image)
        self.image = noise_mask(self.image, noise_prob)
        return self
    
    def noise_saltpepper(self, noise_prob=0.2):
        """Salt-Pepper noise apply to image.

        The salt-pepper noise is based on the signal-to-noise ratio of the image,
        randomly generating the pixel positions in some images all channel,
        and randomly assigning these pixels to 0 or 255.

        Args:
            image: Either a 3-D float Tensor of shape [height, width, depth].
            noise_prob: should be in the interval (0, 1].
                       if int or float, the probability that each element is kept.
                       if tuple or list, randomly picked in the interval
                       `[keep_prob[0], keep_prob[1])`, the probability that each element is kept.
        Returns:
            3-D float Tensor, as per the input.
        Raises:
            ValueError: If `keep_prob` is not in `(0, 1]`.
        """
        if type(self.image)!=np.ndarray:
            self.image_to_array(self.image)
        self.image = noise_saltpepper(self.image, noise_prob)
        return self
    
    def noise_rainbow(image, noise_prob=0.2):
        """Rainbowr noise apply to image.

        The rainbow noise is based on the signal-to-noise ratio of the image,
        randomly generating the pixel positions in some images,
        and randomly assigning these pixels to 0 or 255.

        Args:
            image: Either a 3-D float Tensor of shape [height, width, depth].
            noise_prob: should be in the interval (0, 1].
                       if int or float, the probability that each element is kept.
                       if tuple or list, randomly picked in the interval
                       `[keep_prob[0], keep_prob[1])`, the probability that each element is kept.
        Returns:
            3-D float Tensor, as per the input.
        Raises:
            ValueError: If `keep_prob` is not in `(0, 1]`.
        """
        if type(self.image)!=np.ndarray:
            self.image_to_array(self.image)
        self.image = noise_rainbow(self.image, noise_prob)
        return self
    
    
    
    
