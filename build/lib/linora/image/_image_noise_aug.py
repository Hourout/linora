from linora.image._image_noise import *

class ImageNoiseAug(object):
    def __init__(self, image=None):
        self.image = image
    
    def noise_gaussian(self, scale=1, mean=0.0, std=1.0):
        """Gaussian noise apply to image.

        new pixel = image + gaussian_noise * scale
        Args:
            image: a Image instance.
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
            a Image instance.
        """
        if type(self.image)!=np.ndarray:
            self.image_to_array(self.image)
        self.image = noise_gaussian(self.image, scale, mean, std)
        return self
    
    def noise_poisson(self, scale=1, lam=1.0):
        """Poisson noise apply to image.

        new pixel = image + poisson_noise * scale
        Args:
            image: a Image instance.
            scale: if int or float, value multiply with poisson_noise.
                   if tuple or list, randomly picked in the interval
                   `[scale[0], scale[1])`, value multiply with poisson_noise.
            lam: if int or float, value is poisson distribution lambda.
                 if tuple or list, randomly picked in the interval
                 `[lam[0], lam[1])`, value is poisson distribution lambda.
        Returns:
            a Image instance.
        """
        if type(self.image)!=np.ndarray:
            self.image_to_array(self.image)
        self.image = noise_poisson(self.image, scale, lam)
        return self
    
    def noise_color(self, white_prob=0.05, black_prob=0.05, rainbow_prob=0):
        """Mask noise apply to image with color.

        while: 
            white_prob = black_prob and rainbow_prob=0, is salt-pepper noise

        The salt-pepper noise is based on the signal-to-noise ratio of the image,
        randomly generating the pixel positions in some images all channel,
        and randomly assigning these pixels to 0 or 255.

        while: 
            white_prob = black_prob=0 and rainbow_prob>0, is rainbow noise

        The rainbow noise is based on the signal-to-noise ratio of the image,
        randomly generating the pixel positions in some images,
        and randomly assigning these pixels to 0 or 255.

        Args:
            image: a Image instance.
            white_prob: white pixel prob.
            black_prob: black pixel prob.
            rainbow_prob: rainbow color pixel prob.
        Returns:
            a Image instance.
        """
        self.image = noise_color(self.image, white_prob, black_prob, rainbow_prob)
        return self
