from linora.image._image_noise import *

class ImageNoiseAug(object):
    def __init__(self, image=None, p=1):
        self.image = image
        self._p = p
    
    def noise_gaussian(self, scale=1, mean=0.0, std=1.0, p=None):
        """Gaussian noise apply to image.

        new pixel = image + gaussian_noise * scale
        Args:
            scale: if int or float, value multiply with poisson_noise.
                   if tuple or list, randomly picked in the interval
                   `[scale[0], scale[1])`, value multiply with poisson_noise.
            mean: if int or float, value is gaussian distribution mean.
                  if tuple or list, randomly picked in the interval
                  `[mean[0], mean[1])`, value is gaussian distribution mean.
            std: if int or float, value is gaussian distribution std.
                 if tuple or list, randomly picked in the interval
                 `[std[0], std[1])`, value is gaussian distribution std.
            p: probability that the image does this. Default value is 1.
        """
        if p is None:
            p = self._p
        self.image = noise_gaussian(self.image, scale, mean, std, p)
        return self
    
    def noise_poisson(self, scale=1, lam=1.0, p=None):
        """Poisson noise apply to image.

        new pixel = image + poisson_noise * scale
        Args:
            scale: if int or float, value multiply with poisson_noise.
                   if tuple or list, randomly picked in the interval
                   `[scale[0], scale[1])`, value multiply with poisson_noise.
            lam: if int or float, value is poisson distribution lambda.
                 if tuple or list, randomly picked in the interval
                 `[lam[0], lam[1])`, value is poisson distribution lambda.
            p: probability that the image does this. Default value is 1.
        """
        if p is None:
            p = self._p
        self.image = noise_poisson(self.image, scale, lam, p)
        return self
    
    def noise_color(self, white_prob=0.05, black_prob=0.05, rainbow_prob=0, p=None):
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
            white_prob: white pixel prob.
            black_prob: black pixel prob.
            rainbow_prob: rainbow color pixel prob.
            p: probability that the image does this. Default value is 1.
        """
        if p is None:
            p = self._p
        self.image = noise_color(self.image, white_prob, black_prob, rainbow_prob, p)
        return self
