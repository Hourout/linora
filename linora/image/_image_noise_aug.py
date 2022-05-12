from linora.image._image_noise import *

class ImageNoiseAug(object):
    def __init__(self, image=None, p=1):
        self.image = image
        self._p = p
    
    def noise(self, mode=NoiseMode.Gaussian, wise='pixel', scale=1, prob=0.6, p=None, **kwargs):
        """noise apply to image.

        new pixel = image + gaussian_noise * scale
        Args:
            mode: la.image.NoiseMode
            wise: 'pixel' or 'channel', method of applying noise.
            scale: if int or float, value multiply with noise.
                   if tuple or list, randomly picked in the interval `[scale[0], scale[1])`.
            prob: probability of every pixel or channel being changed.
            p: probability that the image does this. Default value is 1.
        """
        if p is None:
            p = self._p
        self.image = noise(self.image, mode, wise, scale, prob, p, **kwargs)
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
