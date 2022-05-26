from linora.image._image_noise import *

class ImageNoiseAug(object):
    def __init__(self, image=None):
        self.image = image
    
    def mosaic(self, size=(80,80), block=0, axis=None, prob=0.3, p=None):
        """mosaic noise apply to image.

        Args:
            size: if int or float, xsize=ysize, how many mosaic blocks the image is divided into.
                  if 2-tuple, (xsize, ysize), 
                  if 4-tuple, xsize in (size[0], size[1]) and  ysize in (size[2], size[3]).
            block: mosaic area block
            axis: list or tuple, one or more mosaic center axis point.
            prob: probability of numbers of mosaci.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = mosaic(self.image, size, block, axis, prob, p=p)
        return self

    def noise_saltpepper(self, white_prob=0.025, black_prob=0.025, p=None):
        """Mask salt pepper noise apply to image.

        The salt-pepper noise is based on the signal-to-noise ratio of the image,
        randomly generating the pixel positions in some images all channel,
        and randomly assigning these pixels to 0 or 255.

        Args:
            white_prob: white pixel prob.
            black_prob: black pixel prob.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = noise_saltpepper(self.image, white_prob, black_prob, p=p)
        return self

    def noise_gaussian(self, mean=0, std=30, wise='pixel', scale=1, prob=0.2, p=None):
        """gaussian distribution noise apply to image.

        Args:
            mean: if int or float, value is gaussian distribution mean.
                  if tuple or list, randomly picked in the interval `[mean[0], mean[1])`.
            std: if int or float, value is gaussian distribution std.
                 if tuple or list, randomly picked in the interval `[std[0], std[1])`.
            wise: 'pixel' or 'channel' or 'pixelchannel' or list of channel, method of applying noise.
            scale: if int or float, value multiply with noise.
                   if tuple or list, randomly picked in the interval `[scale[0], scale[1])`.
            prob: probability of every pixel or channel being changed.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = noise_gaussian(self.image, mean, std, wise, scale, prob, p=p)
        return self

    def noise_laplace(self, mean=0, lam=30, wise='pixel', scale=1, prob=0.2, p=None):
        """laplace distribution noise apply to image.

        Args:
            mean: if int or float, value is laplace distribution mean.
                  if tuple or list, randomly picked in the interval `[mean[0], mean[1])`.
            lam: if int or float, value is laplace distribution lam.
                 if tuple or list, randomly picked in the interval `[lam[0], lam[1])`.
            wise: 'pixel' or 'channel' or 'pixelchannel' or list of channel, method of applying noise.
            scale: if int or float, value multiply with noise.
                   if tuple or list, randomly picked in the interval `[scale[0], scale[1])`.
            prob: probability of every pixel or channel being changed.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = noise_laplace(self.image, mean, lam, wise, scale, prob, p=p)
        return self

    def noise_poisson(self, lam=30, wise='pixel', scale=1, prob=0.2, p=None):
        """poisson distribution noise apply to image.

        Args:
            lam: if int or float, value is poisson distribution lam.
                 if tuple or list, randomly picked in the interval `[lam[0], lam[1])`.
            wise: 'pixel' or 'channel' or 'pixelchannel' or list of channel, method of applying noise.
            scale: if int or float, value multiply with noise.
                   if tuple or list, randomly picked in the interval `[scale[0], scale[1])`.
            prob: probability of every pixel or channel being changed.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = noise_poisson(self.image, lam, wise, scale, prob, p=p)
        return self


    def noise_uniform(self, lower=-50, upper=50, wise='pixel', scale=1, prob=0.2, p=None):
        """uniform distribution noise apply to image.

        Args:
            lower: if int or float, value is uniform distribution lower.
                   if tuple or list, randomly picked in the interval `[lower[0], lower[1])`.
            upper: if int or float, value is uniform distribution upper.
                   if tuple or list, randomly picked in the interval `[upper[0], upper[1])`.
            wise: 'pixel' or 'channel' or 'pixelchannel' or list of channel, method of applying noise.
            scale: if int or float, value multiply with noise.
                   if tuple or list, randomly picked in the interval `[scale[0], scale[1])`.
            prob: probability of every pixel or channel being changed.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = noise_uniform(self.image, lower, upper, wise, scale, prob, p=p)
        return self

    def noise_speckle(self, wise='pixel', prob=0.2, p=None):
        """speckle noise apply to image.

        pixel = pixel+pixel*gaussian

        Args:
            prob: probability of every pixel or channel being changed.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = noise_speckle(self.image, wise, prob, p=p)
        return self


    def noise_impulse(self, binomial=0.5, wise='pixel', prob=0.2, p=None):
        """impulse noise apply to image.

        Args:
            binomial: if int or float, value is binomial distribution prob.
                      if tuple or list, randomly picked in the interval `[binomial[0], binomial[1])`.
            wise: 'pixel' or 'channel' or 'pixelchannel' or list of channel, method of applying noise.
            prob: probability of every pixel or channel being changed.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = noise_impulse(self.image, binomial, wise, prob, p=p)
        return self
