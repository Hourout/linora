from linora.image._image_rescale import *

class ImageRescaleAug(object):
    def __init__(self, image=None):
        self.image = image
    
    def add(self, scale, wise='pixel', prob=1, p=None):
        """add apply to image.

        new pixel = int(image + scale)
        Args:
            scale: if int or float, value add with image.
                   if tuple or list, randomly picked in the interval `[scale[0], scale[1])`.
            wise: 'pixel' or 'channel' or list of channel, method of applying operate.
            prob: probability of every pixel or channel being changed.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = add(self.image, scale, wise, prob, p)
        return self
    
    def multiply(self, scale, wise='pixel', prob=1, p=None):
        """Rescale apply to image.

        new pixel = int(image * scale)
        Args:
            scale: if int or float, value multiply with image.
                   if tuple or list, randomly picked in the interval `[scale[0], scale[1])`.
            wise: 'pixel' or 'channel' or list of channel, method of applying operate.
            prob: probability of every pixel or channel being changed.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = multiply(self.image, scale, wise, prob, p)
        return self
        
    def normalize_global(self, mean=None, std=None, p=None):
        """Normalize scales `image` to have mean and variance.

        This op computes `(x - mean) / std`.
        Args:
            mean: if None, computes image mean.
                  if int or float, customize image all channels mean.
                  if tuple or list, randomly picked in the interval `[mean[0], mean[1])`
            std: if None, computes image std.
                 if int or float, customize image all channels std.
                 if tuple or list, randomly picked in the interval `[std[0], std[1])`
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = normalize_global(self.image, mean, std, p)
        return self
    
    def normalize_channel(self, mean=None, std=None, p=None):
        """Normalize scales `image` to have mean and variance.

        This op computes `(x - mean) / std`.
        Args:
            mean: if None, computes image mean.
                  if tuple or list, customize image each channels mean, shape should 3 dims.
            std: if None, computes image std.
                 if tuple or list, customize image each channels std, shape should 3 dims.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = normalize_channel(self.image, mean, std, p)
        return self
    
    def normalize_posterize(self, bits, p=None):
        """Reduce the number of bits for each color channel.

        There are up to 2**bits types of pixel values per channel.
        Args:
            bits: int or tuple or list, The number of bits to keep for each channel (1-8).
                  if list or tuple, randomly picked in the interval `[bits[0], bits[1])` value.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = normalize_posterize(self.image, bits, p)
        return self
