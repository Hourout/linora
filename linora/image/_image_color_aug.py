from linora.image._image_color import *

class ImageColorAug(object):
    def __init__(self, image=None):
        self.image = image
    
    def enhance_saturation(self, delta, p=None):
        """Adjust image color balance.
        This class can be used to adjust the colour balance of an image, 
        in a manner similar to the controls on a colour TV set. 
        An enhancement factor of 0.0 gives a black and white image. 
        A factor of 1.0 gives the original image.

        Args:
            delta: A floating point value controlling the enhancement. 
                   delta 1.0 always returns a copy of the original image, 
                   lower factors mean less color, 
                   and higher values more. There are no restrictions on this value.
                   if list, tuple, randomly picked in the interval
                   `[delta[0], delta[1])` , value is float multiplier for adjusting color.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = enhance_saturation(self.image, delta, p)
        return self
    
    def enhance_contrast_linear(self, delta, p=None):
        """Adjust contrast of RGB or grayscale images.
  
        Contrast is adjusted independently for each channel of each image.
        pixel = (x - mean) * delta + mean

        Tips:
            1 means pixel value no change.
            0 means all pixel equal. 
            a suitable interval is (0, 4].
        Args:
            delta: if int, float, a float multiplier for adjusting contrast.
                   if list, tuple, randomly picked in the interval
                   `[delta[0], delta[1])` , value is float multiplier for adjusting contrast.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = enhance_contrast_linear(self.image, delta, p)
        return self
    
    def enhance_contrast_gamma(self, gamma=1, gain=1.0, p=None):
        """Perform gamma correction on an image.

        For gamma greater than 1, the histogram will shift towards left and the output image will be darker than the input image. 
        For gamma less than 1, the histogram will shift towards right and the output image will be brighter than the input image.

        pixel = 255*gain *((v/255)**gamma)
        Values in the range gamma=(0.5, 2.0) seem to be sensible.

        Args:
            gamma: float, Non negative real number, gamma larger than 1 make the shadows darker, 
                   while gamma smaller than 1 make dark regions lighter.
                   if list or tuple, randomly picked in the interval `[gamma[0], gamma[1])`.
            gain: float, The constant multiplier.
                  if list or tuple, randomly picked in the interval `[gain[0], gain[1])`.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = enhance_contrast_gamma(self.image, gamma, gain, p)
        return self
    
    def enhance_contrast_sigmoid(self, cutoff=0.5, gain=10, p=None):
        """Perform sigmoid correction on an image.

        pixel = 255*1/(1+exp(gain*(cutoff-v/255)))
        Values in the range gain=(5, 20) and cutoff=(0.25, 0.75) seem to be sensible.

        Args:
            image: a PIL instance.
            cutoff: float, Cutoff that shifts the sigmoid function in horizontal direction. 
                    Higher values mean that the switch from dark to light pixels happens later. 
                    if list or tuple, randomly picked in the interval `[cutoff[0], cutoff[1])`.
            gain: float, The constant multiplier.
                  if list or tuple, randomly picked in the interval `[gain[0], gain[1])`.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = enhance_contrast_sigmoid(self.image, cutoff, gain, p)
        return self
    
    def enhance_contrast_log(self, gain=1, p=None):
        """Perform log correction on an image.

        pixel = 255*gain*log_2(1+v/255)
        Values in the range gain=[0.6, 1.4] seem to be sensible.

        Args:
            gain: float, The constant multiplier.
                  if list or tuple, randomly picked in the interval `[gain[0], gain[1])`.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = enhance_contrast_log(self.image, gain, p)
        return self
    
    def enhance_brightness(self, delta, p=None):
        """Adjust the brightness of RGB or Grayscale images.
    
        Tips:
            delta extreme value in the interval [-1, 1], >1 to white, <-1 to black.
            a suitable interval is [-0.5, 0.5].
            0 means pixel value no change.
        Args:
            delta: if int, float, Amount to add to the pixel values.
                   if list, tuple, randomly picked in the interval `[delta[0], delta[1])` to add to the pixel values.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = enhance_brightness(self.image, delta, p)
        return self

    def enhance_sharpness(self, delta, p=None):
        """Adjust image sharpness.
        
        This class can be used to adjust the sharpness of an image. 
        An enhancement factor of 0.0 gives a blurred image, 
        a factor of 1.0 gives the original image, 
        and a factor of 2.0 gives a sharpened image.

        Args:
            delta: A floating point value controlling the enhancement. 
                   delta 1.0 always returns a copy of the original image, 
                   lower factors mean less sharpness, 
                   and higher values more. There are no restrictions on this value.
                   if list, tuple, randomly picked in the interval
                       `[delta[0], delta[1])` , value is float multiplier for adjusting color.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = enhance_sharpness(self.image, delta, p)
        return self
    
    def enhance_hue(self, delta, p=None):
        """Adjust hue of an image.

        The image hue is adjusted by converting the image to HSV and
        cyclically shifting the intensities in the hue channel (H).
        The image is then converted back to original image mode.

        `delta` is the amount of shift in H channel and must be in the interval `[-1, 1]`.

        Args:
            delta: How much to shift the hue channel. Should be in [-1, 1]. 
                   1 and -1 give complete reversal of hue channel in
                   HSV space in positive and negative direction respectively.
                   0 means no shift. Therefore, both -1 and 1 will give an image
                   with complementary colors while 0 gives the original image.
                   if list or tuple, randomly picked in the interval `[delta[0], delta[1])`.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = enhance_hue(self.image, delta, p)
        return self
    
    def color_invert(self, lower=None, upper=None, wise='pixel', prob=1, p=None):
        """Invert colors of input PIL image.

        Args:
            lower: int or list or tuple, [0, 255], All pixels below this greyscale level are inverted.
                   if list or tuple, randomly picked in the interval `[lower[0], lower[1])`.
            upper: int or list or tuple, [0, 255], All pixels above this greyscale level are inverted.
                   if list or tuple, randomly picked in the interval `[upper[0], upper[1])`.
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
        self.image = color_invert(self.image, lower, upper, wise, prob, p)
        return self
    
    def color_clip(self, lower=None, upper=None, wise='pixel', prob=1, p=None):
        """Clipped colors of input PIL image.

        Args:
            lower: int or list or tuple, [0, 255], All pixels below this greyscale level are clipped.
                   if list or tuple, randomly picked in the interval `[lower[0], lower[1])`.
            upper: int or list or tuple, [0, 255], All pixels above this greyscale level are clipped.
                   if list or tuple, randomly picked in the interval `[upper[0], upper[1])`.
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
        self.image = color_clip(self.image, lower, upper, wise, prob, p)
        return self

    def equalize(self, p=None):
        """
        Equalize the image histogram. This function applies a non-linear
        mapping to the input image, in order to create a uniform
        distribution of grayscale values in the output image.
        
        Args:
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = equalize(self.image, p)
        return self

    def dropout(self, value=0, wise='pixel', prob=0.1, p=None):
        """Drop random channels from images.

        For image data, dropped channels will be filled with value.

        Args:
            value: int or list, dropped channels will be filled with value.
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
        self.image = dropout(self.image, value, wise, prob, p)
        return self