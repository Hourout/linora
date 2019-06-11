import tensorflow as tf
from linora.image._image_color import *

class ImageColorAug(object):
    def __init__(self, image=None):
        self._image = image
    
    def RandomBrightness(self, delta, seed=None):
        """Adjust the brightness of RGB or Grayscale images.
    
        Tips:
            delta extreme value in the interval [-1, 1], >1 to white, <-1 to black.
            a suitable interval is [-0.5, 0.5].
            0 means pixel value no change.
        Args:
            delta: if int, float, Amount to add to the pixel values.
                   if list, tuple, randomly picked in the interval
                   `[delta[0], delta[1])` to add to the pixel values.
            seed: A Python integer. Used to create a random seed. See
                 `tf.set_random_seed` for behavior.
        Returns:
            A brightness-adjusted tensor of the same shape and type as `image`.
        Raises:
            ValueError: if `delta` type is error.
        """
        self._image = RandomBrightness(self._image, delta, seed, _=True)
        return self
    
    def RandomContrast(self, delta, seed=None):
        """Adjust contrast of RGB or grayscale images.
    
        `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
        interpreted as `[height, width, channels]`.  The other dimensions only
        represent a collection of images, such as `[batch, height, width, channels].`
        Contrast is adjusted independently for each channel of each image.
        For each channel, this Ops computes the mean of the image pixels in the
        channel and then adjusts each component `x` of each pixel to
        `(x - mean) * delta + mean`.
        Tips:
            1 means pixel value no change.
            0 means all pixel equal. 
            a suitable interval is (0, 4].
        Args:
            delta: if int, float, a float multiplier for adjusting contrast.
                   if list, tuple, randomly picked in the interval
                   `[delta[0], delta[1])` , value is float multiplier for adjusting contrast.
            seed: A Python integer. Used to create a random seed. See
                 `tf.set_random_seed` for behavior.
        Returns:
            The contrast-adjusted image or images tensor of the same shape and type as `image`.
        Raises:
            ValueError: if `delta` type is error.
        """
        self._image = RandomContrast(self._image, delta, seed, _=True)
        return self
    
    def RandomHue(self, delta, seed=None):
        """Adjust hue of an RGB image.
    
        `image` is an RGB image.  The image hue is adjusted by converting the
        image to HSV and rotating the hue channel (H) by `delta`.
        The image is then converted back to RGB.
        Tips:
            `delta` should be in the interval `[-1, 1]`, but any value is allowed.
            a suitable interval is [-0.5, 0.5].
            int value means pixel value no change.
        Args:
            delta: if float, How much to add to the hue channel.
                   if list, tuple, randomly picked in the interval
                   `[delta[0], delta[1])` , value is how much to add to the hue channel.
            seed: A Python integer. Used to create a random seed. See
                 `tf.set_random_seed` for behavior.
        Returns:
            The hue-adjusted image or images tensor of the same shape and type as `image`.
        Raises:
            ValueError: if `delta` type is error.
        """
        self._image = RandomHue(self._image, delta, seed, _=True)
        return self
    
    def RandomSaturation(self, delta, seed=None):
        """Adjust saturation of an RGB image.
    
        `image` is an RGB image.  The image saturation is adjusted by converting the
        image to HSV and multiplying the saturation (S) channel by `delta` and clipping.
        The image is then converted back to RGB.
        Tips:
            if delta <= 0, image channels value are equal, image color is gray.
            a suitable interval is delta >0
        Args:
            delta: if int, float, Factor to multiply the saturation by.
                   if list, tuple, randomly picked in the interval
                   `[delta[0], delta[1])` , value is factor to multiply the saturation by.
            seed: A Python integer. Used to create a random seed. See
                 `tf.set_random_seed` for behavior.
        Returns:
            The saturation-adjusted image or images tensor of the same shape and type as `image`.
        Raises:
            ValueError: if `delta` type is error.
        """
        self._image = RandomSaturation(self._image, delta, seed, _=True)
        return self
    
    def RandomGamma(self, gamma, seed=None):
        """Performs Gamma Correction on the input image.
    
        Also known as Power Law Transform. This function transforms the
        input image pixelwise according to the equation `Out = In**gamma`
        after scaling each pixel to the range 0 to 1.
        Tips:
            For gamma greater than 1, the histogram will shift towards left and
            the output image will be darker than the input image.
            For gamma less than 1, the histogram will shift towards right and
            the output image will be brighter than the input image.
            if gamma is 1, image pixel value no change.
        Args:
            gamma : if int, float, Non negative real number.
                    if list, tuple, randomly picked in the interval
                    `[delta[0], delta[1])` , value is Non negative real number.
            seed: A Python integer. Used to create a random seed. See
                  `tf.set_random_seed` for behavior.
        Returns:
            A float Tensor. Gamma corrected output image.
        Raises:
            ValueError: If gamma is negative.
        References:
            [1] http://en.wikipedia.org/wiki/Gamma_correction
        """
        self._image = RandomGamma(self._image, gamma, seed, _=True)
        return self
