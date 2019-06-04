import numpy as np
import tensorflow as tf
from linora.image._image import *

__all__ = ['ImageAug']

class ImageAug():
    def __init__(self, image=None):
        """init numpy array image.
        
        Args:
            image: array, default None.
                   if image is None, you should frist use `ImageAug().read_image()`
        """
        self._image = image
    
    def read_image(self, filename, channel=0, image_format='mix'):
        """Convenience function for read image type one of `bmp`, `gif`, `jpeg`, `jpg`, and `png`.
    
        Detects whether an image is a BMP, GIF, JPEG, JPG, or PNG, and performs the
        appropriate operation to convert the input bytes `string` into a `Tensor`
        of type `dtype`.

        Note: `gif` returns a 4-D array `[num_frames, height, width, 3]`, as
        opposed to `bmp`, `jpeg`, `jpg` and `png`, which return 3-D
        arrays `[height, width, num_channels]`. Make sure to take this into account
        when constructing your graph if you are intermixing GIF files with BMP, JPEG, JPG,
        and/or PNG files.
        Args:
            filename: 0-D `string`. image absolute path.
            channels: An optional `int`. Defaults to `0`. Number of color channels for
                      the decoded image. 1 for `grayscale` and 3 for `rgb`.
            image_format: 0-D `string`. image format type one of `bmp`, `gif`, `jpeg`,
                          `jpg`, `png` and `mix`. `mix` mean contains many types image format.
        Returns:
            `Tensor` with type uint8 and shape `[height, width, num_channels]` for
            BMP, JPEG, and PNG images and shape `[num_frames, height, width, 3]` for
            GIF images.
        Raises:
            ValueError: On incorrect number of channels.
        """
        return ImageAug(read_image(filename, channel, image_format, _=True))
    
    def save_image(self, filename):
        """Writes image to the file at input filename. 
    
        Args:
            image:    A Tensor of type string. scalar. The content to be written to the output file.
            filename: A string. scalar. The name of the file to which we write the contents.
        Raises:
            ValueError: If `filename` is not in `[`jpg`, `jpeg`, `png`]`.
        """
        return save_image(self._image, filename)
    
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
        return ImageAug(RandomBrightness(self._image, delta, seed, _=True))
    
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
        return ImageAug(RandomContrast(self._image, delta, seed, _=True))
    
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
        return ImageAug(RandomHue(self._image, delta, seed, _=True))
    
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
        return ImageAug(RandomSaturation(self._image, delta, seed, _=True))
    
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
        return ImageAug(RandomGamma(self._image, gamma, seed, _=True))
    
    def RandomFlipLeftRight(self, random=True, seed=None):
        """Randomly flip an image horizontally (left to right).
    
        With a 1 in 2 chance, outputs the contents of `image` flipped along the
        second dimension, which is `width`.  Otherwise output the image as-is.
        Args:
            random: bool, default True.
                    if True, random flip left or rignt image.
                    if False, flip left or right image.
            seed: A Python integer. Used to create a random seed. See
                  `tf.set_random_seed` for behavior.
        Returns:
            A tensor of the same type and shape as `image`.
        Raises:
            ValueError: if the shape of `image` not supported or `random` dtype not bool.
        """
        return ImageAug(RandomFlipLeftRight(self._image, random, seed, _=True))
    
    def RandomFlipTopBottom(self, random=True, seed=None):
        """Randomly flips an image vertically (upside down).
    
        With a 1 in 2 chance, outputs the contents of `image` flipped along the first
        dimension, which is `height`.  Otherwise output the image as-is.
        Args:
            random: bool, default True.
                    if True, random flip top or bottom image.
                    if False, flip top or bottom image.
            seed: A Python integer. Used to create a random seed. See
                  `tf.set_random_seed` for behavior.
        Returns:
            A tensor of the same type and shape as `image`.
        Raises:
            ValueError: if the shape of `image` not supported or `random` dtype not bool.
        """
        return ImageAug(RandomFlipTopBottom(self._image, random, seed, _=True))
    
    def RandomTranspose(self, random=True, seed=None):
        """Transpose image(s) by swapping the height and width dimension.
    
        Args:
            random: bool, default True.
                    if True, random transpose image.
                    if False, transpose image.
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            If `image` was 4-D, a 4-D float Tensor of shape `[batch, width, height, channels]`.
            If `image` was 3-D, a 3-D float Tensor of shape `[width, height, channels]`.
        Raises:
            ValueError: if the shape of `image` not supported or `random` dtype not bool.
        """
        return ImageAug(RandomTranspose(self._image, random, seed, _=True))
    
    def RandomRotation(self, k=[0, 1, 2, 3], seed=None):
        """Rotate image(s) counter-clockwise by 90 degrees.
    
        Tips:
            k should be int one of [1, 2, 3] or sublist in the [0, 1, 2, 3].
        Args:
            k: if k is list, random select t form k, rotation image by 90 degrees * t.
               if k is int, rotation image by 90 degrees * k.
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
        A rotated tensor of the same type and shape as `image`.
      Raises:
        ValueError: if the shape of `image` not supported or `k` dtype not int or list.
      """
        return ImageAug(RandomRotation(self._image, k, seed, _=True))
    
    def RandomCropCentralResize(self, central_rate, size, method=0, seed=None):
        """Crop the central region of the image(s) and resize specify shape.
    
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

        Tips:
            method should be one of [0, 1, 2, 3], "0:bilinear", "1:nearest_neighbor", "2:bicubic", "3:area".
        Args:
            central_rate: if int float, should be in the interval (0, 1], fraction of size to crop.
                          if tuple list, randomly picked in the interval
                          `[central_rate[0], central_rate[1])`, value is fraction of size to crop.
            size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.
                  The new size for the images.
            method: int, default 0. resize image shape method.
                    should be one of "0:bilinear", "1:nearest_neighbor", "2:bicubic", "3:area"
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            3-D / 4-D float Tensor, as per the input.
        Raises:
            ValueError: if central_crop_fraction is not within (0, 1].
        """
        return ImageAug(RandomCropCentralResize(self._image, central_rate, size, method, seed, _=True))
    
    def RandomCropPointResize(self, height_rate, width_rate, size, method=0, seed=None):
        """Crop the any region of the image(s) and resize specify shape.
    
        Crop region area = height_rate * width_rate *image_height * image_width

        This function works on either a single image (`image` is a 3-D Tensor), or a
        batch of images (`image` is a 4-D Tensor).

        Tips:
            method should be one of [0, 1, 2, 3], "0:bilinear", "1:nearest_neighbor", "2:bicubic", "3:area".
        Args:
            height_rate: flaot, in the interval (0, 1].
            width_rate: flaot, in the interval (0, 1].
            size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.
                  The new size for the images.
            method: int, default 0. resize image shape method.
                    should be one of "0:bilinear", "1:nearest_neighbor", "2:bicubic", "3:area"
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            3-D / 4-D float Tensor, as per the input.
        Raises:
            ValueError: if central_crop_fraction is not within (0, 1].
        """
        return ImageAug(RandomCropPointResize(self._image, height_rate, width_rate, size, method, seed, _=True))
    
    def Normalize(self, mean=None, std=None):
        """Normalize scales `image` to have mean and variance.
    
        This op computes `(x - mean) / std`.
        Args:
            mean: if None, computes image mean.
                  if int float, customize image all channels mean.
                  if tuple list, customize image each channels mean,
                  shape should 3 dims.
            std: if None, computes image std.
                 if int float, customize image all channels std.
                 if tuple list, customize image each channels std,
                 shape should 3 dims.
        Returns:
            The standardized image with same shape as `image`.
        Raises:
            ValueError: if the shape of 'image' is incompatible with this function.
        """
        return ImageAug(Normalize(self._image, mean, std, _=True))
    
    def RandomRescale(self, scale, seed=None):
        """Rescale apply to image.
    
        new pixel = image * scale
        Args:
            scale: if int float, value multiply with image.
                   if tuple list, randomly picked in the interval
                   `[central_rate[0], central_rate[1])`, value multiply with image.
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            3-D / 4-D float Tensor, as per the input.
        Raises:
            scale type error.
        """
        return ImageAug(RandomRescale(self._image, scale, seed, _=True))
    
    def RandomNoiseGaussian(self, scale=1, mean=0.0, std=1.0, seed=None):
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
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            3-D / 4-D float Tensor, as per the input.
        Raises:
            scale or lam type error.
        """
        return ImageAug(RandomNoiseGaussian(self._image, scale, mean, std, seed, _=True))
    
    def RandomNoisePoisson(self, scale=1, lam=1.0, seed=None):
        """Poisson noise apply to image.
    
        new pixel = image + poisson_noise * scale
        Args:
            scale: if int or float, value multiply with poisson_noise.
                   if tuple or list, randomly picked in the interval
                   `[scale[0], scale[1])`, value multiply with poisson_noise.
            lam: if int or float, value is poisson distribution lambda.
                 if tuple or list, randomly picked in the interval
                 `[lam[0], lam[1])`, value is poisson distribution lambda.
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            3-D / 4-D float Tensor, as per the input.
        Raises:
            scale or lam type error.
        """
        return ImageAug(RandomNoisePoisson(self._image, scale, lam, seed, _=True))
    
    def RandomNoiseMask(self, keep_prob=0.95, seed=None):
        """Mask noise apply to image.
    
        With probability `keep_prob`, outputs the input element scaled up by
        `1`, otherwise outputs `0`. 

        Tips:
            1 mean pixel have no change.
            a suitable interval is [0.9, 1].
        Args:
            image: Either a 3-D float Tensor of shape [height, width, depth], or a 4-D
                   Tensor of shape [batch_size, height, width, depth].
            keep_prob: should be in the interval (0, 1].
                       if int or float, the probability that each element is kept.
                       if tuple or list, randomly picked in the interval
                       `[keep_prob[0], keep_prob[1])`, the probability that each element is kept.
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            3-D / 4-D float Tensor, as per the input.
        Raises:
            ValueError: If `keep_prob` is not in `(0, 1]`.
        """
        return ImageAug(RandomNoiseMask(self._image, keep_prob, seed, _=True))
    
    def RandomNoiseSaltPepper(self, keep_prob=0.95, seed=None):
        """Salt-Pepper noise apply to image.
    
        The salt-pepper noise is based on the signal-to-noise ratio of the image,
        randomly generating the pixel positions in some images all channel,
        and randomly assigning these pixels to 0 or 255.

        Tips:
            1 mean pixel have no change.
            a suitable interval is [0.9, 1].
        Args:
            image: Either a 3-D float Tensor of shape [height, width, depth], or a 4-D
                   Tensor of shape [batch_size, height, width, depth].
            keep_prob: should be in the interval (0, 1].
                       if int or float, the probability that each element is kept.
                       if tuple or list, randomly picked in the interval
                       `[keep_prob[0], keep_prob[1])`, the probability that each element is kept.
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            3-D / 4-D float Tensor, as per the input.
        Raises:
            ValueError: If `keep_prob` is not in `(0, 1]`.
        """
        return ImageAug(RandomNoiseSaltPepper(self._image, keep_prob, seed, _=True))
    
    def RandomNoiseRainbow(self, keep_prob=0.95, seed=None):
        """Rainbowr noise apply to image.
    
        The rainbow noise is based on the signal-to-noise ratio of the image,
        randomly generating the pixel positions in some images,
        and randomly assigning these pixels to 0 or 255.

        Tips:
            1 mean pixel have no change.
            a suitable interval is [0.9, 1].
        Args:
            image: Either a 3-D float Tensor of shape [height, width, depth], or a 4-D
                   Tensor of shape [batch_size, height, width, depth].
            keep_prob: should be in the interval (0, 1].
                       if int or float, the probability that each element is kept.
                       if tuple or list, randomly picked in the interval
                       `[keep_prob[0], keep_prob[1])`, the probability that each element is kept.
            seed: A Python integer. Used to create a random seed.
                  See `tf.set_random_seed` for behavior.
        Returns:
            3-D / 4-D float Tensor, as per the input.
        Raises:
            ValueError: If `keep_prob` is not in `(0, 1]`.
        """
        return ImageAug(RandomNoiseRainbow(self._image, keep_prob, seed, _=True))
    
    def run(self):
        """return numpy array image."""
        return self._image if type(self._image)==np.ndarray else self._image.numpy()
    
    def show(self):
        """plot numpy array image."""
        return tf.keras.preprocessing.image.array_to_img((self._image if type(self._image)==np.ndarray else self._image.numpy()).astype('uint8'))
