from linora.image._image_color import *

class ImageColorAug(object):
    def __init__(self, image=None):
        self.image = image
    
    def enhance_color(self, delta):
        """Adjust image color balance.
        This class can be used to adjust the colour balance of an image, 
        in a manner similar to the controls on a colour TV set. 
        An enhancement factor of 0.0 gives a black and white image. 
        A factor of 1.0 gives the original image.

        Args:
        image: a Image instance.
        delta: A floating point value controlling the enhancement. 
               delta 1.0 always returns a copy of the original image, 
               lower factors mean less color, 
               and higher values more. There are no restrictions on this value.
               if list, tuple, randomly picked in the interval
                   `[delta[0], delta[1])` , value is float multiplier for adjusting color.
        Returns:
            a Image instance.
        """
        self.image = enhance_color(self.image, delta)
        return self
    
    def enhance_contrast(self, delta):
        """Adjust contrast of RGB or grayscale images.
  
        Contrast is adjusted independently for each channel of each image.

        For each channel, this Ops computes the mean of the image pixels in the
        channel and then adjusts each component `x` of each pixel to
        `(x - mean) * delta + mean`.

        Tips:
            1 means pixel value no change.
            0 means all pixel equal. 
            a suitable interval is (0, 4].
        Args:
            images: a Image instance.
            delta: if int, float, a float multiplier for adjusting contrast.
                   if list, tuple, randomly picked in the interval
                   `[delta[0], delta[1])` , value is float multiplier for adjusting contrast.
        Returns:
            a Image instance.
        """
        self.image = enhance_contrast(self.image, delta)
        return self
    
    def enhance_brightness(self, delta):
        """Adjust the brightness of RGB or Grayscale images.
    
        Tips:
            delta extreme value in the interval [-1, 1], >1 to white, <-1 to black.
            a suitable interval is [-0.5, 0.5].
            0 means pixel value no change.
        Args:
            image: Tensor or array. An image.
            delta: if int, float, Amount to add to the pixel values.
                   if list, tuple, randomly picked in the interval
                   `[delta[0], delta[1])` to add to the pixel values.
        Returns:
            a Image instance.
        """
        self.image = enhance_brightness(self.image, delta)
        return self

    def enhance_sharpness(self, delta):
        """Adjust image sharpness.
        This class can be used to adjust the sharpness of an image. 
        An enhancement factor of 0.0 gives a blurred image, 
        a factor of 1.0 gives the original image, 
        and a factor of 2.0 gives a sharpened image.

        Args:
        image: a Image instance.
        delta: A floating point value controlling the enhancement. 
               delta 1.0 always returns a copy of the original image, 
               lower factors mean less sharpness, 
               and higher values more. There are no restrictions on this value.
               if list, tuple, randomly picked in the interval
                   `[delta[0], delta[1])` , value is float multiplier for adjusting color.
        Returns:
            a Image instance.
        """
        self.image = enhance_sharpness(self.image, delta)
        return self
    
    def color_invert(self):
        """
        Invert colors of input PIL image.

        Args:
            image (PIL image): Image to be color inverted.

        Returns:
            image (PIL image), Color inverted image.

        """
        self.image = color_invert(self.image)
        return self

    def equalize(self):
        """
        Equalize the image histogram. This function applies a non-linear
        mapping to the input image, in order to create a uniform
        distribution of grayscale values in the output image.

        Args:
            img (PIL image): Image to be equalized

        Returns:
            img (PIL image), Equalized image.

        """
        self.image = equalize(self.image)
        return self
