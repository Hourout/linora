from linora.image._image_filter import *

class ImageFilterAug(object):
    def __init__(self, image=None):
        self.image = image
    
    def filter_BoxBlur(self, radius=2):
        """
        Blurs the image by setting each pixel to the average value of the pixels 
        in a square box extending radius pixels in each direction. 
        Supports float radius of arbitrary size. Uses an optimized implementation 
        which runs in linear time relative to the size of the image for any radius value.

        Args:
        radius: Size of the box in one direction. Radius 0 does not blur, returns an identical image. 
                Radius 1 takes 1 pixel in each direction, i.e. 9 pixels in total.

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_BoxBlur(self.image, radius)
        return self

    def filter_GaussianBlur(self, radius=2):
        """Gaussian blur filter.

        Args:
        radius: Blur radius.

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_GaussianBlur(self.image, radius)
        return self

    def filter_UnsharpMask(self, radius=2, percent=150, threshold=3):
        """Unsharp mask filter.
        See Wikipedia’s entry on digital unsharp masking for an explanation of the parameters.

        Args:
        radius: Blur Radius
        percent: Unsharp strength, in percent
        threshold: Threshold controls the minimum brightness change that will be sharpened

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_UnsharpMask(self.image, radius, percent, threshold)
        return self

    def filter_Rank(self, size, rank):
        """Create a rank filter. 
        The rank filter sorts all pixels in a window of the given size, and returns the rank’th value.

        Args:
        size: The kernel size, in pixels.
        rank: What pixel value to pick. Use 0 for a min filter, 
              size * size / 2 for a median filter, size * size - 1 for a max filter, etc.

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_Rank(self.image, size, rank)
        return self

    def filter_Median(self, size=3):
        """Create a median filter. Picks the median pixel value in a window with the given size.

        Args:
        size: The kernel size, in pixels.

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_Median(self.image, size)
        return self

    def filter_Min(self, size=3):
        """Create a min filter. Picks the lowest pixel value in a window with the given size.

        Args:
        size: The kernel size, in pixels.

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_Min(self.image, size)
        return self

    def filter_Max(self, size=3):
        """Create a max filter. Picks the largest pixel value in a window with the given size.

        Args:
        size: The kernel size, in pixels.

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_Max(self.image, size)
        return self

    def filter_Mode(self, size=3):
        """Create a mode filter. Picks the most frequent pixel value in a box with the given size. 

        Pixel values that occur only once or twice are ignored; 
        if no pixel value occurs more than twice, 
        the original pixel value is preserved.

        Args:
        size: The kernel size, in pixels.

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_Mode(self.image, size)
        return self

    def filter_BLUR(self):
        """Normal blur.

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_BLUR(self.image)
        return self

    def filter_CONTOUR(self):
        """contour blur.

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_CONTOUR(self.image)
        return self

    def filter_DETAIL(self):
        """detall blur.

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_DETAIL(self.image)
        return self

    def filter_EDGE_ENHANCE(self):
        """Edge enhancement blur.

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_EDGE_ENHANCE(self.image)
        return self

    def filter_EDGE_ENHANCE_MORE(self):
        """Edge enhancement threshold blur.

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_EDGE_ENHANCE_MORE(self.image)
        return self

    def filter_EMBOSS(self):
        """emboss blur.

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_EMBOSS(self.image)
        return self

    def filter_FIND_EDGES(self):
        """Find the edge blur.

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_FIND_EDGES(self.image)
        return self

    def filter_SHARPEN(self):
        """Sharpen blur.

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_SHARPEN(self.image)
        return self

    def filter_SMOOTH(self):
        """Smooth blur.

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_SMOOTH(self.image)
        return self

    def filter_SMOOTH_MORE(self):
        """Smooth threshold blur.

        Returns:
                A Image instance. of the same type and shape as `image`. 
        """
        self.image = filter_SMOOTH_MORE(self.image)
        return self
