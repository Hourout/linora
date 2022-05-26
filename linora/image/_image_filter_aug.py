from linora.image._image_filter import *

class ImageFilterAug(object):
    def __init__(self, image=None):
        self.image = image
    
    def filters(self, mode=FilterMode.Mean, p=None, **kwarg):
        """
        la.image.FilterMode.Box:
            Blurs the image by setting each pixel to the average value of the pixels 
            in a square box extending radius pixels in each direction. 
            Supports float radius of arbitrary size. Uses an optimized implementation 
            which runs in linear time relative to the size of the image for any radius value.

            you should append param `radius`
            radius: Size of the box in one direction. Radius 0 does not blur, returns an identical image. 
                    Radius 1 takes 1 pixel in each direction, i.e. 9 pixels in total.
            eg.
            la.image.filters(image, mode=la.image.FilterMode.Box, radius=2)

        la.image.FilterMode.Gaussian:
            Gaussian blur filter.

            you should append param `radius`
            radius: Blur radius.
            eg.
            la.image.filters(image, mode=la.image.FilterMode.Gaussian, radius=2)

        la.image.FilterMode.Unsharpmask:
            Unsharp mask filter.
            See Wikipedia’s entry on digital unsharp masking for an explanation of the parameters.

            you should append param `radius`,`percent`,`threshold`
            radius: Blur radius.
            percent: Unsharp strength, in percent
            threshold: Threshold controls the minimum brightness change that will be sharpened
            eg.
            la.image.filters(image, mode=la.image.FilterMode.Unsharpmask, radius=2, percent=150, threshold=3)

        la.image.FilterMode.Rank:
            Create a rank filter. The rank filter sorts all pixels in a window of the given size, and returns the rank’th value.

            you should append param `size`,`rank`
            size: The kernel size, in pixels.
            rank: What pixel value to pick. Use 0 for a min filter, 
                  size * size / 2 for a median filter, size * size - 1 for a max filter, etc.
            eg.
            la.image.filters(image, mode=la.image.FilterMode.Rank, size, rank)

        la.image.FilterMode.Median:
            Create a median filter. Picks the median pixel value in a window with the given size.

            you should append param `size`
            size: The kernel size, in pixels.
            eg.
            la.image.filters(image, mode=la.image.FilterMode.Median, size=3)

        la.image.FilterMode.Min:
            Create a min filter. Picks the lowest pixel value in a window with the given size.

            you should append param `size`
            size: The kernel size, in pixels.
            eg.
            la.image.filters(image, mode=la.image.FilterMode.Min, size=3)

        la.image.FilterMode.Max:
            Create a max filter. Picks the largest pixel value in a window with the given size.

            you should append param `size`
            size: The kernel size, in pixels.
            eg.
            la.image.filters(image, mode=la.image.FilterMode.Max, size=3)

        la.image.FilterMode.Mode:
            Create a mode filter. Picks the most frequent pixel value in a box with the given size. 

            Pixel values that occur only once or twice are ignored; 
            if no pixel value occurs more than twice, the original pixel value is preserved.

            you should append param `size`
            size: The kernel size, in pixels.
            eg.
            la.image.filters(image, mode=la.image.FilterMode.Mode, size=3)

        la.image.FilterMode.Mean:
            Normal blur.
            eg.
            la.image.filters(image, mode=la.image.FilterMode.Mean)

        la.image.FilterMode.CONTOUR:
            contour blur.
            eg.
            la.image.filters(image, mode=la.image.FilterMode.CONTOUR)

        la.image.FilterMode.DETAIL:
            detail blur.
            eg.
            la.image.filters(image, mode=la.image.FilterMode.DETAIL)

        la.image.FilterMode.EDGE_ENHANCE:
            Edge enhancement blur.
            eg.
            la.image.filters(image, mode=la.image.FilterMode.EDGE_ENHANCE)
            
        la.image.FilterMode.EDGE_ENHANCE_MORE:
            Edge enhancement more blur.
            eg.
            la.image.filters(image, mode=la.image.FilterMode.EDGE_ENHANCE_MORE)

        la.image.FilterMode.EMBOSS:
            emboss blur.
            eg.
            la.image.filters(image, mode=la.image.FilterMode.EMBOSS)

        la.image.FilterMode.FIND_EDGES:
            Find the edge blur.
            eg.
            la.image.filters(image, mode=la.image.FilterMode.FIND_EDGES)

        la.image.FilterMode.SHARPEN:
            Sharpen blur.
            eg.
            la.image.filters(image, mode=la.image.FilterMode.SHARPEN)

        la.image.FilterMode.SMOOTH:
            Smooth blur.
            eg.
            la.image.filters(image, mode=la.image.FilterMode.SMOOTH)

        la.image.FilterMode.SMOOTH_MORE:
            Smooth threshold blur.
            eg.
            la.image.filters(image, mode=la.image.FilterMode.SMOOTH_MORE)

        Args:
            mode: la.image.FilterMode.
            p: probability that the image does this. Default value is 1.
        """
        if self._max_aug_nums>0:
            if self._nums>self._max_aug_nums:
                return self
            self._nums += 1
        if p is None:
            p = self._p
        self.image = filters(self.image, mode, p, **kwarg)
        return self
