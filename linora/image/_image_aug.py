from linora.image._image_io_aug import ImageIoAug
from linora.image._image_color_aug import ImageColorAug
from linora.image._image_noise_aug import ImageNoiseAug
from linora.image._image_resize_aug import ImageResizeAug
from linora.image._image_rescale_aug import ImageRescaleAug
from linora.image._image_position_aug import ImagePositionAug
from linora.image._image_crop_aug import ImageCropAug
from linora.image._image_filter_aug import ImageFilterAug
from linora.image._image_draw_aug import ImageDrawAug

__all__ = ['ImageAug']


class ImageAug(ImageIoAug, ImageColorAug, ImageNoiseAug, ImageResizeAug, 
               ImageRescaleAug, ImagePositionAug, ImageCropAug, 
               ImageFilterAug, ImageDrawAug
              ):
    """General class of image enhancement, which is used for pipeline image processing
    
    Args:
        image: a PIL instance.
        max_aug_nums: max image aug numbers, only valid for methods with parameter p.
        p: probability that the image does this. Default value is 1.
    """
    def __init__(self, image=None, max_aug_nums=None, p=1):
        super(ImageAug, self).__init__(image=image)
        if max_aug_nums is not None:
            if max_aug_nums>0:
                self._max_aug_nums = max_aug_nums
                self._nums = 1
            else:
                raise ValueError('`max_aug_nums` value error.')
        else:
            self._max_aug_nums = 0
        self._p = p
