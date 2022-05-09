from linora.image._image_io_aug import ImageIoAug
from linora.image._image_color_aug import ImageColorAug
from linora.image._image_noise_aug import ImageNoiseAug
from linora.image._image_resize_aug import ImageResizeAug
from linora.image._image_rescale_aug import ImageRescaleAug
from linora.image._image_position_aug import ImagePositionAug
from linora.image._image_crop_aug import ImageCropAug
from linora.image._image_filter_aug import ImageFilterAug

__all__ = ['ImageAug']


class ImageAug(ImageIoAug, ImageColorAug, ImageNoiseAug, ImageResizeAug, 
               ImageRescaleAug, ImagePositionAug, ImageCropAug, 
               ImageFilterAug
              ):
    """General class of image enhancement, which is used for pipeline image processing
    
    Args:
        image: a PIL instance.
        p: probability that the image does this. Default value is 1.
    """
    def __init__(self, image=None, p=1):
        super(ImageAug, self).__init__(image, p)
