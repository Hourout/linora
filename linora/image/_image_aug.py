from linora.image._image_io_aug import ImageIoAug
from linora.image._image_color_aug import ImageColorAug
from linora.image._image_noise_aug import ImageNoiseAug
from linora.image._image_resize_aug import ImageResizeAug
from linora.image._image_rescale_aug import ImageRescaleAug
from linora.image._image_position_aug import ImagePositionAug

__all__ = ['ImageAug']

class ImageAug(ImageIoAug, ImageColorAug, ImageNoiseAug, ImageResizeAug, ImageRescaleAug, ImagePositionAug):
    def __init__(self, image=None):
        super(ImageAug, self).__init__(image)
