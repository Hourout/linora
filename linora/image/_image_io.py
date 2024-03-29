import io
import base64

import requests
from PIL import Image, ImageOps

from linora.gfile._gfile import exists

__all__ = ['read_image', 'save_image', 'encode_base64', 'decode_base64']


def read_image(filename):
    """Reads the contents of file to a PIL Image instance.
    
    Args:
        filename: str, image absolute path, or io.BytesIO stream to image file.
    Returns:
        a PIL instance.
    """
    if isinstance(filename, str):
        if exists(filename):
            with open(filename, 'rb') as f:
                image = Image.open(io.BytesIO(f.read()))
        else:
            image = Image.open(io.BytesIO(requests.get(filename).content), 'r')
    elif isinstance(filename, io.BytesIO):
        image = Image.open(filename)
    else:
        raise TypeError('path should be absolute path or io.BytesIO')
    return ImageOps.exif_transpose(image)


def save_image(filename, image, file_format=None, **kwargs):
    """Saves an image stored as a Numpy array to a path or file object.
    
    if save gif image, please use save_image(filename, image, file_format=None, save_all=True, append_images=[im1, im2, ...])
    
    Args
        filename: Path or file object.
        image: A PIL instance.
        file_format: Optional file format override. If omitted, the
            format to use is determined from the filename extension.
            If a file object was used instead of a filename, this
            parameter should always be used.
        **kwargs: Additional keyword arguments passed to `PIL.Image.save()`.
        if save gif, param `duration` and `loop` is optional.
    """
    if file_format is None:
        file_format = filename.split('.')[-1]
    if image.mode == 'RGBA' and file_format in ['jpg', 'jpeg']:
        image = image.convert('RGB')
    if filename.lower().endswith('.gif'):
        assert isinstance(image, list), '`image` must be image of list.'
        duration = kwargs['duration'] if 'duration' in kwargs else 100
        loop = kwargs['loop'] if 'loop' in kwargs else 0
        image[0].save(filename, format='GIF', append_images=image[1:], save_all=True, duration=duration, loop=loop)
    else:
        image.save(filename, format=file_format, **kwargs)


def encode_base64(filename):
    """encode image to string.
    
    Args
        filename: image file path.
    Returns:
        a bites string.
    """
    with open(filename, "rb")as f:
        bs64 = base64.b64encode(f.read()).decode()
    return bs64


def decode_base64(filename, image_str):
    """decode image to file.
    
    Args
        filename: image file path.
        image_str: image bites string.
    """
    with open(filename, "wb") as f:
        f.write(base64.b64decode(image_str))
