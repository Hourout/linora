import io

from PIL import Image

__all__ = ['read_image', 'save_image']

def read_image(filename):
    """
    Args:
        filename: str, image absolute path.
    Returns:
        a Image instance.
    """
    with open(filename, 'rb') as f:
        image = Image.open(io.BytesIO(f.read()))
    return image

def save_image(filename, image, file_format=None, **kwargs):
    """Saves an image stored as a Numpy array to a path or file object.
    Args
        filename: Path or file object.
        image: A PIL Image instance.
        file_format: Optional file format override. If omitted, the
            format to use is determined from the filename extension.
            If a file object was used instead of a filename, this
            parameter should always be used.
        **kwargs: Additional keyword arguments passed to `PIL.Image.save()`.
    """
    if image.mode == 'RGBA' and file_format in ['jpg', 'jpeg']:
#         warnings.warn('The JPG format does not support RGBA images, converting to RGB.')
        image = image.convert('RGB')
    image.save(filename, format=file_format, **kwargs)
