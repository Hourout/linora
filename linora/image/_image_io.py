import io
import base64

from PIL import Image

__all__ = ['read_image', 'save_image', 'encode_base64', 'decode_base64']

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
        image = image.convert('RGB')
    image.save(filename, format=file_format, **kwargs)

def encode_base64(file):
    """encode image to string.
    Args
        file: image file path.
    """
    with open(file, "rb")as f:
        bs64 = base64.b64encode(f.read()).decode()
    return bs64

def decode_base64(file, image_str):
    """decode image to file.
    Args
        file: image file path.
        image_str: image bites string.
    """
    with open(file, "wb") as f:
        f.write(base64.b64decode(image_str))
