from PIL import Image

__all__ = ['ResizeMethod', 'resize']

class ResizeMethod:
    NEAREST = Image.NEAREST
    BOX = Image.BOX
    BILINEAR = Image.BILINEAR
    HAMMING = Image.HAMMING
    BICUBIC = Image.BICUBIC
    LANCZOS = Image.LANCZOS
    
def resize(image, size, method=ResizeMethod.BILINEAR):
    if image.mode in ['1', 'P'] and method!=ResizeMethod.NEAREST:
        method = ResizeMethod.NEAREST
    return image.resize(size, resample=method)
