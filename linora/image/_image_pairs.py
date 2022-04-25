from PIL import ImageChops

__all__ = ['pairs', 'PairsMode']


class pairs_mode:
    """pairs image computing method type."""
    Add               = 'add'
    Add_modulo        = 'add_modulo'
    Blend             = 'blend'
    Composite         = 'composite'
    Darker            = 'darker'
    Difference        = 'difference'
    Lighter           = 'lighter'
    Logical_and       = 'logical_and'
    Logical_or        = 'logical_or'
    Logical_xor       = 'logical_xor'
    Multiply          = 'multiply'
    SoftLight         = 'soft_light'
    HardLight         = 'hard_light'
    Overlay           = 'overlay'
    Screen            = 'screen'
    Subtract          = 'subtract'
    Subtract_modulo   = 'subtract_modulo'

    
PairsMode = pairs_mode()


def pairs(image1, image2, mode, **kwargs):
    """
    la.image.PairsMode.Add
        Adds two images, dividing the result by scale and adding the offset. If omitted, scale defaults to 1.0, and offset to 0.0.
        out = ((image1 + image2) / scale + offset)
        
        you should append param `scale` and `offset`
        scale: int or float, defaults to 1.0
        offset: int or float, defaults to 0
        eg.
        la.image.pairs(image1, image2, mode=la.image.PairsMode.Add, scale=1.0, offset=0)
    
    la.image.PairsMode.Add_modulo
        Add two images, without clipping the result.
        out = ((image1 + image2) % 255)
        eg.
        la.image.pairs(image1, image2, mode=la.image.PairsMode.Add_modulo)
        
    la.image.PairsMode.Blend
        Creates a new image by interpolating between two input images.

        using a constant alpha.
        out = image1 * (1.0 - alpha) + image2 * alpha

        If alpha is 0.0, a copy of the first image is returned. 
        If alpha is 1.0, a copy of the second image is returned. 
        There are no restrictions on the alpha value. 
        If necessary, the result is clipped to fit into the allowed output range.
        
        you should append param `alpha`
        alpha: The interpolation alpha factor.
        eg.
        la.image.pairs(image1, image2, mode=la.image.PairsMode.Blend, alpha=0.2)
        
    la.image.PairsMode.Composite
        Create composite image by blending images using a transparency mask.
        
        you should append param `mask`
        mask: A mask image. This image can have mode “1”, “L”, or “RGBA”, and must have the same size as the other two images.
        eg.
        la.image.pairs(image1, image2, la.image.PairsMode.Composite, mask)
        
    la.image.PairsMode.Darker
        Compares the two images, pixel by pixel, and returns a new image containing the darker values.
        out = min(image1, image2)
        
        eg.
        la.image.pairs(image1, image2, mode=la.image.PairsMode.Darker)
        
    la.image.PairsMode.Difference
        Returns the absolute value of the pixel-by-pixel difference between the two images.
        out = abs(image1 - image2)
        
        eg.
        la.image.pairs(image1, image2, mode=la.image.PairsMode.Difference)
    
    la.image.PairsMode.Lighter
        Compares the two images, pixel by pixel, and returns a new image containing the lighter values.
        out = max(image1, image2)
        
        eg.
        la.image.pairs(image1, image2, mode=la.image.PairsMode.Lighter)
        
    la.image.PairsMode.Logical_and
        Logical AND between two images.
        Both of the images must have mode “1”. 
        If you would like to perform a logical AND on an image with a mode other than “1”, 
        try multiply() instead, using a black-and-white mask as the second image.
        out = ((image1 and image2) % 255)
        
        eg.
        la.image.pairs(image1, image2, mode=la.image.PairsMode.Logical_and)
        
    la.image.PairsMode.Logical_or
        Logical OR between two images.
        Both of the images must have mode “1”.
        out = ((image1 or image2) % 255)
        
        eg.
        la.image.pairs(image1, image2, mode=la.image.PairsMode.Logical_or)
        
    la.image.PairsMode.Logical_xor
        Logical XOR between two images.
        Both of the images must have mode “1”.
        out = ((bool(image1) != bool(image2)) % 255)
        
        eg.
        la.image.pairs(image1, image2, mode=la.image.PairsMode.Logical_xor)
        
    la.image.PairsMode.multiply
        Superimposes two images on top of each other.
        If you multiply an image with a solid black image, the result is black. 
        If you multiply with a solid white image, the image is unaffected.
        out = image1 * image2 / 255
        
        eg.
        la.image.pairs(image1, image2, mode=la.image.PairsMode.multiply)
        
    la.image.PairsMode.SoftLight
        Superimposes two images on top of each other using the Soft Light algorithm
        eg.
        la.image.pairs(image1, image2, mode=la.image.PairsMode.SoftLight)
        
    la.image.PairsMode.HardLight
        Superimposes two images on top of each other using the Hard Light algorithm
        eg.
        la.image.pairs(image1, image2, mode=la.image.PairsMode.HardLight)
        
    la.image.PairsMode.Overlay
        Superimposes two images on top of each other using the Overlay algorithm
        eg.
        la.image.pairs(image1, image2, mode=la.image.PairsMode.Overlay)
        
    la.image.PairsMode.Screen
        Superimposes two inverted images on top of each other.
        out = 255 - ((255 - image1) * (255 - image2) / 255)
        eg.
        la.image.pairs(image1, image2, mode=la.image.PairsMode.Screen)
        
    la.image.PairsMode.Subtract
        Subtracts two images, dividing the result by scale and adding the offset. If omitted, scale defaults to 1.0, and offset to 0.0.
        out = ((image1 - image2) / scale + offset)
        
        you should append param `scale` and `offset`
        scale: int or float, defaults to 1.0
        offset: int or float, defaults to 0
        eg.
        la.image.pairs(image1, image2, mode=la.image.PairsMode.Subtract, scale=1.0, offset=0)
        
    la.image.PairsMode.Subtract_modulo
        Subtract two images, without clipping the result.
        out = ((image1 - image2) % 255)
        eg.
        la.image.pairs(image1, image2, mode=la.image.PairsMode.Subtract_modulo)


    Args:
        image1: a PIL instance. The first image.
        image2: a PIL instance. The second image.  Must have the same mode and size as the first image.
        mode: la.image.PairsMode
    Return:
        a PIL instance.
    """
    if 'scale' not in kwargs:
        kwargs['scale'] = 1.
    if 'offset' not in kwargs:
        kwargs['offset'] = 0
    if mode=='add':
        return ImageChops.add(image1, image2, scale=kwargs['scale'], offset=kwargs['offset'])
    elif mode=='add_modulo':
        return ImageChops.add_modulo(image1, image2)
    elif mode=='blend':
        if 'alpha' not in kwargs:
            raise ValueError("Missing parameter `alpha`")
        return ImageChops.blend(image1, image2, alpha=kwargs['alpha'])
    elif mode=='composite':
        if 'mask' not in kwargs:
            raise ValueError("Missing parameter `mask`")
        return ImageChops.composite(image1, image2, mask=kwargs['mask'])
    elif mode=='darker':
        return ImageChops.darker(image1, image2)
    elif mode=='difference':
        return ImageChops.difference(image1, image2)
    elif mode=='lighter':
        return ImageChops.lighter(image1, image2)
    elif mode=='logical_and':
        return ImageChops.logical_and(image1, image2)
    elif mode=='logical_or':
        return ImageChops.logical_or(image1, image2)
    elif mode=='logical_xor':
        return ImageChops.logical_xor(image1, image2)
    elif mode=='multiply':
        return ImageChops.multiply(image1, image2)
    elif mode=='soft_light':
        return ImageChops.soft_light(image1, image2)
    elif mode=='hard_light':
        return ImageChops.hard_light(image1, image2)
    elif mode=='overlay':
        return ImageChops.overlay(image1, image2)
    elif mode=='screen':
        return ImageChops.screen(image1, image2)
    elif mode=='subtract':
        return ImageChops.subtract(image1, image2, scale=kwargs['scale'], offset=kwargs['offset'])
    elif mode=='subtract_modulo':
        return ImageChops.subtract_modulo(image1, image2)
    else:
        raise ValueError("mode must be la.image.PairsMode param")