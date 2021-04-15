from PIL import ImageFilter

__all__ = ['filter_BoxBlur', 'filter_GaussianBlur', 'filter_UnsharpMask', 'filter_Rank', 'filter_Median', 'filter_Min',
           'filter_Max', 'filter_Mode', 'filter_BLUR', 'filter_CONTOUR', 'filter_DETAIL', 'filter_EDGE_ENHANCE', 
           'filter_EDGE_ENHANCE_MORE', 'filter_EMBOSS', 'filter_FIND_EDGES', 'filter_SHARPEN', 'filter_SMOOTH', 
           'filter_SMOOTH_MORE']

def filter_BoxBlur(image, radius=2):
    return image.filter(ImageFilter.BoxBlur(radius))

def filter_GaussianBlur(image, radius=2):
    return image.filter(ImageFilter.GaussianBlur(radius))

def filter_UnsharpMask(image, radius=2, percent=150, threshold=3):
    return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

def filter_Rank(image, size, rank):
    return image.filter(ImageFilter.RankFilter(size, rank))

def filter_Median(image, size=3):
    return image.filter(ImageFilter.MedianFilter(size))

def filter_Min(image, size=3):
    return image.filter(ImageFilter.MinFilter(size))

def filter_Max(image, size=3):
    return image.filter(ImageFilter.MaxFilter(size))

def filter_Mode(image, size=3):
    return image.filter(ImageFilter.ModeFilter(size))

def filter_BLUR(image):
    return image.filter(ImageFilter.BLUR)

def filter_CONTOUR(image):
    return image.filter(ImageFilter.CONTOUR)

def filter_DETAIL(image):
    return image.filter(ImageFilter.DETAIL)

def filter_EDGE_ENHANCE(image):
    return image.filter(ImageFilter.EDGE_ENHANCE)

def filter_EDGE_ENHANCE_MORE(image):
    return image.filter(ImageFilter.EDGE_ENHANCE_MORE)

def filter_EMBOSS(image):
    return image.filter(ImageFilter.EMBOSS)

def filter_FIND_EDGES(image):
    return image.filter(ImageFilter.FIND_EDGES)

def filter_SHARPEN(image):
    return image.filter(ImageFilter.SHARPEN)

def filter_SMOOTH(image):
    return image.filter(ImageFilter.SMOOTH)

def filter_SMOOTH_MORE(image):
    return image.filter(ImageFilter.SMOOTH_MORE)
