import os

from linora.parallel import ProcessLoom
from linora.image._image_util import *
from linora.image._image_io import read_image

__all__ = ['mean_std']

def mean_std(image_file, mode=True):
    """
    Args:
        image_file: str, image path or image folder, image channel should be 3 or 4 channels.
        mode: int or bool, True is 3 channels and False is all channels.
    Returns:
        a dict about mean and std of `image_file`.
    """
    def image_map(path, mode):
        image = read_image(path)
        image = color_convert(image, ColorMode.RGB)
        image = image_to_array(image).reshape(-1, 3)
        return [image.mean(axis=0), image.std(axis=0)] if mode else [image.mean(), image.std()]
    if os.path.isdir(image_file):
        loom = ProcessLoom()
        for i in list_images(image_file):
            loom.add_function(image_map, [i, mode])
        t = loom.execute()
        if mode:
            mean = [[t[i]['output'][0][0] for i in t], [t[i]['output'][0][1] for i in t], [t[i]['output'][0][2] for i in t]]
            std = [[t[i]['output'][1][0] for i in t], [t[i]['output'][1][1] for i in t], [t[i]['output'][1][2] for i in t]]
            result = {'mean':[round(sum(i)/len(i),3) for i in mean], 'std':[round(sum(i)/len(i),3) for i in std]}
        else:
            mean = [t[i]['output'][0] for i in t]
            std = [t[i]['output'][1] for i in t]
            result = {'mean':round(sum(mean)/len(mean),3), 'std':round(sum(std)/len(std),3)}
    else:
        result = image_map(image_file, mode)
        if mode:
            result = {'mean':[round(i,3) for i in result[0].tolist()], 'std':[round(i,3) for i in result[1].tolist()]}
        else:
            result = {'mean':round(result[0],3), 'std':round(result[1],3)}
    return result
