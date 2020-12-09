import os

__all__ = ['list_images']

def list_images(directory, file_format=('jpg', 'jpeg', 'bmp', 'png', 'ppm', 'tif', 'tiff')):
    """Lists all pictures in a directory, including all subdirectories.
    Args:
        directory: string, absolute path to the directory
        file_format: tuple of strings or single string, extensions of the pictures
    Returns:
        a list of paths
    """
    file_format = tuple('.%s' % e for e in ((file_format,) if isinstance(file_format, str) else file_format))
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if f.lower().endswith(file_format)]
