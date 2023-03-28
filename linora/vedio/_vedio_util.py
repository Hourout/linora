from linora.image._image_util import list_images

__all__ = ['list_vedios']


def list_audios(directory, file_format=('mpeg', 'mpg', 'dat', 'mp4', 'avi', 'mov', 'asf', 'wmv', 'mkv', 'flv', 'rmvb')):
    """Lists all vedios in a directory, including all subdirectories.
    
    Args:
        directory: string, absolute path to the directory.
        file_format: tuple of strings or single string, extensions of the vedios.
    Returns:
        a list of vedio paths.
    """
    return list_images(directory, file_format=file_format)