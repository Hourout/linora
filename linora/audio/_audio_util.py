from linora.image._image_util import list_images

__all__ = ['list_audios']


def list_audios(directory, file_format=('mp3', 'flac', 'wav', 'ogg', 'ape')):
    """Lists all audios in a directory, including all subdirectories.
    
    Args:
        directory: string, absolute path to the directory.
        file_format: tuple of strings or single string, extensions of the audios.
    Returns:
        a list of audio paths.
    """
    return list_images(directory, file_format=file_format)