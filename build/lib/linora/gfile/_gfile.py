import os
import shutil

__all__ = ['copy', 'exists', 'isdir', 'isfile', 'listdir', 'makedirs', 'path_join', 'remove', 'rename', 'stat', 'walk',
           ]


def exists(path):
    """Determines whether a path exists or not.
    Args:
        path: string, a path, filepath or dirpath.
    Returns:
        True if the path exists, whether it's a file or a directory. 
        False if the path does not exist and there are no filesystem errors.
    """
    return os.path.exists(path)


def isdir(path):
    """Returns whether the path is a directory or not.
    
    Args:
        path: string, path to a potential directory.
    Returns:
        True, if the path is a directory; False otherwise.
    """
    return os.path.isdir(path)


def isfile(path):
    """Returns whether the path is a regular file or not.
    
    Args:
        path: string, path to a potential file.
    Returns:
        True, if the path is a regular file; False otherwise.
    """
    return os.path.isfile(path)


def listdir(path, full_path=False):
    """Returns a list of entries contained within a directory.
    
    Args:
        path: string, path to a directory.
    Returns:
        [filename1, filename2, ... filenameN] as strings.
    Raises:
        errors. NotFoundError if directory doesn't exist.
    """
    return [path_join(path, i) for i in os.listdir(path)] if full_path else os.listdir(path)


def copy(src, dst, overwrite=False):
    """Copies data from src to dst.
    
    1.copy file to file.  eg:'/data/A/a.txt' to '/data/B/b.txt' --> exist '/data/B/b.txt'
    2.copy file to directory.  eg:'/data/A/a.txt' to '/data/B'  --> exist '/data/B/a.txt'
    3.copy directory to directory. eg:'/data/A' to '/data/B'    --> exist '/data/B/A'

    Args:
        src: string, name of the file or directory whose contents need to be copied
        dst: string, name of the file or directory to which to copy to
        overwrite: boolean, Whether to overwrite the file if existing file.
    """
    assert exists(src), "src not exists."
    if isfile(src):
        if overwrite or not exists(dst):
            return shutil.copy(src, dst)
        else:
            if isdir(dst):
                if not exists(path_join(dst, path_join(src).split('/')[-1])):
                    return shutil.copy(src, dst)
    else: 
        makedirs(dst)
        path = path_join(dst, path_join(src).rstrip('/').split('/')[-1])
        if overwrite or not exists(path):
            remove(path)
            return shutil.copytree(src, path)


def makedirs(path):
    """Creates a directory and all parent/intermediate directories.
    
    Args:
        path: string, name of the directory to be created.
    """
    if not exists(path):
        os.makedirs(path)


def remove(path):
    """Deletes a directory or file.
    
    Args:
        path: string, a path, filepath or dirpath.
    Raises:
        errors. NotFoundError if directory or file doesn't exist.
    """
    if exists(path):
        if isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def rename(src, dst, overwrite=False):
    """Rename or move a file / directory.
    
    Args:
        src: string, pathname for a file or dir.
        dst: string, pathname to which the file needs to be moved.
        overwrite: boolean, Whether to overwrite the file if existing file.
    """
    assert exists(src), "src not exists."
    if exists(dst):
        assert isdir(src)==isdir(dst), "src and dst should same type."
        assert isfile(src)==isfile(dst), "src and dst should same type."
        if overwrite:
            shutil.rmtree(dst)
            os.rename(src, dst)
    else:
        os.rename(src, dst)
    

def stat(path):
    """Returns file or directory statistics for a given path.
    
    Args:
        path: string, a path, filepath or dirpath.
    Returns:
        FileStatistics struct that contains information about the path.
    """
    return os.stat(path)


def walk(path, topdown=True, onerror=None):
    """Recursive directory tree generator for directories.
    
    Args:
        path: string, a Directory name
        topdown: bool, Traverse pre order if True, post order if False.
        onerror: optional handler for errors. Should be a function, 
                 it will be called with the error as argument. 
                 Rethrowing the error aborts the walk. 
                 Errors that happen while listing directories are ignored.
    
    Returns:
        Yields, Each yield is a 3-tuple: the pathname of a directory, 
        followed by lists of all its subdirectories and leaf files. 
        That is, each yield looks like: (dirname, [subdirname, subdirname, ...], [filename, filename, ...]). 
        Each item is a string.
    """
    return list(os.walk(top, topdown, onerror))


def path_join(path, *paths):
    """
    Join two or more pathname components, inserting '/' as needed.
    If any component is an absolute path, all previous path components 
    will be discarded.  An empty last part will result in a path that 
    ends with a separator.
    
    Args:
        path: file or directory
        *paths: file or directory
    Return:
        a new path
    """
    return eval(repr(os.path.join(path, *paths)).replace("\\", '/').replace("//", '/'))
