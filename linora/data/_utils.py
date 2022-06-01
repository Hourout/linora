import requests

from linora import gfile
from linora.utils._progbar import Progbar

__all__ = ['get_file']


def assert_dirs(root, root_dir=None, delete=True, make_root_dir=True):
    if root is None:
        root = './'
    assert gfile.isdir(root), '{} should be directory.'.format(root)
    if root_dir is not None:
        assert isinstance(root_dir, str), '{} should be str.'.format(root_dir)
        task_path = gfile.path_join(root, root_dir)
        if gfile.exists(task_path):
            if delete:
                gfile.remove(task_path)
                gfile.makedirs(task_path)
        else:
            if make_root_dir:
                gfile.makedirs(task_path)
        return task_path
    else:
        gfile.makedirs(root)
        return root


def get_file(url, root_file, verbose=1, retries=3, chunk_size=5120):
    """Request url and download to root_file.
    
    Args:
        url: str, request url.
        root_file: str, downloaded and saved file name.
        verbose: Verbosity mode, 0 (silent), 1 (verbose)
        retries: retry counts.
        chunk_size: the number of bytes it should read into memory.
    Return:
        str, downloaded and saved file name.
    """
    for i in range(retries):
        try:
            r = requests.get(url, stream=True)
            content_type = r.headers.get('Content-Length')
            total_size = None if content_type is None else int(content_type.strip())
            p = Progbar(total_size, verbose=verbose)
            down_size = 0
            with open(root_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size):
                    p.add(chunk_size)
                    f.write(chunk)
                    down_size += len(chunk)
            if down_size==total_size:
                break
            raise 'download failed'
        except:
            if i==retries-1:
                raise f'{url} download failed'
    return root_file