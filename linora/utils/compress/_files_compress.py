import bz2
import gzip
import zipfile
import tarfile
import rarfile

import linora.gfile as gfile

__all__ = ['files_zip', 'files_tar', 'files_bz2']

def files_zip(files, zip_name):
    """Compression files to .zip.
    
    Args:
        files: str or list
               if str, files should be file path;
               if list, files should be file path list.
        zip_name: str, compression files name.
    Return:
        zip_name: str, compression files name.
    """
    assert isinstance(zip_name, str), '`zip_name` should be str.'
    if isinstance(files, str):
        assert not gfile.isdir(files), '`files` should be file path.'
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(files)
    elif isinstance(files, list):
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
            for file in files:
                assert not gfile.isdir(file), 'Elements in the list should be file path.'
                z.write(file)
    else:
        raise ValueError('`files` should be type of str or list.')
    return zip_name

def files_tar(files, tar_name):
    """Compression files to .tar.
    
    Args:
        files: str or list
               if str, files should be file path;
               if list, files should be file path list.
        tar_name: str, compression files name.
    Return:
        tar_name: str, compression files name.
    """
    assert isinstance(tar_name, str), '`tar_name` should be str.'
    if isinstance(files, str):
        assert not gfile.isdir(files), '`files` should be file path.'
        with tarfile.TarFile(tar_name, 'w') as t:
            t.add(files)
    elif isinstance(files, list):
        with tarfile.TarFile(tar_name, 'w') as t:
            for file in files:
                assert not gfile.isdir(file), 'Elements in the list should be file path.'
                t.add(file)
    else:
        raise ValueError('`files` should be type of str or list.')
    return tar_name

def files_bz2(files, bz2_name):
    """Compression files to .bz2.
    
    Args:
        files: str or list
               if str, files should be file path;
               if list, files should be file path list.
        zip_name: str, compression files name.
    Return:
        zip_name: str, compression files name.
    """
    assert isinstance(bz2_name, str), '`bz2_name` should be str.'
    if isinstance(files, str):
        assert not gfile.isdir(files), '`files` should be file path.'
        with bz2.BZ2File(bz2_name, 'w') as b:
            with open(files, 'rb') as f:
                b.write(f.read())
    elif isinstance(files, list):
        with bz2.BZ2File(bz2_name, 'w') as b:
            for file in files:
                assert not gfile.isdir(file), 'Elements in the list should be file path.'
                with open(file, 'rb') as f:
                    b.write(f.read())
    else:
        raise ValueError('`files` should be type of str or list.')
    return bz2_name
© 2022 GitHub, Inc.
Terms
