import os
import bz2
import gzip
import zipfile
import tarfile
import rarfile

import linora.gfile as gfile

__all__ = ['decompress', 'compress_folder', 'compress_files']

def decompress(file, folder=None):
    """Decompression file.
    
    Args:
        file: str, file should be file path;
        folder: str, decompression folder.
    Return:
        folder: str, decompression folder.
    """
    mat = file.split('.')[-1]
    if folder is None:
        folder = file[:len(mat)+1]
    if mat in ['gz']:
        with gzip.GzipFile(file) as g:
            with open(folder, "w+") as f:
                f.write(g.read())
    elif mat in ['tar', 'tgz']:
        with tarfile.open(file, 'r') as g:
            names = g.getnames()
            for name in names:
                g.extract(name, folder)
    elif mat in ['zip']:
        with zipfile.ZipFile(file, 'r') as g:
            names = g.namelist()
            for name in names:
                g.extract(name, folder)
    elif mat in ['rar']:
        with rarfile.RarFile(file, 'r') as g:
            names = g.namelist()
            for name in names:
                g.extract(name, folder)
    elif mat in ['bz2']:
        with bz2.BZ2File(file, 'r') as g:
            with open(folder, 'w') as f:
                f.write(g.read())
    else:
        raise ValueError("`file` should be type of ['.gz', '.tar', '.tgz', '.zip', '.rar', '.bz2'].")
    return folder

def compress_folder(folder, file):
    """Compress all files in the folder to file.
    
    Args:
        folder: str, folder should be folder path.
        file: str, compression files name.
    Return:
        file: str, compression files name.
    """
    assert gfile.isdir(folder), '`folder` should be folder path.'
    mat = file.split('.')[-1]
    if mat in ['zip']:
        with zipfile.ZipFile(file, 'w', zipfile.ZIP_DEFLATED) as z:
            for dirpath, dirnames, filenames in gfile.walk(folder):
                fpath = dirpath.replace(startdir, '')
                fpath = fpath and fpath + os.sep or ''
                for filename in filenames:
                    z.write(gfile.path_join(dirpath, filename), gfile.path_join(fpath, filename))
    elif mat in ['tar']:
        with tarfile.open(file, 'w') as tar:
            for dirpath, dirnames, filenames in gfile.walk(folder):
                fpath = dirpath.replace(startdir, '')
                fpath = fpath and fpath + os.sep or ''
                for filename in filenames:
                    tar.add(gfile.path_join(dirpath, filename), gfile.path_join(fpath, filename))
    else:
        raise ValueError("`file` should be type of ['.tar', '.zip'].")
    return file

def compress_files(files, file):
    """Compression files to file.
    
    Args:
        files: str or list
               if str, files should be file path;
               if list, files should be file path list.
        file: str, compression files name.
    Return:
        file: str, compression files name.
    """
    if isinstance(files, str):
        files = [files]
    mat = file.split('.')[-1]
    if mat in ['zip']:
        with zipfile.ZipFile(file, 'w', zipfile.ZIP_DEFLATED) as z:
            for i in files:
                assert not gfile.isdir(i), 'Elements in the list should be file path.'
                z.write(i)
    elif mat in ['tar']:
        with tarfile.TarFile(file, 'w') as t:
            for i in files:
                assert not gfile.isdir(i), 'Elements in the list should be file path.'
                t.add(i)
    elif mat in ['bz2']:
         with bz2.BZ2File(file, 'w') as b:
            for i in files:
                assert not gfile.isdir(i), 'Elements in the list should be file path.'
                with open(i, 'rb') as f:
                    b.write(f.read())
    else:
        raise ValueError("`file` should be type of ['.tar', '.zip', '.bz2'].")
    return file