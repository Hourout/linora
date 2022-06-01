import os
import bz2
import gzip
import zipfile
import tarfile
import rarfile

import linora.gfile as gfile

__all__ = ['decompress', 'compress']


def decompress(file, folder=None):
    """Decompression file.
    
    Args:
        file: str, file should be file path;
        folder: str, decompression folder.
    Return:
        decompression folder.
    """
    mat = file.split('.')[-1]
    if folder is None:
        folder = file[:-(len(mat)+1)]
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


def compress(files, file):
    """Compress folder or file or list of files to file.
    
    Args:
        files: str or list
               if str, files should be file or folder path;
               if list, files should be file path list.
        file: str, compression files name.
    Return:
        compression files name.
    """
    if gfile.isdir(files):
        mat = file.split('.')[-1]
        if mat in ['zip']:
            with zipfile.ZipFile(file, 'w', zipfile.ZIP_DEFLATED) as z:
                for dirpath, dirnames, filenames in gfile.walk(files):
                    fpath = dirpath.replace(startdir, '')
                    fpath = fpath and fpath + os.sep or ''
                    for filename in filenames:
                        z.write(gfile.path_join(dirpath, filename), gfile.path_join(fpath, filename))
        elif mat in ['tar']:
            with tarfile.open(file, 'w') as tar:
                for dirpath, dirnames, filenames in gfile.walk(files):
                    fpath = dirpath.replace(startdir, '')
                    fpath = fpath and fpath + os.sep or ''
                    for filename in filenames:
                        tar.add(gfile.path_join(dirpath, filename), gfile.path_join(fpath, filename))
        else:
            raise ValueError("`file` should be type of ['.tar', '.zip'].")
    else:
        if isinstance(files, str):
            files = [files]
        assert isinstance(files, list), 'files should be file or list of files path.'
        for i in files:
            assert gfile.isfile(i), 'files should be file or list of files path.'
        mat = file.split('.')[-1]
        if mat in ['zip']:
            with zipfile.ZipFile(file, 'w', zipfile.ZIP_DEFLATED) as z:
                for i in files:
                    z.write(i)
        elif mat in ['tar']:
            with tarfile.TarFile(file, 'w') as t:
                for i in files:
                    t.add(i)
        elif mat in ['bz2']:
             with bz2.BZ2File(file, 'w') as b:
                for i in files:
                    with open(i, 'rb') as f:
                        b.write(f.read())
        else:
            raise ValueError("`file` should be type of ['.tar', '.zip', '.bz2'].")
    return file