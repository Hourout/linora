import bz2
import gzip
import zipfile
import tarfile
import rarfile

__all__ = ['un_gz', 'un_tar', 'un_zip', 'un_rar', 'un_bz2']

def un_gz(gz_file, ugz_name=None):
    """Uncompression .gz file.
    
    Args:
        gz_files: str, gz_file should be file path;
        ugz_name: str, uncompression files name.
    Return:
        ugz_name: str, uncompression files name.
    """
    assert gz_file[-3:]==".gz", '`gz_file` should be `xxx.gz`'
    if ugz_name is None:
        ugz_name = gz_file[:-3]
    with gzip.GzipFile(gz_file) as g_file:
        with open(ugz_name, "w+") as f:
            f.write(g_file.read())
    return ugz_name

def un_tar(tar_file, utar_folder=None):
    """Uncompression .tar file
    
    Args:
        tar_files: str, tar_file should be file path;
        utar_folder: str, uncompression files name.
    Return:
        utar_folder: str, uncompression files name.
    """
    assert tar_file[-4:] in [".tar", '.tgz'], '`tar_file` should be `xxx.tar`'
    if utar_folder is None:
        utar_folder = tar_file[:-4]
    with tarfile.open(tar_file, 'r') as tar:
        names = tar.getnames()
        for name in names:
            tar.extract(name, utar_folder)
    return utar_folder

def un_zip(zip_file, uzip_folder=None):
    """Uncompression .zip file
    
    Args:
        zip_files: str, zip_file should be file path;
        uzip_folder: str, uncompression files name.
    Return:
        uzip_folder: str, uncompression files name.
    """
    assert zip_file[-4:]==".zip", '`zip_file` should be `xxx.zip`'
    if uzip_folder is None:
        uzip_folder = zip_file[:-4]
    with zipfile.ZipFile(zip_file, 'r') as z:
        names = z.namelist()
        for name in names:
            z.extract(name, uzip_folder)
    return uzip_folder

def un_rar(rar_file, urar_folder=None):
    """Uncompression .rar file
    
    Args:
        rar_files: str, rar_file should be file path;
        urar_folder: str, uncompression files name.
    Return:
        urar_folder: str, uncompression files name.
    """
    assert rar_file[-4:]==".rar", '`rar_file` should be `xxx.rar`'
    if urar_folder is None:
        urar_folder = rar_file[:-4]
    with rarfile.RarFile(rar_file, 'r') as r:
        names = r.namelist()
        for name in names:
            r.extract(name, urar_folder)
    return urar_folder

def un_bz2(bz2_file, ubz2_folder=None):
    """Uncompression .bz2 file
    
    Args:
        rar_files: str, rar_file should be file path;
        urar_folder: str, uncompression files name.
    Return:
        urar_folder: str, uncompression files name.
    """
    assert bz2_file[-4:]==".bz2", '`bz2_file` should be `xxx.bz2`'
    if ubz2_folder is None:
        ubz2_folder = bz2_file[:-4]
    with bz2.BZ2File(bz2_file, 'r') as b:
        with open(ubz2_folder, 'w') as f:
            f.write(b.read())
    return ubz2_folder