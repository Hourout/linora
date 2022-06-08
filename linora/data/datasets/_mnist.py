import io
import time
import gzip

import requests
import numpy as np
import pandas as pd

from linora import gfile
from linora.utils._progbar import Progbar
from linora.data._compress import decompress
from linora.data._utils import assert_dirs, get_file
from linora.image._image_io import save_image
from linora.image._image_util import array_to_image
from linora.data.Dataset._dataset import from_class_folder
from linora.data.datasets._config_path import Param

__all__ = ['mnist', 'mnist_fashion', 'mnist_kannada', 'mnist_tibetan',
           'mnist_kuzushiji10', 'mnist_kuzushiji49', 'mnist_kuzushiji_kanji']


def mnist(root=None, dataset=True, verbose=1):
    """MNIST handwritten digits dataset from http://yann.lecun.com/exdb/mnist
    
    Each sample is an gray image (in 3D NDArray) with shape (28, 28, 1).
    
    Attention: if exist dirs `root/mnist`, api will delete it and create it.
    Data storage directory:
    root = `/user/.../mydata`
    mnist data: 
    `root/mnist/train/0/xx.png`
    `root/mnist/train/2/xx.png`
    `root/mnist/train/6/xx.png`
    `root/mnist/test/0/xx.png`
    `root/mnist/test/2/xx.png`
    `root/mnist/test/6/xx.png`
    Args:
        root: str, Store the absolute path of the data directory.
              example:if you want data path is `/user/.../mydata/mnist`,
              root should be `/user/.../mydata`.
        dataset: whether to return a la.data.Dataset object.
        verbose: Verbosity mode, 0 (silent), 1 (verbose)
    Returns:
        Store the absolute path of the data directory, is `root/mnist`.
    """
    p = Progbar(10, verbose=verbose)
    task_path = assert_dirs(root, 'mnist')
    p.add(1)
    for url in Param.mnist:
        get_file(url, gfile.path_join(task_path, url.split('/')[-1]), verbose=0)
        p.add(1)
    with gzip.open(gfile.path_join(task_path, 'train-labels-idx1-ubyte.gz'), 'rb') as lbpath:
        train_label = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(gfile.path_join(task_path, 'train-images-idx3-ubyte.gz'), 'rb') as imgpath:
        train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(train_label), 28, 28)

    with gzip.open(gfile.path_join(task_path, 't10k-labels-idx1-ubyte.gz'), 'rb') as lbpath:
        test_label = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(gfile.path_join(task_path, 't10k-images-idx3-ubyte.gz'), 'rb') as imgpath:
        test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(test_label), 28, 28)
    p.add(1)
    for i in set(train_label):
        gfile.makedirs(gfile.path_join(task_path, 'train', str(i)))
    for i in set(test_label):
        gfile.makedirs(gfile.path_join(task_path, 'test', str(i)))
    p.add(1)
    for idx in range(train.shape[0]):
        save_image(gfile.path_join(task_path, 'train', str(train_label[idx]), str(idx)+'.png'), 
                   array_to_image(train[idx].reshape(28, 28, 1)))
    p.add(1)
    for idx in range(test.shape[0]):
        save_image(gfile.path_join(task_path, 'test', str(test_label[idx]), str(idx)+'.png'), 
                   array_to_image(test[idx].reshape(28, 28, 1)))
    p.add(1)
    for url in Param.mnist:
        gfile.remove(gfile.path_join(task_path, url.split('/')[-1]))
    p.add(1)
    if dataset:
        return (from_class_folder(gfile.path_join(task_path, 'train'), label_encoder=1).split({'train':1})
                .join({'test':from_class_folder(gfile.path_join(task_path, 'test'), label_encoder=1)}))
    return task_path


def mnist_fashion(root=None, dataset=True, verbose=1):
    """A dataset of Zalando's article images consisting of fashion products.
    
    Fashion mnist datasets is a drop-in replacement of the original MNIST dataset
    from https://github.com/zalandoresearch/fashion-mnist.
    Each sample is an gray image (in 3D NDArray) with shape (28, 28, 1).
    
    Attention: if exist dirs `root/mnist_fashion`, api will delete it and create it.
    Data storage directory:
    root = `/user/.../mydata`
    mnist_fashion data: 
    `root/mnist_fashion/train/0/xx.png`
    `root/mnist_fashion/train/2/xx.png`
    `root/mnist_fashion/train/6/xx.png`
    `root/mnist_fashion/test/0/xx.png`
    `root/mnist_fashion/test/2/xx.png`
    `root/mnist_fashion/test/6/xx.png`
    Args:
        root: str, Store the absolute path of the data directory.
              example:if you want data path is `/user/.../mydata/mnist_fashion`,
              root should be `/user/.../mydata`.
        dataset: whether to return a la.data.Dataset object.
        verbose: Verbosity mode, 0 (silent), 1 (verbose)
    Returns:
        Store the absolute path of the data directory, is `root/mnist_fashion`.
    """
    p = Progbar(10, verbose=verbose)
    task_path = assert_dirs(root, 'mnist_fashion')
    p.add(1)
    for url in Param.mnist_fashion:
        get_file(url, gfile.path_join(task_path, url.split('/')[-1]), verbose=0)
        p.add(1)
    with gzip.open(gfile.path_join(task_path, 'train-labels-idx1-ubyte.gz'), 'rb') as lbpath:
        train_label = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(gfile.path_join(task_path, 'train-images-idx3-ubyte.gz'), 'rb') as imgpath:
        train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(train_label), 28, 28)

    with gzip.open(gfile.path_join(task_path, 't10k-labels-idx1-ubyte.gz'), 'rb') as lbpath:
        test_label = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(gfile.path_join(task_path, 't10k-images-idx3-ubyte.gz'), 'rb') as imgpath:
        test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(test_label), 28, 28)
    p.add(1)
    for i in set(train_label):
        gfile.makedirs(gfile.path_join(task_path, 'train', str(i)))
    for i in set(test_label):
        gfile.makedirs(gfile.path_join(task_path, 'test', str(i)))
    p.add(1)
    for idx in range(train.shape[0]):
        save_image(gfile.path_join(task_path, 'train', str(train_label[idx]), str(idx)+'.png'), 
                   array_to_image(train[idx].reshape(28, 28, 1)))
    p.add(1)
    for idx in range(test.shape[0]):
        save_image(gfile.path_join(task_path, 'test', str(test_label[idx]), str(idx)+'.png'), 
                   array_to_image(test[idx].reshape(28, 28, 1)))
    p.add(1)
    for url in Param.mnist_fashion:
        gfile.remove(gfile.path_join(task_path, url.split('/')[-1]))
    p.add(1)
    if dataset:
        return (from_class_folder(gfile.path_join(task_path, 'train'), label_encoder=1).split({'train':1})
                .join({'test':from_class_folder(gfile.path_join(task_path, 'test'), label_encoder=1)}))
    return task_path


def mnist_kannada(root=None, dataset=True, verbose=1):
    """kannada-MNIST from https://github.com/vinayprabhu/Kannada_MNIST.
    
    The Kannada-MNIST dataset was created an a drop-in substitute for the standard MNIST dataset.
    
    Each sample is an gray image (in 3D NDArray) with shape (28, 28, 1).
    
    Attention: if exist dirs `root/mnist_kannada`, api will delete it and create it.
    Data storage directory:
    root = `/user/.../mydata`
    mnist_kannada data: 
    `root/mnist_kannada/train/0/xx.png`
    `root/mnist_kannada/train/2/xx.png`
    `root/mnist_kannada/train/6/xx.png`
    `root/mnist_kannada/test/0/xx.png`
    `root/mnist_kannada/test/2/xx.png`
    `root/mnist_kannada/test/6/xx.png`
    Args:
        root: str, Store the absolute path of the data directory.
              example:if you want data path is `/user/.../mydata/mnist_kannada`,
              root should be `/user/.../mydata`.
        dataset: bool, whether to return a la.data.Dataset object.
        verbose: Verbosity mode, 0 (silent), 1 (verbose)
    Returns:
        Store the absolute path of the data directory, is `root/mnist_kannada` or la.data.Dataset.
    """
    p = Progbar(10, verbose=verbose)
    task_path = assert_dirs(root, 'mnist_kannada')
    p.add(1)
    zip_path = get_file(Param.mnist_kannada[0], task_path+'/kannada_MNIST.zip', verbose=0)
    p.add(2)
    unzip_path = decompress(task_path+'/kannada_MNIST.zip')
    p.add(1)
    train = pd.read_csv(gfile.path_join(task_path, 'kannada_MNIST/kannada_MNIST_train.csv'), header=None, dtype='uint8')
    test = pd.read_csv(gfile.path_join(task_path, 'kannada_MNIST/kannada_MNIST_test.csv'), header=None, dtype='uint8')
    p.add(2)
    for i in set(train[0]):
        gfile.makedirs(gfile.path_join(task_path, 'train', str(i)))
        gfile.makedirs(gfile.path_join(task_path, 'test', str(i)))
    p.add(1)
    for i in range(len(train)):
        save_image(gfile.path_join(task_path, 'train', str(train.iat[i, 0]), str(i)+'.png'),
                   array_to_image(train.iloc[i, 1:].values.reshape(28, 28, 1)))
    p.add(1)
    for i in range(len(test)):
        save_image(gfile.path_join(task_path, 'test', str(test.iat[i, 0]), str(i)+'.png'),
                       array_to_image(test.iloc[i, 1:].values.reshape(28, 28, 1)))
    p.add(1)
    gfile.remove(zip_path)
    gfile.remove(unzip_path)
    p.add(1)
    if dataset:
        return (from_class_folder(gfile.path_join(task_path, 'train'), label_encoder=1).split({'train':1})
                .join({'test':from_class_folder(gfile.path_join(task_path, 'test'), label_encoder=1)}))
    return task_path


def mnist_kuzushiji10(root=None, dataset=True, verbose=1):
    """Kuzushiji-MNIST from https://github.com/rois-codh/kmnist.
    
    Kuzushiji-MNIST is a drop-in replacement for the
    MNIST dataset (28x28 grayscale, 70,000 images), 
    provided in the original MNIST format as well as a NumPy format.
    Since MNIST restricts us to 10 classes, we chose one character to
    represent each of the 10 rows of Hiragana when creating Kuzushiji-MNIST.
    
    Each sample is an gray image (in 3D NDArray) with shape (28, 28, 1).
    
    Attention: if exist dirs `root/mnist_kuzushiji10`, api will delete it and create it.
    Data storage directory:
    root = `/user/.../mydata`
    mnist_kuzushiji10 data: 
    `root/mnist_kuzushiji10/train/0/xx.png`
    `root/mnist_kuzushiji10/train/2/xx.png`
    `root/mnist_kuzushiji10/train/6/xx.png`
    `root/mnist_kuzushiji10/test/0/xx.png`
    `root/mnist_kuzushiji10/test/2/xx.png`
    `root/mnist_kuzushiji10/test/6/xx.png`
    Args:
        root: str, Store the absolute path of the data directory.
              example:if you want data path is `/user/.../mydata/mnist_kuzushiji10`,
              root should be `/user/.../mydata`.
        dataset: whether to return a la.data.Dataset object.
        verbose: Verbosity mode, 0 (silent), 1 (verbose)
    Returns:
        Store the absolute path of the data directory, is `root/mnist_kuzushiji10`.
    """
    p = Progbar(10, verbose=verbose)
    task_path = assert_dirs(root, 'mnist_kuzushiji10')
    p.add(1)
    for url in Param.mnist_kuzushiji10:
        get_file(url, gfile.path_join(task_path, url.split('/')[-1]), verbose=0)
        p.add(1)
    train = np.load(gfile.path_join(task_path, 'kmnist-train-imgs.npz'))['arr_0']
    train_label = np.load(gfile.path_join(task_path, 'kmnist-train-labels.npz'))['arr_0']
    test = np.load(gfile.path_join(task_path, 'kmnist-test-imgs.npz'))['arr_0']
    test_label = np.load(gfile.path_join(task_path, 'kmnist-test-labels.npz'))['arr_0']
    p.add(1)
    for i in set(train_label):
        gfile.makedirs(gfile.path_join(task_path, 'train', str(i)))
    for i in set(test_label):
        gfile.makedirs(gfile.path_join(task_path, 'test', str(i)))
    p.add(1)
    for idx in range(train.shape[0]):
        save_image(gfile.path_join(task_path, 'train', str(train_label[idx]), str(idx)+'.png'), 
                   array_to_image(train[idx].reshape(28, 28, 1)))
    p.add(1)
    for idx in range(test.shape[0]):
        save_image(gfile.path_join(task_path, 'test', str(test_label[idx]), str(idx)+'.png'), 
                   array_to_image(test[idx].reshape(28, 28, 1)))
    p.add(1)
    for url in Param.mnist_kuzushiji10:
        gfile.remove(gfile.path_join(task_path, url.split('/')[-1]))
    p.add(1)
    if dataset:
        return (from_class_folder(gfile.path_join(task_path, 'train'), label_encoder=1).split({'train':1})
                .join({'test':from_class_folder(gfile.path_join(task_path, 'test'), label_encoder=1)}))
    return task_path


def mnist_kuzushiji49(root=None, dataset=True, verbose=1):
    """Kuzushiji-49 from https://github.com/rois-codh/kmnist.
    
    Kuzushiji-49, as the name suggests, has 49 classes (28x28 grayscale, 270,912 images),
    is a much larger, but imbalanced dataset containing 48 Hiragana 
    characters and one Hiragana iteration mark.
    
    Each sample is an gray image (in 3D NDArray) with shape (28, 28, 1).
    
    Attention: if exist dirs `root/mnist_kuzushiji49`, api will delete it and create it.
    Data storage directory:
    root = `/user/.../mydata`
    mnist_kuzushiji49 data: 
    `root/mnist_kuzushiji49/train/0/xx.png`
    `root/mnist_kuzushiji49/train/2/xx.png`
    `root/mnist_kuzushiji49/train/6/xx.png`
    `root/mnist_kuzushiji49/test/0/xx.png`
    `root/mnist_kuzushiji49/test/2/xx.png`
    `root/mnist_kuzushiji49/test/6/xx.png`
    Args:
        root: str, Store the absolute path of the data directory.
              example:if you want data path is `/user/.../mydata/mnist_kuzushiji49`,
              root should be `/user/.../mydata`.
        dataset: whether to return a la.data.Dataset object.
        verbose: Verbosity mode, 0 (silent), 1 (verbose)
    Returns:
        Store the absolute path of the data directory, is `root/mnist_kuzushiji49`.
    """
    p = Progbar(10, verbose=verbose)
    task_path = assert_dirs(root, 'mnist_kuzushiji49')
    p.add(1)
    for url in Param.mnist_kuzushiji49:
        get_file(url, gfile.path_join(task_path, url.split('/')[-1]), verbose=0)
        p.add(1)
    train = np.load(gfile.path_join(task_path, 'k49-train-imgs.npz'))['arr_0']
    train_label = np.load(gfile.path_join(task_path, 'k49-train-labels.npz'))['arr_0']
    test = np.load(gfile.path_join(task_path, 'k49-test-imgs.npz'))['arr_0']
    test_label = np.load(gfile.path_join(task_path, 'k49-test-labels.npz'))['arr_0']
    p.add(1)
    for i in set(train_label):
        gfile.makedirs(gfile.path_join(task_path, 'train', str(i)))
    for i in set(test_label):
        gfile.makedirs(gfile.path_join(task_path, 'test', str(i)))
    p.add(1)
    for idx in range(train.shape[0]):
        save_image(gfile.path_join(task_path, 'train', str(train_label[idx]), str(idx)+'.png'), 
                   array_to_image(train[idx].reshape(28, 28, 1)))
    p.add(1)
    for idx in range(test.shape[0]):
        save_image(gfile.path_join(task_path, 'test', str(test_label[idx]), str(idx)+'.png'), 
                   array_to_image(test[idx].reshape(28, 28, 1)))
    p.add(1)
    for url in Param.mnist_kuzushiji49:
        gfile.remove(gfile.path_join(task_path, url.split('/')[-1]))
    p.add(1)
    if dataset:
        return (from_class_folder(gfile.path_join(task_path, 'train'), label_encoder=1).split({'train':1})
                .join({'test':from_class_folder(gfile.path_join(task_path, 'test'), label_encoder=1)}))
    return task_path


def mnist_kuzushiji_kanji(root=None, dataset=True, verbose=1):
    """Kuzushiji-Kanji dataset from https://github.com/rois-codh/kmnist.
    
    Kuzushiji-Kanji is a large and highly imbalanced 64x64 dataset 
    of 3832 Kanji characters, containing 140,426 images 
    of both common and rare characters.
    
    Attention: if exist dirs `root/mnist_kuzushiji_kanji`, api will delete it and create it.
    Data storage directory:
    root = `/user/.../mydata`
    mnist_kuzushiji_kanji data: 
    `root/mnist_kuzushiji_kanji/train/U+55C7/xx.png`
    `root/mnist_kuzushiji_kanji/train/U+7F8E/xx.png`
    `root/mnist_kuzushiji_kanji/train/U+9593/xx.png`
    Args:
        root: str, Store the absolute path of the data directory.
              example:if you want data path is `/user/.../mydata/mnist_kuzushiji_kanji`,
              root should be `/user/.../mydata`.
        dataset: whether to return a la.data.Dataset object.
        verbose: Verbosity mode, 0 (silent), 1 (verbose)
    Returns:
        Store the absolute path of the data directory, is `root/mnist_kuzushiji_kanji.
    """
    if root is None:
        root = './'
    p = Progbar(10, verbose=verbose)
    task_path = assert_dirs(root, 'mnist_kuzushiji_kanji', make_root_dir=False)
    p.add(1)
    get_file(Param.mnist_kuzushiji_kanji[0], gfile.path_join(root, url.split('/')[-1]), verbose=0)
    p.add(7)
    decompress(gfile.path_join(root, url.split('/')[-1]), task_path)
    p.add(1)
    gfile.rename(gfile.path_join(task_path, 'kkanji2'), gfile.path_join(task_path, 'train'))
    gfile.remove(gfile.path_join(root, 'kkanji.tar'))
    p.add(1)
    if dataset:
        return (from_class_folder(gfile.path_join(task_path, 'train'), label_encoder=1).split({'train':1})
               )
    return task_path


def mnist_tibetan(root=None, dataset=True, verbose=1):
    """Tibetan-MNIST from https://github.com/bat67/TibetanMNIST.
    
    Tibetan-MNIST is a drop-in replacement for the
    MNIST dataset (28x28 grayscale, 70,000 images), 
    provided in the original MNIST format as well as a NumPy format.
    Since MNIST restricts us to 10 classes, we chose one character to
    represent each of the 10 rows of Hiragana when creating Tibetan-MNIST.
    
    Each sample is an gray image (in 3D NDArray) with shape (28, 28, 1).
    
    Attention: if exist dirs `root/mnist_tibetan`, api will delete it and create it.
    Data storage directory:
    root = `/user/.../mydata`
    mnist_tibetan data: 
    `root/mnist_tibetan/train/0/xx.png`
    `root/mnist_tibetan/train/2/xx.png`
    `root/mnist_tibetan/train/6/xx.png`
    `root/mnist_tibetan/test/0/xx.png`
    `root/mnist_tibetan/test/2/xx.png`
    `root/mnist_tibetan/test/6/xx.png`
    Args:
        root: str, Store the absolute path of the data directory.
              example:if you want data path is `/user/.../mydata/mnist_tibetan`,
              root should be `/user/.../mydata`.
        dataset: whether to return a la.data.Dataset object.
        verbose: Verbosity mode, 0 (silent), 1 (verbose)
    Returns:
        Store the absolute path of the data directory, is `root/mnist_tibetan`.
    """
    p = Progbar(10, verbose=verbose)
    task_path = assert_dirs(root, 'mnist_tibetan')
    p.add(1)
    data = pd.DataFrame()
    for url in Param.mnist_tibetan:
        s = requests.get(url).content
        data = pd.concat([data, pd.read_csv(io.StringIO(s.decode('utf-8')), header=None, dtype='uint8')])
        p.add(3)
    train = data.loc[:, 1:].values.reshape(-1, 28, 28)
    train_label = data.loc[:, 0].values
    for i in set(train_label):
        gfile.makedirs(gfile.path_join(task_path, 'train', str(i)))
    p.add(1)
    for idx in range(train.shape[0]):
        save_image(gfile.path_join(task_path, 'train', str(train_label[idx]), str(idx)+'.png'),
                   array_to_image(train[idx].reshape(28, 28, 1)))
    p.add(2)
    if dataset:
        return (from_class_folder(gfile.path_join(task_path, 'train'), label_encoder=1).split({'train':1})
                .join({'test':from_class_folder(gfile.path_join(task_path, 'test'), label_encoder=1)}))
    return task_path