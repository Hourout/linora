import gzip

import numpy as np

from linora import gfile
from linora.utils._progbar import Progbar
from linora.data._utils import assert_dirs, get_file
from linora.image._io import save_image, array_to_image
from linora.data.Dataset._dataset import from_class_folder

__all__ = ['mnist', 'mnist_fashion']


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
    p = la.utils.Progbar(10, verbose=verbose)
    task_path = assert_dirs(root, 'mnist')
    p.add(1)
    url_list = ['https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/train-labels-idx1-ubyte.gz',
                'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/train-images-idx3-ubyte.gz',
                'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/t10k-labels-idx1-ubyte.gz',
                'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/t10k-images-idx3-ubyte.gz']
    for url in url_list:
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
    for url in url_list:
        gfile.remove(gfile.path_join(task_path, url.split('/')[-1]))
    p.add(1)
    if dataset:
        return from_class_folder(task_path, label_encoder=True)
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
    p = la.utils.Progbar(10, verbose=verbose)
    task_path = assert_dirs(root, 'mnist_fashion')
    p.add(1)
    url_list = ['http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
                'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
                'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
                'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz']
    for url in url_list:
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
    for url in url_list:
        gfile.remove(gfile.path_join(task_path, url.split('/')[-1]))
    p.add(1)
    if dataset:
        return from_class_folder(task_path, label_encoder=True)
    return task_path
