from linora.utils._config import Config

Param = Config()

Param.mnist = [
    'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/train-labels-idx1-ubyte.gz',
    'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/train-images-idx3-ubyte.gz',
    'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/t10k-labels-idx1-ubyte.gz',
    'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/t10k-images-idx3-ubyte.gz',
]

Param.mnist_fashion = [
    'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
    'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
    'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
]

Param.mnist_kannada = [
    'https://github.com/Hourout/datasets/releases/download/0.0.1/kannada_MNIST.zip',
]

Param.mnist_kuzushiji10 = [
    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz',
]

Param.mnist_kuzushiji49 = [
    'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz',
]

Param.mnist_kuzushiji_kanji = [
    "http://codh.rois.ac.jp/kmnist/dataset/kkanji/kkanji.tar",
]

Param.mnist_tibetan = [
    'https://raw.githubusercontent.com/Hourout/datasets/master/TibetanMNIST/TibetanMNIST_28_28_01.csv',
    'https://raw.githubusercontent.com/Hourout/datasets/master/TibetanMNIST/TibetanMNIST_28_28_02.csv',
]