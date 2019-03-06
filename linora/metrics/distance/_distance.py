import numpy as np
import pandas as pd

__all__ = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'hamming',
           'jaccard', 'pearson', 'cosine'
          ]

def euclidean(x, y, normalize=False):
    assert isinstance(normalize, bool), "`normalize` should be bool types."
    std = pd.concat([x, y]).std() if normalize else 1
    return np.sqrt(np.sum(np.square((x-y)/std)))

def manhattan(x, y):
    return np.sum(np.abs(x-y))

def chebyshev(x, y):
    return np.max(x-y)

def minkowski(x, y, p):
    return np.sqrt(np.sum(np.power(np.abs(x-y), p)))

def hamming(x, y):
    return len(x)-np.sum(np.equal(x, y))

def jaccard(x, y):
    return 1 - len(set(x).intersection(set(y)))/len(set(x).union(set(y)))

def pearson(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    d = np.sum((x-x_mean)*(y-y_mean))
    d2 = np.sqrt(np.sum(np.square(x-x_mean)))*np.sqrt(np.sum(np.square(y-y_mean)))
    return 1-d/d2

def cosine(x, y):
    return (x*y).sum()/np.sqrt(np.square(x).sum())/np.sqrt(np.square(y).sum())
