import numpy as np
import pandas as pd

__all__ = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'hamming',
           'jaccard', 'pearson', 'cosine', 'levenshtein'
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

def levenshtein(x, y, normalize=False):
    def levenshtein1(s1, s2, normalize=False):
        if len(s1) < len(s2):
            return levenshtein1(s2, s1, normalize)
        if not s2:
            return len(s1)

        a = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            b = [i+1]
            for j, c2 in enumerate(s2):
                b.append(min(a[j+1]+1, b[j]+1, a[j]+(c1 != c2)))
            a = b

        if normalize:
            return (b[-1] / len(s1))
        return b[-1]
    return pd.DataFrame({'label1':x, 'label2':y}).apply(lambda x:levenshtein1(x[0], x[1], normalize=normalize), axis=1).sum()
