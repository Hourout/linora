import math

import numpy as np
import pandas as pd

__all__ = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'hamming', 'jaccard', 'pearson', 'cosine', 
           'levenshtein', 'dice', 'ochiia', 'braycurtis', 'geodesic', 'canberra', 'hausdorff', 'chisquare', 
           'hellinger', 'bhattacharyya', 'wasserstein'
          ]


def euclidean(x, y, normalize=False):
    """euclidean distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        normalize: default False, std=pd.concat([x, y]).std() if normalize else 1.
    Returns:
        euclidean distance value.
    """
    x = pd.Series(x)
    y = pd.Series(y)
    std = pd.concat([x, y], axis=1).std(axis=1) if normalize else 1
    return np.sqrt(np.sum(np.square((x-y)/std)))


def manhattan(x, y):
    """manhattan distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        manhattan distance value.
    """
    return np.sum(np.abs(np.array(x)-np.array(y)))


def chebyshev(x, y):
    """chebyshev distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        chebyshev distance value.
    """
    return np.max(np.array(x)-np.array(y))


def minkowski(x, y, p):
    """minkowski distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        p: int, norm dimension.
    Returns:
        minkowski distance value.
    """
    return np.power(np.sum(np.power(np.abs(np.array(x)-np.array(y)), p)), 1/p)


def hamming(x, y):
    """hamming distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        hamming distance value.
    """
    return len(x)-np.sum(np.equal(np.array(x)-np.array(y)))


def jaccard(x, y):
    """jaccard distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        jaccard distance value.
    """
    return 1 - len(set(x).intersection(set(y)))/len(set(x).union(set(y)))


def pearson(x, y):
    """pearson distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        pearson distance value.
    """
    x = np.array(x)
    y = np.array(y)
    x_mean, y_mean = np.mean(x), np.mean(y)
    d = np.sum((x-x_mean)*(y-y_mean))
    d2 = np.sqrt(np.sum(np.square(x-x_mean)))*np.sqrt(np.sum(np.square(y-y_mean)))
    return 1-d/d2


def cosine(x, y):
    """cosine distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        cosine distance value.
    """
    x = np.array(x)
    y = np.array(y)
    return (x*y).sum()/np.sqrt(np.square(x).sum())/np.sqrt(np.square(y).sum())


def levenshtein(x, y, normalize=False):
    """levenshtein distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        normalize: bool, default False, if normalize is True, levenshtein distance should be distance/max(len(x), len(y)).
    Returns:
        levenshtein distance value.
    """
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


def canberra(x, y):
    """canberra distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        canberra distance value.
    """
    assert len(x)==len(y), 'x shape should be same with y.'
    x = pd.Series(x)
    y = pd.Series(y)
    return ((x-y)/(x.abs()+y.abs())).sum()


def geodesic(x, y):
    """Calculation of latitude and longitude distance by haversine formula.
    
    Args:
        x: list or tuple, Longitude and latitude coordinates, [latitude, longitude]
        y: list or tuple, Longitude and latitude coordinates, [latitude, longitude]
    Return:
        Longitude and latitude distance, unit is 'km'.
    """
    lat0 = math.radians(x[0])
    lat1 = math.radians(y[0])
    lng0 = math.radians(x[1])
    lng1 = math.radians(y[1])
    h = math.sin(math.fabs(lat0 - lat1)/2)**2 + math.cos(lat0) * math.cos(lat1) * math.sin(math.fabs(lng0 - lng1)/2)**2
    return 2 * 6371.393 * math.asin(math.sqrt(h))


def braycurtis(x, y):
    """braycurtis distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        braycurtis distance value.
    """
    x = np.array(x)
    y = np.array(y)
    return np.sum(np.abs(x-y))/(np.sum(x)+np.sum(y))


def ochiia(x, y):
    """ochiia distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        ochiia distance value.
    """
    return 1 - len(set(x).intersection(set(y)))/np.sqrt(len(set(x))*len(set(y)))


def dice(x, y):
    """dice distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        dice distance value.
    """
    return 1 - 2*len(set(x).intersection(set(y)))/(len(set(x))+len(set(y)))


def hausdorff(x, y, method=None):
    """canberra distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        method: distance method, default la.metrics.distance.euclidean, 
            see la.metrics.distance, Some methods are effective.
    Returns:
        canberra distance value.
    """
    x = np.array(x)
    y = np.array(y)
    if method is None:
        method = euclidean
    fhd = max([min([method(i, j) for j in y]) for i in x])
    rhd = max([min([method(i, j) for j in x]) for i in y])
    return max(fhd,rhd)


def chisquare(x, y):
    """chi square distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        chi square distance value.
    """
    x = np.asarray(x, np.int32)
    y = np.asarray(y, np.int32)
    value = np.square(x-y)
    y[np.where(y==0)] = 1
    return np.sum(value/y)


def hellinger(x, y):
    """hellinger distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        hellinger distance value.
    """
    return 1/np.sqrt(2)*np.linalg.norm(np.sqrt(x)-np.sqrt(y))


def bhattacharyya(x, y):
    """bhattacharyya distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        bhattacharyya distance value.
    """
    x = np.array(x)
    y = np.array(y)
    return np.log(np.sum(np.sqrt(x * y)))


def wasserstein(x, y):
    """wasserstein distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        wasserstein distance value.
    """
    x = np.array(x)
    y = np.array(y)
    x_sort = np.argsort(x)
    y_sort = np.argsort(y)

    z = np.concatenate([x, y])
    z.sort(kind='mergesort')

    x_cdf = x[x_sort].searchsorted(z[:-1], 'right')/x.size
    y_cdf = y[y_sort].searchsorted(z[:-1], 'right')/y.size
    return np.sum(np.multiply(np.abs(x_cdf - y_cdf), np.diff(z)))