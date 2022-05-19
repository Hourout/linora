import math

import numpy as np
import pandas as pd

__all__ = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'hamming', 'jaccard', 'pearson', 'cosine', 
           'levenshtein', 'dice', 'ochiia', 'braycurtis', 'geodesic', 'canberra', 'hausdorff', 'chisquare', 
           'hellinger', 'bhattacharyya', 'wasserstein'
          ]


def euclidean(x, y, normalize=False, sample_weight=None):
    """euclidean distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        normalize: default False, std=pd.concat([x, y]).std() if normalize else 1.
        sample_weight: list or array of sample weight.
    Returns:
        euclidean distance value.
    """
    x = pd.Series(x)
    y = pd.Series(y)
    sample_weight = np.ones(len(x)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    std = pd.concat([x, y], axis=1).std(axis=1) if normalize else 1
    return np.sqrt(np.sum(np.square((x-y)*sample_weight/std)))


def manhattan(x, y, sample_weight=None):
    """manhattan distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        sample_weight: list or array of sample weight.
    Returns:
        manhattan distance value.
    """
    sample_weight = np.ones(len(x)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    return np.sum(np.abs(np.array(x)-np.array(y))*sample_weight)


def chebyshev(x, y, sample_weight=None):
    """chebyshev distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        sample_weight: list or array of sample weight.
    Returns:
        chebyshev distance value.
    """
    sample_weight = np.ones(len(x)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    return np.max((np.array(x)-np.array(y))*sample_weight)


def minkowski(x, y, p, sample_weight=None):
    """minkowski distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        p: int, norm dimension.
        sample_weight: list or array of sample weight.
    Returns:
        minkowski distance value.
    """
    sample_weight = np.ones(len(x)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    return np.power(np.sum(np.power(np.abs(np.array(x)-np.array(y))*sample_weight, p)), 1/p)


def hamming(x, y, sample_weight=None):
    """hamming distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        sample_weight: list or array of sample weight.
    Returns:
        hamming distance value.
    """
    sample_weight = np.ones(len(x)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    return len(x)-np.sum(np.equal(np.array(x)-np.array(y))*sample_weight)


def jaccard(x, y):
    """jaccard distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        jaccard distance value.
    """
    return 1 - len(set(x).intersection(set(y)))/len(set(x).union(set(y)))


def pearson(x, y, sample_weight=None):
    """pearson distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        sample_weight: list or array of sample weight.
    Returns:
        pearson distance value.
    """
    x = np.array(x)
    y = np.array(y)
    sample_weight = np.ones(len(x)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    x_mean, y_mean = np.mean(x), np.mean(y)
    d = np.sum((x-x_mean)*(y-y_mean)*sample_weight)
    d2 = np.sqrt(np.sum(np.square(x-x_mean)))*np.sqrt(np.sum(np.square(y-y_mean)))
    return 1-d/d2


def cosine(x, y, sample_weight=None):
    """cosine distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        sample_weight: list or array of sample weight.
    Returns:
        cosine distance value.
    """
    x = np.array(x)
    y = np.array(y)
    sample_weight = np.ones(len(x)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    return (x*y*sample_weight).sum()/np.sqrt(np.square(x).sum())/np.sqrt(np.square(y).sum())


def levenshtein(x, y, normalize=False, sample_weight=None):
    """levenshtein distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        normalize: bool, default False, if normalize is True, levenshtein distance should be distance/max(len(x), len(y)).
        sample_weight: list or array of sample weight.
    Returns:
        levenshtein distance value.
    """
    def levenshtein1(s1, s2, weight, normalize=False):
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
            return (b[-1] / len(s1))*weight
        return b[-1]*weight
    sample_weight = np.ones(len(x)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    return pd.DataFrame({'label1':x, 'label2':y, 'weight':sample_weight}).apply(lambda x:levenshtein1(x[0], x[1], x[2], normalize=normalize), axis=1).sum()


def canberra(x, y, sample_weight=None):
    """canberra distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        sample_weight: list or array of sample weight.
    Returns:
        canberra distance value.
    """
    assert len(x)==len(y), 'x shape should be same with y.'
    x = pd.Series(x)
    y = pd.Series(y)
    sample_weight = np.ones(len(x)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    return ((x-y)*sample_weight/(x.abs()+y.abs())).sum()


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


def braycurtis(x, y, sample_weight=None):
    """braycurtis distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        sample_weight: list or array of sample weight.
    Returns:
        braycurtis distance value.
    """
    x = np.array(x)
    y = np.array(y)
    sample_weight = np.ones(len(x)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    return np.sum(np.abs(x-y)*sample_weight)/(np.sum(x)+np.sum(y))


def ochiia(x, y):
    """ochiia distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        ochiia distance value.
    """
    sample_weight = np.ones(len(x)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    return 1 - len(set(x).intersection(set(y)))/np.sqrt(len(set(x))*len(set(y)))


def dice(x, y):
    """dice distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        dice distance value.
    """
    sample_weight = np.ones(len(x)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    return 1 - 2*len(set(x).intersection(set(y)))/(len(set(x))+len(set(y)))


def hausdorff(x, y, method=euclidean, sample_weight=None):
    """canberra distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        method: distance method, default la.metrics.distance.euclidean, 
                see la.metrics.distance, Some methods are effective.
        sample_weight: list or array of sample weight.
    Returns:
        canberra distance value.
    """
    x = np.array(x)
    y = np.array(y)
    sample_weight = np.ones(len(x)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    fhd = max([min([method(i, j, sample_weight=sample_weight) for j in y]) for i in x])
    rhd = max([min([method(i, j, sample_weight=sample_weight) for j in x]) for i in y])
    return max(fhd,rhd)


def chisquare(x, y, sample_weight=None):
    """chi square distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        sample_weight: list or array of sample weight.
    Returns:
        chi square distance value.
    """
    x = np.asarray(x, np.int32)
    y = np.asarray(y, np.int32)
    sample_weight = np.ones(len(x)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    value = np.square((x-y)*sample_weight)
    y[np.where(y==0)] = 1
    return np.sum(value/y)


def hellinger(x, y, sample_weight=None):
    """hellinger distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        sample_weight: list or array of sample weight.
    Returns:
        hellinger distance value.
    """
    sample_weight = np.ones(len(x)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    return 1/np.sqrt(2)*np.linalg.norm((np.sqrt(x)-np.sqrt(y))*sample_weight)


def bhattacharyya(x, y, sample_weight=None):
    """bhattacharyya distance.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        sample_weight: list or array of sample weight.
    Returns:
        bhattacharyya distance value.
    """
    x = np.array(x)
    y = np.array(y)
    sample_weight = np.ones(len(x)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    return np.log(np.sum(np.sqrt(x * y*sample_weight)))


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
    sample_weight = np.ones(len(x)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    x_sort = np.argsort(x)
    y_sort = np.argsort(y)

    z = np.concatenate([x, y])
    z.sort(kind='mergesort')

    x_cdf = x[x_sort].searchsorted(z[:-1], 'right')/x.size
    y_cdf = y[y_sort].searchsorted(z[:-1], 'right')/y.size
    return np.sum(np.multiply(np.abs(x_cdf - y_cdf), np.diff(z)))