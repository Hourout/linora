from itertools import chain
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

__all__ = ['CountVectorizer', 'TfidfVectorizer']

def CountVectorizer(feature, feature_scale=None, prefix='columns', dtype='int8'):
    """Convert a collection of text documents to a matrix of token counts.
    
    Args:
        feature: pd.Series or np.array or list of Lists, sample feature value.
        feature_scale: list or tuple, feature element set.
        prefix: return DataFrame columns prefix name.
        dtype: return DataFrame dtypes.
    Returns:
         CountVectorizer feature with pd.DataFrame type and feature_scale.
    """
    scale = feature_scale if feature_scale is not None else list(set(chain.from_iterable(feature)))
    class_mapping = defaultdict(int)
    class_mapping.default_factory = class_mapping.__len__
    [class_mapping[i] for i in scale]
    row = []
    col = []
    values = []
    for r, x in enumerate(feature):
        y = Counter(x)
        row.extend([r]*len(y))
        col.extend([class_mapping[i] for i in y])
        values.extend(y.values())
    t = np.zeros([max(row)+1, len(class_mapping)], dtype=dtype)
    t[row, col] = values
    if feature_scale is not None:
        t = t[:, :len(feature_scale)]
    t = pd.DataFrame(t, columns=[prefix+'_'+str(i) for i in scale])
    return t, scale

def TfidfVectorizer(count_vectorizer, norm='l2'):
    """Transform a count matrix to a normalized tf or tf-idf representation.
    
    Args:
        count_vectorizer:la.text.CountVectorizer()[0]
        norm: 'l1', 'l2' or None, optional (default=’l2’).
              Each output row will have unit norm, either: * 'l2': Sum of squares of vector elements is 1. 
              The cosine similarity between two vectors is their dot product when l2 norm has been applied. 
              * 'l1': Sum of absolute values of vector elements is 1.
    Returns:
         sparse matrix or DataFrame, [n_samples, n_features].
    """
    t = (count_vectorizer*(np.log((count_vectorizer.shape[0]+1)/(count_vectorizer.replace({0:None}).count()+1))+1))
    if norm=='l2':
        t = (t.T/np.sqrt(np.sum(np.square(t), axis=1))).T
    elif norm=='l1':
        t = (t.T/np.sum(np.abs(t), axis=1)).T
    return t
