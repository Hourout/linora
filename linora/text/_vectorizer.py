import itertools
import collections
import numpy as np
import pandas as pd

__all__ = ['CountVectorizer', 'TfidfVectorizer']

def CountVectorizer(feature, feature_scale=None, prefix='columns', dtype='int8'):
    scale = feature_scale if feature_scale is not None else list(set(itertools.chain.from_iterable(feature)))
    class_mapping = collections.defaultdict(int)
    class_mapping.default_factory = class_mapping.__len__
    [class_mapping[i] for i in scale]
    row = []
    col = []
    values = []
    for r, x in enumerate(feature):
        y = collections.Counter(x)
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
    t = (count_vectorizer*(np.log((count_vectorizer.shape[0]+1)/(count_vectorizer.replace({0:None}).count()+1))+1))
    if norm=='l2':
        t = (t.T/np.sqrt(np.sum(np.square(t), axis=1))).T
    elif norm=='l1':
        t = (t.T/np.sum(np.abs(t), axis=1)).T
    return t
