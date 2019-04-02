import itertools
import collections
import numpy as np
import pandas as pd

__all__ = ['CountVectorizer']

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
