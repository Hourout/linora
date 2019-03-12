from functools import reduce
import itertools
from collections import defaultdict

import pandas as pd
import numpy as np

__all__ = ['categorical_encoder', 'categorical_hash', 'categorical_crossed',
           'categorical_onehot_binarizer', 'categorical_onehot_multiple']


def categorical_encoder(feature, feature_scale=None):
    scale = feature_scale if feature_scale is not None else {j:i for i,j in feature.drop_duplicates().reset_index(drop=True).to_dict().items()}
    t = feature.replace(scale)
    return t, scale

def categorical_hash(feature, hash_bucket_size):
    return feature.fillna('').astype(str).map(lambda x:hash(x))%hash_bucket_size

def categorical_crossed(feature_list, hash_bucket_size):
    return reduce(lambda x,y:x+y, [i.fillna('').astype(str) for i in feature_list]).map(lambda x:hash(x))%hash_bucket_size

def categorical_onehot_binarizer(feature, feature_scale=None, prefix='columns', dtype='int32'):
    assert not any(feature.isnull()), "`feature' should be not contains NaN"
    scale = feature.drop_duplicates().tolist()
    if feature_scale is not None:
        t = pd.get_dummies(feature.replace({i:'temp_str' for i in set.difference(set(scale), set(feature_scale))}), prefix=prefix, dtype=dtype)
        if prefix+'_temp_str' in t.columns:
            t = t.drop([prefix+'_temp_str'], axis=1)
        for i in set.difference(set(feature_scale), set(scale)):
            if prefix+'_'+str(i) not in t.columns:
                t[prefix+'_'+str(i)] = 0
        scale = feature_scale
    else:
        t = pd.get_dummies(feature, prefix=prefix, dtype=dtype)
    return t, scale

def categorical_onehot_multiple(feature, feature_scale=None, prefix='columns', dtype='int32'):
    assert not any(feature.isnull()), "`feature' should be not contains NaN"
    scale2 = list(set(itertools.chain.from_iterable(temp)))
    scale = feature_scale if feature_scale is not None else scale2
    class_mapping = defaultdict(int)
    class_mapping.default_factory = class_mapping.__len__
    [class_mapping[i] for i in scale]
    col = [class_mapping[i] for i in itertools.chain.from_iterable(temp)]
    row = [i for i in itertools.chain.from_iterable(map(lambda x,y:[x]*len(y), range(len(feature)), feature))]
    t = np.zeros([len(feature), len(scale2)], dtype='int8')
    t[row, col] = 1
    t = pd.DataFrame(t, columns=[prefix+'_'+i for i in scale2], dtype=dtype)
    if t.shape[1]-len(scale)>0:
        t = t.drop(t.columns[len(scale):].tolist(), axis=1)
    for i in set.difference(set(scale), set(scale2)):
        if prefix+'_'+str(i) not in t.columns:
                t[prefix+'_'+str(i)] = 0
    return t, scale
