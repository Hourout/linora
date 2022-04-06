import itertools
from functools import reduce
from collections import defaultdict

import pandas as pd
import numpy as np

__all__ = ['categorical_encoder', 'categorical_hash', 'categorical_crossed',
           'categorical_onehot_binarizer', 'categorical_onehot_multiple',
           'categorical_count', 'categorical_regress', 'categorical_hist']


def categorical_encoder(feature, feature_scale=None, abnormal_value=-1):
    """Encode labels with value between 0 and n_classes-1.
       
       if feature values not in feature_scale dict, return `abnormal_value`.
    
    Args:
        feature: pd.Series, sample feature.
        feature_scale: dict, label parameters dict for this estimator.
        abnormal_value: int, if feature values not in feature_scale dict, return `abnormal_value`.
    Returns:
        return encoded labels and label parameters dict.
    """
    scale = feature_scale if feature_scale is not None else {j:i for i,j in feature.drop_duplicates().reset_index(drop=True).to_dict().items()}
    t = pd.Series([scale.get(i, abnormal_value) for i in feature], index=feature.index)
    return t, scale

def categorical_hash(feature, hash_bucket_size):
    """Hash labels with value between 0 and n_classes-1.
    
    Args:
        feature: pd.Series, sample feature.
        hash_bucket_size: int, number of categories that need hash.
    Returns:
        return hash labels.
    """
    return feature.fillna('').astype(str).map(lambda x:hash(x))%hash_bucket_size

def categorical_crossed(feature_list, hash_bucket_size):
    """Crossed categories and hash labels with value between 0 and n_classes-1.
    
    Args:
        feature_list: pd.Series list, sample feature list.
        hash_bucket_size: int, number of categories that need hash.
    Returns:
        return hash labels.
    """
    return reduce(lambda x,y:x+y, [i.fillna('').astype(str) for i in feature_list]).map(lambda x:hash(x))%hash_bucket_size

def categorical_onehot_binarizer(feature, feature_scale=None, prefix='columns', dtype='int8'):
    """Transform between iterable of iterables and a multilabel format, sample is simple categories.
    
    Args:
        feature: pd.Series, sample feature.
        feature_scale: list, feature categories list.
        prefix: String to append DataFrame column names.
        dtype: default np.uint8. Data type for new columns. Only a single dtype is allowed.
    Returns:
        Dataframe for onehot binarizer.
    """
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
        t = t[[prefix+'_'+str(i) for i in feature_scale]]
    else:
        t = pd.get_dummies(feature, prefix=prefix, dtype=dtype)
        t = t[[prefix+'_'+str(i) for i in scale]]
    return t, scale

def categorical_onehot_multiple(feature, feature_scale=None, prefix='columns', dtype='int8'):
    """Transform between iterable of iterables and a multilabel format, sample is multiple categories.
    
    Args:
        feature: pd.Series, sample feature.
        feature_scale: list, feature categories list.
        prefix: String to append DataFrame column names.
        dtype: default np.uint8. Data type for new columns. Only a single dtype is allowed.
    Returns:
        Dataframe for onehot binarizer.
    """
    assert not any(feature.isnull()), "`feature' should be not contains NaN."
    scale = feature_scale if feature_scale is not None else list(set(itertools.chain.from_iterable(feature)))
    class_mapping = defaultdict(int)
    class_mapping.default_factory = class_mapping.__len__
    [class_mapping[i] for i in scale]
    col = [class_mapping[i] for i in itertools.chain.from_iterable(feature)]
    row = [i for i in itertools.chain.from_iterable(map(lambda x,y:[x]*len(y), range(len(feature)), feature))]
    t = np.zeros([max(row)+1, len(class_mapping)], dtype=dtype)
    t[row, col] = 1
    if feature_scale is not None:
        t = t[:, :len(feature_scale)]
    t = pd.DataFrame(t, columns=[prefix+'_'+str(i) for i in scale])
    return t, scale

def categorical_count(feature, feature_scale=None, abnormal_value=-1):
    """Count labels with value counts.
       
       if feature values not in feature_scale dict, return `abnormal_value`.
    
    Args:
        feature: pd.Series, sample feature.
        feature_scale: dict, label parameters dict for this estimator.
        abnormal_value: int, if feature values not in feature_scale dict, return `abnormal_value`.
    Returns:
        return count labels and label parameters dict.
    """
    scale = feature_scale if feature_scale is not None else feature.value_counts().to_dict()
    t = pd.Series([scale.get(i, abnormal_value) for i in feature], index=feature.index)
    return t, scale

def categorical_regress(feature, label, feature_scale=None, mode='mean'):
    """Regress labels with value counts prob.
    
    Args:
        feature: pd.Series, sample feature.
        label: pd.Series, sample regress label.
        feature_scale: pd.DataFrame, label parameters DataFrame for this estimator.
        mode: 'mean' or 'median'
    Returns:
        return Regress labels and label parameters DataFrame.
    """
    if feature_scale is not None:
        scale = feature_scale
    else:
        scale = pd.concat([feature, label], axis=1).groupby([feature.name])[label.name].agg(mode).to_dict()
    abnormal_value = label.mean()
    t = pd.Series([scale.get(i, abnormal_value) for i in feature], index=feature.index)
    return t, scale

def categorical_hist(feature, label, feature_scale=None):
    """Hist labels with value counts prob.
           
    Args:
        feature: pd.Series, sample feature.
        label: pd.Series, sample categorical label.
        feature_scale: pd.DataFrame, label parameters DataFrame for this estimator.
    Returns:
        return hist labels and label parameters DataFrame.
    """
    if feature_scale is not None:
        scale = feature_scale
    else:
        t = pd.concat([feature, label], axis=1).groupby([feature.name])[label.name].value_counts(normalize=True).unstack()
        t.columns = [t.columns.name+'_'+str(i) for i in t.columns]
        scale = t.reset_index().fillna(0)
    t = feature.to_frame().merge(scale, on=feature.name, how='left')
    return t, scale
