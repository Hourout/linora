import itertools
from functools import reduce
from collections import defaultdict

import pandas as pd
import numpy as np

__all__ = ['categorical_count', 'categorical_crossed','categorical_encoder', 
           'categorical_hash', 'categorical_hist',
           'categorical_onehot_binarizer', 'categorical_onehot_multiple',
           'categorical_regress', 
          ]


def categorical_count(feature, abnormal_value=0, miss_value=0, normalize=True, config=None, name=None, mode=0):
    """Count or frequency of conversion category variables.
    
    Args:
        feature: pd.Series, sample feature.
        abnormal_value: int or float, if feature values not in feature_scale dict, return `abnormal_value`.
        miss_value: int or float, if feature values are missing, return `miss_value`.
        normalize: bool, If True then the object returned will contain the relative frequencies of the unique values.
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
    Returns:
        return count labels and label parameters dict.
    """
    if config is None:
        config = {'feature_scale':feature.value_counts(normalize).to_dict(),
                  'abnormal_value':abnormal_value, 'miss_value':miss_value, 
                  'type':'categorical_count', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        scale = {**config['feature_scale'], **{i:config['abnormal_value'] for i in feature.unique().tolist() if i not in config['feature_scale']}}
        t = feature.replace(scale).fillna(config['miss_value']).rename(config['name_output'])
        return t if mode else (t, config)


def categorical_crossed(feature_list, hash_bucket_size=3, config=None, name=None, mode=0):
    """Crossed categories and hash labels with value between 0 and hash_bucket_size-1.
    
    Args:
        feature_list: pd.Series list, sample feature list.
        hash_bucket_size: int, number of categories that need hash.
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
    Returns:
        return hash labels.
    """
    if config is None:
        config = {'hash_bucket_size':hash_bucket_size, 'type':'categorical_crossed', 
                  'name_input':[i.name for i in feature_list], 'name_output':name}
    if mode==2:
        return config
    else:
        t = reduce(lambda x,y:x+y, [i.fillna('').astype(str) for i in feature_list]).map(lambda x:hash(x))%config['hash_bucket_size']
        if config['name_output'] is not None:
            t = t.rename(config['name_output'])
        return t if mode else (t, config)


def categorical_encoder(feature, abnormal_value=-1, miss_value=-1, config=None, name=None, mode=0):
    """Encode labels with value between 0 and n_classes-1.
    
    Args:
        feature: pd.Series, sample feature.
        abnormal_value: int, if feature values not in feature_scale dict, return `abnormal_value`.
        miss_value: int or float, if feature values are missing, return `miss_value`.
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
    Returns:
        return encoded labels and label parameters dict.
    """
    if config is None:
        config = {'feature_scale':{j:i for i,j in feature.drop_duplicates().reset_index(drop=True).to_dict().items()},
                  'abnormal_value':abnormal_value, 'miss_value':miss_value, 
                  'type':'categorical_encoder', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        scale = {**config['feature_scale'], **{i:config['abnormal_value'] for i in feature.unique().tolist() if i not in config['feature_scale']}}
        t = feature.replace(scale).fillna(config['miss_value']).rename(config['name_output'])
        return t if mode else (t, config)


def categorical_hash(feature, hash_bucket_size=3, config=None, name=None, mode=0):
    """Hash labels with value between 0 and hash_bucket_size-1.
    
    Args:
        feature: pd.Series, sample feature.
        hash_bucket_size: int, number of categories that need hash.
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
    Returns:
        return hash labels.
    """
    if config is None:
        config = {'hash_bucket_size':hash_bucket_size, 'type':'categorical_hash', 
                  'name_input':feature.name, 'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        t = feature.fillna('').astype(str).map(lambda x:hash(x))%config['hash_bucket_size']
        if config['name_output'] is not None:
            t = t.rename(config['name_output'])
        return t if mode else (t, config)


def categorical_hist(feature, label, abnormal_value=0, miss_value=0, config=None, name=None, mode=0):
    """Hist labels with value counts prob.
           
    Args:
        feature: pd.Series, sample feature.
        label: pd.Series, sample categorical label.
        abnormal_value: int, if feature values not in feature_scale dict, return `abnormal_value`.
        miss_value: int or float, if feature values are missing, return `miss_value`.
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
    Returns:
        return hist labels and label parameters DataFrame.
    """
    if config is None:
        config = {'feature_scale':None,
                  'abnormal_value':abnormal_value, 'miss_value':miss_value, 
                  'type':'categorical_hist', 'name_input':[feature.name, label.name], 
                  'name_output':feature.name if name is None else name}
        t = pd.concat([feature, label], axis=1).groupby([feature.name])[label.name].value_counts(normalize=True).unstack()
        t.columns = [config['name_output']+'_'+str(i) for i in t.columns]
        config['feature_scale'] = t.reset_index().to_dict()
        
    if mode==2:
        return config
    else:
        t = (feature.to_frame().merge(pd.DataFrame(config['feature_scale']).fillna(config['miss_value']), on=feature.name, how='left')
             .drop([feature.name], axis=1).fillna(config['abnormal_value']))
        return t if mode else (t, config)


def categorical_onehot_binarizer(feature, feature_scale=None, prefix='columns', dtype='int8'):
    """Transform between iterable of iterables and a multilabel format, sample is simple categories.
    
    Args:
        feature: pd.Series, sample feature.
        feature_scale: list, feature categories list.
        prefix: str, String to append DataFrame column names.
        dtype: str, default int8. Data type for new columns.
    Returns:
        Dataframe for onehot binarizer and feature parameters list.
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
        prefix: str, String to append DataFrame column names.
        dtype: str, default int8. Data type for new columns.
    Returns:
        Dataframe for onehot binarizer and feature parameters list.
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


def categorical_regress(feature, label, method='mean', abnormal_value='mean', miss_value='mean', config=None, name=None, mode=0):
    """Regress labels with value counts prob.
    
    Args:
        feature: pd.Series, sample feature.
        label: pd.Series, sample regress label.
        mode: 'mean' or 'median'
        abnormal_value: int, if feature values not in feature_scale dict, return `abnormal_value`.
        miss_value: int or float, if feature values are missing, return `miss_value`.
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
    Returns:
        return Regress labels and label parameters dict.
    """
    if config is None:
        config = {'feature_scale':pd.concat([feature, label], axis=1).groupby([feature.name])[label.name].agg(mode).to_dict(),
                  'abnormal_value':label.mean() if mode=='mean' else label.median(), 
                  'miss_value':label.mean() if mode=='mean' else label.median(), 
                  'type':'categorical_hist', 'name_input':[feature.name, label.name], 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        scale = {**config['feature_scale'], **{i:config['abnormal_value'] for i in feature.unique().tolist() if i not in config['feature_scale']}}
        t = feature.replace(scale).fillna(config['miss_value']).rename(config['name_output'])
        return t if mode else (t, config)
