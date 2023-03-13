import itertools
from functools import reduce
from collections import defaultdict

import pandas as pd
import numpy as np

__all__ = ['categorical_count', 'categorical_crossed','categorical_encoder', 
           'categorical_hash', 'categorical_hist',
           'categorical_onehot_binarizer', 'categorical_onehot_multiple',
           'categorical_rare', 'categorical_regress', 'categorical_woe'
          ]


def categorical_count(feature, mode=0, normalize=True, abnormal_value=0, miss_value=0, name=None, config=None):
    """Count or frequency of conversion category variables.
    
    Args:
        feature: pd.Series, sample feature.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        normalize: bool, If True then the object returned will contain the relative frequencies of the unique values.
        abnormal_value: int or float, if feature values not in feature_scale dict, return `abnormal_value`.
        miss_value: int or float, if feature values are missing, return `miss_value`.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
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


def categorical_crossed(feature_list, mode=0, hash_bucket_size=3, name=None, config=None):
    """Crossed categories and hash labels with value between 0 and hash_bucket_size-1.
    
    Args:
        feature_list: pd.Series list, sample feature list.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        hash_bucket_size: int, number of categories that need hash.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature_list` and `mode` is invalid.
    Returns:
        return hash labels.
    """
    if config is None:
        config = {'hash_bucket_size':hash_bucket_size, 'type':'categorical_crossed', 
                  'name_input':[i.name for i in feature_list], 'name_output':name}
    if mode==2:
        return config
    else:
        t = reduce(lambda x,y:x+y, [i.fillna('').astype(str) for i in feature_list]).map(lambda x:hash(x)).rename(config['name_output'])%config['hash_bucket_size']
        return t if mode else (t, config)


def categorical_encoder(feature, mode=0, abnormal_value=-1, miss_value=-1, name=None, config=None):
    """Encode labels with value between 0 and n_classes-1.
    
    Args:
        feature: pd.Series, sample feature.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        abnormal_value: int, if feature values not in feature_scale dict, return `abnormal_value`.
        miss_value: int or float, if feature values are missing, return `miss_value`.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
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


def categorical_hash(feature, mode=0, hash_bucket_size=3, name=None, config=None):
    """Hash labels with value between 0 and hash_bucket_size-1.
    
    Args:
        feature: pd.Series, sample feature.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        hash_bucket_size: int, number of categories that need hash.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        return hash labels.
    """
    if config is None:
        config = {'hash_bucket_size':hash_bucket_size, 'type':'categorical_hash', 
                  'name_input':feature.name, 'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        t = feature.fillna('').astype(str).map(lambda x:hash(x)).rename(config['name_output'])%config['hash_bucket_size']
        return t if mode else (t, config)


def categorical_hist(feature, label, mode=0, abnormal_value=0, miss_value=0, name=None, config=None):
    """Hist labels with value counts prob.
           
    Args:
        feature: pd.Series, sample feature.
        label: pd.Series, sample categorical label.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        abnormal_value: int, if feature values not in feature_scale dict, return `abnormal_value`.
        miss_value: int or float, if feature values are missing, return `miss_value`.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid, but `label` must be passed in.
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
             .drop([feature.name], axis=1).fillna(config['abnormal_value']).rename(config['name_output']))
        return t if mode else (t, config)


def categorical_onehot_binarizer(feature, mode=0, abnormal_value=0, miss_value=0, name=None, config=None):
    """Transform between iterable of iterables and a multilabel format, sample is simple categories.
    
    Args:
        feature: pd.Series, sample feature.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        abnormal_value: int, if feature values not in feature_scale dict, return `abnormal_value`.
        miss_value: int or float, if feature values are missing, return `miss_value`.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        Dataframe for onehot binarizer and feature parameters list.
    """
    if config is None:
        config = {'feature_scale':feature.dropna().drop_duplicates().tolist(),
                  'abnormal_value':abnormal_value, 'miss_value':miss_value, 
                  'type':'categorical_onehot_binarizer', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        scale = feature.dropna().drop_duplicates().tolist()
        scale_dict = {i:'temp_str' for i in set.difference(set(scale), set(config['feature_scale']))}
        t = pd.get_dummies(feature.replace(scale_dict), prefix=config['name_output'], dtype='int8', dummy_na=True)
        
        for i in set.difference(set(config['feature_scale']), set(scale)):
            if config['name_output']+'_'+str(i) not in t.columns:
                t[config['name_output']+'_'+str(i)] = 0
                
        if f"{config['name_output']}_temp_str" in t.columns:
            t.loc[t[f"{config['name_output']}_temp_str"]==1, :] = config['abnormal_value']
            t = t.drop([f"{config['name_output']}_temp_str"], axis=1)
            
        if f"{config['name_output']}_nan" in t.columns:
            t.loc[t[f"{config['name_output']}_nan"]==1, :] = config['miss_value']
            t = t.drop([f"{config['name_output']}_nan"], axis=1)
        t = t[[config['name_output']+'_'+str(i) for i in config['feature_scale']]]
        return t if mode else (t, config)


def categorical_onehot_multiple(feature, mode=0, abnormal_value=0, miss_value=0, name=None, config=None):
    """Transform between iterable of iterables and a multilabel format, sample is multiple categories.
    
    Args:
        feature: pd.Series, sample feature.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        abnormal_value: int, if feature values not in feature_scale dict, return `abnormal_value`.
        miss_value: int or float, if feature values are missing, return `miss_value`.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        Dataframe for onehot binarizer and feature parameters list.
    """
    if config is None:
        config = {'feature_scale':list(set(itertools.chain.from_iterable(feature.dropna()))),
                  'abnormal_value':abnormal_value, 'miss_value':miss_value, 
                  'type':'categorical_onehot_multiple', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        scale = list(set(itertools.chain.from_iterable(feature.fillna('_'))))        
        class_mapping = defaultdict(int)
        class_mapping.default_factory = class_mapping.__len__
        [class_mapping[i] for i in scale]
        col = [class_mapping[i] for i in itertools.chain.from_iterable(feature.fillna('_'))]
        row = [i for i in itertools.chain.from_iterable(map(lambda x,y:[x]*len(y), range(len(feature)), feature.fillna('_')))]
        t = np.zeros([max(row)+1, len(class_mapping)], dtype='int8')
        t[row, col] = 1
        t = pd.DataFrame(t, columns=[config['name_output']+'_'+str(i) for i in scale], index=feature.index)
    
        for i in set.difference(set(config['feature_scale']), set(scale)):
            if config['name_output']+'_'+str(i) not in t.columns:
                t[config['name_output']+'_'+str(i)] = 0

        if f"{config['name_output']}__" in t.columns:
            t.loc[t[f"{config['name_output']}__"]==1, :] = config['miss_value']
            t = t.drop([f"{config['name_output']}__"], axis=1)
        
        t = t[[config['name_output']+'_'+str(i) for i in config['feature_scale']]]
        return t if mode else (t, config)


def categorical_rare(feature, mode=0, p=0.05, min_num=None, max_num=None, abnormal_value=-1, miss_value=-1, name=None, config=None):
    """Groups rare or infrequent categories in a new category called “Rare”, or any other name entered by the user.
    
    Args:
        feature: pd.Series, sample feature.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        p: The minimum frequency a label should have to be considered frequent. Categories with frequencies lower than tol will be grouped.
        min_num: The minimum number of categories a variable should have for the encoder to find frequent labels. 
            If the variable contains less categories, all of them will be considered frequent.
        max_num: The maximum number of categories that should be considered frequent. 
            If None, all categories with frequency above the tolerance (tol) will be considered frequent. 
            If you enter 5, only the 4 most frequent categories will be retained and the rest grouped.
        abnormal_value: int or float, if feature values not in feature_scale dict, return `abnormal_value`.
        miss_value: int or float, if feature values are missing, return `miss_value`.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        return count labels and label parameters dict.
    """
    if config is None:
        t = feature.value_counts(True)
        if min_num is None:
            min_num = len(t[t>p])
        if max_num is None:
            max_num = max(min_num, len(t[t>p])+1)
        k = 1 if len(t[t<=p])>0 else 0
        if len(t)<min_num:
            feature_scale = {i:r for r, i in enumerate(t.index)}
        elif len(t[t>p])+k<min_num:
            feature_scale = {**{i:r for r, i in enumerate(t.iloc[:min_num-1].index)},
                             **{i:min_num-1 for i in t.iloc[min_num-1:].index}}
        elif len(t[t>p])+k>max_num:
            feature_scale = {**{i:r for r, i in enumerate(t.iloc[:max_num-1].index)},
                             **{i:max_num-1 for i in t.iloc[max_num-1:].index}}
        else:
            feature_scale = {**{i:r for r, i in enumerate(t[t>p].index)},
                             **{i:len(t[t>p]) for i in t[t<=p].index}}
            
        config = {'feature_scale':feature_scale,
                  'abnormal_value':abnormal_value, 'miss_value':miss_value, 
                  'type':'categorical_rare', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        scale = {**config['feature_scale'], **{i:config['abnormal_value'] for i in feature.unique().tolist() if i not in config['feature_scale']}}
        t = feature.replace(scale).fillna(config['miss_value']).rename(config['name_output'])
        return t if mode else (t, config)


def categorical_regress(feature, label, mode=0, method='mean', abnormal_value='mean', miss_value='mean', name=None, config=None):
    """Regress labels with value counts prob.
    
    Args:
        feature: pd.Series, sample feature.
        label: pd.Series, sample regress label.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        method: 'mean' or 'median'
        abnormal_value: int, if feature values not in feature_scale dict, return `abnormal_value`.
        miss_value: int or float, if feature values are missing, return `miss_value`.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid, but `label` must be passed in.
    Returns:
        return Regress labels and label parameters dict.
    """
    if config is None:
        config = {'feature_scale':pd.concat([feature, label], axis=1).groupby([feature.name])[label.name].agg(method).to_dict(),
                  'abnormal_value':label.mean() if abnormal_value=='mean' else label.median(), 
                  'miss_value':label.mean() if miss_value=='mean' else label.median(), 
                  'type':'categorical_regress', 'name_input':[feature.name, label.name], 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        scale = {**config['feature_scale'], **{i:config['abnormal_value'] for i in feature.unique().tolist() if i not in config['feature_scale']}}
        t = feature.replace(scale).fillna(config['miss_value']).rename(config['name_output'])
        return t if mode else (t, config)

    
def categorical_woe(feature, label, mode=0, pos_label=1, abnormal_value=-1, miss_value=-1, name=None, config=None):
    """Calculate series woe value
    
    Args:
        feature: pd.Series, shape (n_samples,) x variable, model feature.
        label: pd.Series, sample categorical label.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        pos_label: int, default=1, positive label value.
        abnormal_value: int, if feature values not in feature_scale dict, return `abnormal_value`.
        miss_value: int or float, if feature values are missing, return `miss_value`.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid, but `label` must be passed in.
    Return:
        return series woe value and label parameters DataFrame.
    """
    if config is None:
        t = pd.DataFrame({'label':label, 'feature':feature})
        assert t.label.nunique()==2, "`y_true` should be binary classification."
        label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
        t['label'] = t.label.replace(label_dict)
        corr = t.label.sum()/(t.label.count()-t.label.sum())
        t = t.groupby(['feature']).label.apply(lambda x:np.log(x.sum()/(x.count()-x.sum())/corr))
        
        config = {'feature_scale':t.to_dict(),
                  'abnormal_value':abnormal_value, 'miss_value':miss_value, 
                  'type':'categorical_woe', 'name_input':[feature.name, label.name], 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        scale = {**config['feature_scale'], **{i:config['abnormal_value'] for i in feature.unique().tolist() if i not in config['feature_scale']}}
        t = feature.replace(scale).fillna(config['miss_value']).rename(config['name_output'])
        return t if mode else (t, config)
    

