import numpy as np

__all__ = ['normalize_max', 'normalize_maxabs', 'normalize_l1', 'normalize_l2', 
           'normalize_meanminmax', 'normalize_minmax', 'normalize_norm', 'normalize_robust']


def normalize_max(feature, mode=0, name=None, config=None):
    """normalize feature with max method.
    
    Args:
        feature: pd.Series, sample feature value.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        normalize feature and feature_scale.
    """
    if config is None:
        config = {'feature_scale':feature.max(),
                  'type':'normalize_max', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        scale = config['feature_scale']
        t = (feature/scale).rename(config['name_output'])
        return t if mode else (t, config)
    

def normalize_maxabs(feature, mode=0, name=None, config=None):
    """normalize feature with maxabs method.
    
    Args:
        feature: pd.Series, sample feature value.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        normalize feature and feature_scale.
    """
    if config is None:
        config = {'feature_scale':feature.abs().max(),
                  'type':'normalize_maxabs', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        scale = config['feature_scale']
        t = (feature/scale).rename(config['name_output'])
        return t if mode else (t, config)


def normalize_l1(feature, mode=0, name=None, config=None):
    """normalize feature with l1 method.
    
    Args:
        feature: pd.Series, sample feature value.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        normalize feature and feature_scale.
    """
    if config is None:
        config = {'feature_scale':feature.abs().sum(),
                  'type':'normalize_l1', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        scale = config['feature_scale']
        t = (feature/scale).rename(config['name_output'])
        return t if mode else (t, config)


def normalize_l2(feature, mode=0, name=None, config=None):
    """normalize feature with l2 method.
    
    Args:
        feature: pd.Series, sample feature value.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        normalize feature and feature_scale.
    """
    if config is None:
        config = {'feature_scale':np.sqrt(np.sum(np.square(feature))),
                  'type':'normalize_l2', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        scale = config['feature_scale']
        t = (feature/scale).rename(config['name_output'])
        return t if mode else (t, config)


def normalize_meanminmax(feature, mode=0, name=None, config=None):
    """normalize feature with meanminmax method.
    
    Args:
        feature: pd.Series, sample feature value.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        normalize feature and feature_scale.
    """
    if config is None:
        config = {'feature_scale':[feature.mean(), feature.min(), feature.max()],
                  'type':'normalize_meanminmax', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        scale = config['feature_scale']
        t = ((feature-scale[0])/(scale[2]-scale[1])).rename(config['name_output'])
        return t if mode else (t, config)


def normalize_minmax(feature, mode=0, feature_range=(0, 1), name=None, config=None):
    """normalize feature with minmax method.
    
    Args:
        feature: pd.Series, sample feature value.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        feature_range: list or tuple, range of values after feature transformation.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        normalize feature and feature_scale.
    """
    if config is None:
        config = {'feature_scale':[feature.min(), feature.max()],
                  'feature_range':feature_range,
                  'type':'normalize_minmax', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        scale = config['feature_scale']
        ran = config['feature_range']
        t = ((feature-scale[0])/(scale[1]-scale[0])*(ran[1]-ran[0])+ran[0]).rename(config['name_output'])
        return t if mode else (t, config)


def normalize_norm(feature, mode=0, name=None, config=None):
    """normalize feature with norm method.
    
    Args:
        feature: pd.Series, sample feature value.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        normalize feature and feature_scale.
    """
    if config is None:
        config = {'feature_scale':[feature.mean(), feature.std()],
                  'type':'normalize_norm', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        scale = config['feature_scale']
        t = ((feature-scale[0])/scale[1]).rename(config['name_output'])
        return t if mode else (t, config)


def normalize_robust(feature, mode=0, feature_scale=(0.5, 0.5), name=None, config=None):
    """normalize feature with robust method.
    
    Args:
        feature: pd.Series, sample feature value.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        feature_scale: list or tuple, each element is in the [0,1] interval.
                       (feature_scale[0], feature.quantile(0.5+feature_scale[1]/2)-feature.quantile(0.5-feature_scale[1]/2)).
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        normalize feature and feature_scale.
    """
    if config is None:
        config = {'feature_scale':[feature.quantile(feature_scale[0]), feature.quantile(0.5+feature_scale[1]/2)-feature.quantile(0.5-feature_scale[1]/2)],
                  'type':'normalize_robust', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        scale = config['feature_scale']
        t = ((feature-scale[0])/scale[1]).rename(config['name_output'])
        return t if mode else (t, config)
