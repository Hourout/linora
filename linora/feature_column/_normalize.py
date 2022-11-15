import numpy as np

__all__ = ['normalize_max', 'normalize_maxabs', 'normalize_l1', 'normalize_l2', 
           'normalize_meanminmax', 'normalize_minmax', 'normalize_norm', 'normalize_robust']


def normalize_max(feature, config=None, name=None, mode=0):
    """normalize feature with max method.
    
    Args:
        feature: pd.Series, sample feature value.
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
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
        t = feature/scale
        return t if mode else (t, config)
    

def normalize_maxabs(feature, config=None, name=None, mode=0):
    """normalize feature with maxabs method.
    
    Args:
        feature: pd.Series, sample feature value.
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
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
        t = feature/scale
        return t if mode else (t, config)


def normalize_l1(feature, config=None, name=None, mode=0):
    """normalize feature with l1 method.
    
    Args:
        feature: pd.Series, sample feature value.
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
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
        t = feature/scale
        return t if mode else (t, config)


def normalize_l2(feature, config=None, name=None, mode=0):
    """normalize feature with l2 method.
    
    Args:
        feature: pd.Series, sample feature value.
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
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
        t = feature/scale
        return t if mode else (t, config)


def normalize_meanminmax(feature, config=None, name=None, mode=0):
    """normalize feature with meanminmax method.
    
    Args:
        feature: pd.Series, sample feature value.
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
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
        t = (feature-scale[0])/(scale[2]-scale[1])
        return t if mode else (t, config)


def normalize_minmax(feature, feature_range=(0, 1), config=None, name=None, mode=0):
    """normalize feature with minmax method.
    
    Args:
        feature: pd.Series, sample feature value.
        feature_range: list or tuple, range of values after feature transformation.
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
    Returns:
        normalize feature and feature_scale.
    """
    if config is None:
        config = {'feature_scale':[feature.min(), feature.max()],
                  'type':'normalize_minmax', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        scale = config['feature_scale']
        t = (feature-scale[0])/(scale[1]-scale[0])*(feature_range[1]-feature_range[0])+feature_range[0]
        return t if mode else (t, config)


def normalize_norm(feature, config=None, name=None, mode=0):
    """normalize feature with norm method.
    
    Args:
        feature: pd.Series, sample feature value.
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
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
        t = (feature-scale[0])/scale[1]
        return t if mode else (t, config)


def normalize_robust(feature, feature_scale=(0.5, 0.5), config=None, name=None, mode=0):
    """normalize feature with robust method.
    
    Args:
        feature: pd.Series, sample feature value.
        feature_scale: list or tuple, each element is in the [0,1] interval.
                       (feature_scale[0], feature.quantile(0.5+feature_scale[1]/2)-feature.quantile(0.5-feature_scale[1]/2)).
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
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
        t = (feature-scale[0])/scale[1]
        return t if mode else (t, config)
