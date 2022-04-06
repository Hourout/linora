import numpy as np

__all__ = ['normalize_meanminmax', 'normalize_minmax', 'normalize_maxabs', 'normalize_max',
           'normalize_l1', 'normalize_l2', 'normalize_norm', 'normalize_robust']

def normalize_meanminmax(feature, feature_scale=None):
    """normalize feature with meanminmax method.
    
    Args:
        feature: pd.Series, sample feature value.
        feature_scale: list or tuple, [feature.mean(), feature.min(), feature.max()].
    Returns:
        normalize feature and feature_scale.
    """
    scale = feature_scale if feature_scale is not None else (feature.mean(), feature.min(), feature.max())
    t = (feature-scale[0])/(scale[2]-scale[1])
    return t, scale

def normalize_minmax(feature, feature_range=(0, 1), feature_scale=None):
    """normalize feature with minmax method.
    
    Args:
        feature: pd.Series, sample feature value.
        feature_range: list or tuple, range of values after feature transformation.
        feature_scale: list or tuple, [feature.min(), feature.max()].
    Returns:
        normalize feature and feature_scale.
    """
    scale = feature_scale if feature_scale is not None else (feature.min(), feature.max())
    t = (feature-scale[0])/(scale[1]-scale[0])*(feature_range[1]-feature_range[0])+feature_range[0]
    return t, scale

def normalize_maxabs(feature, feature_scale=None):
    """normalize feature with maxabs method.
    
    Args:
        feature: pd.Series, sample feature value.
        feature_scale: int or float, feature.abs().max().
    Returns:
        normalize feature and feature_scale.
    """
    scale = abs(feature_scale) if feature_scale is not None else feature.abs().max()
    t = feature/scale
    return t, scale

def normalize_max(feature, feature_scale=None):
    """normalize feature with max method.
    
    Args:
        feature: pd.Series, sample feature value.
        feature_scale: int or float, feature.max().
    Returns:
        normalize feature and feature_scale.
    """
    scale = feature_scale if feature_scale is not None else feature.max()
    t = feature/scale
    return t, scale

def normalize_l1(feature, feature_scale=None):
    """normalize feature with l1 method.
    
    Args:
        feature: pd.Series, sample feature value.
        feature_scale: int or float, feature.abs().sum().
    Returns:
        normalize feature and feature_scale.
    """
    scale = feature_scale if feature_scale is not None else feature.abs().sum()
    t = feature/scale
    return t, scale

def normalize_l2(feature, feature_scale=None):
    """normalize feature with l2 method.
    
    Args:
        feature: pd.Series, sample feature value.
        feature_scale: int or float, np.sqrt(np.sum(np.square(feature))).
    Returns:
        normalize feature and feature_scale.
    """
    scale = feature_scale if feature_scale is not None else np.sqrt(np.sum(np.square(feature)))
    t = feature/scale
    return t, scale

def normalize_norm(feature, feature_scale=None):
    """normalize feature with norm method.
    
    Args:
        feature: pd.Series, sample feature value.
        feature_scale: list or tuple, [feature.mean(), feature.std()].
    Returns:
        normalize feature and feature_scale.
    """
    scale = feature_scale if feature_scale is not None else (feature.mean(), feature.std())
    t = (feature-scale[0])/scale[1]
    return t, scale

def normalize_robust(feature, feature_scale=(None, 0.5)):
    """normalize feature with robust method.
    
    Args:
        feature: pd.Series, sample feature value.
        feature_scale: list or tuple, [feature.median(), feature.quantile(0.75)-feature.quantile(0.25)];
                       if feature_scale[0] is not None, 
                       (feature_scale[0], feature.quantile(0.5+feature_scale[1]/2)-feature.quantile(0.5-feature_scale[1]/2)).
    Returns:
        normalize feature and feature_scale.
    """
    if feature_scale[0] is not None:
        scale = (feature_scale[0], feature.quantile(0.5+feature_scale[1]/2)-feature.quantile(0.5-feature_scale[1]/2))
    else:
        scale = (feature.median(), feature.quantile(0.75)-feature.quantile(0.25))
    t = (feature-scale[0])/scale[1]
    return t, scale
