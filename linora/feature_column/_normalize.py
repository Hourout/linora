import numpy as np

__all__ ['normalize_minmax', 'normalize_maxabs', 'normalize_max', 'normalize_l1', 'normalize_l2']

def normalize_minmax(feature, feature_range=(0, 1), feature_scale=None):
    scale = feature_scale if feature_scale is not None else (feature.min(), feature.max())
    t = (feature-scale[0])/(scale[1]-scale[0])*(feature_range[1]-feature_range[0])+feature_range[0]
    return t, scale

def normalize_maxabs(feature, feature_scale=None):
    scale = abs(feature_scale) if feature_scale is not None else feature.abs().max()
    t = feature/scale
    return t, scale

def normalize_max(feature, feature_scale=None):
    scale = feature_scale if feature_scale is not None else feature.max()
    t = feature/scale
    return t, scale

def normalize_l1(feature, feature_scale=None):
    scale = feature_scale if feature_scale is not None else feature.abs().sum()
    t = feature/scale
    return t, scale

def normalize_l2(feature, feature_scale=None):
    scale = feature_scale if feature_scale is not None else np.sqrt(np.sum(np.square(feature)))
    t = feature/scale
    return t, scale
