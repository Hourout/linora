from scipy.stats import norm


__all__ = ['numeric_binarizer', 'numeric_bucketized', 'numeric_padding', 'numeric_outlier']

def numeric_binarizer(feature, feature_scale=None):
    """Encode labels with value between 0 and 1.
    
    Args:
        feature: pd.Series, sample feature.
        feature_scale: float, feature.mean().
    Returns:
        normalize feature and feature_scale.
    """
    scale = feature_scale if feature_scale is not None else feature.mean()
    t = feature.clip(upper=scale).replace({scale:scale+0.1}).clip(scale).replace({scale:0, scale+0.1:1}).astype('int')
    return t, scale

def numeric_bucketized(feature, boundaries, miss_pad=-1, dtype='int64'):
    """feature bucket.
    
    Buckets include the left boundary, and exclude the right boundary. 
    Namely, boundaries=[0., 1., 2.] generates buckets (-inf, 0.), [0., 1.), [1., 2.), and [2., +inf).
    
    Args:
        feature: pd.Series, sample feature.
        boundaries: A sorted list or tuple of floats specifying the boundaries.
        miss_pad: default -1, feature fillna value.
        dtype: default 'int64', return transfrom dtypes.
    Returns:
        normalize feature.
    """
    t = feature.copy()
    bound = sorted(boundaries)
    t[feature<bound[0]] = 0
    for r, i in enumerate(bound):
        t[feature>=i] = r+1
    return t.fillna(miss_pad).astype(dtype)

def numeric_padding(feature, method='mean', feature_scale=None):
    """feature fillna method.
    
    Args:
        feature: pd.Series, sample feature.
        method: default 'mean', one of 'mean' or 'median'.
        feature_scale: float, feature.mean() or feature.median().
    Returns:
        normalize feature and feature_scale.
    """
    assert method in ['mean', 'median'], "`method` should be one of ['mean', 'median']."
    scale_dict = {'mean':feature.mean(), 'median':feature.median()}
    scale = feature_scale if feature_scale is not None else scale_dict[method]
    t = feature.fillna(scale)
    return t, scale

def numeric_outlier(feature, keep_rate=0.9545, mode='right', feature_scale=None):
    """feature clip outlier.
    
    Args:
        feature: pd.Series, sample feature.
        keep_rate: default 0.9545, 
        method: default 'right', one of ['left', 'right', 'both'], statistical distribution boundary.
        feature_scale: list or tuple, [feature.mean(), feature.std()].
    Returns:
        normalize feature and feature_scale.
    """
    assert mode in ['left', 'right', 'both'], "`mode` should be one of ['left', 'right', 'both']."
    scale = feature_scale if feature_scale is not None else (feature.mean(), feature.std())
    if mode=='both':
        clip_dict = (feature.mean()+norm.ppf((1-keep_rate)/2)*feature.std(), feature.mean()+norm.ppf(keep_rate+(1-keep_rate)/2)*feature.std())
    elif mode=='right':
        clip_dict = (feature.min(), feature.mean()+norm.ppf(keep_rate)*feature.std())
    else:
        clip_dict = (feature.mean()+norm.ppf(1-keep_rate)*feature.std(), feature.max())
    t = feature.clip(clip_dict[0], clip_dict[1])
    return t, scale
