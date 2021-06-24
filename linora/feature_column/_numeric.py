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

def numeric_bucketized(feature, boundaries, miss_pad=-1, score=None, miss_score=None, dtype='int64', mode=1):
    """feature bucket.
    
    if mode is True:
        Buckets include the right boundary, and exclude the left boundary. 
        Namely, boundaries=[0., 1., 2.] generates buckets (-inf, 0.], (0., 1.], (1., 2.], and (2., +inf).
    else:
        Buckets include the left boundary, and exclude the right boundary. 
        Namely, boundaries=[0., 1., 2.] generates buckets (-inf, 0.), [0., 1.), [1., 2.), and [2., +inf).
        
    Args:
        feature: pd.Series, sample feature.
        boundaries: A sorted list or tuple of floats specifying the boundaries.
        miss_pad: default -1, feature fillna value.
        dtype: default 'int64', return transfrom dtypes.
        score: None, A score list or tuple of floats specifying the boundaries.
        miss_score: None, score fillna value.
        mode: True.
    Returns:
        normalize feature.
    """
    t = feature.copy()
    bound = sorted(boundaries)
    if mode:
        for i in range(len(bound)):
            if i==0:
                t[feature<=bound[i]] = i
            else:
                t[(feature>bound[i-1])&(feature<=bound[i])] = i
        t[feature>bound[i]] = i+1
    else:
        t[feature<bound[0]] = 0
        for r, i in enumerate(bound):
            t[feature>=i] = r+1
    t = t.fillna(miss_pad).astype(dtype)
    if isinstance(score, (tuple, list)):
        t = t.replace({i:j for i,j in enumerate(score)})
        if miss_score is not None:
            t = t.replace({miss_pad:miss_score}) 
    return t

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
    from scipy.stats import norm
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
