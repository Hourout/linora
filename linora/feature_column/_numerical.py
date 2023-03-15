__all__ = ['numerical_binarizer', 'numerical_bucketized', 
           'numerical_padding', 'numerical_outlier']


def numerical_binarizer(feature, mode=0, method='mean', name=None, config=None):
    """Encode labels with value between 0 and 1.
    
    Args:
        feature: pd.Series, sample feature.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        method: default 'mean', one of 'mean' or 'median' or float or int.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        normalize feature and feature_scale.
    """
    if config is None:
        config = {'feature_scale':None,
                  'type':'numerical_binarizer', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
        if method=='mean':
            config['feature_scale'] = feature.mean()
        elif method=='median':
            config['feature_scale'] = feature.median()
        else:
            config['feature_scale'] = method
    if mode==2:
        return config
    else:
        scale = config['feature_scale']
        t = (feature.clip(upper=scale).replace({scale:scale+0.1}).clip(scale)
             .replace({scale:0, scale+0.1:1}).astype('int8').rename(config['name_output']))
        return t if mode else (t, config)


def numerical_bucketized(feature, boundaries, mode=0, miss_pad=-1, score=None, miss_score=None, method=1, name=None, config=None):
    """feature bucket.
    
    if method is True:
        Buckets include the right boundary, and exclude the left boundary. 
        Namely, boundaries=[0., 1., 2.] generates buckets (-inf, 0.], (0., 1.], (1., 2.], and (2., +inf).
    else:
        Buckets include the left boundary, and exclude the right boundary. 
        Namely, boundaries=[0., 1., 2.] generates buckets (-inf, 0.), [0., 1.), [1., 2.), and [2., +inf).
        
    Args:
        feature: pd.Series, sample feature.
        boundaries: list, A sorted list or tuple of floats specifying the boundaries.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        miss_pad: int, default -1, feature fillna value.
        score: None, A score list or tuple of floats specifying the boundaries.
        miss_score: int or float, None, score fillna value.
        method: True.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid, but `boundaries` must be passed in.
    Returns:
        normalize feature.
    """
    if config is None:
        config = {'feature_scale':boundaries, 'miss_pad':miss_pad, 'score':score, 
                  'miss_score':miss_score, 'method':method, 
                  'type':'numerical_bucketized', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
    if mode==2:
        return config
    else:
        t = feature.copy()
        bound = sorted(config['boundaries'])
        if config['method']:
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
        t = t.fillna(config['miss_pad']).astype('int64').rename(config['name_output'])
        if isinstance(config['score'], (tuple, list)):
            t = t.replace({i:j for i,j in enumerate(config['score'])})
            if config['miss_score'] is not None:
                t = t.replace({config['miss_pad']:config['miss_score']})
        return t if mode else (t, config)


def numerical_padding(feature, mode=0, method='mean', name=None, config=None):
    """feature fillna method.
    
    Args:
        feature: pd.Series, sample feature.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        method: default 'mean', one of 'mean' or 'median' or float or int.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        normalize feature and feature_scale.
    """
    if config is None:
        config = {'feature_scale':None,
                  'type':'numerical_padding', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
        if method=='mean':
            config['feature_scale'] = feature.mean()
        elif method=='median':
            config['feature_scale'] = feature.median()
        else:
            config['feature_scale'] = method
    if mode==2:
        return config
    else:
        scale = config['feature_scale']
        t = feature.fillna(scale).rename(config['name_output'])
        return t if mode else (t, config)


def numerical_outlier(feature, mode=0, method='norm', delta=0.9545, tail='right', name=None, config=None):
    """feature clip outlier.
    
    Caps maximum and/or minimum values of a variable at automatically determined values, and optionally adds indicators
    The extreme values beyond which an observation is considered an outlier are determined 
    
    using:
    - norm
    - a Gaussian approximation
    - the inter-quantile range proximity rule (IQR)
    - MAD-median rule (MAD)
    - percentiles

    norm limits:
        right tail: mean + ppf((1-delta)/2)* std
        left tail: mean - ppf(delta+(1-delta)/2)* std
    
    Gaussian limits(Delta is recommended to be 3):
        right tail: mean + 3* std
        left tail: mean - 3* std

    IQR limits(Delta is recommended to be 3):
        right tail: 75th quantile + 3* IQR
        left tail: 25th quantile - 3* IQR
    where IQR is the inter-quartile range: 75th quantile - 25th quantile.

    MAD limits(Delta is recommended to be 3):
        right tail: median + 3* MAD
        left tail: median - 3* MAD
    where MAD is the median absoulte deviation from the median.

    percentiles:
        right tail: 95th percentile
        left tail: 5th percentile

    You can select how far out to cap the maximum or minimum values with the parameter 'delta'.
    If capping_method='gaussian' delta gives the value to multiply the std.
    If capping_method='iqr' delta is the value to multiply the IQR.
    If capping_method='mad' delta is the value to multiply the MAD.
    If capping_method='quantiles', delta is the percentile on each tail that should be censored. 
        For example, if delta=0.05, the limits will be the 5th and 95th percentiles. 
        If delta=0.1, the limits will be the 10th and 90th percentiles.
    Args:
        feature: pd.Series, sample feature.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        method: must be one of ['norm', 'gaussian', 'iqr', 'mad', 'quantiles']
        delta: float, default 0.9545
        tail: str, default 'right', one of ['left', 'right', 'both'], statistical distribution boundary.
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        normalize feature and feature_scale.
    """
    if config is None:
        config = {'method':method, 'tail':tail, 'delta':delta,
                  'type':'numerical_outlier', 'name_input':feature.name, 
                  'name_output':feature.name if name is None else name}
        if method =='norm':
            if not 0<delta<1:
                raise ValueError("`delta` must be 0<delta<1")
            from scipy.stats import norm
            config['feature_scale'] = [feature.mean(), feature.std()]
        elif method=='gaussian':
            config['feature_scale'] = [feature.mean(), feature.std()]
        elif method=='iqr':
            config['feature_scale'] = [feature.quantile(0.25), feature.quantile(0.75)]
        elif method=='mad':
            config['feature_scale'] = [feature.median(), (feature-feature.median()).abs().median()]
        elif method=='quantiles':
            if not 0<delta<0.5:
                raise ValueError("`delta` must be 0<delta<0.5")
            config['feature_scale'] = [feature.quantile(delta), feature.quantile(1-delta)]
        else:
            raise ValueError("`method` must be one of ['norm', 'gaussian', 'iqr', 'mad', 'quantiles']")
    if mode==2:
        return config
    else:
        scale = config['feature_scale']
        delta = config['delta']
        if method =='norm':
            if config['tail']=='both':
                clip = (scale[0]+norm.ppf((1-delta)/2)*scale[1], scale[0]+norm.ppf(delta+(1-delta)/2)*scale[1])
            elif config['tail']=='right':
                clip = (feature.min(), scale[0]+norm.ppf(delta)*scale[1])
            else:
                clip = (scale[0]+norm.ppf(1-delta)*scale[1], feature.max())
        elif method in ['gaussian', 'mad']:
            if config['tail']=='both':
                clip = (scale[0]-delta*scale[1], scale[0]+delta*scale[1])
            elif config['tail']=='right':
                clip = (feature.min(), scale[0]+delta*scale[1])
            else:
                clip = (scale[0]-delta*scale[1], feature.max())
        elif method =='iqr':
            if config['tail']=='both':
                clip = (scale[0]-delta*(scale[1]-scale[0]), scale[1]+delta*(scale[1]-scale[0]))
            elif config['tail']=='right':
                clip = (feature.min(), scale[1]+delta*(scale[1]-scale[0]))
            else:
                clip = (scale[0]-delta*(scale[1]-scale[0]), feature.max())
        elif method =='quantiles':
            if config['tail']=='both':
                clip = (scale[0], scale[1])
            elif config['tail']=='right':
                clip = (feature.min(), scale[1])
            else:
                clip = (scale[0], feature.max())
        else:
            raise ValueError("`method` must be one of ['norm', 'gaussian', 'iqr', 'mad', 'quantiles']")
        t = feature.clip(clip[0], clip[1]).rename(config['name_output'])
        return t if mode else (t, config)
