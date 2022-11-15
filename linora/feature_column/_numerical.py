__all__ = ['numerical_binarizer', 'numerical_bucketized', 
           'numerical_padding', 'numerical_outlier']


def numerical_binarizer(feature, method='mean', config=None, name=None, mode=0):
    """Encode labels with value between 0 and 1.
    
    Args:
        feature: pd.Series, sample feature.
        method: default 'mean', one of 'mean' or 'median' or float or int.
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
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


def numerical_bucketized(feature, boundaries, miss_pad=-1, score=None, miss_score=None, method=1, config=None, name=None, mode=0):
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
        miss_pad: int, default -1, feature fillna value.
        dtype: default 'int64', return transfrom dtypes.
        score: None, A score list or tuple of floats specifying the boundaries.
        miss_score: int or float, None, score fillna value.
        method: True.
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
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
        t = t.fillna(config['miss_pad']).astype('int64').rename(config['name_output'])
        if isinstance(config['score'], (tuple, list)):
            t = t.replace({i:j for i,j in enumerate(config['score'])})
            if config['miss_score'] is not None:
                t = t.replace({config['miss_pad']:config['miss_score']})
        return t if mode else (t, config)


def numerical_padding(feature, method='mean', config=None, name=None, mode=0):
    """feature fillna method.
    
    Args:
        feature: pd.Series, sample feature.
        method: default 'mean', one of 'mean' or 'median' or float or int.
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
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


def numerical_outlier(feature, keep_rate=0.9545, method='right', config=None, name=None, mode=0):
    """feature clip outlier.
    
    Args:
        feature: pd.Series, sample feature.
        keep_rate: float, default 0.9545, 
        method: str, default 'right', one of ['left', 'right', 'both'], statistical distribution boundary.
        config: dict, label parameters dict for this estimator. if config is not None,  other parameter is invalid.
        name: str, output feature name, if None, name is feature.name .
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
    Returns:
        normalize feature and feature_scale.
    """
    from scipy.stats import norm
    
    if config is None:
        config = {'feature_scale':[feature.mean(), feature.std()], 
                  'method':method, 'keep_rate':keep_rate,
                  'type':'numerical_outlier', 'name_input':feature.name, 
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
        if config['method']=='both':
            clip_dict = (scale[0]+norm.ppf((1-config['keep_rate'])/2)*scale[1], scale[0]+norm.ppf(config['keep_rate']+(1-config['keep_rate'])/2)*scale[1])
        elif config['method']=='right':
            clip_dict = (feature.min(), scale[0]+norm.ppf(config['keep_rate'])*scale[1])
        else:
            clip_dict = (scale[0]+norm.ppf(1-config['keep_rate'])*scale[1], feature.max())
        t = feature.clip(clip_dict[0], clip_dict[1]).rename(config['name_output'])
        return t if mode else (t, config)

