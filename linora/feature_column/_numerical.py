import numpy as np


__all__ = ['numerical_binarizer', 'numerical_bucketized', 'numerical_cyclical',
           'numerical_combine', 'numerical_math', 'numerical_padding', 'numerical_outlier',
           'numerical_relative']


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
        Refer to params `mode` explanation.
    """
    if config is None:
        config = {'param':{'method':method, 'name':feature.name if name is None else name},
                  'type':'numerical_binarizer', 'variable':feature.name, 'keep':keep}
        if method=='mean':
            config['param']['feature_scale'] = feature.mean()
        elif method=='median':
            config['param']['feature_scale'] = feature.median()
        else:
            config['param']['feature_scale'] = method
    if mode==2:
        return config
    else:
        scale = config['param']['feature_scale']
        t = (feature.clip(upper=scale).replace({scale:scale+0.1}).clip(scale)
             .replace({scale:0, scale+0.1:1}).astype('int8').rename(config['param']['name']))
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
        Refer to params `mode` explanation.
    """
    if config is None:
        config = {'param':{'boundaries':boundaries, 'miss_pad':miss_pad,
                           'score':score, 'miss_score':miss_score, 'method':method,
                           'name':feature.name if name is None else name},
                  'type':'numerical_bucketized', 'variable':feature.name}
    if mode==2:
        return config
    else:
        t = feature.copy()
        bound = sorted(config['param']['boundaries'])
        if config['param']['method']:
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
        t = t.fillna(config['param']['miss_pad']).astype('int64').rename(config['param']['name'])
        if isinstance(config['param']['score'], (tuple, list)):
            t = t.replace({i:j for i,j in enumerate(config['param']['score'])})
            if config['param']['miss_score'] is not None:
                t = t.replace({config['param']['miss_pad']:config['param']['miss_score']})
        return t if mode else (t, config)


def numerical_combine(feature, function, mode=0, name=None, config=None):
    """feature combine transform.
    
    Args:
        feature: pd.DataFrame, sample feature.
        function: function or str, Function to use for aggregating the data. 
            If a function, must either work when passed a DataFrame or when passed to DataFrame.apply.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.        
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        Refer to params `mode` explanation.
    """
    if config is None:
        config = {'param':{'function':function, 
                          'name':'combine_'+'_'.join([i.name for i in feature]) if name is None else name},
                  'type':'numerical_combine', 'variable':[i for i in feature.columns]}
    if mode==2:
        return config
    else:
        t = feature.agg(function, axis=1).rename(config['param']['name'])
        return t if mode else (t, config)


def numerical_cyclical(feature, mode=0, name=None, config=None):
    """feature cyclical transform.
    
    applies cyclical transformations to numerical variables, returning 2 new features per variable.
    according to:
        var_sin = sin(variable * (2. * pi / max_value))
        var_cos = cos(variable * (2. * pi / max_value))

    where max_value is the maximum value in the variable, and pi is 3.14…
    
    Args:
        feature: pd.Series, sample feature.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.        
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        Refer to params `mode` explanation.
    """
    if config is None:
        if name is None:
            name = feature.name
        config = {'param':{'max_value':feature.max(), 'name':[f'{name}_sin', f'{name}_cos']},
                  'type':'numerical_cyclical', 'variable':feature.name}
    if mode==2:
        return config
    else:
        t = feature*(2.*np.pi/config['param']['max_value'])
        t = pd.concat([np.sin(t), np.cos(t)], axis=1)
        t.columns = config['param']['name']
        return t if mode else (t, config)


def numerical_math(feature, mode=0, method=['log'], log='e', c='auto', power=0.5, name=None, config=None):
    """feature math transform.
    
    using:
    - log
    - power
    - arcsin
    - reciprocal
    - BoxCox
    - YeoJohnson
    
    log:
        x = log(x+c)
        if c='auto', c = x.min().abs() + 1
        if x<0, x=0
        
        if log='e', x = log-e(x+c)
        if log='2', x = log-2(x+c)
        if log='10', x = log-10(x+c)
    
    power:
        x = x^power
        if power<1, x[x<=0] = 1
        
    arcsin:
        x = arcsin(sqrt(x))
        x must be 0~1
    
    reciprocal:
        x = 1/x
        if x=0, x=0
    
    BoxCox:
        applies the BoxCox transformation to numerical variables.
        The Box-Cox transformation is defined as:
        - T(Y)=(Y exp(λ)−1)/λ if λ!=0
        - log(Y) otherwise
        where Y is the response variable and λ is the transformation parameter. 
        λ varies, typically from -5 to 5. 
        In the transformation, all values of λ are considered and the optimal value for a given variable is selected.
        `Box and Cox. “An Analysis of Transformations”. Read at a RESEARCH MEETING, 1964.`
    
    YeoJohnson:
        applies the Yeo-Johnson transformation to the numerical variables.
        `Yeo, In-Kwon and Johnson, Richard (2000). 
        A new family of power transformations to improve normality or symmetry. Biometrika, 87, 954-959.`
    
    Args:
        feature: pd.Series, sample feature.
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.
        method: list, must be one or more of ['log', 'power', 'arcsin', 'reciprocal']
        log: Indicates if the natural or base 10 logarithm should be applied. Can take values ‘e’ or ‘10’, '2'.
        c: The constant C to add to the variable before the logarithm, i.e., log(x + C).
        power: The power (or exponent).
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        Refer to params `mode` explanation.
    """
    method_list = ['log', 'power', 'arcsin', 'reciprocal', 'BoxCox', 'YeoJohnson']
    if config is None:
        if isinstance(method, str):
            method = [method]
        config = {'param':{'method':method, 'log':log, 'c':c,
                          'negative_replace':negative_replace, 'power':power,
                          'name':feature.name if name is None else name},
                  'type':'numerical_math', 'variable':feature.name}
        for i in method:
            if i not in method_list:
                raise ValueError(f"`method` must be one of {method_list}.")
        if len(method)!=len(set(method)):
            raise ValueError("`method` must be unique.")
        if 'log' in method:
            if c=='auto':
                config['param']['c'] = feature.min().abs() + 1
        if 'BoxCox' in method:
            import scipy.stats.boxcox as boxcox
            _, config['param']['boxcox_lambda'] = boxcox(feature)
        if 'YeoJohnson' in method:
            import scipy.stats.yeojohnson as yeojohnson
            _, config['param']['yeojohnson_lambda'] = yeojohnson(feature)
            
    if mode==2:
        return config
    else:
        log = config['param']['log']
        power = config['param']['power']
        
        t = feature.copy().rename(config['param']['name'])
        for i in config['param']['method']:
            if i=='log':
                t = t+config['param']['c']
                temp = t<=0
                if log=='e':
                    t = np.log(t)
                elif log=='2':
                    t = np.log2(t)
                elif log=='10':
                    t = np.log10(t)
                t[temp] = 0
            elif i=='power':
                if power<1:
                    t[t<=0] = 1
                t = np.power(t, power)
            elif i=='arcsin':
                t = np.arcsin(np.sqrt(t))
            elif i=='reciprocal':
                temp = t==0
                t = 1/t
                t[temp] = 0
            elif i=='BoxCox':
                import scipy.stats.boxcox as boxcox
                t = boxcox(t, lmbda=config['param']['boxcox_lambda'])
            elif i=='YeoJohnson':
                import scipy.stats.yeojohnson as yeojohnson
                t = yeojohnson(t, lmbda=config['param']['yeojohnson_lambda'])
            else:
                raise ValueError(f"`method` must be one of {method_list}.")
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
        Refer to params `mode` explanation.
    """
    if config is None:
        config = {'param':{'method':method, 'name':feature.name if name is None else name},
                  'type':'numerical_padding', 'variable':feature.name}
        if method=='mean':
            config['param']['feature_scale'] = feature.mean()
        elif method=='median':
            config['param']['feature_scale'] = feature.median()
        else:
            config['param']['feature_scale'] = method
    if mode==2:
        return config
    else:
        scale = config['param']['feature_scale']
        t = feature.fillna(scale).rename(config['param']['name'])
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
        Refer to params `mode` explanation.
    """
    if config is None:
        config = {'param':{'method':method, 'delta':delta, 'tail':tail,
                           'name':feature.name if name is None else name},
                  'type':'numerical_outlier', 'variable':feature.name}
        if method =='norm':
            if not 0<delta<1:
                raise ValueError("`delta` must be 0<delta<1")
            from scipy.stats import norm
            config['param']['feature_scale'] = [feature.mean(), feature.std()]
        elif method=='gaussian':
            config['param']['feature_scale'] = [feature.mean(), feature.std()]
        elif method=='iqr':
            config['param']['feature_scale'] = [feature.quantile(0.25), feature.quantile(0.75)]
        elif method=='mad':
            config['param']['feature_scale'] = [feature.median(), (feature-feature.median()).abs().median()]
        elif method=='quantiles':
            if not 0<delta<0.5:
                raise ValueError("`delta` must be 0<delta<0.5")
            config['param']['feature_scale'] = [feature.quantile(delta), feature.quantile(1-delta)]
        else:
            raise ValueError("`method` must be one of ['norm', 'gaussian', 'iqr', 'mad', 'quantiles']")
    if mode==2:
        return config
    else:
        scale = config['param']['feature_scale']
        delta = config['param']['delta']
        if method =='norm':
            if config['param']['tail']=='both':
                clip = (scale[0]+norm.ppf((1-delta)/2)*scale[1], scale[0]+norm.ppf(delta+(1-delta)/2)*scale[1])
            elif config['param']['tail']=='right':
                clip = (feature.min(), scale[0]+norm.ppf(delta)*scale[1])
            else:
                clip = (scale[0]+norm.ppf(1-delta)*scale[1], feature.max())
        elif method in ['gaussian', 'mad']:
            if config['param']['tail']=='both':
                clip = (scale[0]-delta*scale[1], scale[0]+delta*scale[1])
            elif config['param']['tail']=='right':
                clip = (feature.min(), scale[0]+delta*scale[1])
            else:
                clip = (scale[0]-delta*scale[1], feature.max())
        elif method =='iqr':
            if config['param']['tail']=='both':
                clip = (scale[0]-delta*(scale[1]-scale[0]), scale[1]+delta*(scale[1]-scale[0]))
            elif config['param']['tail']=='right':
                clip = (feature.min(), scale[1]+delta*(scale[1]-scale[0]))
            else:
                clip = (scale[0]-delta*(scale[1]-scale[0]), feature.max())
        elif method =='quantiles':
            if config['param']['tail']=='both':
                clip = (scale[0], scale[1])
            elif config['param']['tail']=='right':
                clip = (feature.min(), scale[1])
            else:
                clip = (scale[0], feature.max())
        else:
            raise ValueError("`method` must be one of ['norm', 'gaussian', 'iqr', 'mad', 'quantiles']")
        t = feature.clip(clip[0], clip[1]).rename(config['param']['name'])
        return t if mode else (t, config)


def numerical_relative(feature, reference, function, mode=0, name=None, config=None):
    """feature combine transform.
    
    supported method:
        ['add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod', 'pow', 'max', 'min', 'mean']
    
    Args:
        feature: pd.Series or pd.DataFrame, sample feature.
        reference: pd.Series or pd.DataFrame, sample feature.
        function: str or str of list, 
            one of ['add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod', 'pow', 'max', 'min', 'mean']
        mode: if 0, output (transform feature, config); if 1, output transform feature; if 2, output config.        
        name: str, output feature name, if None, name is feature.name .
        config: dict, label parameters dict for this estimator. 
            if config is not None, only parameter `feature` and `mode` is invalid.
    Returns:
        Refer to params `mode` explanation.
    """
    if type(feature)==pd.core.series.Series:
        feature = feature.to_frame()
    if type(reference)==pd.core.series.Series:
        reference = reference.to_frame()
    if isinstance(function, str):
        function = [function]
    function_list = ['add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod', 'pow', 'max', 'min', 'mean']
    for i in function:
        assert i in function_list, f'`function` must be one of {function_list}.'
    
    if config is None:
        config = {'param':{'function':function, 'reference':reference.columns.tolist(),
                          'name':'relative' if name is None else name},
                  'type':'numerical_relative', 'variable':feature.columns.tolist()}
    if mode==2:
        return config
    else:
        t = []
        
        for i in feature.columns:
            for j in reference.columns:
                for k in function:
                    if k=='add':
                        temp = feature[i].add(reference[j])
                    elif k=='sub':
                        temp = feature[i].sub(reference[j])
                    elif k=='mul':
                        temp = feature[i].mul(reference[j])
                    elif k=='div':
                        temp = feature[i].div(reference[j])
                    elif k=='truediv':
                        temp = feature[i].truediv(reference[j])
                    elif k=='floordiv':
                        temp = feature[i].floordiv(reference[j])
                    elif k=='mod':
                        temp = feature[i].mod(reference[j])
                    elif k=='pow':
                        temp = feature[i].pow(reference[j])
                    elif k=='max':
                        temp = pd.concat([feature[i], reference[j]], axis=1).max(axis=1)
                    elif k=='min':
                        temp = pd.concat([feature[i], reference[j]], axis=1).min(axis=1)
                    elif k=='mean':
                        temp = pd.concat([feature[i], reference[j]], axis=1).mean(axis=1)
                    t.append(temp.rename(f"{config['param']['name']}_{i}_{k}_{j}"))
        if len(t)>1:
            t = pd.concat(t, axis=1)
        return t if mode else (t, config)


