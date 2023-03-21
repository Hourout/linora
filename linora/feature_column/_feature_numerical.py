from linora.feature_column._numerical import *

class FeatureNumerical(object):
    def __init__(self):
        self.pipe = {}
        
    def numerical_binarizer(self, variable, method='mean', name=None, keep=True):
        """Encode labels with value between 0 and 1.

        Args:
            variable: str, feature variable name.
            method: default 'mean', one of 'mean' or 'median' or float or int.
            name: str, output feature name, if None, name is variable.
            keep: If the `name` is output only once in the calculation, the `name` will be kept in the final result.
        """
        config = {'param':{'method':method, 'name':variable if name is None else name},
                  'type':'numerical_binarizer', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def numerical_bucketized(self, variable, boundaries, miss_pad=-1, score=None, miss_score=None, method=1, name=None, keep=True):
        """feature bucket.

        if method is True:
            Buckets include the right boundary, and exclude the left boundary. 
            Namely, boundaries=[0., 1., 2.] generates buckets (-inf, 0.], (0., 1.], (1., 2.], and (2., +inf).
        else:
            Buckets include the left boundary, and exclude the right boundary. 
            Namely, boundaries=[0., 1., 2.] generates buckets (-inf, 0.), [0., 1.), [1., 2.), and [2., +inf).

        Args:
            variable: str, feature variable name.
            boundaries: list, A sorted list or tuple of floats specifying the boundaries.
            miss_pad: int, default -1, feature fillna value.
            score: None, A score list or tuple of floats specifying the boundaries.
            miss_score: int or float, None, score fillna value.
            method: True.
            name: str, output feature name, if None, name is variable.
            keep: If the `name` is output only once in the calculation, the `name` will be kept in the final result.
        """
        config = {'param':{'boundaries':boundaries, 'miss_pad':miss_pad,
                           'score':score, 'miss_score':miss_score, 'method':method,
                           'name':variable if name is None else name},
                  'type':'numerical_bucketized', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def numerical_padding(self, variable, method='mean', name=None, keep=True):
        """feature fillna method.

        Args:
            variable: str, feature variable name.
            method: default 'mean', one of 'mean' or 'median' or float or int.
            name: str, output feature name, if None, name is variable.
            keep: If the `name` is output only once in the calculation, the `name` will be kept in the final result.
        """
        config = {'param':{'method':method, 'name':variable if name is None else name},
                  'type':'numerical_padding', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def numerical_outlier(self, variable, method='norm', delta=0.9545, tail='right', name=None, keep=True):
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
            variable: str, feature variable name.
            method: must be one of ['norm', 'gaussian', 'iqr', 'mad', 'quantiles']
            delta: float, default 0.9545
            tail: str, default 'right', one of ['left', 'right', 'both'], statistical distribution boundary.
            name: str, output feature name, if None, name is variable.
            keep: If the `name` is output only once in the calculation, the `name` will be kept in the final result.
        """
        config = {'param':{'method':method, 'delta':delta, 'tail':tail,
                           'name':variable if name is None else name},
                  'type':'numerical_outlier', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    
    
    