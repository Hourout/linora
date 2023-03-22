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
    
    def numerical_combine(self, variable, function, name=None, keep=True):
        """feature combine transform.

        Args:
            variable: str of list, feature variable name.
            function: function or str, Function to use for aggregating the data. 
                If a function, must either work when passed a DataFrame or when passed to DataFrame.apply.
            name: str, output feature name, if None, name is feature.name .
            keep: If the `name` is output only once in the calculation, the `name` will be kept in the final result.
        """
        config = {'param'{'func':func, 
                          'name':'combine_'+'_'.join(variable) if name is None else name}
                  'type':'numerical_combine', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def numerical_cyclical(self, variable, name=None, keep=True):
        """feature cyclical transform.

        applies cyclical transformations to numerical variables, returning 2 new features per variable.
        according to:
            var_sin = sin(variable * (2. * pi / max_value))
            var_cos = cos(variable * (2. * pi / max_value))

        where max_value is the maximum value in the variable, and pi is 3.14…

        Args:
            variable: str, feature variable name.
            name: str, output feature name, if None, name is variable.
            keep: If the `name` is output only once in the calculation, the `name` will be kept in the final result.
        """
        config = {'param':{'name':variable if name is None else name},
                  'type':'numerical_cyclical', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self

    def numerical_math(self, variable, method=['log'], log='e', c='auto', power=0.5, name=None, keep=True):
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
            variable: str, feature variable name.
            method: list, must be one or more of ['log', 'power', 'arcsin', 'reciprocal', 'BoxCox', 'YeoJohnson']
            log: Indicates if the natural or base 10 logarithm should be applied. Can take values ‘e’ or ‘10’, '2'.
            c: The constant C to add to the variable before the logarithm, i.e., log(x + C).
            power: The power (or exponent).
            name: str, output feature name, if None, name is feature.name .
            keep: If the `name` is output only once in the calculation, the `name` will be kept in the final result.
        """
        config = {'param':{'method':method, 'log':log, 'c':c, 'power':power,
                           'name':variable if name is None else name},
                  'type':'numerical_math', 'variable':variable, 'keep':keep}
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
    
    
    
    