import numpy as np

from linora.metrics._utils import _sample_weight

__all__ = ['normal_loss', 'mean_absolute_error', 'mean_squared_error',
           'mean_absolute_percentage_error', 'hinge', 'explained_variance_score',
           'median_absolute_error', 'r2_score', 'report_regression',
           'mean_relative_error', 'poisson', 'log_cosh_error', 'max_error',
           'mean_tweedie_deviance', 'mean_poisson_deviance', 'mean_gamma_deviance',
           'mean_pinball_error'
          ]


def normal_loss(y_true, y_pred, k, log=False, root=False, sample_weight=None):
    """Mean normal error regression loss.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        k: int, loss = np.sqrt(loss, 1/k).
        log: default False, whether to log the variable.
        root: default False, whether to sqrt the variable, if True, return rmse loss.
        sample_weight: list or array of sample weight.
    Returns:
        regression loss values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sample_weight = _sample_weight(y_true, sample_weight)
    if log:
        loss = (np.power(np.abs(np.log1p(y_true)-np.log1p(y_pred))*sample_weight, k)).mean()
    else:
        loss = (np.power(np.abs(y_true-y_pred)*sample_weight, k)).mean()
    if root:
        loss = np.power(loss, 1/k)
    return loss


def mean_absolute_error(y_true, y_pred, log=False, sample_weight=None):
    """Mean absolute error regression loss.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        log: default False, whether to log the variable.
        sample_weight: list or array of sample weight.
    Returns:
        regression loss values.
    """
    return normal_loss(y_true, y_pred, k=1, log=log, root=False, sample_weight=sample_weight)


def mean_squared_error(y_true, y_pred, log=False, root=False, sample_weight=None):
    """Mean squared error regression loss.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        log: default False, whether to log the variable.
        root: default False, whether to sqrt the variable, if True, return rmse loss.
        sample_weight: list or array of sample weight.
    Returns:
        regression loss values.
    """
    return normal_loss(y_true, y_pred, k=2, log=log, root=root, sample_weight=sample_weight)


def mean_absolute_percentage_error(y_true, y_pred, sample_weight=None):
    """Mean absolute percentage error regression loss.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        sample_weight: list or array of sample weight.
    Returns:
        regression loss values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sample_weight = _sample_weight(y_true, sample_weight)
    value = np.abs(y_pred - y_true)
    y_true[np.where(y_true==0)] = 1
    return (value/np.abs(y_true)*sample_weight).mean()


def max_error(y_true, y_pred):
    """The max_error metric calculates the maximum residual error.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
    Returns:
        regression loss values.
    """
    return np.max(np.abs(np.array(y_true) - np.array(y_pred)))


def hinge(y_true, y_pred, k=1, sample_weight=None):
    """hinge regression loss.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        k: int, pow() function dim.
        sample_weight: list or array of sample weight.
    Returns:
        regression loss values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sample_weight = _sample_weight(y_true, sample_weight)
    return np.power((1-y_true*y_pred).clip(min=0)*sample_weight, k).mean()


def explained_variance_score(y_true, y_pred, sample_weight=None):
    """explained variance regression loss.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        sample_weight: list or array of sample weight.
    Returns:
        regression loss values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sample_weight = _sample_weight(y_true, sample_weight)
    return 1-((y_true-y_pred)*sample_weight).std()**2/y_true.std()**2


def median_absolute_error(y_true, y_pred, sample_weight=None):
    """Median absolute error regression loss.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        sample_weight: list or array of sample weight.
    Returns:
        regression loss values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sample_weight = _sample_weight(y_true, sample_weight)
    return np.median(np.abs(y_true-y_pred)*sample_weight)


def r2_score(y_true, y_pred, sample_weight=None):
    """r2 regression loss.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        sample_weight: list or array of sample weight.
    Returns:
        regression loss values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sample_weight = _sample_weight(y_true, sample_weight)
    return 1-np.power((y_true-y_pred)*sample_weight, 2).sum()/np.power((y_true-y_true.mean())*sample_weight, 2).sum()


def report_regression(y_true, y_pred, sample_weight=None, printable=False):
    """regression report
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels.
        sample_weight: list or array of sample weight.
        printable: bool, print report.
    Returns:
        dict, regression report.
    """
    result = {'mean_absolute_error':mean_absolute_error(y_true, y_pred, sample_weight=sample_weight),
              'mean_squared_error':mean_squared_error(y_true, y_pred, sample_weight=sample_weight),
              'mean_absolute_percentage_error':mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weight),
              'hinge_loss':hinge(y_true, y_pred, sample_weight=sample_weight),
              'explained_variance_score':explained_variance_score(y_true, y_pred, sample_weight=sample_weight),
              'median_absolute_error':median_absolute_error(y_true, y_pred, sample_weight=sample_weight),
              'r2_score':r2_score(y_true, y_pred, sample_weight=sample_weight)
             }
    if printable:
        print("\nRegression Report")
        print("mean_absolute_error: %.4f" % result['mean_absolute_error'])
        print("mean_squared_error: %.4f" % result['mean_squared_error'])
        print("mean_absolute_percentage_error: %.4f" % result['mean_absolute_percentage_error'])
        print("hinge_loss: %.4f" % result['hinge_loss'])
        print("explained_variance_score: %.4f" % result['explained_variance_score'])
        print("median_absolute_error: %.4f" % result['median_absolute_error'])
        print("r2_score: %.4f" % result['r2_score'])
    return result


def mean_relative_error(y_true, y_pred, normalizer, sample_weight=None):
    """Computes the mean relative error by normalizing with the given values.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        normalizer: The normalizer values with same shape as y_pred.
        sample_weight: list or array of sample weight.
    Returns:
        regression loss values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sample_weight = _sample_weight(y_true, sample_weight)
    return (np.abs(y_true-y_pred)/np.array(normalizer)*sample_weight).mean()


def poisson(y_true, y_pred, sample_weight=None):
    """Computes the Poisson loss between y_true and y_pred.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        sample_weight: list or array of sample weight.
    Returns:
        regression loss values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sample_weight = _sample_weight(y_true, sample_weight)
    return np.mean((y_pred - y_true * np.log(y_pred+ 1e-7))*sample_weight)


def log_cosh_error(y_true, y_pred, sample_weight=None):
    """Computes the logarithm of the hyperbolic cosine of the prediction error.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        sample_weight: list or array of sample weight.
    Returns:
        regression loss values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    x = (y_pred-y_true)*sample_weight
    sample_weight = _sample_weight(y_true, sample_weight)
    return np.mean(x + np.log(np.exp(-2. * x) + 1.) - np.log(2.))


def mean_tweedie_deviance(y_true, y_pred, p, sample_weight=None):
    """Mean Tweedie deviance regression loss.
    
    when p=0 it is equivalent to mean_squared_error.
    when p=1 it is equivalent to mean_poisson_deviance.
    when p=2 it is equivalent to mean_gamma_deviance.

    p < 0: Extreme stable distribution. Requires: y_pred > 0.
    p = 0 : Normal distribution, output corresponds to mean_squared_error. y_true and y_pred can be any real numbers.
    p = 1 : Poisson distribution. Requires: y_true >= 0 and y_pred > 0.
    1 < p < 2 : Compound Poisson distribution. Requires: y_true >= 0 and y_pred > 0.
    p = 2 : Gamma distribution. Requires: y_true > 0 and y_pred > 0.
    p = 3 : Inverse Gaussian distribution. Requires: y_true > 0 and y_pred > 0.
    otherwise : Positive stable distribution. Requires: y_true > 0 and y_pred > 0.

    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        p: Tweedie power parameter. Either p <= 0 or p >= 1.
           The higher p the less weight is given to extreme deviations between true and predicted targets.
        sample_weight: list or array of sample weight.
    Returns:
        A non-negative floating point value (the best value is 0.0).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if p < 0:
        if (y_pred <= 0).any():
            raise ValueError("p < 0: Extreme stable distribution. Requires: y_pred > 0.")
    elif p == 0:
        pass
    elif 0 < p < 1:
        raise ValueError("Tweedie deviance is only defined for p<=0 and p>=1.")
    elif 1 <= p < 2:
        if (y_true < 0).any() or (y_pred <= 0).any():
            raise ValueError("1 < p < 2 : Compound Poisson distribution. Requires: y_true >= 0 and y_pred > 0.")
    else:
        if (y_true <= 0).any() or (y_pred <= 0).any():
            raise ValueError("p>=2, Positive stable distribution. Requires: y_true > 0 and y_pred > 0.")

    if p==0:
        t = np.square(y_true-y_pred)
    elif p==1:
        t = 2*(y_true*np.log(y_true/y_pred)+y_pred-y_true)
    elif p==2:
        t = 2*(np.log(y_pred/y_true)+y_true/y_pred - 1)
    else:
        t = 2 * (
            np.power(np.maximum(y_true, 0), 2 - p) / ((1 - p) * (2 - p))
            - y_true * np.power(y_pred, 1 - p) / (1 - p)
            + np.power(y_pred, 2 - p) / (2 - p))
    sample_weight = _sample_weight(y_true, sample_weight)
    return np.average(t, weights=sample_weight)


def mean_poisson_deviance(y_true, y_pred, sample_weight=None):
    """Mean Poisson deviance regression loss.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        sample_weight: list or array of sample weight.
    Returns:
        A non-negative floating point value (the best value is 0.0).
    """
    return mean_tweedie_deviance(y_true, y_pred, p=1, sample_weight=sample_weight)


def mean_gamma_deviance(y_true, y_pred, sample_weight=None):
    """Mean Gamma deviance regression loss.

    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        sample_weight: list or array of sample weight.
    Returns:
        A non-negative floating point value (the best value is 0.0).
    """
    return mean_tweedie_deviance(y_true, y_pred, p=2, sample_weight=sample_weight)


def mean_pinball_error(y_true, y_pred, alpha=0.5, sample_weight=None):
    """Pinball loss for quantile regression.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        alpha: this loss is equivalent to Mean absolute error when alpha=0.5, 
               alpha=0.95 is minimized by estimators of the 95th percentile.
        sample_weight: list or array of sample weight.
    Returns:
        A non-negative floating point value (the best value is 0.0).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sample_weight = _sample_weight(y_true, sample_weight)
    t = alpha*np.maximum(y_true-y_pred, 0)+(1-alpha)*np.maximum(y_pred-y_true, 0)
    return np.average(t, weights=sample_weight)