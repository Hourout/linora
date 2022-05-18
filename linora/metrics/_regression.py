import numpy as np

__all__ = ['normal_loss', 'mean_absolute_error', 'mean_squared_error',
           'mean_absolute_percentage_error', 'hinge', 'explained_variance_score',
           'median_absolute_error', 'r2_score', 'report_regression',
           'mean_relative_error', 'poisson', 'log_cosh_error'
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
    sample_weight = np.ones(len(y_true)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
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
    sample_weight = np.ones(len(y_true)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    value = np.abs(y_pred - y_true)
    y_true[np.where(y_true==0)] = 1
    return (value/np.abs(y_true)*sample_weight).mean()


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
    sample_weight = np.ones(len(y_true)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
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
    sample_weight = np.ones(len(y_true)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
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
    sample_weight = np.ones(len(y_true)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
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
    sample_weight = np.ones(len(y_true)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
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
    sample_weight = np.ones(len(y_true)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
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
    sample_weight = np.ones(len(y_true)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
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
    sample_weight = np.ones(len(y_true)) if sample_weight is None else np.array(sample_weight)
    sample_weight = sample_weight/sample_weight.sum()*len(sample_weight)
    return np.mean(x + np.log(np.exp(-2. * x) + 1.) - np.log(2.))
