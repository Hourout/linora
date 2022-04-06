import numpy as np

__all__ = ['normal_loss', 'mean_absolute_error', 'mean_squared_error',
           'mean_absolute_percentage_error', 'hinge', 'explained_variance_score',
           'median_absolute_error', 'r2_score', 'regression_report',
           'mean_relative_error', 'poisson', 'log_cosh_error'
          ]

def normal_loss(y_true, y_pred, k, log=False, root=False):
    """Mean normal error regression loss.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        k: int, loss = np.sqrt(loss, 1/k).
        log: default False, whether to log the variable.
        root: default False, whether to sqrt the variable, if True, return rmse loss.
    Returns:
        regression loss values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if log:
        loss = np.power(np.abs(np.log1p(y_true)-np.log1p(y_pred)), k).mean()
    else:
        loss = np.power(np.abs(y_true-y_pred), k).mean()
    if root:
        loss = np.sqrt(loss, 1/k)
    return loss

def mean_absolute_error(y_true, y_pred, log=False):
    """Mean absolute error regression loss.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        log: default False, whether to log the variable.
    Returns:
        regression loss values.
    """
    return normal_loss(y_true, y_pred, k=1, log=log, root=False)

def mean_squared_error(y_true, y_pred, log=False, root=False):
    """Mean squared error regression loss.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        log: default False, whether to log the variable.
        root: default False, whether to sqrt the variable, if True, return rmse loss.
    Returns:
        regression loss values.
    """
    return normal_loss(y_true, y_pred, k=2, log=log, root=root)

def mean_absolute_percentage_error(y_true, y_pred):
    """Mean absolute percentage error regression loss.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
    Returns:
        regression loss values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.abs((y_true-y_pred)/y_true).mean()

def hinge(y_true, y_pred, k=1):
    """hinge regression loss.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        k: int, pow() function dim.
    Returns:
        regression loss values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.power((1-y_true*y_pred).clip(min=0), k).mean()

def explained_variance_score(y_true, y_pred):
    """explained variance regression loss.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
    Returns:
        regression loss values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return 1-(y_true-y_pred).std()**2/y_true.std()**2

def median_absolute_error(y_true, y_pred):
    """Median absolute error regression loss.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
    Returns:
        regression loss values.
    """
    return np.median(np.abs(np.array(y_true)-np.array(y_pred)))

def r2_score(y_true, y_pred):
    """r2 regression loss.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
    Returns:
        regression loss values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return 1-np.power(y_true-y_pred, 2).sum()/np.power(y_true-y_true.mean(), 2).sum()

def regression_report(y_true, y_pred, printable=False, printinfo='Regression Report'):
    """
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels.
    Returns:
        regression report.
    """
    result = {'mean_absolute_error':mean_absolute_error(y_true, y_pred),
              'mean_squared_error':mean_squared_error(y_true, y_pred),
              'mean_absolute_percentage_error':mean_absolute_percentage_error(y_true, y_pred),
              'hinge_loss':hinge(y_true, y_pred),
              'explained_variance_score':explained_variance_score(y_true, y_pred),
              'median_absolute_error':median_absolute_error(y_true, y_pred),
              'r2_score':r2_score(y_true, y_pred)
             }
    if printable:
        print("\n{}".format(printinfo))
        print("mean_absolute_error: %.4f" % result['mean_absolute_error'])
        print("mean_squared_error: %.4f" % result['mean_squared_error'])
        print("mean_absolute_percentage_error: %.4f" % result['mean_absolute_percentage_error'])
        print("hinge_loss: %.4f" % result['hinge_loss'])
        print("explained_variance_score: %.4f" % result['explained_variance_score'])
        print("median_absolute_error: %.4f" % result['median_absolute_error'])
        print("r2_score: %.4f" % result['r2_score'])
    return result

def mean_relative_error(y_true, y_pred, normalizer):
    """Computes the mean relative error by normalizing with the given values.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
        normalizer: The normalizer values with same shape as y_pred.
    Returns:
        regression loss values.
    """
    return (np.abs(np.array(y_true)-np.array(y_pred))/np.array(normalizer)).mean()

def poisson(y_true, y_pred):
    """Computes the Poisson loss between y_true and y_pred.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
    Returns:
        regression loss values.
    """
    return np.mean(np.array(y_pred) - np.array(y_true) * np.log(np.array(y_pred)+ 1e-7))

def log_cosh_error(y_true, y_pred):
    """Computes the logarithm of the hyperbolic cosine of the prediction error.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a regression.
    Returns:
        regression loss values.
    """
    x = np.array(y_pred) - np.array(y_true)
    return np.mean(x + np.log(np.exp(-2. * x) + 1.) - np.log(2.))
