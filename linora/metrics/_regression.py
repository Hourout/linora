import numpy as np
import pandas as pd

__all__ = ['normal_loss', 'mean_absolute_error', 'mean_squared_error',
           'mean_absolute_percentage_error', 'hinge', 'explained_variance_score',
           'median_absolute_error', 'r2_score', 'regression_report']

def normal_loss(y_true, y_pred, k, log=False, root=False):
    """Mean normal error regression loss.
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted values, as returned by a regression.
        k: int, loss = np.sqrt(loss, 1/k).
        log: default False, whether to log the variable.
        root: default False, whether to sqrt the variable, if True, return rmse loss.
    Returns:
        regression loss values.
    """
    if log:
        loss = (np.log1p(y_true)-np.log1p(y_pred)).abs().pow(k).mean()
    else:
        loss = (y_true-y_pred).abs().pow(k).mean()
    if root:
        loss = np.sqrt(loss, 1/k)
    return loss

def mean_absolute_error(y_true, y_pred, log=False):
    """Mean absolute error regression loss.
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted values, as returned by a regression.
        log: default False, whether to log the variable.
    Returns:
        regression loss values.
    """
    return normal_loss(y_true, y_pred, k=1, log=log, root=False)

def mean_squared_error(y_true, y_pred, log=False, root=False):
    """Mean squared error regression loss.
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted values, as returned by a regression.
        log: default False, whether to log the variable.
        root: default False, whether to sqrt the variable, if True, return rmse loss.
    Returns:
        regression loss values.
    """
    return normal_loss(y_true, y_pred, k=2, log=log, root=root)

def mean_absolute_percentage_error(y_true, y_pred):
    """Mean absolute percentage error regression loss.
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted values, as returned by a regression.
    Returns:
        regression loss values.
    """
    return ((y_true-y_pred)/y_true).abs().mean()

def hinge(y_true, y_pred, k=1):
    """hinge regression loss.
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted values, as returned by a regression.
        k: int, pow() function dim.
    Returns:
        regression loss values.
    """
    return (1-y_true*y_pred).clip(lower=0).pow(k).mean()

def explained_variance_score(y_true, y_pred):
    """explained variance regression loss.
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted values, as returned by a regression.
    Returns:
        regression loss values.
    """
    return 1-(y_true-y_pred).std()**2/y_true.std()**2

def median_absolute_error(y_true, y_pred):
    """Median absolute error regression loss.
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted values, as returned by a regression.
    Returns:
        regression loss values.
    """
    return (y_true-y_pred).abs().median()

def r2_score(y_true, y_pred):
    """r2 regression loss.
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted values, as returned by a regression.
    Returns:
        regression loss values.
    """
    return 1-(y_true-y_pred).pow(2).sum()/(y_true-y_true.mean()).pow(2).sum()

def regression_report(y_true, y_pred, printable=False, printinfo='Regression Report'):
    """
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted labels.
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
