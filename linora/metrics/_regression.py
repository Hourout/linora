import numpy as np
import pandas as pd

__all__ = ['normal_loss', 'mean_absolute_error', 'mean_squared_error',
           'mean_absolute_percentage_error', 'hinge', 'explained_variance_score',
           'median_absolute_error', 'r2_score']

def normal_loss(y_true, y_pred, k, log=False, root=False):
    if log:
        loss = (np.log1p(y_true)-np.log1p(y_pred)).abs().pow(k).mean()
    else:
        loss = (y_true-y_pred).abs().pow(k).mean()
    if root:
        loss = np.sqrt(loss, 1/k)
    return loss

def mean_absolute_error(y_true, y_pred, log=False, root=False):
    return normal_loss(y_true, y_pred, k=1, log=log, root=root)

def mean_squared_error(y_true, y_pred, log=False, root=False):
    return normal_loss(y_true, y_pred, k=2, log=log, root=root)

def mean_absolute_percentage_error(y_true, y_pred):
    return ((y_true-y_pred)/y_true).abs().mean()

def hinge(y_true, y_pred, k=1):
    return (1-y_true*y_pred).clip(lower=0).pow(k).mean()

def explained_variance_score(y_true, y_pred):
    return 1-(y_true-y_pred).std()**2/y_true.std()**2

def median_absolute_error(y_true, y_pred):
    return (y_true-y_pred).abs().median()

def r2_score(y_true, y_pred):
    return 1-(y_true-y_pred).pow(2).sum()/(y_true-y_true.mean()).pow(2).sum()
