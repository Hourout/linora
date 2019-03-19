import numpy as np
import pandas as pd

__all__ = ['normal_loss', 'mean_absolute_error', 'mean_squared_error',
           'mean_absolute_percentage_error', 'hinge']

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
