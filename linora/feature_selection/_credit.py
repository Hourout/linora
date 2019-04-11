import numpy as np
import pandas as pd

__all__ = ['woe', 'iv']

def woe(y_true, feature, pos_label=1):
    """Calculate series woe value
    
    Parameters
    ----------
    y_true   : pd.Series, shape (n_samples,)
        The target variable for supervised learning problems.
    feature  : pd.Series, shape (n_samples,)
        x variable, model feature.
    pos_label : int, default=1
        positive label value.
    
    Returns
    -------
    t : series woe value
    """
    t = pd.DataFrame({'label':y_true, 'feature':feature})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    corr = t.label.sum()/(t.label.count()-t.label.sum())
    t = t.groupby(['feature']).label.apply(lambda x:np.log(x.sum()/(x.count()-x.sum())/corr))
    return t

def iv(y_true, feature, pos_label=1):
    """Calculate series iv value
    
    Parameters
    ----------
    y_true   : pd.Series, shape (n_samples,)
        The target variable for supervised learning problems.
    feature  : pd.Series, shape (n_samples,)
        x variable, model feature.
    pos_label : int, default=1
        positive label value.
    
    Returns
    -------
    t : series iv value
    """
    t = pd.DataFrame({'label':y_true, 'feature':feature})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    def _iv(group, df):
        a = group.sum()/df.label.sum()
        b = (group.count()-group.sum())/(t.label.count()-t.label.sum())
        return (a-b)/np.log(a/b)
    corr = t.label.sum()/(t.label.count()-t.label.sum())
    t = t.groupby(['feature']).label.apply(lambda x:_iv(x, t)).sum()
    return t
