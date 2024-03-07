
import itertools

import numpy as np
import pandas as pd

from linora.metrics import psi

__all__ = ['woe', 'iv', 'psi_columns']


def woe(feature, y_true, max_bins=5, pos_label=1):
    """Calculate series woe value
    
    Args:
        feature: pd.Series, shape (n_samples,) x variable, model feature.
        y_true: pd.Series, shape (n_samples,) The target variable for supervised learning problems.
	max_bins: maximum number of boxes.
        pos_label: int, default=1, positive label value.
    
    Return:
        series woe value
    """
    t = pd.DataFrame({'label':y_true, 'feature':feature})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    if t.feature.nunique()>max_bins:
        t['feature'] = pd.qcut(x=t.feature, q=max_bins, duplicates='drop')
    df = pd.crosstab(index=t.feature, columns=t.label, margins=False).rename(columns={0:'neg',1:'pos'})
    df['pos_rate'] = df['pos'].replace({0:1}) / df['pos'].sum()
    df['neg_rate'] = df['neg'].replace({0:1}) / df['neg'].sum()
    df['woe'] = np.log(df['pos_rate'] / df['neg_rate'])
    return df


def iv(feature, y_true, max_bins=5, pos_label=1):
    """Calculate series iv value
    
    Args:
        feature: pd.Series, shape (n_samples,) x variable, model feature.
        y_true: pd.Series, shape (n_samples,) The target variable for supervised learning problems.
	max_bins: maximum number of boxes.
        pos_label: int, default=1, positive label value.
    
    Return:
        series iv value
    """
    t = pd.DataFrame({'label':y_true, 'feature':feature})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    if t.feature.nunique()>max_bins:
        t['feature'] = pd.qcut(x=t.feature, q=max_bins, duplicates='drop')
    df = pd.crosstab(index=t.feature, columns=t.label, margins=False).rename(columns={0:'neg',1:'pos'})
    df['pos_rate'] = df['pos'].replace({0:1}) / df['pos'].sum()
    df['neg_rate'] = df['neg'].replace({0:1}) / df['neg'].sum()
    df['woe'] = np.log(df['pos_rate'] / df['neg_rate'])
    df['iv'] = (df['pos_rate'] - df['neg_rate']) * df['woe']
    iv = df['iv'].sum()
    return iv


def psi_columns(feature_list, data_dict, bins=4):
    name = list(itertools.combinations([i for i in data_dict], 2))
    t = pd.DataFrame({'feature_name':feature_list})
    for i,j in name:
        t[f'{i}_{j}_psi'] = [psi(data_dict[i][f], data_dict[j][f], bins=bins) for f in feature_list]
    return t

