import pandas as pd


__al__ = ['statistical_bins']


def statistical_bins(y_true, y_pred, bins=10, method='quantile', pos_label=1):
    """
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        bins: Number of boxes.
        method: 'quantile' is Equal frequency bin, 'uniform' is Equal width bin.
        pos_label: positive label.
    Returns:
        
    """
    logic = False
    while not logic:
        t = pd.DataFrame({'bins':y_pred, 'label':y_true})
        if method=='quantile':
            t['bins'] = pd.qcut(t['bins'], q=bins, duplicates='drop')
        else:
            t['bins'] = pd.cut(t['bins'], bins, duplicates='drop')
        assert t.label.nunique()==2, "`y_true` should be binary classification."
        label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
        t['label'] = t.label.replace(label_dict)
        t = t.groupby('bins', observed=True).label.agg(['sum', 'count']).sort_index(ascending=logic).reset_index()
        t.columns = ['bins', 'bad_num', 'sample_num']
        t['bad_rate'] = t['bad_num']/t['sample_num']
        t['bad_num_cum'] = t['bad_num'].cumsum()
        t['sample_num_cum'] = t['sample_num'].cumsum()
        t['bad_rate_cum'] = t['bad_num_cum']/t['sample_num_cum']
        t['good_num'] = t['sample_num']-t['bad_num']
        t['good_rate'] = t['good_num']/t['sample_num']
        t['good_num_cum'] = t['good_num'].cumsum()
        t['good_rate_cum'] = t['good_num_cum']/t['sample_num_cum']
        t['ks'] = (t['bad_rate_cum']-t['good_rate_cum']).abs()
        t['lift'] = t['bad_num']/t['sample_num']/t['bad_num'].sum()*t['sample_num'].sum()
        t['cum_lift'] = t['bad_num'].cumsum()/t['sample_num'].cumsum()/t['bad_num'].sum()*t['sample_num'].sum()
        if t['cum_lift'].values[0]>t['cum_lift'].values[-1]:
            logic = True
    return t

