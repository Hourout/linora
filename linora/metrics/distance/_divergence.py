import numpy as np
import pandas as pd

__all__ = ['kl_divergence', 'js_divergence']

def kl_divergence(x, y, continuous=False, bucket_num=1000):
    if continuous:
        x1 = (x-min(x.min(), y.min()))/(max(x.max(), y.max())-min(x.min(), y.min()))
        y1 = (y-min(x.min(), y.min()))/(max(x.max(), y.max())-min(x.min(), y.min()))
        x1 = pd.cut(x1, bucket_num, labels=range(bucket_num))
        y1 = pd.cut(y1, bucket_num, labels=range(bucket_num))
        t = x1.value_counts(normalize=True).reset_index().merge(y1.value_counts(normalize=True).reset_index(), on='index', how='left').fillna(0.)
    else:
        t = x.value_counts(normalize=True).reset_index().merge(y.value_counts(normalize=True).reset_index(), on='index', how='left').fillna(0.)
    t.columns = ['label', 'prob_x', 'prob_y']
    t = np.sum(np.log((t.prob_x+0.00001)/(t.prob_y+0.00001))*t.prob_x)
    return t

def js_divergence(x, y, continuous=False, bucket_num=1000):
    if continuous:
        x1 = (x-min(x.min(), y.min()))/(max(x.max(), y.max())-min(x.min(), y.min()))
        y1 = (y-min(x.min(), y.min()))/(max(x.max(), y.max())-min(x.min(), y.min()))
        x1 = pd.cut(x1, bucket_num, labels=range(bucket_num))
        y1 = pd.cut(y1, bucket_num, labels=range(bucket_num))
        t = x1.value_counts(normalize=True).reset_index().merge(y1.value_counts(normalize=True).reset_index(), on='index', how='outer').fillna(0.)
    else:
        t = x.value_counts(normalize=True).reset_index().merge(y.value_counts(normalize=True).reset_index(), on='index', how='outer').fillna(0.)
    t.columns = ['label', 'prob_x', 'prob_y']
    t['prob_m'] = (t.prob_x+t.prob_y)/2
    t1 = np.sum(np.log((t.prob_x+0.00001)/(t.prob_m+0.00001))*t.prob_x)
    t2 = np.sum(np.log((t.prob_y+0.00001)/(t.prob_m+0.00001))*t.prob_y)
    return 0.5*(t1+t2)
