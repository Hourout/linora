import numpy as np
import pandas as pd

__all__ = ['kl_divergence', 'js_divergence', 'mutual_information_rate', 'pointwise_mutual_information_rate']

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

def mutual_information(feature1, feature2):
    t = feature1.value_counts(normalize=True).reset_index()
    t.columns = ['label', 'label_mean']
    t['temp'] = 0
    t1 = feature2.value_counts(normalize=True).reset_index()
    t1.columns = ['label1', 'label1_mean']
    t1['temp'] = 0
    t = t.merge(t1, on='temp')
    t['label2'] = t.label.astype(str)+'-'+t.label1.astype(str)
    t2 = (feature1.astype(str)+'-'+feature2.astype(str)).value_counts(normalize=True).reset_index()
    t2.columns = ['label2', 'label2_mean']
    t = t.merge(t2, on='label2', how='left').dropna()
    t['mutual_information'] = t.label2_mean*np.log2(t.label2_mean/t.label_mean/t.label1_mean)
    return t.mutual_information.sum()

def pointwise_mutual_information(feature1, feature2):
    t = feature1.value_counts(normalize=True).reset_index()
    t.columns = ['label', 'label_mean']
    t['temp'] = 0
    t1 = feature2.value_counts(normalize=True).reset_index()
    t1.columns = ['label1', 'label1_mean']
    t1['temp'] = 0
    t = t.merge(t1, on='temp')
    t['label2'] = t.label.astype(str)+'-'+t.label1.astype(str)
    t2 = (feature1.astype(str)+'-'+feature2.astype(str)).value_counts(normalize=True).reset_index()
    t2.columns = ['label2', 'label2_mean']
    t = t.merge(t2, on='label2', how='left').dropna()
    return np.power(np.cumprod(np.log2(t.label2_mean/t.label_mean/t.label1_mean)).max(), 1/len(t))

def mutual_information_rate(feature1, feature2):
    return mutual_information(feature1, feature2)/min(mutual_information(feature1, feature1), mutual_information(feature2, feature2))

def pointwise_mutual_information_rate(feature1, feature2):
    return pointwise_mutual_information(feature1, feature2)/min(pointwise_mutual_information(feature1, feature1), pointwise_mutual_information(feature2, feature2))
