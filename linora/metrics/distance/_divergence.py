import numpy as np
import pandas as pd

__all__ = ['divergence_kl', 'divergence_js', 'divergence_F',
           'mutual_information', 'mutual_information_rate', 
           'pointwise_mutual_information', 'pointwise_mutual_information_rate'
          ]


def divergence_kl(x, y, continuous=False, bucket_num=1000):
    """kl divergence.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        continuous: bool, default False, whether it is a continuous value. if True, normalized x and y.
        bucket_num: int, default 1000, if continuous is True, perform bucket operations on features.
    Returns:
        kl divergence value.
    """
    x = pd.Series(x)
    y = pd.Series(y)
    if continuous:
        x = (x-min(x.min(), y.min()))/(max(x.max(), y.max())-min(x.min(), y.min()))
        y = (y-min(x.min(), y.min()))/(max(x.max(), y.max())-min(x.min(), y.min()))
        x = pd.cut(x, bucket_num, labels=range(bucket_num))
        y = pd.cut(y, bucket_num, labels=range(bucket_num))
    t = x.value_counts(normalize=True).reset_index().merge(y.value_counts(normalize=True).reset_index(), on='index', how='left').fillna(0.)
    t.columns = ['label', 'prob_x', 'prob_y']
    t = np.sum(np.log((t.prob_x+0.00001)/(t.prob_y+0.00001))*t.prob_x)
    return t


def divergence_js(x, y, continuous=False, bucket_num=1000):
    """js divergence.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
        continuous: default False, whether it is a continuous value. if True, normalized x and y.
        bucket_num: default 1000, if continuous is True, perform bucket operations on features.
    Returns:
        js divergence value.
    """
    x = pd.Series(x)
    y = pd.Series(y)
    if continuous:
        x = (x-min(x.min(), y.min()))/(max(x.max(), y.max())-min(x.min(), y.min()))
        y = (y-min(x.min(), y.min()))/(max(x.max(), y.max())-min(x.min(), y.min()))
        x = pd.cut(x, bucket_num, labels=range(bucket_num))
        y = pd.cut(y, bucket_num, labels=range(bucket_num))
    t = x.value_counts(normalize=True).reset_index().merge(y.value_counts(normalize=True).reset_index(), on='index', how='outer').fillna(0.)
    t.columns = ['label', 'prob_x', 'prob_y']
    t['prob_m'] = (t.prob_x+t.prob_y)/2
    t1 = np.sum(np.log((t.prob_x+0.00001)/(t.prob_m+0.00001))*t.prob_x)
    t2 = np.sum(np.log((t.prob_y+0.00001)/(t.prob_m+0.00001))*t.prob_y)
    return 0.5*(t1+t2)


def divergence_F(x, y):
    """F divergence.
    
    Args:
        x: pd.Series or array or list, sample n dim feature value.
        y: pd.Series or array or list, sample n dim feature value.
    Returns:
        F divergence value.
    """
    x = np.array(x)
    y = np.array(y)
    value = y*x
    x[np.where(x==0)] = 1
    y[np.where(y==0)] = 1
    return np.sum(value/y*np.log(x/y))


def mutual_information(feature1, feature2):
    """mutual information.
    
    Args:
        feature1: pd.Series, sample feature value.
        feature2: pd.Series, sample feature value.
    Returns:
        mutual information value.
    """
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
    """pointwise mutual information.
    
    Args:
        feature1: pd.Series, sample feature value.
        feature2: pd.Series, sample feature value.
    Returns:
        pointwise mutual information value.
    """
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
    """mutual information rate.
    
    Args:
        feature1: pd.Series, sample feature value.
        feature2: pd.Series, sample feature value.
    Returns:
        mutual information rate value.
    """
    return mutual_information(feature1, feature2)/min(mutual_information(feature1, feature1), mutual_information(feature2, feature2))


def pointwise_mutual_information_rate(feature1, feature2):
    """pointwise mutual information rate.
    
    Args:
        feature1: pd.Series, sample feature value.
        feature2: pd.Series, sample feature value.
    Returns:
        pointwise mutual information rate value.
    """
    return pointwise_mutual_information(feature1, feature2)/min(pointwise_mutual_information(feature1, feature1), pointwise_mutual_information(feature2, feature2))
