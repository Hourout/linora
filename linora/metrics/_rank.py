import pandas as pd

from linora.metrics._utils import _sample_weight

__all__  = ['mapk', 'hit_ratio', 'mean_reciprocal_rank']


def mapk(y_true, y_pred, k, sample_weight=None):
    """Mean Average Precision k
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a rank.
        k: int, top k predict values.
        sample_weight: list or array of sample weight.
    Returns:
        Mean Average Precision k values.
    """
    def apk(actual, predict, weight, k):
        if len(predict)>k:
            predict = predict[:k]
        score = 0.0
        nums = 0.0
        for i,p in enumerate(predict):
            if p in actual and p not in predict[:i]:
                nums += 1.0
                score += nums / (i+1.0)
        return score / min(len(actual), k)*weight if actual else 0.0
    sample_weight = _sample_weight(y_true, sample_weight)
    return pd.DataFrame({'label1':y_true, 'label2':y_pred, 'weight':sample_weight}).apply(lambda x:apk(x[0], x[1], x[2], k=k), axis=1).mean()


def hit_ratio(y_true, y_pred, k, sample_weight=None):
    """Hit Ratio k
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a rank.
        k: int, top k predict values.
        sample_weight: list or array of sample weight.
    Returns:
        Hit Ratio k values.
    """
    sample_weight = _sample_weight(y_true, sample_weight)
    t = pd.DataFrame({'label1':y_true, 'label2':y_pred, 'weight':sample_weight})
    return t.apply(lambda x:len(set(x[0]).intersection(set(x[1][:k])))*x[2], axis=1).sum()/t.label1.map(lambda x:len(set(x))).sum()


def mean_reciprocal_rank(y_true, y_pred, k, sample_weight=None):
    """Mean Reciprocal Rank
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted values, as returned by a rank.
        k: int, top k predict values.
        sample_weight: list or array of sample weight.
    Returns:
        mean reciprocal rank k values.
    """
    def mrr(actual, predict, weight, k):
        try:
            rank = 1./(predict[:k].index(actual)+1)*weight
        except:
            rank = 0
        return rank
    sample_weight = _sample_weight(y_true, sample_weight)
    return pd.DataFrame({'label1':y_true, 'label2':y_pred, 'weight':sample_weight}).apply(lambda x: mrr(x[0], x[1], x[2], k=k), axis=1).mean()
