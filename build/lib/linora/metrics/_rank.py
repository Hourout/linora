import pandas as pd

__all__  = ['mapk', 'hit_ratio', 'mean_reciprocal_rank']

def mapk(y_true, y_pred, k):
    """Mean Average Precision k
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted values, as returned by a rank.
        k: int, top k predict values.
    Returns:
        Mean Average Precision k values.
    """
    def apk(actual, predict, k):
        if len(predict)>k:
            predict = predict[:k]
        score = 0.0
        nums = 0.0
        for i,p in enumerate(predict):
            if p in actual and p not in predict[:i]:
                nums += 1.0
                score += nums / (i+1.0)
        return score / min(len(actual), k) if actual else 0.0
    return pd.DataFrame({'label1':y_true, 'label2':y_pred}).apply(lambda x:apk(x[0], x[1], k=k), axis=1).mean()

def hit_ratio(y_true, y_pred, k):
    """Hit Ratio k
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted values, as returned by a rank.
        k: int, top k predict values.
    Returns:
        Hit Ratio k values.
    """
    t = pd.DataFrame({'label1':y_true, 'label2':y_pred})
    a = t.apply(lambda x:len(set(x[0]).intersection(set(x[1]))) if len(x[1])<=k else len(set(x[0]).intersection(set(x[1][:k]))), axis=1).sum()
    b = t.label1.map(lambda x:len(set(x))).sum()
    return a/b

def mean_reciprocal_rank(y_true, y_pred, k):
    """Mean Reciprocal Rank
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted values, as returned by a rank.
        k: int, top k predict values.
    Returns:
        mean reciprocal rank k values.
    """
    def mrr(actual, predict, k):
        if len(predict)>k:
            predict = predict[:k]
        try:
            rank = 1./(predict.index(actual)+1)
        except:
            rank = 0
        return rank
    return pd.DataFrame({'label1':y_true, 'label2':y_pred}).apply(lambda x: mrr(x[0], x[1], k=k), axis=1).mean()
