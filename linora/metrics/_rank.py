import pandas as pd

__all__  = ['mapk']

def mapk(y_true, y_pred, k):
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
