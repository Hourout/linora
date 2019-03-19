import numpy as np
import pandas as pd

__all__ = ['binary_accuracy', 'categorical_accuracy', 'recall', 'precision', 'confusion_matrix',
           'fbeta_score', 'f1_score', 'auc_roc', 'auc_pr', 'binary_crossentropy', 
           'categorical_crossentropy', 'ks', 'gini', 'psi']

def classified_func(y_true, y_pred, prob=0.5, pos_label=1):
    t = pd.DataFrame({'prob':y_pred, 'label':y_true})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    if t.prob.nunique()!=2:
        label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
        t['label'] = t.label.replace(label_dict)
        t.loc[t.prob>=prob, 'prob'] = 1
        t.loc[t.prob<prob, 'prob'] = 0
    return t

def binary_accuracy(y_true, y_pred, prob=0.5, pos_label=1):
    """
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted labels, as returned by a classifier.
        prob: probability threshold.
        pos_label: positive label.
    Returns:
        the fraction of correctly classified samples (float).
    """
    t = classified_func(y_true, y_pred, prob=prob, pos_label=pos_label)
    return (t.label==t.prob).mean()

def categorical_accuracy(y_true, y_pred):
    """
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted labels, as returned by a classifier.
    Returns:
        the fraction of correctly classified samples (float).
    """
    return (y_true==y_pred).mean()

def recall(y_true, y_pred, prob=0.5, pos_label=1):
    """
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted labels, as returned by a classifier.
        prob: probability threshold.
        pos_label: positive label.
    Returns:
        Recall of the positive class in binary classification.
    """
    t = classified_func(y_true, y_pred, prob=prob, pos_label=pos_label)
    return t.prob[t.label==pos_label].mean()

def precision(y_true, y_pred, prob=0.5, pos_label=1):
    """
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted labels, as returned by a classifier.
        prob: probability threshold.
        pos_label: positive label.
    Returns:
        Precision of the positive class in binary classification.
    """
    t = classified_func(y_true, y_pred, prob=prob, pos_label=pos_label)
    return t.label[t.prob==pos_label].mean()

def confusion_matrix(y_true, y_pred):
    """
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted labels, as returned by a classifier.
    Returns:
        Confusion matrix.
    """
    t = pd.DataFrame({'actual':y_true, 'predict':y_pred})
    t = pd.crosstab(t.predict, t.actual)
    return t

def fbeta_score(y_true, y_pred, beta, prob=0.5, pos_label=1):
    """
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted labels, as returned by a classifier.
        beta : weight of precision in harmonic mean.
        prob: probability threshold.
        pos_label: positive label.
    Returns:
        Fbeta score of the positive class in binary classification.
    """
    r = recall(y_true, y_pred, prob, pos_label)
    p = precision(y_true, y_pred, prob, pos_label)
    return r*p*(1+np.power(beta, 2))/(np.power(beta, 2)*p+r)

def f1_score(y_true, y_pred, prob=0.5, pos_label=1):
    """
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted labels, as returned by a classifier.
        prob: probability threshold.
        pos_label: positive label.
    Returns:
        F1 score of the positive class in binary classification.
    """
    return fbeta_score(y_true, y_pred, beta=1, prob=prob, pos_label=pos_label)

def auc_roc(y_true, y_pred, pos_label=1):
    """
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted labels, as returned by a classifier.
        pos_label: positive label.
    Returns:
        Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    """
    assert y_true.nunique()==2, "`y_true` should be binary classification."
    t = pd.concat([y_true, y_pred], axis=1)
    t.columns = ['label', 'prob']
    t.insert(0, 'target', t[t.label!=pos_label].label.unique()[0])
    t = t[t.label!=pos_label].merge(t[t.label==pos_label], on='target')
    auc = (t.prob_y>t.prob_x).mean()+(t.prob_y==t.prob_x).mean()/2
    return auc

def auc_pr(y_true, y_pred, pos_label=1):
    """
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted labels, as returned by a classifier.
        pos_label: positive label.
    Returns:
        Area Under the Receiver Operating Characteristic Curve (PR AUC) from prediction scores.
    """
    t = pd.DataFrame({'prob':y_pred, 'label':y_true})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    t = t.sort_values(['prob', 'label'], ascending=False).reset_index(drop=True)   
    t['tp'] = t.label.cumsum()
    t['fp'] = t.index+1-t.tp
    t['recall'] = t.tp/t.label.sum()
    t['precision'] = t.tp/(t.tp+t.fp)
    auc = t.sort_values(['recall', 'precision']).drop_duplicates(['recall'], 'last').precision.mean()
    return auc

def binary_crossentropy(y_true, y_pred):
    t = np.exp(y_pred-np.max(y_pred))
    t = -(np.log(t/t.sum())*y_true).mean()
    return t

def categorical_crossentropy(y_true, y_pred, one_hot=True):
    assert y_pred.shape[1]==y_true.nunique(), "`y_pred` and `y_true` dim not same."
    t = np.exp(y_pred.T-np.max(y_pred, axis=1))
    if one_hot:
        t = -(np.log(t/np.sum(t, axis=0)).T*pd.get_dummies(y_true)).sum(axis=1).mean()
    else:
        t = -(np.log(t/np.sum(t, axis=0)).T*y_true).sum(axis=1).mean()
    return t

def ks(y_true, y_pred, pos_label=1):
    """
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted probability, as returned by a classifier.
        pos_label: positive label.
    Returns:
        Ks score of the positive class in binary classification.
    """
    t = pd.DataFrame({'prob':y_pred, 'label':y_true})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    t = t.sort_values(['prob', 'label'], ascending=False).reset_index(drop=True)   
    t['tp'] = t.label.cumsum()
    t['fp'] = t.index+1-t.tp
    t['tpr'] = t.tp/t.label.sum()
    t['fpr'] = t.fp/(t.label.count()-t.label.sum())
    ks = (t.tpr-t.fpr).abs().max()
    return ks

def gini(y_true, y_pred, pos_label=1):
    """
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted probability, as returned by a classifier.
        pos_label: positive label.
    Returns:
        Gini score of the positive class in binary classification.
    """
    t = pd.DataFrame({'prob':y_pred, 'label':y_true})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    t = t.sort_values(['prob', 'label'], ascending=False).reset_index(drop=True)
    gini = (t.label.cumsum().sum()/t.label.sum()-(t.label.count()+1)/2)/t.label.count()
    return gini

def psi(y_true, y_pred, threshold):
    actual = (y_true-y_true.min())/(y_true.max()-y_true.min())
    predict = (y_pred-y_pred.min())/(y_pred.max()-y_pred.min())
    actual = pd.cut(actual, threshold, labels=range(1, len(threshold))).value_counts(normalize=True).reset_index()
    actual.columns = ['label', 'prob1']
    predict = pd.cut(predict, threshold, labels=range(1, len(threshold))).value_counts(normalize=True).reset_index()
    predict.columns = ['label', 'prob2']
    predict = actual.merge(predict, on='label', how='outer')
    psi = ((predict.prob1-predict.prob2)*np.log((predict.prob1/(predict.prob2+0.00000001)))).sum()
    return psi
