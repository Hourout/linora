import numpy as np
import pandas as pd

__all__ = ['binary_accuracy', 'categorical_accuracy', 'recall', 'precision', 'confusion_matrix',
           'fbeta_score', 'f1_score', 'auc_roc', 'auc_pr', 'binary_crossentropy', 
           'categorical_crossentropy', 'ks', 'gini', 'psi', 'fmi', 'binary_report',
           'top_k_categorical_accuracy']

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
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
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
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
    Returns:
        the fraction of correctly classified samples (float).
    """
    if not isinstance(y_true[0], int):
        y_true = np.argmax(y_true, axis=1)
    if not isinstance(y_pred[0], int):
        y_pred = np.argmax(y_pred, axis=1)
    return (pd.Series(y_true)==pd.Series(y_pred)).mean()

def recall(y_true, y_pred, prob=0.5, pos_label=1):
    """
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
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
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
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
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
    Returns:
        Confusion matrix.
    """
    t = pd.DataFrame({'actual':y_true, 'predict':y_pred})
    t = pd.crosstab(t.predict, t.actual)
    return t

def fbeta_score(y_true, y_pred, beta, prob=0.5, pos_label=1):
    """
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
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
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        prob: probability threshold.
        pos_label: positive label.
    Returns:
        F1 score of the positive class in binary classification.
    """
    return fbeta_score(y_true, y_pred, beta=1, prob=prob, pos_label=pos_label)

def auc_roc(y_true, y_pred, pos_label=1):
    """
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        pos_label: positive label.
    Returns:
        Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    """
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
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
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
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
    """Computes the crossentropy metric between the labels and predictions.
    
    This is the crossentropy metric class to be used when there are only two label classes (0 and 1).
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted probability, as returned by a classifier.
    Returns:
        binary crossentropy of the positive class in binary classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    t = np.exp(y_pred-np.max(y_pred))
    t = -(np.log(t/t.sum())*y_true).mean()
    return t

def categorical_crossentropy(y_true, y_pred, one_hot=False):
    """Computes the crossentropy metric between the labels and predictions.
    
    This is the crossentropy metric class to be used when there are multiple label classes (2 or more). 
    Here we assume that labels are given as a one_hot representation. 
    eg., When labels values are [2, 0, 1], y_true = [[0, 0, 1], [1, 0, 0], [0, 1, 0]].
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted probability, as returned by a classifier.
        one_hot: default True, Whether y_true is a one_hot variable.
    Returns:
        categorical crossentropy of the positive class in categorical classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
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
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted probability, as returned by a classifier.
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
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted probability, as returned by a classifier.
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
    """
    Args:
        y_true: pd.Series or array or list, a feature variable.
        y_pred: pd.Series or array or list, a feature variable.
        threshold: list, a threshold list.
    Returns:
        psi value of two variable.
    """
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    actual = (y_true-y_true.min())/(y_true.max()-y_true.min())
    predict = (y_pred-y_pred.min())/(y_pred.max()-y_pred.min())
    actual = pd.cut(actual, threshold, labels=range(1, len(threshold))).value_counts(normalize=True).reset_index()
    actual.columns = ['label', 'prob1']
    predict = pd.cut(predict, threshold, labels=range(1, len(threshold))).value_counts(normalize=True).reset_index()
    predict.columns = ['label', 'prob2']
    predict = actual.merge(predict, on='label', how='outer')
    psi = ((predict.prob1-predict.prob2)*np.log((predict.prob1/(predict.prob2+0.00000001)))).sum()
    return psi

def fmi(y_true, y_pred, prob=0.5, pos_label=1):
    """
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        prob: probability threshold.
        pos_label: positive label.
    Returns:
        FMI of the positive class in binary classification.
    """
    t = classified_func(y_true, y_pred, prob=prob, pos_label=pos_label)
    t = pd.crosstab(t.label, t.prob)
    return t.iat[1, 1]/np.sqrt((t.iat[1, 1]+t.iat[1, 0])*(t.iat[1, 1]+t.iat[0, 1]))

def binary_report(y_true, y_pred, prob=0.5, pos_label=1, printable=False, printinfo='Binary Classification Report'):
    """
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        prob: probability threshold.
        pos_label: positive label.
    Returns:
        binary report of the positive class in binary classification.
    """
    t = classified_func(y_true, y_pred, prob=prob, pos_label=pos_label)
    cm = pd.crosstab(t.label, t.prob)
    tp = cm.iat[1, 1]
    fn = cm.iat[1, 0]
    fp = cm.iat[0, 1]
    tn = cm.iat[0, 0]
    result = {'accuracy':round((tp+tn)/(tp+tn+fp+fn), 4),
              'precision':round(tp/(tp+fp), 4),
              'recall':round(tp/(tp+fn), 4),
              'f1_score':round(2*(tp/(tp+fp))*(tp/(tp+fn))/((tp/(tp+fp))+(tp/(tp+fn))), 4),
              'auc_roc':round(auc_roc(t.label, t.prob, pos_label=pos_label), 4),
              'auc_pr':round(auc_pr(t.label, t.prob, pos_label=pos_label), 4),
              'fmi':round(tp/np.sqrt((tp+fp)*(tp+fn)), 4),
              'gini':round(gini(t.label, t.prob), 4),
              'ks':round(ks(t.label, t.prob), 4),
              'RMSE_label':round(np.sqrt(np.mean(np.square((t.label-t.prob)))), 4),
              'RMSE_prob':round(np.sqrt(np.mean(np.square((t.label-y_pred)))), 4)
             }
    if printable:
        print("\n{}".format(printinfo))
        print("Accuracy: %.4f" % result['accuracy'])
        print("Precision: %.4f" % result['precision'])
        print("Recall: %.4f" % result['recall'])
        print("F1_score: %.4f" % result['f1_score'])
        print("AUC_ROC Score: %.4f" % result['auc_roc'])
        print("AUC_PR Score: %.4f" % result['auc_pr'])
        print("FMI: %.4f" % result['fmi'])
        print("KS: %.4f" % result['ks'])
        print("Gini: %.4f" % result['gini'])
        print("RMSE_label: %.4f" % result['RMSE_label'])
        print("RMSE_prob: %.4f" % result['RMSE_prob'])
    return result

def top_k_categorical_accuracy(y_true, y_pred, k):
    """
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted probs, as returned by a classifier.
    Returns:
        the fraction of correctly classified samples (float).
    """
    if not isinstance(y_true[0], int):
        y_true = np.argmax(y_true, axis=1)
    y_pred = [[s.index(i) for i in sorted(s)[-k:]] for s in y_pred]
    return np.mean([1 if j in y_pred[i] else 0 for i,j in enumerate(y_true)])
