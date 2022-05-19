import numpy as np
import pandas as pd

from linora.metrics._regression import mean_squared_error

__all__ = ['accuracy_binary', 'accuracy_categorical', 'recall', 'precision', 'confusion_matrix',
           'fbeta_score', 'f1_score', 'auc_roc', 'auc_pr', 'crossentropy_binary', 
           'crossentropy_categorical', 'ks', 'gini', 'psi', 'fmi', 'report_binary',
           'accuracy_categorical_top_k', 'iou_binary', 'iou_categorical',
           'precision_on_recall', 'recall_on_precision', 
           'specificity_on_sensitivity', 'sensitivity_on_specificity'
          ]


def classified_func(y_true, y_pred, prob=0.5, pos_label=1):
    t = pd.DataFrame({'prob':y_pred, 'label':y_true})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    if t.prob.nunique()!=2:
        label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
        t['label'] = t.label.replace(label_dict)
        t.loc[t.prob>=prob, 'prob'] = 1
        t.loc[t.prob<prob, 'prob'] = 0
    return t


def _sample_weight(y_true, sample_weight):
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))
    elif isinstance(sample_weight, dict):
        sample_weight = [sample_weight[i] for i in y_true]
    else:
        sample_weight = np.array(sample_weight)
    return sample_weight/sample_weight.sum()*len(sample_weight)


def accuracy_binary(y_true, y_pred, sample_weight=None, prob=0.5, pos_label=1):
    """Calculates how often predictions match binary labels.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        sample_weight: list or array or dict of sample weight.
        prob: probability threshold.
        pos_label: positive label.
    Returns:
        the fraction of correctly classified samples (float).
    """
    sample_weight = _sample_weight(y_true, sample_weight)
    t = classified_func(y_true, y_pred, prob=prob, pos_label=pos_label)
    return ((t.label==t.prob)*sample_weight).mean()


def accuracy_categorical(y_true, y_pred, sample_weight=None):
    """Calculates how often predictions match categorical labels.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        sample_weight: list or array or dict of sample weight.
    Returns:
        the fraction of correctly classified samples (float).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.ndim!=1:
        y_true = np.argmax(y_true, axis=-1)
    if y_pred.ndim!=1:
        y_pred = np.argmax(y_pred, axis=-1)
    sample_weight = _sample_weight(y_true, sample_weight)
    return ((pd.Series(y_true)==pd.Series(y_pred))*sample_weight).mean()


def iou_binary(y_true, y_pred, sample_weight=None, prob=0.5, pos_label=1):
    """Computes the Intersection-Over-Union metric for class 0 and/or 1.
    
    iou = true_positives / (true_positives + false_positives + false_negatives)
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        sample_weight: list or array or dict of sample weight.
        prob: probability threshold.
        pos_label: positive label.
    Returns:
        the fraction of correctly classified samples (float).
    """
    sample_weight = _sample_weight(y_true, sample_weight)
    t = classified_func(y_true, y_pred, prob=prob, pos_label=pos_label)
    t['weight'] = sample_weight
    tp = (((t.label==pos_label)&(t.prob==pos_label))*t.weight).sum()
    fp = (((t.label==pos_label)&(t.prob!=pos_label))*t.weight).sum()
    fn = (((t.label!=pos_label)&(t.prob==pos_label))*t.weight).sum()
    return tp/(tp+fp+fn)


def iou_categorical(y_true, y_pred, sample_weight=None, target_class_ids=None):
    """Computes mean Intersection-Over-Union metric for one-hot encoded or categorical labels.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        sample_weight: list or array or dict of sample weight.
        target_class_ids: A tuple or list of target class ids for which the metric is returned. 
    Returns:
        the fraction of correctly classified samples (float).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.ndim!=1:
        y_true = np.argmax(y_true, axis=-1)
    if y_pred.ndim!=1:
        y_pred = np.argmax(y_pred, axis=-1)
    sample_weight = _sample_weight(y_true, sample_weight)
    t = pd.DataFrame({'prob':y_pred.flatten(), 'label':y_true.flatten(), 'weight':sample_weight})
    if target_class_ids is None:
        target_class_ids = t.label.unique().tolist()
    result = {'IoU_mean':[], 'IoU_class':{i:0 for i in target_class_ids}}
    for i in t.label.unique():
        if i in target_class_ids:
            tp = (((t.label==i)&(t.prob==i))*t.weight).sum()
            fp = (((t.label==i)&(t.prob!=i))*t.weight).sum()
            fn = (((t.label!=i)&(t.prob==i))*t.weight).sum()
            result['IoU_mean'].append(tp/(tp+fp+fn))
            result['IoU_class'][i] = tp/(tp+fp+fn)
    result['IoU_mean'] = sum(result['IoU_mean'])/len(result['IoU_mean'])
    return result


def recall(y_true, y_pred, sample_weight=None, prob=0.5, pos_label=1):
    """Computes the recall of the predictions with respect to the labels.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        sample_weight: list or array or dict of sample weight.
        prob: probability threshold.
        pos_label: positive label.
    Returns:
        Recall of the positive class in binary classification.
    """
    sample_weight = _sample_weight(y_true, sample_weight)
    t = classified_func(y_true, y_pred, prob=prob, pos_label=pos_label)
    return (t.prob*sample_weight)[t.label==pos_label].mean()


def precision(y_true, y_pred, sample_weight=None, prob=0.5, pos_label=1):
    """Computes the precision of the predictions with respect to the labels.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        sample_weight: list or array or dict of sample weight.
        prob: probability threshold.
        pos_label: positive label.
    Returns:
        Precision of the positive class in binary classification.
    """
    sample_weight = _sample_weight(y_true, sample_weight)
    t = classified_func(y_true, y_pred, prob=prob, pos_label=pos_label)
    return (t.label*sample_weight)[t.prob==pos_label].mean()


def _cumsum_confusion_matrix(y_true, y_pred, sample_weight=None, pos_label=1):
    sample_weight = _sample_weight(y_true, sample_weight)
    t = pd.DataFrame({'prob':y_pred, 'label':y_true, 'weight':sample_weight})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    t = t.sort_values(['prob', 'label'], ascending=False).reset_index(drop=True)   
    t['tp'] = (t.label*t.weight).cumsum()
    t['fp'] = t.weight.cumsum()-t.tp
    t['tn'] = ((1-t.label)*t.weight).iloc[::-1].cumsum()
    t['fn'] = t.weight.iloc[::-1].cumsum()-t.tn
    t['recall'] = t.tp/(t.tp + t.fn)
    t['precision'] = t.tp/(t.tp+t.fp)
    t['specificity'] = t.tn/(t.tn + t.fp)
    t['sensitivity'] = t.tp/(t.tp + t.fn)
    return t


def precision_on_recall(y_true, y_pred, recall=0.5, sample_weight=None, pos_label=1):
    """Computes best precision where recall is >= specified value.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        recall: A scalar value in range [0, 1].
        sample_weight: list or array or dict of sample weight.
        pos_label: positive label.
    Returns:
        prediction scores.
    """
    t = _cumsum_confusion_matrix(y_true, y_pred, sample_weight=sample_weight, pos_label=pos_label)
    return t[t.recall>=recall].precision.max()


def recall_on_precision(y_true, y_pred, precision=0.5, sample_weight=None, pos_label=1):
    """Computes best recall where precision is >= specified value.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        precision: A scalar value in range [0, 1].
        sample_weight: list or array or dict of sample weight.
        pos_label: positive label.
    Returns:
        recall scores.
    """
    t = _cumsum_confusion_matrix(y_true, y_pred, sample_weight=sample_weight, pos_label=pos_label)
    return t[t.precision>=precision].recall.max()


def sensitivity_on_specificity(y_true, y_pred, specificity=0.5, sample_weight=None, pos_label=1):
    """Computes best sensitivity where specificity is >= specified value.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        specificity: A scalar value in range [0, 1].
        sample_weight: list or array or dict of sample weight.
        pos_label: positive label.
    Returns:
        sensitivity scores.
    """
    t = _cumsum_confusion_matrix(y_true, y_pred, sample_weight=sample_weight, pos_label=pos_label)
    return t[t.specificity>=specificity].sensitivity.max()


def specificity_on_sensitivity(y_true, y_pred, sensitivity=0.5, sample_weight=None, pos_label=1):
    """Computes best specificity where sensitivity is >= specified value.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        sensitivity: A scalar value in range [0, 1].
        sample_weight: list or array or dict of sample weight.
        pos_label: positive label.
    Returns:
        specificity scores.
    """
    t = _cumsum_confusion_matrix(y_true, y_pred, sample_weight=sample_weight, pos_label=pos_label)
    return t[t.sensitivity>=sensitivity].specificity.max()


def confusion_matrix(y_true, y_pred):
    """
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        sample_weight: list or array of sample weight.
    Returns:
        Confusion matrix.
    """
    t = pd.DataFrame({'actual':y_true, 'predict':y_pred})
    t = pd.crosstab(t.predict, t.actual)
    return t


def fbeta_score(y_true, y_pred, beta, sample_weight=None, prob=0.5, pos_label=1):
    """
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        beta : int or float, weight of precision in harmonic mean.
        sample_weight: list or array or dict of sample weight.
        prob: probability threshold.
        pos_label: positive label.
    Returns:
        Fbeta score of the positive class in binary classification.
    """
    sample_weight = _sample_weight(y_true, sample_weight)
    r = recall(y_true, y_pred, sample_weight, prob, pos_label)
    p = precision(y_true, y_pred, sample_weight, prob, pos_label)
    return r*p*(1+np.power(beta, 2))/(np.power(beta, 2)*p+r)


def f1_score(y_true, y_pred, sample_weight=None, prob=0.5, pos_label=1):
    """
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        sample_weight: list or array or dict of sample weight.
        prob: probability threshold.
        pos_label: positive label.
    Returns:
        F1 score of the positive class in binary classification.
    """
    sample_weight = _sample_weight(y_true, sample_weight)
    return fbeta_score(y_true, y_pred, beta=1, sample_weight=sample_weight, prob=prob, pos_label=pos_label)


def auc_roc(y_true, y_pred, sample_weight=None, pos_label=1):
    """Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        sample_weight: list or array or dict of sample weight.
        pos_label: positive label.
    Returns:
        Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    """
    sample_weight = _sample_weight(y_true, sample_weight)
    t = pd.DataFrame({'prob':y_pred, 'label':y_true, 'weight':sample_weight})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    t.insert(0, 'target', t[t.label!=pos_label].label.unique()[0])
    t = t[t.label!=pos_label].merge(t[t.label==pos_label], on='target')
    auc = ((t.prob_y>t.prob_x)*(t.weight_y+t.weight_x)/2).mean()+((t.prob_y==t.prob_x)*(t.weight_y+t.weight_x)/2).mean()/2
    return auc


def auc_pr(y_true, y_pred, sample_weight=None, pos_label=1):
    """Area Under the Receiver Operating Characteristic Curve (PR AUC)
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        sample_weight: list or array or dict of sample weight.
        pos_label: positive label.
    Returns:
        Area Under the Receiver Operating Characteristic Curve (PR AUC) from prediction scores.
    """
    sample_weight = _sample_weight(y_true, sample_weight)
    t = pd.DataFrame({'prob':y_pred, 'label':y_true, 'weight':sample_weight})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    t = t.sort_values(['prob', 'label'], ascending=False).reset_index(drop=True)   
    t['tp'] = (t.label*t.weight).cumsum()
    t['fp'] = t.weight.cumsum()-t.tp
    t['recall'] = t.tp/(t.label*t.weight).sum()
    t['precision'] = t.tp/(t.tp+t.fp)
    auc = t.sort_values(['recall', 'precision']).drop_duplicates(subset=['recall'], keep='last').precision.mean()
    return auc


def crossentropy_binary(y_true, y_pred, sample_weight=None):
    """Computes the crossentropy metric between the labels and predictions.
    
    This is the crossentropy metric class to be used when there are only two label classes (0 and 1).
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted probability, as returned by a classifier.
        sample_weight: list or array or dict of sample weight.
    Returns:
        binary crossentropy of the positive class in binary classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sample_weight = _sample_weight(y_true, sample_weight)
    t = np.exp((y_pred-np.max(y_pred))*sample_weight)
    t = -(np.log(t/t.sum())*y_true).mean()
    return t


def crossentropy_categorical(y_true, y_pred, sample_weight=None, one_hot=False):
    """Computes the crossentropy metric between the labels and predictions.
    
    This is the crossentropy metric class to be used when there are multiple label classes (2 or more). 
    Here we assume that labels are given as a one_hot representation. 
    eg., When labels values are [2, 0, 1], y_true = [[0, 0, 1], [1, 0, 0], [0, 1, 0]].
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted probability, as returned by a classifier.
        sample_weight: list or array or dict of sample weight.
        one_hot: default True, Whether y_true is a one_hot variable.
    Returns:
        categorical crossentropy of the positive class in categorical classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sample_weight = _sample_weight(y_true, sample_weight)
    assert y_pred.shape[1]==np.unique(np.array(y_true)).size, "`y_pred` and `y_true` dim not same."
    t = np.exp(y_pred.T-np.max(y_pred, axis=1))
    if one_hot:
        t = -((np.log(t/np.sum(t, axis=0)).T*pd.get_dummies(y_true)).sum(axis=1)*sample_weight).mean()
    else:
        t = -((np.log(t/np.sum(t, axis=0)).T*y_true).sum(axis=1)*sample_weight).mean()
    return t


def ks(y_true, y_pred, sample_weight=None, pos_label=1):
    """Kolmogorov-Smirnov metrics.
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted probability, as returned by a classifier.
        sample_weight: list or array or dict of sample weight.
        pos_label: positive label.
    Returns:
        KS score of the positive class in binary classification.
    """
    sample_weight = _sample_weight(y_true, sample_weight)
    t = pd.DataFrame({'prob':y_pred, 'label':y_true, 'weight':sample_weight})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    t = t.sort_values(['prob', 'label'], ascending=False).reset_index(drop=True)   
    t['tp'] = (t.label*t.weight).cumsum()
    t['fp'] = t.weight.cumsum()-t.tp
    t['tpr'] = t.tp/(t.label*t.weight).sum()
    t['fpr'] = t.fp/(t.weight.sum()-(t.label*t.weight).sum())
    ks = (t.tpr-t.fpr).abs().max()
    return ks


def gini(y_true, y_pred, sample_weight=None, pos_label=1):
    """Gini Coefficient
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted probability, as returned by a classifier.
        sample_weight: list or array or dict of sample weight.
        pos_label: positive label.
    Returns:
        Gini score of the positive class in binary classification.
    """
    sample_weight = _sample_weight(y_true, sample_weight)
    t = pd.DataFrame({'prob':y_pred, 'label':y_true, 'weight':sample_weight})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    t = t.sort_values(['prob', 'label'], ascending=False).reset_index(drop=True)
    gini = ((t.label*t.weight).cumsum().sum()/(t.label*t.weight).sum()-(t.weight.sum()+1)/2)/t.weight.sum()
    return gini


def psi(y_true, y_pred, threshold):
    """population stability index
    
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


def fmi(y_true, y_pred, sample_weight=None, prob=0.5, pos_label=1):
    """
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        sample_weight: list or array or dict of sample weight.
        prob: probability threshold.
        pos_label: positive label.
    Returns:
        FMI of the positive class in binary classification.
    """
    sample_weight = _sample_weight(y_true, sample_weight)
    t = classified_func(y_true, y_pred, prob=prob, pos_label=pos_label)
    t['weight'] = sample_weight
    tp = (((t.label==pos_label)&(t.prob==pos_label))*t.weight).sum()
    fp = (((t.label==pos_label)&(t.prob!=pos_label))*t.weight).sum()
    fn = (((t.label!=pos_label)&(t.prob==pos_label))*t.weight).sum()
    return tp/np.sqrt((tp+fp)*(tp+fn))


def report_binary(y_true, y_pred, sample_weight=None, prob=0.5, pos_label=1, printable=False):
    """binary metrics report
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        sample_weight: list or array or dict of sample weight.
        prob: probability threshold.
        pos_label: positive label.
        printable: bool, print report.
    Returns:
        binary report of the positive class in binary classification.
    """
    t = classified_func(y_true, y_pred, prob=prob, pos_label=pos_label)
    result = {'accuracy':round(accuracy_binary(y_true, y_pred, sample_weight=sample_weight, prob=prob, pos_label=pos_label), 4),
              'precision':round(precision(y_true, y_pred, sample_weight=sample_weight, prob=prob, pos_label=pos_label), 4),
              'recall':round(recall(y_true, y_pred, sample_weight=sample_weight, prob=prob, pos_label=pos_label), 4),
              'f1_score':round(f1_score(y_true, y_pred, sample_weight=sample_weight, prob=prob, pos_label=pos_label), 4),
              'auc_roc':round(auc_roc(y_true, y_pred, sample_weight=sample_weight, pos_label=pos_label), 4),
              'auc_pr':round(auc_pr(y_true, y_pred, sample_weight=sample_weight, pos_label=pos_label), 4),
              'fmi':round(fmi(y_true, y_pred, sample_weight=sample_weight, prob=prob, pos_label=pos_label), 4),
              'gini':round(gini(y_true, y_pred, sample_weight=sample_weight, pos_label=pos_label), 4),
              'ks':round(ks(y_true, y_pred, sample_weight=sample_weight, pos_label=pos_label), 4),
              'RMSE_label':round(mean_squared_error(t.label, t.prob, log=False, root=True, sample_weight=sample_weight), 4),
              'RMSE_prob':round(mean_squared_error(y_true, y_pred, log=False, root=True, sample_weight=sample_weight), 4)
             }
    if printable:
        print("\nBinary Classification Report")
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


def accuracy_categorical_top_k(y_true, y_pred, k, sample_weight=None):
    """top k categorical accuracy
    
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted probs, as returned by a classifier.
        sample_weight: list or array or dict of sample weight.
        k: Number of top elements to look at for computing accuracy.
    Returns:
        Top K categorical accuracy value.
    """
    y_true = np.array(y_true)
    if y_true.ndim!=1:
        y_true = np.argmax(y_true, axis=-1)
    sample_weight = _sample_weight(y_true, sample_weight)
    y_pred = [[s.index(i) for i in sorted(s)[-k:]] for s in y_pred]
    return np.mean([sample_weight[i] if j in y_pred[i] else 0 for i,j in enumerate(y_true)])
