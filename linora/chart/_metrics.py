import numpy as np
import pandas as pd
import pyecharts as pe
from linora.metrics import confusion_matrix

__all__ = ['ks_curve', 'roc_curve', 'pr_curve', 'lift_curve', 'gain_curve', 'gini_curve',
           'confusion_matrix_map']

def ks_curve(y_true, y_pred, pos_label=1, jupyter=True, path='Kolmogorov-Smirnov Curve.html'):
    t = pd.DataFrame({'prob':y_pred, 'label':y_true})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    t = t.sort_values(['prob', 'label'], ascending=False).reset_index(drop=True)
    t['tp'] = t.label.cumsum()
    t['fp'] = t.index+1-t.tp
    t['tpr'] = t.tp/t.label.sum()
    t['fpr'] = t.fp/(t.label.count()-t.label.sum())
    t['ks'] = (t.tpr-t.fpr).abs().round(4)
    t.index = np.round(((t.index+1)/len(t)), 2)
    line = pe.Line("Kolmogorov-Smirnov Curve")
    line.add("TPR", [0]+t.index.tolist(), [0]+t.tpr.round(4).tolist(),
             is_smooth=True, yaxis_type='value', xaxis_type='value')
    line.add("FPR", [0]+t.index.tolist(), [0]+t.fpr.round(4).tolist(),
             is_smooth=True, yaxis_type='value', xaxis_type='value')
    line.add("KS", [0]+t.index.tolist(), [0]+t.ks.tolist(),
             is_smooth=True, mark_point=["max"], yaxis_type='value', xaxis_type='value')
    return line if jupyter else line.render(path)

def roc_curve(y_true, y_pred, pos_label=1, jupyter=True, path='Receiver Operating Characteristic Curve.html'):
    t = pd.DataFrame({'prob':y_pred, 'label':y_true})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    t = t.sort_values(['prob', 'label'], ascending=False).reset_index(drop=True)
    t['tp'] = t.label.cumsum()
    t['fp'] = t.index+1-t.tp
    t['tpr'] = t.tp/t.label.sum()
    t['fpr'] = t.fp/(t.label.count()-t.label.sum())
    line = pe.Line("ROC Curve")
    line.add("ROC", [0]+t.fpr.round(4).tolist(), [0]+t.tpr.round(4).tolist(),
             is_smooth=True, is_fill=True, area_opacity=0.4,
             yaxis_type='value', xaxis_type='value')
    line.add("Random", [0]+t.fpr.round(4).tolist(), [0]+t.fpr.round(4).tolist(),
             yaxis_type='value', xaxis_type='value', xaxis_name='FPR', yaxis_name='TPR')
    return line if jupyter else line.render(path)

def pr_curve(y_true, y_pred, pos_label=1, jupyter=True, path='Precision Recall Curve.html'):
    t = pd.DataFrame({'prob':y_pred, 'label':y_true})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    t = t.sort_values(['prob', 'label'], ascending=False).reset_index(drop=True)
    t['tp'] = t.label.cumsum()
    t['fp'] = t.index+1-t.tp
    t['recall'] = t.tp/t.label.sum()
    t['precision'] = t.tp/(t.tp+t.fp)
    line = pe.Line("PR Curve")
    line.add("PR", [0]+t.recall.round(4).tolist(), [1]+t.precision.round(4).tolist(),
             is_smooth=True, is_fill=True, area_opacity=0.4,
             yaxis_type='value', xaxis_type='value', xaxis_name='recall', yaxis_name='precision')
    return line if jupyter else line.render(path)

def lift_curve(y_true, y_pred, pos_label=1, jupyter=True, path='Lift Curve.html'):
    t = pd.DataFrame({'prob':y_pred, 'label':y_true})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    t = t.sort_values(['prob', 'label'], ascending=False).reset_index(drop=True)
    t['tp'] = t.label.cumsum()
    t['fp'] = t.index+1-t.tp
    t['precision'] = t.tp/(t.tp+t.fp)
    t['lift'] = t.precision/t.label.sum()*t.label.count()
    t.index = np.round(((t.index+1)/len(t)), 2)
    line = pe.Line("Lift Curve")
    line.add("Lift", t.index.tolist(), t.lift.round(3).tolist(), is_smooth=True,
             yaxis_type='value', xaxis_type='value')
    line.add("Random", t.index.tolist(), [1]*len(t), is_smooth=True,
             yaxis_type='value', xaxis_type='value')
    return line if jupyter else line.render(path)

def gain_curve(y_true, y_pred, pos_label=1, jupyter=True, path='Gain Curve.html'):
    t = pd.DataFrame({'prob':y_pred, 'label':y_true})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    t = t.sort_values(['prob', 'label'], ascending=False).reset_index(drop=True)
    t['tp'] = t.label.cumsum()
    t['fp'] = t.index+1-t.tp
    t['precision'] = t.tp/(t.tp+t.fp)
    t.index = np.round(((t.index+1)/len(t)), 2)
    line = pe.Line("Gain Curve")
    line.add("Gain", t.index.tolist(), t.precision.round(4).tolist(),
             is_smooth=True, yaxis_type='value', xaxis_type='value')
    return line if jupyter else line.render(path)

def gini_curve(y_true, y_pred, pos_label=1, jupyter=True, path='Gini Curve.html'):
    t = pd.DataFrame({'prob':y_pred, 'label':y_true})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    t = t.sort_values(['prob', 'label']).reset_index(drop=True)
    t['cum'] = t.label.cumsum()/t.label.sum()
    t.index = np.round(((t.index+1)/len(t)), 2)
    line = pe.Line("Gini Curve")
    line.add("Random", [0]+t.index.tolist(), [0]+t.index.tolist(),
             is_fill=True, area_opacity=0.4,
             is_smooth=True, yaxis_type='value', xaxis_type='value')
    line.add("Gini", [0]+t.index.tolist(), [0]+t.cum.round(4).tolist(),
             is_fill=True, area_opacity=0.4,
             is_smooth=True, yaxis_type='value', xaxis_type='value')
    return line if jupyter else line.render(path)

def confusion_matrix_map(y_true, y_pred, jupyter=True, path="Confusion Matrix Map.html"):
    t = confusion_matrix(y_true, y_pred)
    t = pd.DataFrame([[i, m,n ] for i,j in t.to_dict().items() for m, n in j.items()],
                     columns=['actual', 'predict', 'over_values'])
    heatmap = pe.HeatMap("Confusion Matrix Map")
    heatmap.add("Confusion Matrix", t.drop_duplicates(['actual']).actual.values, t.drop_duplicates(['predict']).predict.values,
                t.values, is_visualmap=True, xaxis_name='Actual', yaxis_name='Predict',
                visual_text_color="#000", visual_orient="horizontal", visual_range=[0, t.over_values.max()])
    return  heatmap if jupyter else heatmap.render(path)
