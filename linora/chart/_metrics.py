import numpy as np
import pandas as pd
import pyecharts as pe
from linora.metrics import confusion_matrix

__all__ = ['ks_curve', 'roc_curve', 'pr_curve', 'lift_curve', 'gain_curve', 'gini_curve',
           'confusion_matrix_map']

def ks_curve(y_true, y_pred, pos_label=1, jupyter=True, path='Kolmogorov-Smirnov Curve.html'):
    """plot Kolmogorov-Smirnov Curve.
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted labels, as returned by a classifier.
        pos_label: positive label.
        jupyter: if plot in jupyter, default True.
        path: if jupyter is False, result save a html file.
    Returns:
        A pyecharts polt object.
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
    t['ks'] = (t.tpr-t.fpr).abs().round(4)
    t.index = np.round(((t.index+1)/len(t)), 2)
    line = (pe.charts.Line()
        .add_xaxis([0]+t.index.tolist())
        .add_yaxis("TPR", [0]+t.tpr.round(4).tolist(), is_smooth=True)
        .add_yaxis("FPR", [0]+t.fpr.round(4).tolist(), is_smooth=True)
        .add_yaxis("KS", [0]+t.ks.tolist(), is_smooth=True, markpoint_opts=pe.options.MarkPointOpts(data=[pe.options.MarkPointItem(type_='max', name='最大值')]))
        .set_series_opts(label_opts=pe.options.LabelOpts(is_show=False))
        .set_global_opts(title_opts=pe.options.TitleOpts(title="Kolmogorov-Smirnov Curve"),
                         xaxis_opts=pe.options.AxisOpts(is_scale=True, type_='value')))
    return line.render_notebook() if jupyter else line.render(path)

def roc_curve(y_true, y_pred, pos_label=1, jupyter=True, path='Receiver Operating Characteristic Curve.html'):
    """plot Receiver Operating Characteristic Curve.
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted labels, as returned by a classifier.
        pos_label: positive label.
        jupyter: if plot in jupyter, default True.
        path: if jupyter is False, result save a html file.
    Returns:
        A pyecharts polt object.
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
    line = (pe.charts.Line()
        .add_xaxis([0]+t.fpr.round(4).tolist())
        .add_yaxis("ROC", [0]+t.tpr.round(4).tolist(), is_smooth=True)
        .add_yaxis("Random", [0]+t.fpr.round(4).tolist())
        .set_series_opts(label_opts=pe.options.LabelOpts(is_show=False),
                         areastyle_opts=pe.options.AreaStyleOpts(0.4))
        .set_global_opts(title_opts=pe.options.TitleOpts(title="ROC Curve"),
                         xaxis_opts=pe.options.AxisOpts('FPR', is_scale=True, type_='value'),
                         yaxis_opts=pe.options.AxisOpts('TPR')))
    return line.render_notebook() if jupyter else line.render(path)
    
def pr_curve(y_true, y_pred, pos_label=1, jupyter=True, path='Precision Recall Curve.html'):
    """plot Precision Recall Curve.
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted labels, as returned by a classifier.
        pos_label: positive label.
        jupyter: if plot in jupyter, default True.
        path: if jupyter is False, result save a html file.
    Returns:
        A pyecharts polt object.
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
    line = (pe.charts.Line()
        .add_xaxis([0]+t.recall.round(4).tolist())
        .add_yaxis("PR", [1]+t.precision.round(4).tolist(), is_smooth=True)
        .set_series_opts(label_opts=pe.options.LabelOpts(is_show=False),
                         areastyle_opts=pe.options.AreaStyleOpts(0.4))
        .set_global_opts(title_opts=pe.options.TitleOpts(title="PR Curve"),
                         xaxis_opts=pe.options.AxisOpts('recall', is_scale=True, type_='value'),
                         yaxis_opts=pe.options.AxisOpts('precision')))
    return line.render_notebook() if jupyter else line.render(path)

def lift_curve(y_true, y_pred, pos_label=1, jupyter=True, path='Lift Curve.html'):
    """plot Lift Curve.
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted labels, as returned by a classifier.
        pos_label: positive label.
        jupyter: if plot in jupyter, default True.
        path: if jupyter is False, result save a html file.
    Returns:
        A pyecharts polt object.
    """
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
    line = (pe.charts.Line()
        .add_xaxis(t.index.tolist())
        .add_yaxis("Lift", t.lift.round(3).tolist(), is_smooth=True)
        .add_yaxis("Random", [1]*len(t), is_smooth=True)
        .set_series_opts(label_opts=pe.options.LabelOpts(is_show=False))
        .set_global_opts(title_opts=pe.options.TitleOpts(title="Lift Curve"),
                         xaxis_opts=pe.options.AxisOpts(is_scale=True, type_='value')))
    return line.render_notebook() if jupyter else line.render(path)

def gain_curve(y_true, y_pred, pos_label=1, jupyter=True, path='Gain Curve.html'):
    """plot Gain Curve.
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted labels, as returned by a classifier.
        pos_label: positive label.
        jupyter: if plot in jupyter, default True.
        path: if jupyter is False, result save a html file.
    Returns:
        A pyecharts polt object.
    """
    t = pd.DataFrame({'prob':y_pred, 'label':y_true})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    t = t.sort_values(['prob', 'label'], ascending=False).reset_index(drop=True)
    t['tp'] = t.label.cumsum()
    t['fp'] = t.index+1-t.tp
    t['precision'] = t.tp/(t.tp+t.fp)
    t.index = np.round(((t.index+1)/len(t)), 2)
    line = (pe.charts.Line()
        .add_xaxis(t.index.tolist())
        .add_yaxis("Gain", t.precision.round(4).tolist(), is_smooth=True)
        .set_series_opts(label_opts=pe.options.LabelOpts(is_show=False))
        .set_global_opts(title_opts=pe.options.TitleOpts(title="Gain Curve"),
                         xaxis_opts=pe.options.AxisOpts(is_scale=True, type_='value')))
    return line.render_notebook() if jupyter else line.render(path)

def gini_curve(y_true, y_pred, pos_label=1, jupyter=True, path='Gini Curve.html'):
    """plot Gini Curve.
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted labels, as returned by a classifier.
        pos_label: positive label.
        jupyter: if plot in jupyter, default True.
        path: if jupyter is False, result save a html file.
    Returns:
        A pyecharts polt object.
    """
    t = pd.DataFrame({'prob':y_pred, 'label':y_true})
    assert t.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in t.label.unique()}
    t['label'] = t.label.replace(label_dict)
    t = t.sort_values(['prob', 'label']).reset_index(drop=True)
    t['cum'] = t.label.cumsum()/t.label.sum()
    t.index = np.round(((t.index+1)/len(t)), 2)
    line = (pe.charts.Line()
        .add_xaxis([0]+t.index.tolist())
        .add_yaxis("Gini", [0]+t.cum.round(4).tolist(), is_smooth=True)
        .add_yaxis("Random", [0]+t.index.tolist())
        .set_series_opts(label_opts=pe.options.LabelOpts(is_show=False),
                         areastyle_opts=pe.options.AreaStyleOpts(0.4))
        .set_global_opts(title_opts=pe.options.TitleOpts(title="Gini Curve"),
                         xaxis_opts=pe.options.AxisOpts(is_scale=True, type_='value')))
    return line.render_notebook() if jupyter else line.render(path)

def confusion_matrix_map(y_true, y_pred, jupyter=True, path="Confusion Matrix Map.html"):
    """plot Confusion Matrix Map.
    
    Args:
        y_true: pd.Series, ground truth (correct) labels.
        y_pred: pd.Series, predicted labels, as returned by a classifier.
        jupyter: if plot in jupyter, default True.
        path: if jupyter is False, result save a html file.
    Returns:
        A pyecharts polt object.
    """
    t = confusion_matrix(y_true, y_pred)
    t = pd.DataFrame([[i, m,n ] for i,j in t.to_dict().items() for m, n in j.items()],
                     columns=['actual', 'predict', 'over_values'])
    heatmap = (
        pe.charts.HeatMap()
        .add_xaxis(
            xaxis_data=t.drop_duplicates(['actual']).actual.values.tolist())
        .add_yaxis(
            "Confusion Matrix", 
            yaxis_data=t.drop_duplicates(['predict']).predict.values.tolist(), 
            value=t.values.tolist(),
            label_opts=pe.options.LabelOpts(is_show=True, color="#fff", position='inside', horizontal_align="50%")
        )
        .set_global_opts(
            title_opts=pe.options.TitleOpts(title="Confusion Matrix Map"),
            xaxis_opts=pe.options.AxisOpts(
                type_="category", 
                name='Actual', 
                splitarea_opts=pe.options.SplitAreaOpts(
                    is_show=True, areastyle_opts=pe.options.AreaStyleOpts(opacity=1))),
            yaxis_opts=pe.options.AxisOpts(
                type_="category", 
                name='Predict', 
                splitarea_opts=pe.options.SplitAreaOpts(
                    is_show=True, areastyle_opts=pe.options.AreaStyleOpts(opacity=1))),
            visualmap_opts=pe.options.VisualMapOpts(min_=0, max_=int(t.over_values.max()))))
    return  heatmap.render_notebook() if jupyter else heatmap.render(path)
