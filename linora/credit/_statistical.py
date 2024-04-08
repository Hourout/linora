import os

import numpy as np
import pandas as pd

from linora.metrics._classification import ks, psi
from linora.feature_selection._credit import iv

__all__ = ['statistical_bins', 'statistical_feature', 'statistical_report_score', 'statistical_report_feature', 'statistical_bins1']



def statistical_bins(y_true, y_pred, bins=10, method='quantile', feature=True, pos_label=1):
    """Statistics ks and lift for feature and label.

    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted score, as returned by a classifier.
        bins: Number of boxes.
        method: 'quantile' is Equal frequency bin, 'uniform' is Equal width bin,
                'monotonicity' is Optimal bin, 'onevaluebins' is Discrete bin.
        feature: True means y_pred is a feature, and False means y_pred is a score.
        pos_label: positive label.
    Returns:
        Statistical dataframe
    """
    data = pd.DataFrame({'bins':y_pred, 'label':y_true}).dropna(subset=['label'], ignore_index=True)
    # if_iv = False if data['bins'].nunique()<=bins else True
    if method=='quantile':
        data['bins'] = pd.qcut(data['bins'], q=bins, duplicates='drop')
    elif method=='uniform':
        if data['bins'].nunique()>bins:
            data['bins'] = pd.cut(data['bins'], bins, duplicates='drop')
    elif method=='monotonicity':
        from optbinning import OptimalBinning
        optb = OptimalBinning(name='bins', max_n_bins=bins, dtype="numerical", solver="cp")
        optb.fit(data['bins'], data['label'])
        optb_t = optb.binning_table.build()[['Bin', 'Event', 'Count']]
        optb_t = pd.concat([optb_t[:-3], optb_t[optb_t['Bin']=='Missing']], ignore_index=True)
    elif method=='onevaluebins':
        pass

    if method=='monotonicity':
        t = optb_t.copy()
        t.loc[t['Bin']=='Missing', 'Bin'] = np.nan
    else:
        assert data.label.nunique()==2, "`y_true` should be binary classification."
        label_dict = {i:1 if i==pos_label else 0 for i in data.label.unique()}
        data['label'] = data.label.replace(label_dict)
        t = data.groupby('bins', dropna=False, observed=True).label.agg(['sum', 'count']).sort_index(ascending=True).reset_index()

    logic = False
    while True:
        t.columns = ['bins', 'bad', 'sample']
        t['bad'] = t['bad'].astype('int')
        t['good'] = t['sample']-t['bad']
        t['bad_rate'] = (t['bad']/t['sample']).fillna(0)
        t['good_rate'] = (t['good']/t['sample']).fillna(0)
        t['sample_rate'] = t['sample']/t['sample'].sum()
        
        t.loc[t['bins'].notna(), 'bad_cum'] = t.loc[t['bins'].notna(), 'bad'].cumsum()
        t.loc[t['bins'].notna(), 'bad_rate_cum'] = t.loc[t['bins'].notna(), 'bad_cum']/t.loc[t['bins'].notna(), 'bad'].sum()
        t.loc[t['bins'].notna(), 'sample_cum'] = t.loc[t['bins'].notna(), 'sample'].cumsum()
        t.loc[t['bins'].notna(), 'sample_rate_cum'] = t.loc[t['bins'].notna(), 'sample_cum']/t.loc[t['bins'].notna(), 'sample'].sum()
        t.loc[t['bins'].notna(), 'good_cum'] = t.loc[t['bins'].notna(), 'good'].cumsum()
        t.loc[t['bins'].notna(), 'good_rate_cum'] = t.loc[t['bins'].notna(), 'good_cum']/t.loc[t['bins'].notna(), 'good'].sum()
        t.loc[t['bins'].notna(), 'KS'] = (t.loc[t['bins'].notna(), 'bad_rate_cum']-t.loc[t['bins'].notna(), 'good_rate_cum']).abs()
        t.loc[t['bins'].notna(), 'Lift'] =  (t.loc[t['bins'].notna(), 'bad_rate']
                                             /t.loc[t['bins'].notna(), 'bad'].sum()
                                             *t.loc[t['bins'].notna(), 'sample'].sum())
        t.loc[t['bins'].notna(), 'cum_lift'] = (t.loc[t['bins'].notna(), 'bad'].cumsum()
                                                /t.loc[t['bins'].notna(), 'sample'].cumsum()
                                                /t.loc[t['bins'].notna(), 'bad'].sum()
                                                *t.loc[t['bins'].notna(), 'sample'].sum())
        t.loc[t['bins'].notna(), 'cum_lift_reversed'] = (t.loc[t['bins'].notna(), 'bad'][::-1].cumsum()
                                                         /t.loc[t['bins'].notna(), 'sample'][::-1].cumsum()
                                                         /t.loc[t['bins'].notna(), 'bad'].sum()
                                                         *t.loc[t['bins'].notna(), 'sample'].sum())
        t['pos_rate'] = t['bad'].replace({0:1})/t['bad'].sum()
        t['neg_rate'] = t['good'].replace({0:1})/t['good'].sum()
        t['WoE'] = np.log(t['pos_rate']/t['neg_rate'])
        t['IV'] = (t['pos_rate'] - t['neg_rate']) * t['WoE']
        t.loc[t['bins'].isna(), 'WoE'] = np.nan
        t.loc[t['bins'].isna(), 'IV'] = np.nan

        if feature or logic:
            break
        elif method=='monotonicity':
            break
        elif t.loc[t['bins'].notna(), 'cum_lift'].values[0]>t.loc[t['bins'].notna(), 'cum_lift'].values[-1]:
            break
        else:
            t = data.groupby('bins', dropna=False, observed=True).label.agg(['sum', 'count']).sort_index(ascending=False).reset_index()
            logic = True

    t['bins'] = t['bins'].astype(str).replace('nan', 'Missing')
    if 'Missing' not in t['bins'].tolist():
        miss_list = ['bins', 'bad', 'good', 'sample', 'bad_rate', 'good_rate', 'sample_rate']
        t = pd.concat([t, pd.DataFrame({i:['Missing'] if i=='bins' else [0] for i in miss_list})], ignore_index=True)
    
    t.insert(0, '序号', t.reset_index().index)
    t = pd.concat([t.drop(['pos_rate', 'neg_rate'], axis=1), 
                   pd.DataFrame({
        '序号':['Totals'], 'bad':[t['bad'].sum()], 'sample':[t['sample'].sum()], 
        'bad_rate':[t['bad'].sum()/t['sample'].sum()], 
        'sample_rate':[1.], 
        'good':[t['good'].sum()], 'good_rate':[t['good'].sum()/t['sample'].sum()],
        'KS':[t['KS'].max()], 'IV':[t['IV'].sum()]})], ignore_index=True)

    return t[['序号', 'bins', 'bad', 'good', 'sample', 'bad_rate', 'good_rate', 'sample_rate',
     'bad_cum', 'good_cum', 'sample_cum', 'bad_rate_cum', 'good_rate_cum', 'sample_rate_cum',
     'WoE', 'IV', 'KS', 'Lift', 'cum_lift', 'cum_lift_reversed']]



def statistical_bins1(y_true, y_pred, bins=10, method='quantile', pos_label=1):
    """Statistics ks and lift for feature and label.
    Args:
        y_true: pd.Series or array or list, ground truth (correct) labels.
        y_pred: pd.Series or array or list, predicted labels, as returned by a classifier.
        bins: Number of boxes.
        method: 'quantile' is Equal frequency bin, 'uniform' is Equal width bin.
        pos_label: positive label.
    Returns:
        Statistical dataframe
    """
    data = pd.DataFrame({'bins':y_pred, 'label':y_true})
    if method=='quantile':
        data['bins'] = pd.qcut(data['bins'], q=bins, duplicates='drop')
    else:
        data['bins'] = pd.cut(data['bins'], bins, duplicates='drop')
    data['bins'] = data['bins'].astype(str).replace('nan', 'Missing')
    assert data.label.nunique()==2, "`y_true` should be binary classification."
    label_dict = {i:1 if i==pos_label else 0 for i in data.label.unique()}
    data['label'] = data.label.replace(label_dict)
    logic = True
    t = data.groupby('bins', observed=True).label.agg(['sum', 'count']).sort_index(ascending=True).reset_index()
    while True:
        t.columns = ['bins', 'bad_num', 'sample_num']
        t['bad_rate'] = t['bad_num']/t['sample_num']
        t['bad_num_cum'] = t['bad_num'].cumsum()
        t['bad_rate_cum'] = t['bad_num_cum']/t['bad_num'].sum()
        t['sample_rate'] = t['sample_num']/t['sample_num'].sum()
        t['sample_num_cum'] = t['sample_num'].cumsum()
        t['sample_rate_cum'] = t['sample_num_cum']/t['sample_num'].sum()
        t['good_num'] = t['sample_num']-t['bad_num']
        t['good_rate'] = t['good_num']/t['sample_num']
        t['good_num_cum'] = t['good_num'].cumsum()
        t['good_rate_cum'] = t['good_num_cum']/t['good_num'].sum()
        t['ks'] = (t['bad_rate_cum']-t['good_rate_cum']).abs()
        t['lift'] = t['bad_num']/t['sample_num']/t['bad_num'].sum()*t['sample_num'].sum()
        t['cum_lift'] = t['bad_num'].cumsum()/t['sample_num'].cumsum()/t['bad_num'].sum()*t['sample_num'].sum()
        
        t['pos_rate'] = t['bad_num'].replace({0:1})/t['bad_num'].sum()
        t['neg_rate'] = t['good_num'].replace({0:1})/t['good_num'].sum()
        t['WoE'] = np.log(t['pos_rate']/t['neg_rate'])
        t['IV'] = (t['pos_rate'] - t['neg_rate']) * t['WoE']
        if t['cum_lift'].values[0]>t['cum_lift'].values[-1] or not logic:
            break
        else:
            t = data.groupby('bins', observed=True).label.agg(['sum', 'count']).sort_index(ascending=False).reset_index()
            logic = False
    if 'Missing' not in t['bins'].tolist():
        t = pd.concat([t, pd.DataFrame({i:['Missing'] if i=='bins' else [0] for i in t.columns})]).reset_index(drop=True)
    return t.drop(['pos_rate', 'neg_rate'], axis=1)


def statistical_feature(data, label_list, score_list, method='quantile', feature=True):
    """Statistics ks and lift for feature list and label list.
    Args:
        data: DataFrame columns include label_list and score_list
        label_list: y label name list.
        score_list: score name list.
        method: 'quantile' is Equal frequency bin, 'uniform' is Equal width bin,
                'monotonicity' is Optimal bin, 'onevaluebins' is Discrete bin.
        feature: True means score_list is a feature list, and False means score_list is a score list.
    Return:
        Statistical dataframe
    """
    score_result = []
    for label in label_list:
        for score in score_list:
            df = data.loc[data[label].isin([1,0]), [label, score]].dropna(subset=[label], ignore_index=True)
            stat_dict = {'y标签':label, '特征分数':score, '坏样本量':df[label].sum(), 
                         '坏样本率':df[label].sum()/max(len(df),1), '总样本量':len(df)}
            for i in ['IV', 'KS', 'KS_10箱', 'KS_20箱', '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift',
                      '累计lift_10箱单调变化次数', '累计lift_10箱是否单调', '累计lift_20箱单调变化次数', '累计lift_20箱是否单调',
                      '单值单箱头部累计最小lift', '单值单箱尾部累计最大lift', '单值单箱最小lift', '单值单箱最大lift']:
                stat_dict[i] = np.nan
            
            if df[label].nunique()==2:
                try:
                    stat_dict['KS'] = ks(df[label], df[score])
                except:
                    pass

                try:
                    stat_dict['IV'] = iv(df[score], df[label], max_bins=5 if feature else 10)
                except:
                    pass

                if df[score].nunique()/max(df[score].count(),1)<0.5:
                    try:
                        df_bin = statistical_bins(df[label], df[score], bins=10, method='onevaluebins', feature=feature)
                        stat_dict['单值单箱最小lift'] = df_bin['Lift'].min()
                        stat_dict['单值单箱最大lift'] = df_bin['Lift'].max()
                        stat_dict['单值单箱头部累计最小lift'] = df_bin['cum_lift_reversed'].min()
                        stat_dict['单值单箱尾部累计最大lift'] = df_bin['cum_lift'].max()
                    except:
                        pass
                    
            # 分10箱
            if len(df)>10 and df[label].nunique()==2:
                try:
                    df_10bin = statistical_bins(df[label], df[score], bins=10, method=method, feature=feature)
                    df_10bin = df_10bin[df_10bin.bins!='Missing']
                    df_10bin = df_10bin[df_10bin['序号']!='Totals']
                    stat_dict['KS_10箱'] = round(df_10bin['KS'].max(),4)
                    stat_dict['尾部10%lift'] = round(df_10bin['Lift'].values[0] ,4)
                    stat_dict['头部10%lift'] = round(df_10bin['Lift'].values[-1] ,4)
                    cnt = sum((df_10bin['cum_lift'].values[:-1]-df_10bin['cum_lift'].values[1:])<0)
                    stat_dict['累计lift_10箱单调变化次数'] = cnt
                    stat_dict['累计lift_10箱是否单调'] = '是' if cnt==0 else '否'
                except:
                    pass
                
            # 分20箱
            if len(df)>20 and df[label].nunique()==2:
                try:
                    df_20bin = statistical_bins(df[label], df[score], bins=20, method=method, feature=feature)
                    df_20bin = df_20bin[df_20bin.bins!='Missing']
                    df_20bin = df_20bin[df_20bin['序号']!='Totals']
                    stat_dict['KS_20箱'] = round(df_20bin['KS'].max(),4)
                    stat_dict['尾部5%lift'] = round(df_20bin['Lift'].values[0] ,4)
                    stat_dict['头部5%lift'] = round(df_20bin['Lift'].values[-1] ,4)
                    cnt = sum((df_20bin['cum_lift'].values[:-1]-df_20bin['cum_lift'].values[1:])<0)
                    stat_dict['累计lift_20箱单调变化次数'] = cnt
                    stat_dict['累计lift_20箱是否单调'] = '是' if cnt==0 else '否'
                except:
                    pass
            score_result.append(stat_dict)
    return pd.DataFrame.from_dict(score_result)


def statistical_report_score(data, label_list, score_list, tag_name=None, excel='标品分数统计报告.xlsx'):
    """score and label statistics and result output to excel.
    Args:
        data: DataFrame columns include label_list and score_list
        label_list: y label name list.
        score_list: score name list.
        tag_name: sample distinguishing columns, such as pre-lending and in-lending.
        excel: excel path.
    Return:
        excel path.
    """
    if tag_name is None:
        tag_list = ['总体']
    elif tag_name in data.columns:
        tag_list = data[tag_name].unique().tolist()
    else:
        raise ValueError('tag_name should be in `data.columns`.')
    tag_lists = tag_list.copy() if '总体' in tag_list else ['总体']+tag_list

    css_indexes = 'background-color: steelblue; color: white;'
    headers = {'selector': 'th.col_heading','props': 'background-color: #00688B; color: white;'}
    
    with pd.ExcelWriter(excel) as writer:
        pd.DataFrame(['样本情况']).to_excel(writer, sheet_name='样本情况', startrow=1, index=False, header=None)
    with pd.ExcelWriter(excel, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
        
        for rank, c in enumerate(tag_list):
            temp = data[data[tag_name]==c].reset_index(drop=True)
            n = writer.sheets['样本情况'].max_column+2 if rank else 0
            r = temp['apply_month'].nunique()+1
            pd.DataFrame([c]).to_excel(writer, sheet_name='样本情况', index=False, header=None, startrow=4, startcol=n)

            (temp.groupby(['apply_month'])[label_list].agg('mean').map(lambda x:format(x, '.2%')).replace({'nan%':''})
             .reset_index().rename(columns={'apply_month':'坏样本率'})
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name='样本情况', index=False, startrow=4+1, startcol=n, engine='openpyxl'))
            (temp.groupby(['apply_month'], as_index=False)[label_list].agg('count').rename(columns={'apply_month':'样本量'})
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name='样本情况', index=False, startrow=4+r+2, startcol=n))
            (temp.groupby(['apply_month'], as_index=False)[label_list].agg('sum').rename(columns={'apply_month':'坏样本量'})
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name='样本情况', index=False, startrow=4+r*2+2, startcol=n))
        
        
        # 计算整体指标
        for c in tag_list:
            pd.DataFrame(['性能']).to_excel(writer, sheet_name=f'{c}评估', startrow=1, index=False, header=None)
            result = statistical_feature(data[data[tag_name]==c], label_list, score_list, method='quantile', feature=False)
            result = result[['特征分数', 'y标签', '坏样本量', '坏样本率', '总样本量', 'KS', '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']]
            
            result = result[result['特征分数'].isin(result.groupby('特征分数')[['尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']].count().max(axis=1).where(lambda x:x>0).dropna().index.tolist())]
            score_list1 = result.groupby(['特征分数'])[['尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']].max().max(axis=1).sort_values(ascending=False).index.tolist()
            for i in ['坏样本率', 'KS']:
                result[i] = result[i].map(lambda x:format(x, '.1%')).replace({'nan%':''})
            for i in ['尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']:
                result[i] = result[i].round(2)
            (result.loc[[result[(result['特征分数']==i)&(result['y标签']==j)].index[0] for i in score_list1 for j in label_list]]
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, index=False, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row+1))
            

        for c in tag_list:
            #计算psi
            pd.DataFrame(['稳定性-PSI']).to_excel(writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row+3, index=False, header=None)
            pd.DataFrame(["注：\n1. PSI分十箱计算；\n2. 基期为样本按时间顺序排序之后取前50%的样本；"]).to_excel(
                writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row+1, index=False, header=None)
            
            temp = data[data[tag_name]==c].reset_index(drop=True)
            df_list = []
            psi_list = ['PSI-按基期']
            for i in score_list:
                try:
                    psi_list.append(round(psi(temp[i][:int(len(temp)/2)], temp[i][int(len(temp)/2):]), 4))
                except:
                    psi_list.append(None)
            df_list.append(psi_list)
            df_list.append([temp.apply_month.min()]+['/']*len(score_list))
            month_list = temp.apply_month.drop_duplicates().sort_values().tolist()
            for m, n in zip(month_list[:-1], month_list[1:]):
                psi_list = [n]
                for i in score_list:
                    try:
                        psi_list.append(round(psi(temp[temp.apply_month==m][i].values, temp[temp.apply_month==n][i].values), 4))
                    except:
                        psi_list.append(None)
                df_list.append(psi_list)
            (pd.DataFrame(df_list, columns=['PSI']+score_list).set_index('PSI').T
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row))

        # 计算相关性
        for c in tag_list:
            pd.DataFrame(['相关性']).to_excel(writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row+3, index=False, header=None)
            (data[data[tag_name]==c][score_list].corr().map(lambda x:format(x, '.0%'))
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row+1))

        
        # 计算psi
        # for c in tag_list:
        #     pd.DataFrame(['稳定性-PSI']).to_excel(writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row+3, index=False, header=None)
        #     pd.DataFrame(["注：\n1. PSI分十箱计算；\n2. 基期为样本按时间顺序排序之后取前50%的样本；"]).to_excel(
        #         writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row+1, index=False, header=None)
            
        #     temp = data[data[tag_name]==c].reset_index(drop=True)
        #     df_list = []
        #     # df_list.append([None]+score_list)
        #     df_list.append(['PSI-按基期']+[round(psi(temp[i][:int(len(temp)/2)], temp[i][int(len(temp)/2):]), 4) for i in score_list])
        #     df_list.append([temp.apply_month.min()]+['/']*len(score_list))
        #     month_list = temp.apply_month.drop_duplicates().sort_values().tolist()
        #     for m, n in zip(month_list[:-1], month_list[1:]):
        #         df_list.append([n]+[round(psi(temp[temp.apply_month==m][i].values, temp[temp.apply_month==n][i].values), 4) for i in score_list])
        #     (pd.DataFrame(df_list, columns=['PSI']+score_list).set_index('PSI')
        #      .style.map_index(lambda _: css_indexes, axis=1)
        #      .to_excel(writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row))

        # 计算逐月指标
        for c in tag_list:
            pd.DataFrame(['性能-稳定性']).to_excel(writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row+3, index=False, header=None)
            temp = data[data[tag_name]==c].reset_index(drop=True)
            df = []
            month_list = temp.apply_month.drop_duplicates().sort_values().tolist()
            for m in month_list:
                result = statistical_feature(temp[temp.apply_month==m].reset_index(drop=True), label_list, score_list, method='quantile', feature=False)
                for i in ['KS', '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']:
                    for k in score_list:
                        df.append([i, k, m]+[round(result.loc[(result['y标签']==j)&(result['特征分数']==k), i].values[0], 2) for j in label_list])
            df = pd.DataFrame(df, columns=['metrics', 'score', 'month']+label_list).sort_values(['metrics', 'score', 'month'])
            df = pd.concat([df[df.metrics==i] for i in ['KS', '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']])

            df_list = []
            for i in ['KS', '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']:
                for j in df[df['metrics']==i].groupby(['score'])[label_list].count().max(axis=1).where(lambda x:x>0).dropna().index.tolist():
                    df_list.append(df.loc[(df['metrics']==i)&(df['score']==j)])
            df = pd.concat(df_list)
            
            df_list = []
            for i in ['KS', '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']:
                for j in df[df['metrics']==i].groupby(['score'])[label_list].max().max(axis=1).sort_values(ascending=False).index.tolist():
                    df_list.append(df.loc[(df['metrics']==i)&(df['score']==j)])
            df = pd.concat(df_list)
            
            (df[df.metrics=='KS'].set_index(['metrics', 'score', 'month'])
             .map(lambda x:format(x, '.0%')).replace({'nan%':''}).reset_index()
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row+1, index=False))
            (df[df.metrics!='KS']
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row, index=False, header=None))

        df_list = []
        for c in tag_list:
            for bins in [10, 20, 40, 50]:
                for i in score_list:
                    for j in label_list:
                        temp = data[data[tag_name]==c].reset_index(drop=True)
                        temp = statistical_bins(temp[j], temp[i], bins=bins, method='quantile', feature=False)
                        temp['流量'] = c
                        temp['分箱类型'] = f'{bins}等频'
                        temp['标品名称'] = i
                        temp['y标签'] = j
                        temp = temp[['流量', '分箱类型', 'y标签', '标品名称', '序号', 'bins', 'bad', 'good', 'sample', 
                                     'bad_rate', 'sample_rate', 'KS', 'Lift', 'cum_lift']]
                        df_list.append(temp)
        df = pd.concat(df_list)
        for i,j in [('bad_rate',1), ('sample_rate',1), ('KS',0)]:
            df[i] = df[i].map(lambda x:format(x, f'.{j}%')).replace({'nan%':''})
        for i in ['Lift', 'cum_lift']:
            df[i] = df[i].round(2)
        (df.rename(columns={'cum_lift':'尾部累积Lift'})
         .style.map_index(lambda _: css_indexes, axis=1)
         .to_excel(writer, sheet_name=f'分箱明细', startrow=2, index=False))
    return os.path.join(os.getcwd(), excel)


def statistical_report_feature(data, label_list, score_list, tag_name=None, excel='样本特征统计报告.xlsx'):
    """score and label statistics and result output to excel.
    Args:
        data: DataFrame columns include label_list and score_list
        label_list: y label name list.
        score_list: score name list.
        tag_name: sample distinguishing columns, such as pre-lending and in-lending.
        excel: excel path.
    Return:
        excel path.
    """
    if tag_name is None:
        tag_list = ['总体']
    elif tag_name in data.columns:
        tag_list = data[tag_name].unique().tolist()
    else:
        raise ValueError('tag_name should be in `data.columns`.')
    tag_lists = tag_list.copy() if '总体' in tag_list else ['总体']+tag_list

    css_indexes = 'background-color: steelblue; color: white;'
    headers = {'selector': 'th.col_heading','props': 'background-color: #00688B; color: white;'}
    
    with pd.ExcelWriter(excel) as writer:
        pd.DataFrame(['样本情况']).to_excel(writer, sheet_name='样本情况', startrow=1, index=False, header=None)
    with pd.ExcelWriter(excel, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
        #计算样本情况
        for rank, c in enumerate(tag_list):
            temp = data[data[tag_name]==c].reset_index(drop=True)
            n = writer.sheets['样本情况'].max_column+2 if rank else 0
            r = temp['apply_month'].nunique()+1
            pd.DataFrame([c]).to_excel(writer, sheet_name='样本情况', index=False, header=None, startrow=4, startcol=n)

            (temp.groupby(['apply_month'])[label_list].agg('mean').map(lambda x:format(x, '.2%')).replace({'nan%':''})
             .reset_index().rename(columns={'apply_month':'坏样本率'})
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name='样本情况', index=False, startrow=4+1, startcol=n, engine='openpyxl'))
            (temp.groupby(['apply_month'], as_index=False)[label_list].agg('count').rename(columns={'apply_month':'样本量'})
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name='样本情况', index=False, startrow=4+r+2+1, startcol=n))
            (temp.groupby(['apply_month'], as_index=False)[label_list].agg('sum').rename(columns={'apply_month':'坏样本量'})
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name='样本情况', index=False, startrow=4+r*2+2*2+1, startcol=n))
        
        # 计算整体指标
        for rank, c in enumerate(tag_list):
            ncol = writer.sheets['性能评估'].max_column+3 if rank else 0
            pd.DataFrame([f'{c}']).to_excel(writer, sheet_name='性能评估', startrow=1, startcol=ncol, index=False, header=None)
            pd.DataFrame(['性能']).to_excel(writer, sheet_name='性能评估', startrow=2, startcol=ncol, index=False, header=None)
            result = statistical_feature(data[data[tag_name]==c], label_list, score_list, method='quantile', feature=True)
            result = result[['特征分数', 'y标签', '坏样本量', '坏样本率', '总样本量', 'IV', 'KS', 
                             '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift', 
                             '单值单箱头部累计最小lift', '单值单箱尾部累计最大lift', 
                             '单值单箱最小lift', '单值单箱最大lift']]
            result = result[result['特征分数'].isin(result.groupby('特征分数')[['尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']].count().max(axis=1).where(lambda x:x>0).dropna().index.tolist())]
            score_list1 = result.groupby(['特征分数'])[['尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']].max().max(axis=1).sort_values(ascending=False).index.tolist()
            for i in ['坏样本率', 'KS']:
                result[i] = result[i].map(lambda x:format(x, '.2%')).replace({'nan%':''})
            for i in ['IV', '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift', 
                      '单值单箱头部累计最小lift', '单值单箱尾部累计最大lift', '单值单箱最小lift', '单值单箱最大lift']:
                result[i] = result[i].round(2)
            (result.loc[[result[(result['特征分数']==i)&(result['y标签']==j)].index[0] for i in score_list1 for j in label_list]]
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, index=False, sheet_name='性能评估', startrow=4, startcol=ncol))

        #性能稳定性评估
        for rank, c in enumerate(tag_list):
            # 计算psi
            ncol = writer.sheets['稳定性评估'].max_column+3 if rank else 0
            pd.DataFrame([f'{c}']).to_excel(writer, sheet_name='稳定性评估', startrow=1, startcol=ncol, index=False, header=None)
            pd.DataFrame(['稳定性-PSI']).to_excel(writer, sheet_name='稳定性评估', startrow=2, startcol=ncol, index=False, header=None)
            pd.DataFrame(["注：\n1. PSI分十箱计算；\n2. 基期为样本按时间顺序排序之后取前50%的样本；"]).to_excel(
                writer, sheet_name='稳定性评估', startrow=4, startcol=ncol, index=False, header=None)
            
            temp = data[data[tag_name]==c].reset_index(drop=True)
            df_list = []
            psi_list = ['PSI-按基期']
            for i in score_list:
                try:
                    psi_list.append(round(psi(temp[i][:int(len(temp)/2)], temp[i][int(len(temp)/2):]), 4))
                except:
                    psi_list.append(None)
            df_list.append(psi_list)
            df_list.append([temp.apply_month.min()]+['/']*len(score_list))
            month_list = temp.apply_month.drop_duplicates().sort_values().tolist()
            for m, n in zip(month_list[:-1], month_list[1:]):
                psi_list = [n]
                for i in score_list:
                    try:
                        psi_list.append(round(psi(temp[temp.apply_month==m][i].values, temp[temp.apply_month==n][i].values), 4))
                    except:
                        psi_list.append(None)
                df_list.append(psi_list)
            (pd.DataFrame(df_list, columns=['PSI']+score_list).T.reset_index().rename(columns={'index':'PSI'})
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name='稳定性评估', startrow=6, startcol=ncol, index=False, header=None))

            # 计算逐月指标
            if rank==0:
                r = writer.sheets['稳定性评估'].max_row+3
                pd.DataFrame(['稳定性-性能(KS/Lift)']).to_excel(writer, sheet_name='稳定性评估', startrow=r, startcol=ncol, index=False, header=None)
            temp = data[data[tag_name]==c].reset_index(drop=True)
            df = []
            month_list = temp.apply_month.drop_duplicates().sort_values().tolist()
            for m in month_list:
                result = statistical_feature(temp[temp.apply_month==m].reset_index(drop=True), label_list, score_list, method='quantile', feature=True)
                for i in ['KS', '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']:
                    for k in score_list:
                        df.append([k, i, m]+[round(result.loc[(result['y标签']==j)&(result['特征分数']==k), i].values[0], 2) for j in label_list])
            df = pd.DataFrame(df, columns=['score', 'metrics', 'month']+label_list).sort_values(['metrics', 'score', 'month'])
            df = pd.concat([df[df.metrics==i] for i in ['KS', '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']])

            df_list = []
            for i in ['KS', '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']:
                for j in df[df['metrics']==i].groupby(['score'])[label_list].count().max(axis=1).where(lambda x:x>0).dropna().index.tolist():
                    df_list.append(df.loc[(df['metrics']==i)&(df['score']==j)])
            df = pd.concat(df_list)
            
            df_list = []
            for i in ['KS', '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']:
                for j in df[df['metrics']==i].groupby(['score'])[label_list].max().max(axis=1).sort_values(ascending=False).index.tolist():
                    df_list.append(df.loc[(df['metrics']==i)&(df['score']==j)])
            df = pd.concat(df_list)

            
            (df[df.metrics=='KS'].set_index(['score', 'metrics', 'month'])
             .map(lambda x:format(x, '.0%')).replace({'nan%':''}).reset_index()
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name='稳定性评估', startrow=r+3+2, startcol=ncol, index=False))
            (df[df.metrics!='KS']
             .to_excel(writer, sheet_name='稳定性评估', startrow=r+3+2+len(df[df.metrics=='KS'])+1, startcol=ncol, index=False, header=None))

        # 计算相关性
        for rank, c in enumerate(tag_list):
            r = writer.sheets['相关性明细'].max_row if rank else 0
            pd.DataFrame([f'{c}']).to_excel(writer, sheet_name='相关性明细', startrow=r, index=False, header=None)
            pd.DataFrame(['相关性']).to_excel(writer, sheet_name='相关性明细', startrow=writer.sheets['相关性明细'].max_row, index=False, header=None)
            (data[data[tag_name]==c][score_list].fillna(-999).corr().map(lambda x:format(x, '.0%')).replace({'nan%':''})
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name='相关性明细', startrow=writer.sheets['相关性明细'].max_row+1))
            
        # 分箱明细
        df_list = []
        for c in tag_list:
            for j in label_list:
                for k in score_list:
                    for b in [10, 20]:
                        for m1,m2 in zip(['等距'], ['uniform']):
                            temp = statistical_bins(data[data.user_type==c][j], data[data.user_type==c][k], bins=b, method=m2, feature=True)
                            temp.insert(0, 'feature', k)
                            temp.insert(0, 'y_label', j)
                            temp.insert(0, '分箱类型', f'{b}{m1}')
                            temp.insert(0, 'user_type', c)
                            df_list.append(temp)
                    # try:
                    #     temp = statistical_bins(data[data.user_type==c][j], data[data.user_type==c][k], bins=5, method='monotonicity', sort_bins=True)
                    #     temp.insert(0, 'feature', k)
                    #     temp.insert(0, 'y_label', j)
                    #     temp.insert(0, '分箱类型', '最优分箱')
                    #     temp.insert(0, 'user_type', c)
                    #     df_list.append(temp)
                    # except:
                    #     print(c,j,k)
                    #     pass
        df = pd.concat(df_list)
        (df[['user_type', '分箱类型', 'y_label', 'feature', '序号', 'bins', 'bad', 'good', 'sample', 'bad_rate', 'sample_rate', 'WoE', 'IV', 'Lift', 'cum_lift', 'cum_lift_reversed']]
         .rename(columns={'cum_lift':'尾部累计Lift', 'cum_lift_reversed':'头部累计Lift'})
         .style.map_index(lambda _: css_indexes, axis=1)
         .to_excel(writer, sheet_name=f'分箱明细', startrow=2, index=False))

    return os.path.join(os.getcwd(), excel)

