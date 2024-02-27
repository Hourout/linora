import os

import numpy as np
import pandas as pd
from linora.metrics import ks, psi

__all__ = ['statistical_bins', 'statistical_feature', 'risk_statistics']


def statistical_bins(y_true, y_pred, bins=10, method='quantile', pos_label=1):
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
        if t['cum_lift'].values[0]>t['cum_lift'].values[-1] or not logic:
            break
        else:
            t = data.groupby('bins', observed=True).label.agg(['sum', 'count']).sort_index(ascending=False).reset_index()
            logic = False
    return t


def statistical_feature(data, label_list, score_list):
    """Statistics ks and lift for feature list and label list.
    Args:
        data: DataFrame columns include label_list and score_list
        label_list: y label name list.
        score_list: score name list.
    Return:
        Statistical dataframe
    """
    score_result = []
    for label in label_list:
        for score in score_list:
            df = data.loc[data[label].isin([1,0])&(data[score]>0), [label, score]].dropna().reset_index(drop=True)
            stat_dict = {'y标签':label, '标准分数':score, '坏样本量':df[label].sum(), 
                         '坏样本率':df[label].sum()/len(df), '总样本量':len(df)}
            for i in ['KS', 'KS_10箱', 'KS_20箱', '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift',
                      '累计lift_10箱单调变化次数', '累计lift_10箱是否单调', '累计lift_20箱单调变化次数', '累计lift_20箱是否单调']:
                stat_dict[i] = np.nan
            
            if df[label].nunique()==2:
                stat_dict['KS'] = ks(df[label], df[score])

            # 分10箱
            if len(df)>10 and df[label].nunique()==2:
                try:
                    df_10bin = statistical_bins(df[label], df[score], bins=10)
                    stat_dict['KS_10箱'] = round(df_10bin['ks'].max(),4)
                    stat_dict['尾部10%lift'] = round(df_10bin['lift'].values[0] ,4)
                    stat_dict['头部10%lift'] = round(df_10bin['lift'].values[-1] ,4)
                    cnt = sum((df_10bin['cum_lift'].values[:-1]-df_10bin['cum_lift'].values[1:])<0)
                    stat_dict['累计lift_10箱单调变化次数'] = cnt
                    stat_dict['累计lift_10箱是否单调'] = '是' if cnt==0 else '否'
                except:
                    pass
                
            # 分20箱
            if len(df)>20 and df[label].nunique()==2:
                try:
                    df_20bin = statistical_bins(df[label], df[score], bins=20)
                    stat_dict['KS_20箱'] = round(df_20bin['ks'].max(),4)
                    stat_dict['尾部5%lift'] = round(df_20bin['lift'].values[0] ,4)
                    stat_dict['头部5%lift'] = round(df_20bin['lift'].values[-1] ,4)
                    cnt = sum((df_20bin['cum_lift'].values[:-1]-df_20bin['cum_lift'].values[1:])<0)
                    stat_dict['累计lift_20箱单调变化次数'] = cnt
                    stat_dict['累计lift_20箱是否单调'] = '是' if cnt==0 else '否'
                except:
                    pass
            score_result.append(stat_dict)
    return pd.DataFrame.from_dict(score_result)


def risk_statistics(data, label_list, score_list, tag_name=None, excel='样本统计.xlsx'):
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
    
    with pd.ExcelWriter(excel) as writer:
        pd.DataFrame(['样本情况']).to_excel(writer, sheet_name='样本情况', startrow=1, index=False, header=None)
    with pd.ExcelWriter(excel, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
        
        for rank, c in enumerate(tag_list):
            temp = data[data[tag_name]==c].reset_index(drop=True)
            n = writer.sheets['样本情况'].max_column+2 if rank else 0
            r = temp['apply_month'].nunique()+1
            pd.DataFrame([c]).to_excel(writer, sheet_name='样本情况', index=False, header=None, startrow=4, startcol=n)

            headers = {'selector': 'th.col_heading','props': 'background-color: #00688B; color: white;'}

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
            result = statistical_feature(data[data[tag_name]==c], label_list, score_list)
            result = result[['标准分数', 'y标签', '坏样本量', '坏样本率', '总样本量', 'KS', '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']]
            for i in ['坏样本率', 'KS']:
                result[i] = result[i].map(lambda x:format(x, '.1%'))
            for i in ['尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']:
                result[i] = result[i].round(2)
            (result.loc[[result.loc[(result['标准分数']==i)&(result['y标签']==j)].index[0] for i in score_list for j in label_list]]
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, index=False, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row+1))

        # 计算相关性
        for c in tag_list:
            pd.DataFrame(['相关性']).to_excel(writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row+3, index=False, header=None)
            (data[data[tag_name]==c][score_list].corr().map(lambda x:format(x, '.0%'))
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row+1))
        
        # 计算psi
        for c in tag_list:
            pd.DataFrame(['稳定性-PSI']).to_excel(writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row+3, index=False, header=None)
            pd.DataFrame(["注：\n1. PSI分十箱计算；\n2. 基期为样本按时间顺序排序之后取前50%的样本；"]).to_excel(
                writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row+1, index=False, header=None)
            
            temp = data[data[tag_name]==c].reset_index(drop=True)
            df_list = []
            # df_list.append([None]+score_list)
            df_list.append(['PSI-按基期']+[round(psi(temp[i][:int(len(temp)/2)], temp[i][int(len(temp)/2):]), 4) for i in score_list])
            df_list.append([temp.apply_month.min()]+['/']*len(score_list))
            month_list = temp.apply_month.drop_duplicates().sort_values().tolist()
            for m, n in zip(month_list[:-1], month_list[1:]):
                df_list.append([n]+[round(psi(temp[temp.apply_month==m][i].values, temp[temp.apply_month==n][i].values), 4) for i in score_list])
            (pd.DataFrame(df_list, columns=['PSI']+score_list).set_index('PSI')
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row))

        # 计算逐月指标
        for c in tag_list:
            pd.DataFrame(['性能-稳定性']).to_excel(writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row+3, index=False, header=None)
            temp = data[data[tag_name]==c].reset_index(drop=True)
            df = []
            month_list = temp.apply_month.drop_duplicates().sort_values().tolist()
            for m in month_list:
                result = statistical_feature(temp[temp.apply_month==m].reset_index(drop=True), label_list, score_list)
                for i in ['KS', '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']:
                    for k in score_list:
                        df.append([i, k, m]+[round(result.loc[(result['y标签']==j)&(result['标准分数']==k), i].values[0], 2) for j in label_list])
            df = pd.DataFrame(df, columns=['metrics', 'score', 'month']+label_list).sort_values(['metrics', 'score', 'month'])
            df = pd.concat([df[df.metrics==i] for i in ['KS', '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']])
            
            (df[df.metrics=='KS'].set_index(['metrics', 'score', 'month'])
             .map(lambda x:format(x, '.0%')).replace({'nan%':''}).reset_index()
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row+1, index=False))
            (df[df.metrics!='KS']
             .style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name=f'{c}评估', startrow=writer.sheets[f'{c}评估'].max_row, index=False, header=None))

        for c in tag_list:
            df = pd.DataFrame()
            for bins in [10, 20, 40, 50]:
                for i in score_list:
                    for j in label_list:
                        temp = data[data[tag_name]==c].reset_index(drop=True)
                        temp = statistical_bins(temp[j], temp[i], bins=bins, method='quantile')
                        temp['流量'] = c
                        temp['分箱类型'] = f'{bins}等频'
                        temp['标品名称'] = i
                        temp['y标签'] = j
                        temp = temp.rename(columns={'sample_num':'样本量', 'sample_num_cum':'累积样本量', 'sample_rate_cum':'累积样本率',
                                                    'bad_num':'坏样本量', 'bad_rate':'坏样本率', 'bad_rate_cum':'累积坏样本率'})
                        
                        temp = temp[['流量', '分箱类型', 'y标签', '标品名称', 'bins', '样本量', '累积样本量', '累积样本率', 
                                     '坏样本量', '坏样本率', '累积坏样本率', 'lift', 'cum_lift', 'ks']]
                        df = pd.concat([df, temp])
            for i,j in [('坏样本率',1), ('累积样本率',0), ('累积坏样本率',0), ('ks',0)]:
                df[i] = df[i].map(lambda x:format(x, f'.{j}%')).replace({'nan%':''})
            for i in ['lift', 'cum_lift']:
                df[i] = df[i].round(2)
            (df.style.map_index(lambda _: css_indexes, axis=1)
             .to_excel(writer, sheet_name=f'分箱明细-{c}', startrow=2, index=False))
        
    #     if tag_name is not None:
    #         for i in tag_list:
    #             df_temp1 = pd.DataFrame()
    #             for j in ['10等频', '20等频', '40等频']:
    #                 for m in label_list:
    #                     df_temp = pd.DataFrame()
    #                     df_list = [df[(df['流量']==i)&(df['分箱类型']==j)&(df['y标签']==m)&(df['标品名称']==n)] for n in score_list]
    #                     df_len = max([len(x) for x in df_list])
    #                     for x in df_list:
    #                         df_temp = pd.concat([df_temp, pd.DataFrame([x.columns.tolist()]+x.values.tolist()), 
    #                                              pd.DataFrame(['']*df_len, columns=['']), pd.DataFrame(['']*df_len, columns=[''])], axis=1)
    #                         df_temp.columns = range(0, df_temp.shape[1])
    #                     df_temp1 = pd.concat([df_temp1, pd.DataFrame([['']*df_temp.shape[1], ['']*df_temp.shape[1]]), df_temp])
    #             df_temp1.to_excel(writer, sheet_name=f'{i}明细', index=False)
    #     df.to_excel(writer, sheet_name=f'分箱明细', index=False)
    return os.path.join(os.getcwd(), excel)
