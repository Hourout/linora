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
        t['sample_num_cum'] = t['sample_num'].cumsum()
        t['sample_rate_cum'] = t['sample_num_cum']/t['sample_num'].sum()
        t['bad_rate_cum'] = t['bad_num_cum']/t['bad_num'].sum()
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
    score_result = pd.DataFrame()
    for label in label_list:
        for score in score_list:
            df = data.loc[data[label].isin([1,0]), [label, score]].dropna().reset_index(drop=True)
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
            score_result = score_result.append(stat_dict, ignore_index = True)
    return score_result

def risk_statistics(data, label_list, score_list, tag_name=None, excel='样本统计.xlsx'):
    """score and label statistics and result output to excel.
    Args:
        data: DataFrame columns include label_list and score_list
        label_list: y label name list.
        score_list: score name list.
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
      
    with pd.ExcelWriter(excel) as writer:
        for c in tag_lists:
            if c=='总体':
                temp = data.groupby(['user_guid', 'apply_date'], as_index=False)[label_list].max()
                temp['apply_month'] = temp.apply_date.map(lambda x:x[:7])
            else:
                temp = data[data[tag_name]==c].reset_index(drop=True)
            temp.groupby(['apply_month'])[label_list].count().to_excel(writer, sheet_name=f'{c}样本量')
            temp.groupby(['apply_month'])[label_list].sum().to_excel(writer, sheet_name=f'{c}坏样本量')
            temp.groupby(['apply_month'])[label_list].mean().to_excel(writer, sheet_name=f'{c}坏样本率')
        for c in tag_lists:
            if c=='总体':
                data[score_list].corr().to_excel(writer, sheet_name='总体相关系数')
            else:
                data[data[tag_name]==c][score_list].corr().to_excel(writer, sheet_name=f'{c}相关系数')
        #计算psi
        for c in tag_lists:
            if c=='总体':
                temp = data.copy()
            else:
                temp = data[data[tag_name]==c].reset_index(drop=True)
            df_list = []
            df_list.append([None]+score_list)
            df_list.append(['PSI-按基期']+[psi(temp[i][:int(len(temp)/2)], temp[i][int(len(temp)/2):]) for i in score_list])
            df_list.append([temp.apply_month.min()]+['/']*len(score_list))
            month_list = temp.apply_month.drop_duplicates().sort_values().tolist()
            for m, n in zip(month_list[:-1], month_list[1:]):
                df_list.append([n]+[psi(temp[temp.apply_month==m][i].values, temp[temp.apply_month==n][i].values) for i in score_list])
            pd.DataFrame(df_list).to_excel(writer, sheet_name=f'{c}psi')
        #计算逐月指标
        for c in tag_lists:
            if c=='总体':
                temp = (data.groupby(['user_guid', 'apply_date'], as_index=False)[label_list].max()
                        .merge(data[['user_guid', 'apply_date']+score_list].drop_duplicates(), how='left', on=['user_guid', 'apply_date']))
                temp['apply_month'] = temp.apply_date.map(lambda x:x[:7])
            else:
                temp = data[data[tag_name]==c].reset_index(drop=True)
            df = []
            month_list = temp.apply_month.drop_duplicates().sort_values().tolist()
            for m in month_list:
                result = statistical_feature(temp[temp.apply_month==m].reset_index(drop=True), label_list, score_list)
                for i in ['KS', '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']:
                    for k in score_list:
                        df.append([i, k, m]+[result.loc[(result['y标签']==j)&(result['标准分数']==k), i].values[0] for j in label_list])
            (pd.DataFrame(df, columns=['metrics', 'score', 'month']+label_list)
             .sort_values(['metrics', 'score', 'month']).set_index(['metrics', 'score', 'month'])
             .to_excel(writer, sheet_name=f'{c}逐月指标'))

        for c in tag_lists:
            if c=='总体':
                result = statistical_feature(data, label_list, score_list)
            else:
                result = statistical_feature(data[data[tag_name]==c], label_list, score_list)
            result = result[['标准分数', 'y标签', '坏样本量', '坏样本率', '总样本量','KS', '尾部5%lift', '尾部10%lift', '头部5%lift', '头部10%lift']]
            (result.loc[[result.loc[(result['标准分数']==i)&(result['y标签']==j)].index[0] for i in score_list for j in label_list]]
             .to_excel(writer, sheet_name=f'{c}指标'))
    
        df = pd.DataFrame()
        for bins in [10, 20, 40]:
            for c in tag_lists:
                for i in score_list:
                    for j in label_list:
                        if c=='总体':
                            temp = data.copy()
                        else:
                            temp = data[data[tag_name]==c].reset_index(drop=True)
                        temp = statistical_bins(temp[j], temp[i], bins=10, method='quantile')
                        temp['流量'] = c
                        temp['分箱类型'] = f'{bins}等频'
                        temp['标品名称'] = i
                        temp['y标签'] = j
                        temp = temp.rename(columns={'sample_num':'样本量', 'sample_num_cum':'累积样本量', 'sample_rate_cum':'累积样本率',
                                                    'bad_num':'坏样本量', 'bad_rate':'坏样本率', 'bad_rate_cum':'累积坏样本率'})
                        
                        temp = temp[['流量', '分箱类型', 'y标签', '标品名称', 'bins', '样本量', '累积样本量', '累积样本率', 
                                     '坏样本量', '坏样本率', '累积坏样本率', 'lift', 'cum_lift', 'ks']]
                        df = pd.concat([df, temp])
        
        if tag_name is not None:
            for i in tag_list:
                df_temp1 = pd.DataFrame()
                for j in ['10等频', '20等频', '40等频']:
                    for m in label_list:
                        df_temp = pd.DataFrame()
                        df_list = [df[(df['流量']==i)&(df['分箱类型']==j)&(df['y标签']==m)&(df['标品名称']==n)] for n in score_list]
                        df_len = max([len(x) for x in df_list])
                        for x in df_list:
                            df_temp = pd.concat([df_temp, pd.DataFrame([x.columns.tolist()]+x.values.tolist()), 
                                                 pd.DataFrame(['']*df_len, columns=['']), pd.DataFrame(['']*df_len, columns=[''])], axis=1)
                            df_temp.columns = range(0, df_temp.shape[1])
                        df_temp1 = pd.concat([df_temp1, pd.DataFrame([['']*df_temp.shape[1], ['']*df_temp.shape[1]]), df_temp])
                df_temp1.to_excel(writer, sheet_name=f'{i}明细', index=False)
        df.to_excel(writer, sheet_name=f'分箱明细', index=False)
    return os.path.join(os.getcwd(), excel)