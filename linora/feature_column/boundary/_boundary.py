import pandas as pd

__all__ = ['uniform', 'quantile', 'probability_categorical']

def uniform(feature, bins):
    """Equal width bin, take a uniform distribution for the sample value range.
    
    Args:
        feature: pd.Series, model feature values.
        bins: int, split bins of feature.
    Returns:
        the list of split threshold of feature.
    """
    t = (feature.max()-feature.min())/bins
    m = feature.min()
    return [t*i+m for i in range(bins)]+[feature.max()]

def quantile(feature, bins):
    """Equal frequency bin, take a uniform distribution of the sample size.
    
    Args:
        feature: pd.Series, model feature values.
        bins: int, split bins of feature.
    Returns:
        the list of split threshold of feature.
    """
    t = feature.sort_values().values
    w = round(len(t)/bins)
    return [t[w*i] for i in range(bins)]+[feature.max()]

def probability_categorical(feature, label):
    """Probability grouping of category variables
    
    Args:
        feature: pd.Series, model feature values.
        label: pd.Series, Target value in supervised learning.
    Returns:
        a dict.{'group': grouping dict, 'data': grouping data, 'distance': grouping distance, 'std': grouping std}
    """
    assert feature.nunique()>2, 'feature category nums must be greater than 2.'
    t = pd.DataFrame({'feature':feature, 'label':label})
    cat = label.unique()
    cat = [(cat[i], cat[i+1]) for i in range(len(cat)-1)]
    prob = label.value_counts(1).to_dict()
    slope = [prob.get(i[0], 0)-prob.get(i[1], 0) for i in cat]
    
    slope_dict = t.feature.value_counts(1).to_dict()
    prob = t.groupby([ 'feature']).label.value_counts(1).to_dict()
    slope_dict = {i:{'category_rate':slope_dict[i], 'slope':[prob.get((i,j[0]), 0)-prob.get((i,j[1]), 0) for j in cat]} for i in slope_dict}
    for i in slope_dict:
        slope_dict[i]['slope_diff'] = sum([abs(slope[0]-slope_dict[i]['slope'][0]), abs(slope[1]-slope_dict[i]['slope'][1])])
    value1 = sorted([[[i], slope_dict[i]['slope_diff'], slope_dict[i]['category_rate']] for i in slope_dict], key=lambda x:x[1], reverse=1)
    distance = sorted([value1[i][1]-value1[i+1][1] for i in range(len(value1)-1)])
    std = pd.Series([i[1] for i in value1]).std()
    coupe = value1
    dis = distance[0]
    for k in distance:
        value = value1
        while 1:
            for i in range(len(value)-1):
                if value[i][1]-k<value[i+1][1]:
                    value[i+1][0] = value[i][0]+value[i+1][0]
                    value[i+1][1] = value[i][1]*value[i][2]/(value[i][2]+value[i+1][2])+value[i+1][1]*value[i+1][2]/(value[i][2]+value[i+1][2])
                    value[i+1][2] = value[i][2]+value[i+1][2]
                    value.remove(value[i])
                    break
            if i==len(value)-2:
                break
        if pd.Series([i[1] for i in value]).std()>std:
            coupe = value
            std = pd.Series([i[1] for i in value]).std()
            dis = k
    return {'group':{k:i for i,j in enumerate(coupe) for k in j[0]}, 'data':coupe, 
            'distance':dis, 'distance_index':f'{distance.index(dis)+1}/{len(distance)}', 'std':std}