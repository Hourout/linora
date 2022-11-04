import itertools

import numpy as np
import pandas as pd

__all__ = ['sampling_random', 'sampling_systematic', 'sampling_stratify']


def sampling_random(feature, n=None, frac=None, replace=False, weights=None, seed=None):
    """simple random sampling.
    
    Args:
        feature: pd.DataFrame or pd.Series or np.array or list.
        n: int, optional, Number of items from axis to return. 
           Cannot be used with `frac`. Default = 1 if `frac` = None.
        frac: float, optional, Fraction of axis items to return. Cannot be used with `n`.
        replace: bool, default False, Allow or disallow sampling of the same row more than once.
        weights: str or pd.Series or np.array or list, optional, 
                 If passed a str, using a `feature` column as weights.
                 If passed a Series, will align with target object on index. 
                 Index values in weights not found in sampled object will be ignored and 
                 index values in sampled object not in weights will be assigned weights of zero.
                 If weights do not sum to 1, they will be normalized to sum to 1.
                 Missing values in the weights column will be treated as zero.
        seed: int, random seed.
    Returns:
        a feature index list of simple random sampling.
    """
    if not isinstance(feature, (pd.DataFrame, pd.Series)):
        feature = pd.Series(range(len(feature)))
    return feature.sample(n=n, frac=frac, replace=replace, weights=weights, random_state=seed).index.to_list()


def sampling_systematic(feature, n=None, frac=None, seed=None):
    """systematic sampling.
    
    Args:
        feature: pd.DataFrame or pd.Series or np.array or list.
        n: int, optional, Number of items from axis to return. 
           Cannot be used with `frac`. Default = 1 if `frac` = None.
        frac: float, optional, Fraction of axis items to return. Cannot be used with `n`.
        seed: int, random seed.
    Returns:
        a feature index list of simple systematic sampling.
    """
    if not isinstance(feature, (pd.DataFrame, pd.Series)):
        t = np.array(range(len(feature)))
    else:
        t = feature.index.to_numpy()
    if n is None and frac is None:
        raise ValueError('Only one of `n` and `frac` can be None.')
    elif n is not None and frac is not None:
        raise ValueError('Only one of `n` and `frac` can be None.')
    elif n is not None:
        frac = min(1, n/len(feature))
    else:
        frac = max(0, min(1, frac))
    n = int(frac*len(feature))
    interval = max(2, int(np.floor(1/frac)))
    seed = np.random.choice(range(interval), 1)[0] if seed is None else seed%interval
    index = list(range(seed, len(feature), interval))
    print(index)
    while len(index)>n:
        if (len(index)-n)%2:
            index.pop(int(len(index)*(seed%100/100)))
        else:
            index.pop(int(len(index)*(seed%10/10)))
    while len(index)<n:
        s = list(set(range(0, len(feature))).difference(set(index)))
        if (n-len(index))%2:
            index.append(s.pop(int(len(s)*(seed%100/100))))
        else:
            index.append(s.pop(int(len(s)*(seed%10/10))))
    return t[sorted(index)]


def _sampling_stratify(feature, n=None, frac=None, seed=None):
    var_numerical = [i for i in feature.columns if feature[i].nunique()>len(feature)*0.6 and feature[i].dtype.name[:3] in ['int', 'flo']]
    var_categorical = [i for i in feature.columns if i not in var_numerical]
    if len(var_categorical)>0:
        var_categorical = [j[0] for j in sorted([(i, feature[i].nunique()) for i in var_categorical], key=lambda x:x[1])]
    
    for r, i in enumerate(var_numerical):
        feature.loc[feature[i]<feature[i].median(), f'la_fea_{r}'] = 0
        feature.loc[feature[i]>=feature[i].median(), f'la_fea_{r}'] = 1
    var_categorical += [i for i in feature.columns if 'la_fea_' in i]
    
    sample = []
    group = feature[var_categorical].value_counts()
    if group[group>1].count()>0:
        for i in group[group>1].index:
            sample.append(feature.query('&'.join([str(m)+'=='+str(n) for m,n in zip(var_categorical, i)])).index.tolist())
    
    if group[group==1].count()>0:
        t = feature.query('('+')|('.join(['&'.join([str(m)+'=='+str(n) for m,n in zip(var_categorical, i)]) for i in group[group==1].index])+')')
        while len(t)>0:
            var_categorical = var_categorical[:-1]
            if len(var_categorical)==0:
                sample.append(t.index.tolist())
                break
            group = t[var_categorical].value_counts()
            if group[group>1].count()>0:
                for i in group[group>1].index:
                    sample.append(t.query('&'.join([str(m)+'=='+str(n) for m,n in zip(var_categorical, i)])).index.tolist())
            if group[group==1].count()==0:
                break
            t = t.query('('+')|('.join(['&'.join([str(m)+'=='+str(n) for m,n in zip(var_categorical, i)]) for i in group[group==1].index])+')')
    if n is not None:
        frac = n/len(feature)
    sample = [pd.Series(i).sample(n=max(1, int(np.ceil(len(i)*frac))), random_state=seed).to_list() for i in sample]
    sample = list(itertools.chain.from_iterable(sample))
    return sample[:n] if n is not None else sample[:int(np.ceil(len(feature)*frac))]


def sampling_stratify(feature, stratify=None, n=None, frac=None, seed=None):
    """stratify sampling.
    
    Args:
        feature: pd.DataFrame or pd.Series.
        stratify: pd.Series, shape (n_samples,), The target variable for supervised learning problems.
        n: int, optional, Number of items from axis to return. 
           Cannot be used with `frac`. Default = 1 if `frac` = None.
        frac: float, optional, Fraction of axis items to return. Cannot be used with `n`.
        seed: int, random seed.
    Returns:
        a feature index list of stratified stratify sampling.
    """
    if n is None and frac is None:
        raise ValueError('Only one of `n` and `frac` can be None.')
    elif n is not None and frac is not None:
        raise ValueError('Only one of `n` and `frac` can be None.')
    elif n is not None:
        frac = min(1, n/len(feature))
    else:
        frac = max(0, min(1, frac))
    
    if not isinstance(feature, pd.DataFrame):
        raise ValueError('feature type must be pd.DataFrame.')
    if stratify is not None:
        if not isinstance(stratify, pd.Series):
            raise ValueError('stratify type must be pd.Series.')
        if stratify.nunique()>len(stratify)*0.6 and stratify.dtype.name[:3] in ['int', 'flo']:
            return _sampling_stratify(feature, n=n, seed=seed)
        else:
            index = [_sampling_stratify(feature[stratify==i], frac=frac, seed=seed) for i in stratify.unique()]
            index = list(itertools.chain.from_iterable(index))
            return index[:n] if n is not None else index[:int(np.ceil(len(feature)*frac))]
    else:
        index = _sampling_stratify(feature, frac=frac, seed=seed)
        return index[:n] if n is not None else index