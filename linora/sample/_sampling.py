import numpy as np
import pandas as pd

__all__ = ['sampling_random', 'sampling_systematic']


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
        a feature index list of simple random sampling.
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