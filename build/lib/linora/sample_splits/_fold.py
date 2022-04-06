import random
from functools import reduce

import numpy as np


__all__ = ['kfold', 'train_test_split']

def kfold(df, stratify=None, n_splits=3, shuffle=False, random_state=None):
    """K-Folds cross-validator
    
    Provides train/test indices to split data in train/test sets. 
    Split dataset into k consecutive folds (without shuffling by default).
    Each fold is then used once as a validation while the k - 1 remaining folds form the training set.
    
    Parameters
    ----------
    df       : pd.DataFrame, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    stratify : pd.Series, shape (n_samples,)
        The target variable for supervised learning problems.
    n_splits : int, default=3
        Number of folds. Must be at least 2.
    shuffle  : boolean, default=False, optional
        Whether to shuffle the data before splitting into batches.
    random_state : int or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        Used when ``shuffle`` == True.
    
    Returns
    -------
    t : list, length=n_splits
        List each list containing train-test split of inputs.
    """
    t = df.sample(frac=1, random_state=random_state).index if shuffle else df.index
    if stratify is None:
        m = int(np.floor(len(t)/n_splits))
        fold = [t[i*m:(i+1)*m].tolist() for i in range(n_splits-1)]+[t[(n_splits-1)*m:].tolist()]
    else:
        t = stratify[t]
        fold = []
        for label in t.unique():
            a = t[t==label].index
            m = int(np.floor(len(a)/n_splits))
            fold.append([a[i*m:(i+1)*m].tolist() for i in range(n_splits-1)]+[a[(n_splits-1)*m:].tolist()])
        t = [[] for i in range(n_splits)]
        for i in fold:
            for j in range(n_splits):
                t[j] = t[j]+i[j]
        fold = t.copy()
    t = [[] for i in range(n_splits)]
    for i in range(n_splits):
        j = fold.copy()
        j.pop(i)
        j = reduce(lambda x,y:x+y, j)
        t[i] = t[i]+[j, fold[i]]
    return t

def train_test_split(df, stratify=None, test_size=0.2, shuffle=False, random_state=None):
    """Split DataFrame or matrices into random train and test subsets
    
    Parameters
    ----------
    df       : pd.DataFrame, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    stratify : pd.Series, shape (n_samples,)
        The target variable for supervised learning problems.
    test_size: float, optional (default=0.2)
        Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
    shuffle  : boolean, default=False, optional
        Whether to shuffle the data before splitting into batches.
    random_state : int or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        Used when ``shuffle`` == True.
    
    Returns
    -------
    t : list, length=2
        List containing train-test split of inputs.
    """
    t = df.sample(frac=1, random_state=random_state).index if shuffle else df.index
    if stratify is None:
        t = [t[0:round(len(t)*(1-test_size))].tolist(), t[round(len(t)*(1-test_size)):].tolist()]
    else:
        t = stratify[t]
        fold = []
        for label in t.unique():
            a = t[t==label].index
            fold.append([a[0:round(len(a)*(1-test_size))].tolist(), a[round(len(a)*(1-test_size)):].tolist()])
        t = [[], []]
        for i in fold:
            for j in range(2):
                t[j] = t[j]+i[j]
    if shuffle:
        random.shuffle(t[0], lambda :0.5)
        random.shuffle(t[1], lambda :0.5)
    return t
