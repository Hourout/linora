import numpy as np
from functools import reduce

__all__ = ['kfold', 'train_test_split']

def kfold(df, stratify=None, n_splits=5, shuffle=False, random_state=None):
    t = df.sample(frac=1, random_state=random_state).index if shuffle else df.index
    if stratify is None:
        m = np.floor(len(t)/n_splits)
        fold = [t[i*m:(i+1)*m].tolist() for i in range(n_splits-1)]+[t[(n_splits-1)*m:].tolist()]
    else:
        t = stratify[t]
        fold = []
        for label in t.unique():
            a = t[t==label].index
            m = np.floor(len(a)/n_splits)
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
    return t
