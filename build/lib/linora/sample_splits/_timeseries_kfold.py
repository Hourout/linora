import numpy as np

__all__ = ['timeseries_train_test_split', 'timeseries_walk_forward_fold', 'timeseries_kfold']

def timeseries_train_test_split(df, test_size=0.2, gap=0, ascending=True):
    """Split DataFrame or matrices into random train and test subsets for timeseries
    
    Parameters
    ----------
    df       : pd.DataFrame, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    test_size: float, optional (default=0.2)
        Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
    gap      : int, default=0
        Represents the absolute number of the dropped samples between training set and test set.
    ascending  : boolean, default=True, optional
        Whether timeseries is ascending.
    
    Returns
    -------
    t : list, length=2
        List each list containing train-test split of inputs.
    """
    assert gap>=0, "`gap` should be great than or equal to 0."
    t = df.index.tolist()
    if not ascending:
        t.reverse()
    train_len = round(len(t)*(1-test_size))-gap
    return [t[0:train_len], t[train_len+gap:]]

def timeseries_walk_forward_fold(df, n_splits=3, gap=0, ascending=True):
    """Walk Forward Folds cross-validator for timeseries
    
    Provides train/test indices to split data in train/test sets. 
    Split dataset into k consecutive folds.
    Each fold train set length is not equal.
    
    Parameters
    ----------
    df       : pd.DataFrame, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    n_splits : int, default=3
        Number of folds. Must be at least 2.
    gap      : int, default=0
        Represents the absolute number of the dropped samples between training set and test set.
    ascending  : boolean, default=True, optional
        Whether timeseries is ascending.
    
    Returns
    -------
    t : list, length=n_splits
        List each list containing train-test split of inputs.
    """
    assert gap>=0, "`gap` should be great than or equal to 0."
    t = df.index.tolist()
    if not ascending:
        t.reverse()
    t_len = len(t)
    m = int(np.floor(t_len/(n_splits+1)))
    return [[t[:t_len-m*(i+1)-gap], t[t_len-m*(i+1):t_len-m*i]] for i in range(n_splits)]

def timeseries_kfold(df, train_size=0.3, test_size=0.1, gap=0, ascending=True):
    """K-Folds cross-validator for timeseries
    
    Provides train/test indices to split data in train/test sets. 
    Split dataset into k consecutive folds.
    Each fold train set length is equal.
    
    Parameters
    ----------
    df       : pd.DataFrame, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    train_size: float, optional (default=0.3)
        Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
    test_size: float, optional (default=0.1)
        Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
    gap      : int, default=0
        Represents the absolute number of the dropped samples between training set and test set.
    ascending  : boolean, default=True, optional
        Whether timeseries is ascending.
    
    Returns
    -------
    t : list, length=`int(np.floor((len(df)-len(trainset)-gap)/len(testset)))`
        List each list containing train-test split of inputs.
    """
    assert gap>=0, "`gap` should be great than or equal to 0."
    assert train_size+test_size<1, "`train_size`+`test_size` should be less than 1."
    t = df.index.tolist()
    if not ascending:
        t.reverse()
    t_len = len(t)
    train_len = int(np.floor(t_len*train_size))
    test_len = int(np.floor(t_len*test_size))
    all_len = train_len+test_len+gap
    n_splits = int(np.floor((t_len-all_len)/test_len)+1)
    return [[t[test_len*i:test_len*i+train_len], t[test_len*i+train_len+gap:test_len*i+all_len]] for i in range(n_splits)]
