import numpy as np

__all__ = ['missing_columns', 'single_columns', 'correlation_columns',
           'cv_columns']


def missing_columns(df, missing_rate=0.):
    """Find missing features
    
    Args:
        df: pd.DataFrame, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.
        threshold: float, default=0.,[0,1)
            Count all features with a missing rate greater than `missing_rate`.
    
    Return:
        t : All features with a missing rate greater than `missing_rate`
    """
    assert 1>=missing_rate>=0, "`missing_rate` should be one of [0, 1]."
    t = (1-df.count()/len(df)).reset_index()
    t.columns = ['feature_name', 'missing_rate']
    return t[t.missing_rate>=missing_rate].reset_index(drop=True)


def single_columns(df):
    """Find single value features
    
    Args:
        df: pd.DataFrame, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.
    Return:
        All features with single value
    """
    return df.nunique().rename('single').reset_index().query('single==1')['index'].to_list()


def correlation_columns(df, corr_rate=0.9):
    """Find features whose correlation coefficient is greater than a certain threshold.
    
    Args:
        df: pd.DataFrame, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.
        corr_rate : float, default=0.9, [0,1]
            Count all features with corr greater than `corr_rate`.
    Return:
        All features with corr greater than `corr_rate`.
    """
    t = df.corr()
    t = t.where(np.triu(np.ones(t.shape), k=1).astype(bool)).fillna(0)
    return [[i,j,t.loc[i, j]] for i in t for j in t[i].index if abs(t.loc[i, j])>corr_rate]


def cv_columns(df, cv_rate=0.):
    """Find features with coefficients of variation less than a certain threshold.
    
    Args:
        df: pd.DataFrame, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.
        cv_rate: float, default=0.
            Count all features with cv less than `cv_rate`.
    
    Return:
        All features with cv less than `cv_rate`.
    """
    t = df.replace({np.PINF:np.nan, np.NINF:np.nan})
    t = ((t-t.min())/(t.max()-t.min())*900)+100
    t = (t.std()/t.mean()).reset_index()
    t.columns = ['feature_name', 'cv_rate']
    t = t[t.cv_rate<cv_rate].feature_name.tolist()
    return t

