import numpy as np

__all__ = ['missing_columns', 'single_columns', 'correlation_columns',
           'cv_columns']

def missing_columns(df, missing_threshold=0.6):
    """Find missing features
    
    Parameters
    ----------
    df       : pd.DataFrame, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    missing_threshold  : float, default=0.6
        Count all features with a missing rate greater than `missing_threshold`.
    
    Returns
    -------
    t : All features with a missing rate greater than `missing_threshold`
    """
    assert 1>=missing_threshold>=0, "`missing_threshold` should be one of [0, 1]."
    t = (1-df.count()/len(df)).reset_index()
    t.columns = ['feature_name', 'missing_rate']
    t = t[t.missing_rate>=missing_threshold].reset_index(drop=True)
    return t

def single_columns(df):
    """Find single value features
    
    Parameters
    ----------
    df       : pd.DataFrame, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    
    Returns
    -------
    t : All features with single value
    """
    t = train.nunique().reset_index()
    t.columns = ['feature_name', 'single']
    t = t[t.single==1].feature_name.tolist()
    return t

def correlation_columns(df, correlation_threshold=0.9):
    """Find features whose correlation coefficient is greater than a certain threshold.
    
    Parameters
    ----------
    df       : pd.DataFrame, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    correlation_threshold : float, default=0.9
        Count all features with corr greater than `correlation_threshold`.
    
    Returns
    -------
    t : All features with corr greater than `correlation_threshold`.
    """
    t = df.corr()
    t = t.where(np.triu(np.ones(t.shape), k=1).astype(np.bool))
    return [column for column in t.columns if any(t[column].abs()>=correlation_threshold)]

def cv_columns(df, cv_threshold=0.):
    """Find features with coefficients of variation less than a certain threshold.
    
    Parameters
    ----------
    df       : pd.DataFrame, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    cv_threshold : float, default=0.
        Count all features with cv less than `cv_threshold`.
    
    Returns
    -------
    t : All features with cv less than `cv_threshold`.
    """
    t = df.replace({np.PINF:np.nan, np.NINF:np.nan})
    t = ((t-t.min())/(t.max()-t.min())*900)+100
    t = (t.std()/t.mean()).reset_index()
    t.columns = ['feature_name', 'cv_rate']
    t = t[t.cv_rate<cv_threshold].feature_name.tolist()
    return t

