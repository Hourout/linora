import numpy as np

__all__ = ['missing_columns', 'single_columns', 'correlation_columns',
           'cv_columns']

def missing_columns(df, missing_threshold=0.6):
    assert 1>=missing_threshold>=0, "`missing_threshold` should be one of [0, 1]."
    t = (1-df.count()/len(train)).reset_index()
    t.columns = ['feature_name', 'missing_rate']
    t = t[t.missing_rate>=missing_threshold].reset_index(drop=True)
    return t

def single_columns(df):
    t = train.nunique().reset_index()
    t.columns = ['feature_name', 'single']
    t = t[t.single==1].feature_name.tolist()
    return t

def correlation_columns(df, correlation_threshold=0.9):
    t = df.corr()
    t = t.where(np.triu(np.ones(t.shape), k=1).astype(np.bool))
    return [column for column in t.columns if any(t[column].abs()>=correlation_threshold)]

def cv_columns(df, cv_threshold=0.):
    t = df.replace({np.PINF:np.nan, np.NINF:np.nan})
    t = ((t-t.min())/(t.max()-t.min())*900)+100
    t = (t.std()/t.mean()).reset_index()
    t.columns = ['feature_name', 'cv_rate']
    t = t[t.cv_rate<cv_threshold].feature_name.tolist()
    return t

