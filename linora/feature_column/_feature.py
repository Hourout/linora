import pandas as pd

from linora.utils._config import Config
from linora.feature_column._categorical import *
from linora.feature_column._normalize import *
from linora.feature_column._numerical import *

__all__ = ['Feature']


class Feature():
    def __init__(self):
        self._params = Config()
        self._params.function = [
            ('categorical_count', categorical_count), ('categorical_crossed', categorical_crossed), 
            ('categorical_encoder', categorical_encoder), ('categorical_hash', categorical_hash), 
            ('categorical_hist', categorical_hist), ('categorical_regress', categorical_regress),
            ('categorical_onehot_binarizer', categorical_onehot_binarizer),
            ('categorical_onehot_multiple', categorical_onehot_multiple),
            ('numerical_binarizer', numerical_binarizer), ('numerical_bucketized', numerical_bucketized),
            ('numerical_padding', numerical_padding), ('numerical_outlier', numerical_outlier),
            ('normalize_max', normalize_max), ('normalize_maxabs', normalize_maxabs), 
            ('normalize_l1', normalize_l1), ('normalize_l2', normalize_l2),
            ('normalize_meanminmax', normalize_meanminmax), ('normalize_minmax', normalize_minmax),
            ('normalize_norm', normalize_norm), ('normalize_robust', normalize_robust)
           ]
        self.pipe = {}
        
    def categorical_count(self, variable, normalize=True, abnormal_value=0, miss_value=0, name=None, keep=False):
        """Count or frequency of conversion category variables.
    
        Args:
            variable: str, feature variable name.
            normalize: bool, If True then the object returned will contain the relative frequencies of the unique values.
            abnormal_value: int or float, if feature values not in feature_scale dict, return `abnormal_value`.
            miss_value: int or float, if feature values are missing, return `miss_value`.
            name: str, output feature name, if None, name is variable.
            keep: if name is not None, name whether to keep in the final output.
        Returns:
            return count labels and label parameters dict.
        """
        config = {'param':{'normalize':normalize, 'abnormal_value':abnormal_value, 
                           'miss_value':miss_value, 'name':variable if name is None else name},
                  'type':'categorical_count', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
        
    def add_pipe(self, config):
        self.pipe[len(self.pipe)] = config[0] if isinstance(config, tuple) else config
        
    def fit(self, df):
        for r in range(len(self.pipe)):
            config = self.pipe[r]
            for i in self._params.function:
                if config['type']==i[0]:
                    self.pipe[r] = i[1](df[config['variable']], mode=2, **config['param'])
                    break
        return self
        
    def transform(self, df, keep_columns=None):
        if isinstance(keep_columns, str):
            keep_columns = [keep_columns]
        self._params.data = pd.DataFrame() if keep_columns is None else df[keep_columns].copy()
        for r in range(len(self.pipe)):
            config = self.pipe[r]
            for i in self._params.function:
                if config['type']==i[0]:
                    self._run_function(i[1], config, df)
                    break
        return self._params.data
                    
    def _run_function(self, function, config, df):
        print(config['variable'])
        if config['variable'] in df.columns:
            self._params.data[config['param']['name']] = function(df[config['variable']], config=config, mode=1)
        elif config['variable'] in self._params.data.columns:
            self._params.data[config['param']['name']] = function(self._params.data[config['variable']], config=config, mode=1)
        else:
            raise ValueError(f"variable `{config['type']}` not exist.")
