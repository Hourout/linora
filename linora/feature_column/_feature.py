from collections import defaultdict

import pandas as pd

from linora.utils._config import Config
from linora.feature_column._categorical import *
from linora.feature_column._normalize import *
from linora.feature_column._numerical import *
from linora.feature_column._feature_categorical import FeatureCategorical
from linora.feature_column._feature_numerical import FeatureNumerical
from linora.feature_column._feature_normalize import FeatureNormalize


__all__ = ['Feature']


class Feature(FeatureCategorical, FeatureNumerical, FeatureNormalize):
    def __init__(self):
        super(Feature, self).__init__()
        self._params = Config()
        self._params.function = {
            'categorical_count':categorical_count, 
            'categorical_crossed':categorical_crossed, 
            'categorical_encoder':categorical_encoder, 
            'categorical_hash':categorical_hash, 
            'categorical_hist':categorical_hist,
            'categorical_regress':categorical_regress,
            'categorical_onehot_binarizer':categorical_onehot_binarizer,
            'categorical_onehot_multiple':categorical_onehot_multiple,
            'categorical_rare':categorical_rare,
            'categorical_regress':categorical_regress,
            'categorical_woe':categorical_woe,
            'numerical_binarizer':numerical_binarizer,
            'numerical_bucketized':numerical_bucketized,
            'numerical_padding':numerical_padding, 
            'numerical_outlier':numerical_outlier,
            'normalize_max':normalize_max, 
            'normalize_maxabs':normalize_maxabs, 
            'normalize_l1':normalize_l1, 
            'normalize_l2':normalize_l2,
            'normalize_meanminmax':normalize_meanminmax,
            'normalize_minmax':normalize_minmax,
            'normalize_norm':normalize_norm, 
            'normalize_robust':normalize_robust
        }
        self.pipe = {}
        
    def fit(self, df):
        _ = self._fit_transform(df, fit=True)
        return self
        
    def fit_transform(self, df, keep_columns=None):
        return self._fit_transform(df, keep_columns=keep_columns, fit=True)
        
    def transform(self, df, keep_columns=None):
        return self._fit_transform(df, keep_columns=keep_columns, fit=False)
    
    def _fit_transform(self, df, keep_columns=None, fit=True):
        if isinstance(keep_columns, str):
            keep_columns = [keep_columns]
        data = pd.DataFrame() if keep_columns is None else df[keep_columns].copy()
        name_dict = defaultdict(lambda: 0)
        name_drop = [self.pipe[i]['param']['name'] for i in self.pipe if not self.pipe[i]['keep']]
        for r in range(len(self.pipe)):
            config = self.pipe[r].copy()
            t = self._run_function(self._params.function[config['type']], config, df, data, fit)
            data[config['param']['name']] = t[0]
            name_dict[config['param']['name']] += 1
            if fit:
                self.pipe[r]['param'] = t[1]
        name_if = [i for i in name_dict if name_dict[i]==1]
        drop = [i for i in name_if if i in name_drop]
        keep = [i for i in data.columns if i not in drop]
        return data[keep]
    
    def _run_function(self, function, config, df, data, fit):
        if config['variable'] in df.columns:
            if 'lable' in config:
                if fit:
                    t = function(df[config['variable']], label=df[config['lable']], **config['param'])
                else:
                    t = function(df[config['variable']], label=df[config['lable']], config=config)
            else:
                if fit:
                    t = function(df[config['variable']], **config['param'])
                else:
                    t = function(df[config['variable']], config=config)
        elif config['variable'] in data.columns:
            if 'lable' in config:
                if fit:
                    t = function(data[config['variable']], label=df[config['lable']], **config['param'])
                else:
                    t = function(data[config['variable']], label=df[config['lable']], config=config)
            else:
                if fit:
                    t = function(data[config['variable']], **config['param'])
                else:
                    t = function(data[config['variable']], config=config)
        else:
            raise ValueError(f"variable `{config['type']}` not exist.")
        return t
        
            
            
            
            
            
