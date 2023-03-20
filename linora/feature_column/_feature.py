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
            'normalize_minmax', normalize_minmax,
            'normalize_norm':normalize_norm, 
            'normalize_robust':normalize_robust
        }
        self.pipe = {}
        
    def fit(self, df):
        for r in range(len(self.pipe)):
            config = self.pipe[r]
            if 'lable' in config:
                self._params.function[config['type']](df[config['variable']], label=df[config['lable']], mode=2, **config['param'])
            else:
                self._params.function[config['type']](df[config['variable']], mode=2, **config['param'])
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
