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
            ('categorical_onehot_multiple', categorical_onehot_multiple)
           ]
        self.pipe = {}
        
    def add_pipe(self, config):
        self.pipe[len(self.pipe)] = config[0] if isinstance(config, tuple) else config
        
    def transform(self, df, keep_columns=None):
        keep_columns = keep_columns if isinstance(keep_columns, list) else [keep_columns]
        self._params.data = pd.DataFrame() if keep_columns is None else df[keep_columns].copy()
        for r, config in self.pipe.items():
            for i in function:
                if config['type']==i[0]:
                    self._run_function(i[1], config, df)
                    break
        return self._params.data
                    
    def _run_function(self, function, config, df):
        if config['name_input'] in df.colomns:
            self._params.data[config['name_output']] = function(df[config['name_input']], config=config, mode=1)
        elif config['name_input'] in self._params.data.columns:
            self._params.data[config['name_output']] = function(self._params.data[config['name_input']], config=config, mode=1)
        else:
            raise ValueError(f"variable `{config['type']}` not exist.")
