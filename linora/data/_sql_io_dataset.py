from collections import defaultdict

import numpy as np
import pandas as pd

from linora.utils._config import Config
from linora.utils.pip._pip import install

__all__ = ['MysqlIoDataset', 'HiveIoDataset']


class DataSet():
    def __init__(self):
        self._params_init()
    
    def _params_init(self):
        self._params = Config()
        self._params.batch = 0
        self._params.batch_size = 1
        self._params.take_size = -1
        self._params.shuffle_size = 1
        self._params.prefetch_size = 1
        self._params.repeat_size = 1
        self._params.sample = 0
        self._params.step = 1
        self._params.tensor_mode = 'numpy'
        self._params.options = defaultdict(dict)
        
    def batch(self, batch_size, drop_remainder=False):
        """Combines consecutive elements of this dataset into batches.
        
        Args:
            batch_size: representing the number of consecutive elements of this dataset to combine in a single batch.
            drop_remainder: representing whether the last batch should be dropped in the case it has fewer than batch_size elements; 
                            the default behavior is not to drop the smaller batch.
        """
        assert 'batch' not in self._params.options, '`batch` already exists.'
        assert isinstance(batch_size, int) and batch_size>0, '`batch_size` type should be int and greater than 0.'
        self._params.batch_size = batch_size
        self._params.drop_remainder = drop_remainder
        self._params.options['batch'].update({self._params.step: {'batch_size':batch_size, 'drop_remainder':drop_remainder}})
        self._params.step += 1
        return self
        
    def enumerate(self, start=0):
        """Enumerates the elements of this dataset.
        
        Args:
            start: int, representing the start value for enumeration.
        """
        assert 'enumerate' not in self._params.options, '`enumerate` already exists.'
        self._params.enumerate = start
        self._params.options['enumerate'].update({self._params.step: {'start':start}})
        self._params.step += 1
        return self
    
    def map(self, map_func):
        """Maps map_func across the elements of this dataset.
        
        Args:
            map_func: A function mapping a dataset element to another dataset element.
            map_size: representing the number elements to process asynchronously in parallel. 
        """
        assert 'map' not in self._params.options, '`map` already exists.'
        self._params.map_func = map_func
        self._params.options['map'].update({self._params.step: {'map_func':map_func}})
        self._params.step += 1
        return self
    
    def options(self):
        """Returns the options for this dataset and its inputs."""
        return self._params.options
    
    def prefetch(self, prefetch_size):
        """Creates a Dataset that prefetches elements from this dataset.
        
        Args:
            prefetch_size: representing the maximum number of elements that will be buffered when prefetching.
        """
        assert 'prefetch' not in self._params.options, '`prefetch` already exists.'
        assert isinstance(prefetch_size, int) and prefetch_size>0, '`prefetch_size` type should be int and greater than 0.'
        self._params.prefetch_size = prefetch_size
        self._params.options['prefetch'].update({self._params.step: {'prefetch_size':prefetch_size}})
        self._params.step += 1
        return self

    def read_sql(self, sql, limit_size):
        """Read sql form database.
        
        Args:
            sql: query database content.
            limit_size: int, read data amount.
        """
        self._params.sql = sql
        if 'limit' not in self._params.sql:
            self._params.sql = f"{self._params.sql} limit {limit_size}"
        self._params.options['read_sql'].update({self._params.step: {'read_sql':sql}})
        self._params.step += 1
        return self
        
    def repeat(self, repeat_size):
        """Repeats this dataset so each original value is seen count times.
        
        Args:
            repeat_size: representing the number of times the dataset should be repeated.
        """
        assert isinstance(repeat_size, int) and repeat_size>0, '`repeat_size` type should be int and greater than 0.'
        self._params.repeat_size = repeat_size
        self._params.options['repeat'].update({self._params.step: {'repeat_size':repeat_size}})
        self._params.step += 1
        return self
    
    def shuffle(self, shuffle_size, seed=None):
        """Randomly shuffles the elements of this dataset.
        
        Args:
            shuffle_size: representing the number of elements from this dataset from which the new dataset will sample.
            seed: representing the random seed that will be used to create the distribution.
        """
        assert 'shuffle' not in self._params.options, '`shuffle` already exists.'
        assert isinstance(shuffle_size, int) and shuffle_size>-2 and shuffle_size!=0, '`shuffle_size` type should be int and greater than 0 or equal to -1.'
        self._params.shuffle_size = shuffle_size
        self._params.options['shuffle'].update({self._params.step: {'shuffle_size':shuffle_size, 'seed':seed}})
        self._params.step += 1
        return self
        
    def take(self, take_size):
        """Creates a Dataset with at most count elements from this dataset.
        
        Args:
            take_size: representing the number of elements of this dataset that should be taken to form the new dataset. 
                       If count is -1, or if count is greater than the size of this dataset, 
                       the new dataset will contain all elements of this dataset.
        """
        assert 'take' not in self._params.options, '`take` already exists.'
        assert isinstance(take_size, int) and take_size>-2 and take_size!=0, '`take_size` type should be int and greater than 0 or equal to -1.'
        self._params.take_size = take_size
        self._params.options['take'].update({self._params.step: {'take_size':take_size}})
        self._params.step += 1
        return self

    def to_tensor(self, mode='tf'):
        """Transform data from numpy array to tensor.
        
        Args:
            mode: Deep learning framework name, one of ['tf', 'pytorch', 'paddle', 'mxnet', 'mindspore'].
        """
        assert 'to_tensor' not in self._params.options, '`to_tensor` already exists.'
        assert 'take_while' not in self._params.options, '`take` must be placed in `take_while` front.'
        if mode in ['tf', 'tensorflow']:
            from tensorflow import convert_to_tensor
            self._params.framework = convert_to_tensor
        elif mode in ['pytorch', 'torch']:
            from torch import as_tensor
            self._params.framework = as_tensor
        elif mode in ['paddle', 'paddlepaddle']:
            from paddle import to_tensor
            self._params.framework = to_tensor
        elif mode in ['mx', 'mxnet']:
            from mxnet.ndarray import array
            self._params.framework = array
        elif mode in ['mindspore']:
            from mindspore.numpy import array
            self._params.framework = array
        else:
            raise ValueError('`mode` value error.')
        self._params.tensor_mode = mode
        self._params.options['to_tensor'].update({self._params.step: {'mode':mode}})
        self._params.step += 1
        return self
    
    def _to_tensor(self, data):
        if self._params.tensor_mode=='numpy':
            return data
        return self._params.framework(data)
    
    def __iter__(self):
        self._params.shuffle_size = int(np.ceil(max(self._params.shuffle_size, self._params.prefetch_size, 1)/self._params.batch_size)*self._params.batch_size)
        self._params.df = pd.read_sql(self._params.sql, self._params.engine, chunksize=self._params.shuffle_size)
        self._params.values = next(self._params.df)
        if 'shuffle' in self._params.options:
            self._params.values = self._params.values.sample(frac=1, random_state=self._params.shuffle_seed).reset_index(drop=True)
        self._params.repeat_size -= 1
        self._params.batch_index = 0
        return self
        
    def __next__(self):
        values = self._params.values.loc[self._params.batch_size*self._params.batch_index:self._params.batch_size*(self._params.batch_index+1)]
        if len(values)<self._params.batch_size:
            self._params.batch_index = 0
            try:
                self._params.values = next(self._params.df).reset_index(drop=True)
            except StopIteration:
                if self._params.repeat_size==0:
                    raise StopIteration
                self._params.df = pd.read_sql(self._params.sql, self._params.engine, chunksize=self._params.shuffle_size)
                self._params.values = next(self._params.df)
                self._params.repeat_size -= 1
            if 'shuffle' in self._params.options:
                self._params.values = self._params.values.sample(frac=1, random_state=self._params.shuffle_seed).reset_index(drop=True)
            values = self._params.values.loc[0:self._params.batch_size]
            
        self._params.batch += 1
        self._params.batch_index += 1
        if self._params.take_size>0:
            if self._params.sample>=self._params.take_size:
                raise StopIteration
            self._params.sample += len(values)
        if 'map' in self._params.options:
            return self._to_tensor(values.apply(self._params.map_func, axis=1).values)
        return self._to_tensor(values.values)


class MysqlIoDataset(DataSet):
    """Create a data set composed of mysql database.

    Args:
        host: mysql database host.
        port: mysql database port.
        user: mysql database user.
        password: mysql database password.
    """
    def __init__(self, host, port=3306, user=None, password=None, **kwargs):
        super(MysqlIoDataset, self).__init__()
        try:
            import pymysql
        except:
            install('PyMySQL')
        try:
            import sqlalchemy as sa
        except:
            install('SQLAlchemy')
            import sqlalchemy as sa
        if user is not None and password is None:
            url = f"mysql+pymysql://{user}@{host}:{port}"
        elif user is None and password is None:
            url = f"mysql+pymysql://{host}:{port}"
        elif user is not None and password is not None:
            url = f"mysql+pymysql://{user}:{password}@{host}:{port}"
        else:
            raise ValueError('mysql database address error.')
        self._params.engine = sa.create_engine(url, **kwargs)
        

class HiveIoDataset(DataSet):
    """Create a data set composed of hive database.

    Args:
        host: hive database host.
        port: hive database port.
        user: hive database user.
        password: hive database password.
    """
    def __init__(self, host, port=10000, user=None, password=None, **kwargs):
        super(HiveIoDataset, self).__init__()
        try:
            import pyhive
        except:
            install('pyhive')
        try:
            import sqlalchemy as sa
        except:
            install('SQLAlchemy')
            import sqlalchemy as sa
        if user is not None and password is None:
            url = f"hive://{user}@{host}:{port}"
        elif user is None and password is None:
            url = f"hive://{host}:{port}"
        elif user is not None and password is not None:
            url = f"hive://{user}:{password}@{host}:{port}"
        else:
            raise ValueError('hive database address error.')
        self._params.engine = sa.create_engine(url, **kwargs)
        
        