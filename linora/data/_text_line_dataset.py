import random
import functools
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from linora import gfile
from linora.utils._config import Config

__all__ = ['TextLineDataset']


class DataSet():
    def __init__(self):
        self._params_init()
    
    def _params_init(self):
        self.params = Config()
        self.params.batch = 0
        self.params.batch_size = 1
        self.params.skip_size = None
        self.params.take_size = -1
        self.params.shuffle_size = 1
        self.params.prefetch_size = 1
        self.params.sample = 0
        self.params.step = 1
        self.params.tensor_mode = 'numpy'
        self.params.options = defaultdict(dict)
        
    def batch(self, batch_size, drop_remainder=False):
        """Combines consecutive elements of this dataset into batches.
        
        Args:
            batch_size: representing the number of consecutive elements of this dataset to combine in a single batch.
            drop_remainder: representing whether the last batch should be dropped in the case it has fewer than batch_size elements; 
                            the default behavior is not to drop the smaller batch.
        """
        assert 'batch' not in self.params.options, '`batch` already exists.'
        assert isinstance(batch_size, int) and batch_size>0, '`batch_size` type should be int and greater than 0.'
        self.params.batch_size = batch_size
        self.params.drop_remainder = drop_remainder
        self.params.options['batch'].update({self.params.step: {'batch_size':batch_size, 'drop_remainder':drop_remainder}})
        self.params.step += 1
        return self
    
    def concatenate(self, dataset):
        """Creates a Dataset by concatenating the given dataset with this dataset.
        
        Args:
            dataset: la.data.TextLineDataset object, Dataset to be concatenated.
        """
        self.params.data.append(dataset.params.data)
        self.params.options['concatenate'].update({self.params.step: None})
        self.params.step += 1
        return self
        
    def enumerate(self, start=0):
        """Enumerates the elements of this dataset.
        
        Args:
            start: int, representing the start value for enumeration.
        """
        assert 'enumerate' not in self.params.options, '`enumerate` already exists.'
        self.params.enumerate = start
        self.params.options['enumerate'].update({self.params.step: {'start':start}})
        self.params.step += 1
        return self
    
#     def filter(self, filter_func):
#         """A transformation that filter dataset based on a filter_func.
        
#         Args:
#             filter_func: A function that return True or False
#         """
#         if self.params.data_mode=='list':
#             filter_list = [i for i in range(len(self.params.data[0])) if filter_func([j[i] for j in self.params.data])]
#         else:
#             filter_list = [r for r, i in enumerate(self.params.data) if filter_func(i)]
#         if filter_list:
#             self.params.data_index = [i for i in self.params.data_index if i not in filter_list]
#         self.params.options['filter'].append((self.params.step, filter_func))
#         self.params.step += 1
#         return self
    
    def map(self, map_func):
        """Maps map_func across the elements of this dataset.
        
        Args:
            map_func: A function mapping a dataset element to another dataset element.
            map_size: representing the number elements to process asynchronously in parallel. 
        """
        assert 'map' not in self.params.options, '`map` already exists.'
        self.params.map_func = map_func
        self.params.options['map'].update({self.params.step: {'map_func':map_func}})
        self.params.step += 1
        return self
    
    def options(self):
        """Returns the options for this dataset and its inputs."""
        return self.params.options
    
    def prefetch(self, prefetch_size):
        """Creates a Dataset that prefetches elements from this dataset.
        
        Args:
            prefetch_size: representing the maximum number of elements that will be buffered when prefetching.
        """
        assert 'prefetch' not in self.params.options, '`prefetch` already exists.'
        assert 'take_while' not in self.params.options, '`prefetch` must be placed in `take_while` front.'
        assert isinstance(prefetch_size, int) and prefetch_size>0, '`prefetch_size` type should be int and greater than 0.'
        self.params.prefetch_size = prefetch_size
        self.params.options['prefetch'].update({self.params.step: {'prefetch_size':prefetch_size}})
        self.params.step += 1
        return self
        
#     def reduce(self, reduce_func):
#         """Reduces the input dataset to a single element.
        
#         Args:
#             reduce_func: A function that maps to new_state. It must take two arguments and return a new element
#         """
#         if self.params.data_mode=='list':
#             return [functools.reduce(reduce_func, i[self.params.data_index]) for i in self.params.data]
#         return functools.reduce(reduce_func, self.params.data[self.params.data_index])
    
    def repeat(self, repeat_size):
        """Repeats this dataset so each original value is seen count times.
        
        Args:
            repeat_size: representing the number of times the dataset should be repeated.
        """
        assert 'take_while' not in self.params.options, '`repeat` must be placed in `take_while` front.'
        assert isinstance(repeat_size, int) and repeat_size>0, '`repeat_size` type should be int and greater than 0.'
        self.params.data = self.params.data*(repeat_size+1)
        self.params.options['repeat'].update({self.params.step: {'repeat_size':repeat_size}})
        self.params.step += 1
        return self
        
#     def shard(self, shard_size, shard_index):
#         """Creates a Dataset that includes only 1/num_shards of this dataset.
        
#         Args:
#             shard_size: representing the number of shards operating in parallel.
#             shard_index: representing the worker index.
#         """
#         assert 'take_while' not in self.params.options, '`shard` must be placed in `take_while` front.'
#         assert isinstance(shard_size, int) and shard_size>0, '`shard_size` type should be int and greater than 0.'
#         assert isinstance(shard_index, int) and shard_index>=0, '`shard_index` type should be int and greater than or equal to 0.'
#         self.params.data_index = [self.params.data_index[i] for i in range(shard_index, len(self.params.data_index), shard_size)]
#         self.params.options['shard'].append((self.params.step, shard_size, shard_index))
#         self.params.step += 1
#         return self
    
    def shuffle(self, shuffle_size, seed=None):
        """Randomly shuffles the elements of this dataset.
        
        Args:
            shuffle_size: representing the number of elements from this dataset from which the new dataset will sample.
            seed: representing the random seed that will be used to create the distribution.
        """
        assert 'shuffle' not in self.params.options, '`shuffle` already exists.'
        assert 'take_while' not in self.params.options, '`shuffle` must be placed in `take_while` front.'
        assert isinstance(shuffle_size, int) and shuffle_size>-2 and shuffle_size!=0, '`shuffle_size` type should be int and greater than 0 or equal to -1.'
        self.params.shuffle_size = shuffle_size
        self.params.options['shuffle'].update({self.params.step: {'shuffle_size':shuffle_size, 'seed':seed}})
        self.params.step += 1
        return self
    
    def skip(self, skip_size):
        """Creates a Dataset that skips count elements from this dataset.
        
        Skip all data for the first file at most.
        
        Args:
            skip_size: representing the number of elements of this dataset that should be skipped to form the new dataset. 
                       If count is greater than the size of this dataset, the new dataset will contain no elements.
        """
        assert 'skip' not in self.params.options, '`skip` already exists.'
        assert 'take_while' not in self.params.options, '`skip` must be placed in `take_while` front.'
        assert isinstance(skip_size, int) and skip_size>0, '`skip_size` type should be int and greater than 0.'
        self.params.skip_size = skip_size
        self.params.options['skip'].update({self.params.step: {'skip_size':skip_size}})
        self.params.step += 1
        return self
        
    def take(self, take_size):
        """Creates a Dataset with at most count elements from this dataset.
        
        Args:
            take_size: representing the number of elements of this dataset that should be taken to form the new dataset. 
                       If count is -1, or if count is greater than the size of this dataset, 
                       the new dataset will contain all elements of this dataset.
        """
        assert 'take' not in self.params.options, '`take` already exists.'
        assert 'take_while' not in self.params.options, '`take` must be placed in `take_while` front.'
        assert isinstance(take_size, int) and take_size>-2 and take_size!=0, '`take_size` type should be int and greater than 0 or equal to -1.'
        self.params.take_size = take_size
        self.params.options['take'].update({self.params.step: {'take_size':take_size}})
        self.params.step += 1
        return self
    
#     def take_while(self, take_func):
#         """A transformation that stops dataset iteration based on a take_func.
        
#         Args:
#             take_func: A function that return True or False
#         """
#         temp = set()
#         index = self.params.data_index[:max([self.params.data_index.index(i) for i in range(len(self.params.data))])+1]
#         for r, i in enumerate(index):
#             if i in temp:
#                 continue
#             temp.add(i)
#             if self.params.data_mode=='list':
#                 if take_func([j[i] for j in self.params.data]):
#                     self.params.data_index = self.params.data_index[:r]
#                     break
#             else:
#                 if take_func(self.params.data[i]):
#                     self.params.data_index = self.params.data_index[:r]
#                     break
#         self.params.options['take_while'].append((self.params.step, take_func))
#         self.params.step += 1
#         return self

    def to_tensor(self, mode='tf'):
        """Transform data from numpy array to tensor.
        
        Args:
            mode: Deep learning framework name, one of ['tf', 'pytorch', 'paddle'].
        """
        assert 'to_tensor' not in self.params.options, '`to_tensor` already exists.'
        assert 'take_while' not in self.params.options, '`take` must be placed in `take_while` front.'
        if mode in ['tf', 'tensorflow']:
            from tensorflow import convert_to_tensor
            self.params.framework = convert_to_tensor
        elif mode in ['pytorch', 'torch']:
            from torch import as_tensor
            self.params.framework = as_tensor
        elif mode in ['paddle', 'paddlepaddle']:
            from paddle import to_tensor
            self.params.framework = to_tensor
        else:
            raise ValueError('`mode` value error.')
        self.params.tensor_mode = mode
        self.params.options['to_tensor'].update({self.params.step: {'mode':mode}})
        self.params.step += 1
        return self
    
    def _to_tensor(self, data):
        if self.params.tensor_mode=='numpy':
            return data
        return self.params.framework(data)
    
    def __iter__(self):
        self.params.shuffle_size = np.ceil(max(self.params.shuffle_size, self.params.prefetch_size, 1)/self.params.batch_size)*self.params.batch_size
        self.params.df = pd.read_csv(self.params.data[self.params.batch_file], sep=self.params.sep, 
                                     iterator=True, header=self.params.header, skiprows=self.params.skip_size)
        self.params.values = self.params.df.get_chunk(self.params.shuffle_size)
        if 'shuffle' in self.params.options:
            self.params.values = self.params.values.sample(frac=1, random_state=self.params.shuffle_seed).reset_index(drop=True)
        self.params.batch_file += 1
        self.params.batch_index = 0
        return self
        
    def __next__(self):
        values = self.params.values.loc[self.params.batch_size*self.params.batch_index:self.params.batch_size*(self.params.batch_index+1)]
        if len(values)<self.params.batch_size:
            self.params.batch_index = 0
            try:
                self.params.values = self.params.df.get_chunk(self.params.shuffle_size).reset_index(drop=True)
            except StopIteration:
                if self.params.batch_file==len(self.params.data):
                    raise StopIteration
                self.params.df = pd.read_csv(self.params.data[self.params.batch_file], sep=self.params.sep, header=self.params.header, iterator=True)
                self.params.values = self.params.df.get_chunk(self.params.shuffle_size)
                self.params.batch_file += 1
            if 'shuffle' in self.params.options:
                self.params.values = self.params.values.sample(frac=1, random_state=self.params.shuffle_seed).reset_index(drop=True)
            values = self.params.values.loc[0:self.params.batch_size]
            
        self.params.batch += 1
        self.params.batch_index += 1
        if self.params.take_size>0:
            if self.params.sample>=self.params.take_size:
                raise StopIteration
            self.params.sample += len(values)
        if 'map' in self.params.options:
            return self._to_tensor(values.apply(self.params.map_func, axis=1).values)
        return self._to_tensor(values.values)


class TextLineDataset(DataSet):
    """Represents a potentially large set of elements from text file."""
    def __init__(self):
        super(TextLineDataset, self).__init__()
        self.params.rank_list = None
        self.params.batch_index = 0
        self.params.batch_file = 0
        self.params.sample = 0
        
    def from_tensor(self, data, sep=',', header=None):
        """Creates a Dataset comprising lines from one or more text files.

        Args:
            data: file path string or list of file path.
            sep: str, default ',', Delimiter to use.
            header: int, list of int, Row number(s) to use as the column names, and the start of the data.
        """
        self.params.sep = sep
        self.params.header = header
        data = list(data) if isinstance(data, (tuple, list)) else [data]
        for i in data:
            if not gfile.isfile(i):
                raise ValueError(f'`{i}` not a file path.')
        self.params.data = data
        return self