import random
import itertools

import numpy as np
import pandas as pd

from linora.utils._config import Config

    
class DataSet():
    def __init__(self):
        self._params_init()
    
    def _params_init(self):
        self.params = Config()
        self.params.rank = dict()
        self.params.batch = 0
        self.params.batch_size = 1
        
    def batch(self, batch_size, drop_remainder=False):
        """Combines consecutive elements of this dataset into batches.
        
        Args:
            batch_size: representing the number of consecutive elements of this dataset to combine in a single batch.
            drop_remainder: representing whether the last batch should be dropped in the case it has fewer than batch_size elements; 
                        the default behavior is not to drop the smaller batch.
        """
        assert isinstance(batch_size, int) and batch_size>0, '`batch_size` type should be int and greater than 0'
        self.params.batch_size = batch_size
        self.params.drop_remainder = drop_remainder
        return self
        
    def prefetch(self, buffer_size):
        """Creates a Dataset that prefetches elements from this dataset.
        
        Args:
            buffer_size: representing the maximum number of elements that will be buffered when prefetching.
        """
        if 'prefetch' not in self.params.rank:
            assert isinstance(buffer_size, int) and buffer_size>0, '`buffer_size` type should be int and greater than 0'
            self.params.rank['prefetch'] = len(self.params.rank)+1
            self.params.prefetch_size = buffer_size
        return self
        
    def repeat(self, count):
        """Repeats this dataset so each original value is seen count times.
        
        Args:
            count: representing the number of times the dataset should be repeated.
        """
        assert isinstance(count, int) and count>0, '`count` type should be int and greater than 0'
        if 'repeat' not in self.params.rank:
            self.params.rank['repeat'] = len(self.params.rank)+1
        self.params.repeat_size = count
        self.params.data_index = self.params.data_index*(self.params.repeat_size+1)
        return self
        
    def shuffle(self, buffer_size, seed=None):
        """Randomly shuffles the elements of this dataset.
        
        Args:
            buffer_size: representing the number of elements from this dataset from which the new dataset will sample.
            seed: representing the random seed that will be used to create the distribution.
        """
        assert isinstance(buffer_size, int) and buffer_size>-2 and buffer_size!=0, '`buffer_size` type should be int and greater than 0 or equal to -1.'
        if 'shuffle' not in self.params.rank:
            self.params.rank['shuffle'] = len(self.params.rank)+1
        self.params.shuffle_size = buffer_size
        self.params.shuffle_seed = seed
        if isinstance(self.params.data_index, list):
            self.params.data_index = pd.Series(index=self.params.data_index, data=1).index
        if self.params.shuffle_size > 0:
            t = [self.params.data_index[self.params.shuffle_size*i:self.params.shuffle_size*(i+1)].to_list() for i in range(len(self.params.data_index)//self.params.shuffle_size+1)]
            [random.shuffle(i, random=lambda :((seed if seed is not None else random.randint(1, 99))+self.params.batch)%10/10) for i in t]
            self.params.data_index = list(itertools.chain.from_iterable(t))
        else:
            self.params.data_index = self.params.data_index.to_series().sample(frac=1, random_state=self.params.shuffle_seed).tolist()
        return self
    
    def skip(self, count):
        """Creates a Dataset that skips count elements from this dataset.
        
        Args:
            count: representing the number of elements of this dataset that should be skipped to form the new dataset. 
                   If count is greater than the size of this dataset, the new dataset will contain no elements.
        """
        assert isinstance(count, int) and count>0, '`count` type should be int and greater than 0'
        if 'skip' not in self.params.rank:
            self.params.rank['skip'] = len(self.params.rank)+1
        self.params.skip = count
        self.params.data_index = self.params.data_index[self.params.skip:]
        return self
        
    def take(self, count):
        """Creates a Dataset with at most count elements from this dataset.
        
        Args:
            count: representing the number of elements of this dataset that should be taken to form the new dataset. 
                   If count is -1, or if count is greater than the size of this dataset, 
                   the new dataset will contain all elements of this dataset.
        """
        assert isinstance(count, int) and count>-2 and count!=0, '`count` type should be int and greater than 0 or equal to -1.'
        if 'take' not in self.params.rank:
            self.params.rank['take'] = len(self.params.rank)+1
        self.params.take = count
        if self.params.take != -1:
            self.params.data_index = self.params.data_index[:self.params.take]
        return self
    
    def shard(self, num_shards, index):
        """Creates a Dataset that includes only 1/num_shards of this dataset.
        
        Args:
            num_shards: representing the number of shards operating in parallel.
            index: representing the worker index.
        """
        assert isinstance(num_shards, int) and num_shards>0, '`num_shards` type should be int and greater than 0.'
        assert isinstance(index, int) and index>=0, '`index` type should be int and greater than or equal to 0.'
        if 'shard' not in self.params.rank:
            self.params.rank['shard'] = len(self.params.rank)+1
        self.params.shard_step = num_shards
        self.params.shard_index = index
        self.params.data_index = [self.params.data_index[i] for i in range(self.params.shard_index, len(self.params.data_index), self.params.shard_step)]
        return self
    
    def map(self, map_func):
        """Maps map_func across the elements of this dataset.
        
        Args:
            map_func: A function mapping a dataset element to another dataset element.
        """
        if 'map' not in self.params.rank:
            self.params.rank['map'] = len(self.params.rank)+1
            self.params.map_func = map_func
#         self.params.num_parallel_calls = num_parallel_calls
#         self.params.deterministic = deterministic
        return self

    def cardinality(self):
        """Returns the cardinality of the dataset, if known."""
        return len(self.params.data_index)
    
    def range(self, *args, **kwargs):
        """Creates a Dataset of a step-separated range of values."""
        self._params_init()
        self.params.data_mode = 'array'
        self.params.data_index = list(range(*args, **kwargs))
        self.params.data = np.array(range(*args, **kwargs))
        return self