import random
import functools
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from linora.utils._config import Config

    
class DataSet():
    def __init__(self):
        self._params_init()
    
    def _params_init(self):
        self.params = Config()
        self.params.batch = 0
        self.params.batch_size = 1
        self.params.step = 1
        self.params.options = defaultdict(dict)
        
    def batch(self, batch_size, drop_remainder=False):
        """Combines consecutive elements of this dataset into batches.
        
        Args:
            batch_size: representing the number of consecutive elements of this dataset to combine in a single batch.
            drop_remainder: representing whether the last batch should be dropped in the case it has fewer than batch_size elements; 
                            the default behavior is not to drop the smaller batch.
        """
        assert isinstance(batch_size, int) and batch_size>0, '`batch_size` type should be int and greater than 0.'
        self.params.batch_size = batch_size
        self.params.drop_remainder = drop_remainder
        self.params.options['batch'].update({self.params.step: {'batch_size':batch_size, 'drop_remainder':drop_remainder}})
        self.params.step += 1
        return self
    
    def cardinality(self):
        """Returns the cardinality of the dataset, if known."""
        return len(self.params.data_index)
    
    def concatenate(self, dataset):
        """Creates a Dataset by concatenating the given dataset with this dataset.
        
        Args:
            dataset: Dataset to be concatenated.
        """
        assert 'take_while' not in self.params.options, '`concatenate` must be placed in `take_while` front.'
        assert self.params.data_mode==dataset.params.data_mode, 'The data types of the two data sets are inconsistent.'
        t = len(self.params.data[0]) if self.params.data_mode=='list' else len(self.params.data)
        if self.params.data_mode=='list':
            assert len(self.params.data)==len(dataset.params.data), 'Width needs to be consistent between data.'
            self.params.data = [np.concatenate([self.params.data[i], dataset.params.data[i]]) for i in range(len(self.params.data))]
        else:
            self.params.data = np.concatenate([self.params.data, dataset.params.data])
        self.params.data_index = self.params.data_index+[i+t for i in dataset.params.data_index]
        self.params.options['concatenate'].update({self.params.step: None})
        self.params.step += 1
        return self
        
    def enumerate(self, start=0):
        """Enumerates the elements of this dataset.
        
        Args:
            start: int, representing the start value for enumeration.
        """
        self.params.enumerate = start
        self.params.options['enumerate'].update({self.params.step: {'start':start}})
        self.params.step += 1
        return self
    
    def filter(self, filter_func):
        """A transformation that filter dataset based on a filter_func.
        
        Args:
            filter_func: A function that return True or False
        """
        if self.params.data_mode=='list':
            filter_list = [i for i in range(len(self.params.data[0])) if filter_func([j[i] for j in self.params.data])]
        else:
            filter_list = [r for r, i in enumerate(self.params.data) if filter_func(i)]
        if filter_list:
            self.params.data_index = [i for i in self.params.data_index if i not in filter_list]
        self.params.options['filter'].update({self.params.step: {'filter_func':filter_func}})
        self.params.step += 1
        return self
    
    def map(self, map_func, map_size=8):
        """Maps map_func across the elements of this dataset.
        
        Args:
            map_func: A function mapping a dataset element to another dataset element.
            map_size: representing the number elements to process asynchronously in parallel. 
        """
        assert 'map' not in self.params.options, '`map` can only be set once.'
        assert isinstance(map_size, int) and map_size>0, '`map_size` type should be int and greater than 0.'
        self.params.map_func = map_func
        self.params.map_size = map_size
        self.params.options['map'].update({self.params.step: {'map_func':map_func, 'map_size':map_size}})
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
        assert 'take_while' not in self.params.options, '`prefetch` must be placed in `take_while` front.'
        assert isinstance(prefetch_size, int) and prefetch_size>0, '`prefetch_size` type should be int and greater than 0.'
        self.params.options['prefetch'].update({self.params.step: {'prefetch_size':prefetch_size}})
        self.params.step += 1
        return self
        
    def range(self, *args, **kwargs):
        """Creates a Dataset of a step-separated range of values."""
        assert 'take_while' not in self.params.options, '`range` must be placed in `take_while` front.'
        self._params_init()
        self.params.data_mode = 'array'
        self.params.data = np.array(range(*args, **kwargs))
        self.params.data_index = list(range(len(self.params.data)))
        self.params.options['range'].update({self.params.step: {'args':args, 'kwargs':kwargs}})
        self.params.step += 1
        return self
    
    def random(self, size, lower=0, upper=10, seed=None):
        """Creates a Dataset of pseudorandom values. The dataset generates a sequence of uniformly distributed integer values.
        
        Args:
            size: shape of output values.
            lower: min random values.
            upper: max random values.
            seed: random seed.
        """
        assert 'take_while' not in self.params.options, '`random` must be placed in `take_while` front.'
        if isinstance(size, int):
            t = (list(range(lower, upper))*(size//10+1))[:size]
        else:
            t = 1
            for i in size:
                t *= i
            t = (list(range(lower, upper))*(t//10+1))[:t]
        random.shuffle(t, random=lambda :((seed if seed is not None else random.randint(1, 99)))%10/10)
        self._params_init()
        self.params.data_mode = 'array'
        self.params.data = np.array(t).reshape(size)
        self.params.data_index = list(range(len(self.params.data)))
        self.params.options['random'].update({self.params.step: {'size':size, 'lower':lower, 'upper':upper, 'seed':seed}})
        self.params.step += 1
        return self
    
    def reduce(self, reduce_func):
        """Reduces the input dataset to a single element.
        
        Args:
            reduce_func: A function that maps to new_state. It must take two arguments and return a new element
        """
        if self.params.data_mode=='list':
            return [functools.reduce(reduce_func, i[self.params.data_index]) for i in self.params.data]
        return functools.reduce(reduce_func, self.params.data[self.params.data_index])
    
    def repeat(self, repeat_size):
        """Repeats this dataset so each original value is seen count times.
        
        Args:
            repeat_size: representing the number of times the dataset should be repeated.
        """
        assert 'take_while' not in self.params.options, '`repeat` must be placed in `take_while` front.'
        assert isinstance(repeat_size, int) and repeat_size>0, '`repeat_size` type should be int and greater than 0.'
        self.params.data_index = self.params.data_index*(repeat_size+1)
        self.params.options['repeat'].update({self.params.step: {'repeat_size':repeat_size}})
        self.params.step += 1
        return self
        
    def shard(self, shard_size, shard_index):
        """Creates a Dataset that includes only 1/num_shards of this dataset.
        
        Args:
            shard_size: representing the number of shards operating in parallel.
            shard_index: representing the worker index.
        """
        assert 'take_while' not in self.params.options, '`shard` must be placed in `take_while` front.'
        assert isinstance(shard_size, int) and shard_size>0, '`shard_size` type should be int and greater than 0.'
        assert isinstance(shard_index, int) and shard_index>=0, '`shard_index` type should be int and greater than or equal to 0.'
        self.params.data_index = [self.params.data_index[i] for i in range(shard_index, len(self.params.data_index), shard_size)]
        self.params.options['shard'].update({self.params.step: {'shard_size':shard_size, 'shard_index':shard_index}})
        self.params.step += 1
        return self
    
    def shuffle(self, shuffle_size, seed=None):
        """Randomly shuffles the elements of this dataset.
        
        Args:
            shuffle_size: representing the number of elements from this dataset from which the new dataset will sample.
            seed: representing the random seed that will be used to create the distribution.
        """
        assert 'take_while' not in self.params.options, '`shuffle` must be placed in `take_while` front.'
        assert isinstance(shuffle_size, int) and shuffle_size>-2 and shuffle_size!=0, '`shuffle_size` type should be int and greater than 0 or equal to -1.'
        if isinstance(self.params.data_index, list):
            self.params.data_index = pd.Series(index=self.params.data_index, data=1).index
        if shuffle_size > 0:
            t = [self.params.data_index[shuffle_size*i:shuffle_size*(i+1)].to_list() for i in range(len(self.params.data_index)//shuffle_size+1)]
            [random.shuffle(i, random=lambda :((seed if seed is not None else random.randint(1, 99))+self.params.batch)%10/10) for i in t]
            self.params.data_index = list(itertools.chain.from_iterable(t))
        else:
            self.params.data_index = self.params.data_index.to_series().sample(frac=1, random_state=seed).tolist()
        self.params.options['shuffle'].update({self.params.step: {'shuffle_size':shuffle_size, 'seed':seed}})
        self.params.step += 1
        return self
    
    def skip(self, skip_size):
        """Creates a Dataset that skips count elements from this dataset.
        
        Args:
            skip_size: representing the number of elements of this dataset that should be skipped to form the new dataset. 
                       If count is greater than the size of this dataset, the new dataset will contain no elements.
        """
        assert 'take_while' not in self.params.options, '`skip` must be placed in `take_while` front.'
        assert isinstance(skip_size, int) and skip_size>0, '`skip_size` type should be int and greater than 0.'
        self.params.data_index = self.params.data_index[skip_size:]
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
        assert 'take_while' not in self.params.options, '`take` must be placed in `take_while` front.'
        assert isinstance(take_size, int) and take_size>-2 and take_size!=0, '`take_size` type should be int and greater than 0 or equal to -1.'
        if self.params.take != -1:
            self.params.data_index = self.params.data_index[:take_size]
        self.params.options['take'].update({self.params.step: {'take_size':take_size}})
        self.params.step += 1
        return self
    
    def take_while(self, take_func):
        """A transformation that stops dataset iteration based on a take_func.
        
        Args:
            take_func: A function that return True or False
        """
        temp = set()
        index = self.params.data_index[:max([self.params.data_index.index(i) for i in range(len(self.params.data))])+1]
        for r, i in enumerate(index):
            if i in temp:
                continue
            temp.add(i)
            if self.params.data_mode=='list':
                if take_func([j[i] for j in self.params.data]):
                    self.params.data_index = self.params.data_index[:r]
                    break
            else:
                if take_func(self.params.data[i]):
                    self.params.data_index = self.params.data_index[:r]
                    break
        self.params.options['take_while'].update({self.params.step: {'take_func':take_func}})
        self.params.step += 1
        return self
    
    def __iter__(self):
        if self.params.data_mode=='list':
            if 'map' in self.params.options:
                self._batch_func = self._batch_list_map
            else:
                self._batch_func = self._batch_list
        elif 'map' in self.params.options:
            self._batch_func = self._batch_map
        else:
            self._batch_func = self._batch
        return self
    
    def __next__(self):
        loc = self.params.data_index[self.params.batch_size*self.params.batch:self.params.batch_size*(self.params.batch+1)]
        if len(loc)==0:
            raise StopIteration
        elif len(loc)<self.params.batch_size:
            if self.params.drop_remainder:
                raise StopIteration
        self.params.batch += 1
        if 'enumerate' in self.params.options:
            self.params.enumerate += 1
            return (self.params.enumerate-1, self._batch_func(loc))
        return self._batch_func(loc)