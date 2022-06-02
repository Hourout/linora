import random
import functools
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from linora.gfile._gfile import isfile
from linora.utils._config import Config


class DataSet():
    def __init__(self):
        self._params_init()
    
    def _params_init(self):
        self._params = Config()
        self._params.batch = 0
        self._params.step = 1
        self._params.tensor_mode = 'numpy'
        self._params.index_mode = 'total'
        self._params.data_from = 'tensor'
        self._params.data_index = defaultdict(list)
        self._params.map_func = defaultdict()
        self._params.batch_func = defaultdict()
        self._params.batch_func[self._params.index_mode] = [0, False]
        self._params.options = defaultdict(dict)
        
    def batch(self, batch_size, drop_remainder=False):
        """Combines consecutive elements of this dataset into batches.
        
        Args:
            batch_size: representing the number of consecutive elements of this dataset to combine in a single batch.
            drop_remainder: representing whether the last batch should be dropped in the case it has fewer than batch_size elements; 
                            the default behavior is not to drop the smaller batch.
        """
        assert isinstance(batch_size, int) and batch_size>0, '`batch_size` type should be int and greater than 0.'
        self._params.batch_func[self._params.index_mode] = [batch_size, drop_remainder]
        self._params.options['batch'].update({self._params.step: {'batch_size':batch_size, 'drop_remainder':drop_remainder}})
        self._params.step += 1
        return self
    
    def cardinality(self):
        """Returns the cardinality of the dataset, if known."""
        return len(self._params.data_index[self._params.index_mode])
    
    def concatenate(self, datasets):
        """Creates a Dataset by concatenating the given dataset with this dataset.
        
        Args:
            datasets: la.data.Dataset or list of la.data.Dataset to be concatenated.
        """
        assert 'take_while' not in self._params.options, '`concatenate` must be placed in `take_while` front.'
        if not isinstance(datasets, list):
            self._concatenate(datasets)
        else:
            for dataset in datasets:
                assert self._params.data_mode==dataset._params.data_mode, 'The data types of the two data sets are inconsistent.'
            for dataset in datasets:
                self._concatenate(dataset)
        self._params.options['concatenate'].update({self._params.step: None})
        self._params.step += 1
        return self
    
    def _concatenate(self, dataset):
        t = len(self._params.data[0]) if 'list' in self._params.data_mode else len(self._params.data)
        if 'list' in self._params.data_mode:
            assert len(self._params.data)==len(dataset._params.data), 'Width needs to be consistent between data.'
            self._params.data = [np.concatenate([self._params.data[i], dataset._params.data[i]]) for i in range(len(self._params.data))]
        else:
            self._params.data = np.concatenate([self._params.data, dataset._params.data])
        self._params.data_index[self._params.index_mode] += [i+t for i in dataset._params.data_index[dataset._params.index_mode]]
        
    def drop(self, name):
        """Drop current dataset.
        
        Args:
            name: drop dataset name.
        """
        assert name!='total', "`name` can't be 'total'."
        if name in self._params.data_index:
            self._params.data_index.pop(name)
            
        if name in self._params.map_func:
            self._params.map_func.pop(name)
            
        if name in self._params.batch_func:
            self._params.batch_func.pop(name)
            
        if self._params.index_mode==name:
            self._params.index_mode = 'total'
        return self
        
    def enumerate(self, start=0):
        """Enumerates the elements of this dataset.
        
        Args:
            start: int, representing the start value for enumeration.
        """
        self._params.enumerate = start
        self._params.options['enumerate'].update({self._params.step: {'start':start}})
        self._params.step += 1
        return self
    
    def filter(self, filter_func):
        """A transformation that filter dataset based on a filter_func.
        
        Args:
            filter_func: A function that return True or False, datasets that are kept as True.
        """
        if self._params.data_mode=='list':
            filter_list = [i for i in range(len(self._params.data[0])) if filter_func([j[i] for j in self._params.data])]
        else:
            filter_list = [r for r, i in enumerate(self._params.data) if filter_func(i)]
        if filter_list:
            self._params.data_index[self._params.index_mode] = [i for i in self._params.data_index[self._params.index_mode] if i in filter_list]
        self._params.options['filter'].update({self._params.step: {'filter_func':filter_func}})
        self._params.step += 1
        return self
    
    def get(self, name):
        """Select current dataset.
        
        Args:
            name: split dataset name.
        """
        assert name in self._params.data_index, '`name` not in split dataset.'
        if self._params.batch==len(self._params.data_index[self._params.index_mode]):
            self._params.batch_size=0
        self._params.index_mode = name
        self._params.batch = 0
        return self
    
#     def join(self, join_dict, drop_exist_dataset=True):
#         """Join Dataset.
        
#         Args:
#             join_dict: dict, {data_name:Dataset}, eg.{'train':la.data.Dataset.from_tensor()}.
#         """
#         for i in join_dict:
#             assert self._params.data_mode==dataset._params.data_mode, 'The data types of the two data sets are inconsistent.'
            
#         return self
    
    def map(self, map_func, map_size=8):
        """Maps map_func across the elements of this dataset.
        
        Args:
            map_func: A function mapping a dataset element to another dataset element.
            map_size: representing the number elements to process asynchronously in parallel. 
        """
        assert isinstance(map_size, int) and map_size>0, '`map_size` type should be int and greater than 0.'
        self._params.map_func[self._params.index_mode] = [map_func, map_size]
        self._params.options['map'].update({self._params.step: {'map_func':map_func, 'map_size':map_size}})
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
        assert 'take_while' not in self._params.options, '`prefetch` must be placed in `take_while` front.'
        assert isinstance(prefetch_size, int) and prefetch_size>0, '`prefetch_size` type should be int and greater than 0.'
        self._params.options['prefetch'].update({self._params.step: {'prefetch_size':prefetch_size}})
        self._params.step += 1
        return self
    
    def reduce(self, reduce_func):
        """Reduces the input dataset to a single element.
        
        Args:
            reduce_func: A function that maps to new_state. It must take two arguments and return a new element
        """
        if self._params.data_mode=='list':
            return [functools.reduce(reduce_func, i[self._params.data_index[self._params.index_mode]]) for i in self._params.data]
        return functools.reduce(reduce_func, self._params.data[self._params.data_index[self._params.index_mode]])
    
    def rename(self, name_dict):
        """Rename current dataset.
        
        Args:
            name_dict: rename dataset name dict, eg.{'train':'train_set'}.
        """
        for name in name_dict:
            assert name!='total', "`name` can't be 'total'."
        for name in name_dict:
            if name in self._params.data_index:
                self._params.data_index[name_dict[name]] = self._params.data_index.pop(name)
            
            if name in self._params.map_func:
                self._params.map_func[name_dict[name]] = self._params.map_func.pop(name)

            if name in self._params.batch_func:
                self._params.batch_func[name_dict[name]] = self._params.batch_func.pop(name)

            if self._params.index_mode==name:
                self._params.index_mode = name_dict[name]
        return self
    
    def repeat(self, repeat_size):
        """Repeats this dataset so each original value is seen count times.
        
        Args:
            repeat_size: representing the number of times the dataset should be repeated.
        """
        assert 'take_while' not in self._params.options, '`repeat` must be placed in `take_while` front.'
        assert isinstance(repeat_size, int) and repeat_size>0, '`repeat_size` type should be int and greater than 0.'
        self._params.data_index[self._params.index_mode] = self._params.data_index[self._params.index_mode]*(repeat_size+1)
        self._params.options['repeat'].update({self._params.step: {'repeat_size':repeat_size}})
        self._params.step += 1
        return self
    
    def shard(self, shard_size, shard_index):
        """Creates a Dataset that includes only 1/num_shards of this dataset.
        
        Args:
            shard_size: representing the number of shards operating in parallel.
            shard_index: representing the worker index.
        """
        assert 'take_while' not in self._params.options, '`shard` must be placed in `take_while` front.'
        assert isinstance(shard_size, int) and shard_size>0, '`shard_size` type should be int and greater than 0.'
        assert isinstance(shard_index, int) and shard_index>=0, '`shard_index` type should be int and greater than or equal to 0.'
        self._params.data_index[self._params.index_mode] = [self._params.data_index[self._params.index_mode][i] for i in range(shard_index, len(self._params.data_index[self._params.index_mode]), shard_size)]
        self._params.options['shard'].update({self._params.step: {'shard_size':shard_size, 'shard_index':shard_index}})
        self._params.step += 1
        return self
    
    def shuffle(self, shuffle_size, seed=None):
        """Randomly shuffles the elements of this dataset.
        
        Args:
            shuffle_size: representing the number of elements from this dataset from which the new dataset will sample.
            seed: representing the random seed that will be used to create the distribution.
        """
        assert 'take_while' not in self._params.options, '`shuffle` must be placed in `take_while` front.'
        assert isinstance(shuffle_size, int) and shuffle_size>-2 and shuffle_size!=0, '`shuffle_size` type should be int and greater than 0 or equal to -1.'
        if isinstance(self._params.data_index[self._params.index_mode], list):
            self._params.data_index[self._params.index_mode] = pd.Series(index=self._params.data_index[self._params.index_mode], data=1).index
        if shuffle_size > 0:
            t = [self._params.data_index[self._params.index_mode][shuffle_size*i:shuffle_size*(i+1)].to_list() for i in range(len(self._params.data_index[self._params.index_mode])//shuffle_size+1)]
            [random.shuffle(i, random=lambda :((seed if seed is not None else random.randint(1, 99))+self._params.batch)%10/10) for i in t]
            self._params.data_index[self._params.index_mode] = list(itertools.chain.from_iterable(t))
        else:
            self._params.data_index[self._params.index_mode] = self._params.data_index[self._params.index_mode].to_series().sample(frac=1, random_state=seed).tolist()
        self._params.options['shuffle'].update({self._params.step: {'shuffle_size':shuffle_size, 'seed':seed}})
        self._params.step += 1
        return self
    
    def skip(self, skip_size):
        """Creates a Dataset that skips count elements from this dataset.
        
        Args:
            skip_size: representing the number of elements of this dataset that should be skipped to form the new dataset. 
                       If count is greater than the size of this dataset, the new dataset will contain no elements.
        """
        assert 'take_while' not in self._params.options, '`skip` must be placed in `take_while` front.'
        assert isinstance(skip_size, int) and skip_size>0, '`skip_size` type should be int and greater than 0.'
        self._params.data_index[self._params.index_mode] = self._params.data_index[self._params.index_mode][skip_size:]
        self._params.options['skip'].update({self._params.step: {'skip_size':skip_size}})
        self._params.step += 1
        return self
    
    def split(self, split_dict, shuffle=True, seed=None):
        """Split Dataset.
        
        Args:
            split_dict: dict, {data_name:data_rate}, eg.{'train':0.7, 'test':0.3}.
            shuffle: whether randomly shuffles the elements of this dataset
            seed: random seed.
        """
        t = sum(split_dict[i] for i in split_dict)
        t = {i:split_dict[i]/t for i in split_dict}
        for i in t:
            assert i!='total', "`split_dict` key can't be 'total'."
        if self._params.data_from in ['from_folder', 'from_class_folder']:
            if isinstance(self._params.data, list):
                label = self._params.data[1][self._params.data_index[self._params.index_mode]]
                index = np.array(self._params.data_index[self._params.index_mode])
                for i in np.unique(label):
                    index1 = index[label==i].tolist()
                    n = 0
                    for j in t:
                        self._params.data_index[j] += index1[n:n+int(t[j]*len(index1))]
                        n += int(t[j]*len(index1))
                if shuffle:
                    for i in t:
                        self._params.data_index[i] = pd.Series(self._params.data_index[i]).sample(frac=1, random_state=seed).tolist()
            else:
                self._split(t, shuffle, seed)
        else:
            self._split(t, shuffle, seed)
        return self
    
    def _split(self, t, shuffle, seed):
        if shuffle:
            index = pd.Series(self._params.data_index[self._params.index_mode]).sample(frac=1, random_state=seed).tolist()
        else:
            index = self._params.data_index[self._params.index_mode]
        n = 0
        for i in t:
            self._params.data_index[i] = index[n:n+int(t[i]*len(index))]
            n += int(t[i]*len(index))
        
    def take(self, take_size):
        """Creates a Dataset with at most count elements from this dataset.
        
        Args:
            take_size: representing the number of elements of this dataset that should be taken to form the new dataset. 
                       If count is -1, or if count is greater than the size of this dataset, 
                       the new dataset will contain all elements of this dataset.
        """
        assert 'take_while' not in self._params.options, '`take` must be placed in `take_while` front.'
        assert isinstance(take_size, int) and take_size>-2 and take_size!=0, '`take_size` type should be int and greater than 0 or equal to -1.'
        if take_size != -1:
            self._params.data_index[self._params.index_mode] = self._params.data_index[self._params.index_mode][:take_size]
        self._params.options['take'].update({self._params.step: {'take_size':take_size}})
        self._params.step += 1
        return self
    
    def take_while(self, take_func):
        """A transformation that stops dataset iteration based on a take_func.
        
        Args:
            take_func: A function that return True or False
        """
        temp = set()
        index = self._params.data_index[self._params.index_mode][:max([self._params.data_index[self._params.index_mode].index(i) for i in range(len(self._params.data))])+1]
        for r, i in enumerate(index):
            if i in temp:
                continue
            temp.add(i)
            if 'list' in self._params.data_mode:
                if take_func([j[i] for j in self._params.data]):
                    self._params.data_index[self._params.index_mode] = self._params.data_index[self._params.index_mode][:r]
                    break
            else:
                if take_func(self._params.data[i]):
                    self._params.data_index[self._params.index_mode] = self._params.data_index[self._params.index_mode][:r]
                    break
        self._params.options['take_while'].update({self._params.step: {'take_func':take_func}})
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
    
    def unbatch(self):
        """Splits elements of a dataset into multiple elements."""
        assert not isinstance(self._params.data, list), 'Input data cannot be a tuple.'
        assert self._params.index_mode=='total', f'{self._params.index_mode} dataset not supported.'
        self._params.data = np.array(list(itertools.chain.from_iterable(self._params.data)))
        self._params.data_index[self._params.index_mode] = list(range(len(self._params.data)))
        return self
    
    def unique(self):
        """A transformation that discards duplicate elements of a Dataset."""
        if isinstance(self._params.data, list):
            return tuple([np.unique(i) for i in self._params.data[self._params.data_index[self._params.index_mode]]])
        else:
            return np.unique(self._params.data[self._params.data_index[self._params.index_mode]])
    
    def _to_tensor(self, data):
        if self._params.tensor_mode=='numpy':
            return data
        return self._params.framework(data)
    
    def _data_mode(self):
        self._params.data_mode = 'list_array' if isinstance(self._params.data, list) else 'array'
        t = [i[0] for i in self._params.data] if isinstance(self._params.data, list) else self._params.data[0]
        if isinstance(t, str):
            if isfile(t):
                if t.split('.')[-1] in ['png', 'jpg', 'jpeg', 'bmp', 'rgb', 'tif', 'tiff', 'webp']:
                    self._params.data_mode = 'image'
        elif isinstance(t, list):
            for i in t:
                if isinstance(i, str):
                    if isfile(i):
                        if i.split('.')[-1] in ['png', 'jpg', 'jpeg', 'bmp', 'rgb', 'tif', 'tiff', 'webp']:
                            self._params.data_mode = 'list_image'
    
    def __iter__(self):
        if 'list' in self._params.data_mode:
            if 'map' in self._params.options:
                self._batch_func = self._batch_list_map
            else:
                self._batch_func = self._batch_list
        elif 'map' in self._params.options:
            self._batch_func = self._batch_map
        else:
            self._batch_func = self._batch
        return self
    
    def __next__(self):
        if self._params.batch_func[self._params.index_mode][0]==0:
            self._params.batch_func[self._params.index_mode][0] = 1
            self._params.batch = len(self._params.data_index[self._params.index_mode])
            if 'enumerate' in self._params.options:
                self._params.enumerate += 1
                return (self._params.enumerate-1, self._to_tensor(self._batch_func(self._params.data_index[self._params.index_mode])))
            return self._to_tensor(self._batch_func(self._params.data_index[self._params.index_mode]))
        loc = self._params.data_index[self._params.index_mode][self._params.batch_func[self._params.index_mode][0]*self._params.batch:self._params.batch_func[self._params.index_mode][0]*(self._params.batch+1)]
        if len(loc)==0:
            raise StopIteration
        elif len(loc)<self._params.batch_func[self._params.index_mode][0]:
            if self._params.batch_func[self._params.index_mode][1]:
                raise StopIteration
        self._params.batch += 1
        if 'enumerate' in self._params.options:
            self._params.enumerate += 1
            return (self._params.enumerate-1, self._to_tensor(self._batch_func(loc)))
        return self._to_tensor(self._batch_func(loc))