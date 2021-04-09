import random
import itertools

import pandas as pd
from linora.sample.Dataset._dataset import DataSet

__all__ = ['from_Dataframe', 'from_Array', 'from_Image']

        
class file_no(DataSet):
    def __init__(self):
        super(file_no, self).__init__()
        
    def _repeat(self):
        self.params.data_index = self.params.data_index.append([self.params.data_index for i in range(self.params.repeat_size)])
        
    def _shuffle(self):
        if isinstance(self.params.data_index, list):
            self.params.data_index = pd.Series(index=self.params.data_index, data=1).index
        if self.params.shuffle_size > 0:
            t = [self.params.data_index[self.params.shuffle_size*i:self.params.shuffle_size*(i+1)].to_list() for i in range(len(self.params.data_index)//self.params.shuffle_size+1)]
            [random.shuffle(i, random=lambda :(self.params.shuffle_seed+self.params.batch)%100/100) for i in t]
            self.params.data_index = list(itertools.chain.from_iterable(t))
        elif self.params.shuffle_size == -1:
            self.params.data_index = self.params.data_index.to_series().sample(frac=1, random_state=self.params.shuffle_seed).tolist()
            
    def _skip(self):
        self.params.data_index = self.params.data_index[self.params.skip:] if self.params.skip != -1 else []
        
    def _take(self):
        self.params.data_index = self.params.data_index[:self.params.take] if self.params.take != -1 else self.params.data_index
    
    def _shard(self):
        self.params.data_index = [self.params.data_index[i] for i in range(self.params.shard_index, len(self.params.data_index), self.params.shard_step)]

    def _batch_list_map(self, loc):
        data = list(map(self.params.map_func, *(i[loc] for i in self.params.data)))
        return [np.array(list(map(lambda x:x[i], data))) for i in range(len(data[0]))]
    
    def _batch_list(self, loc):
        return [i[loc] for i in self.params.data]
    
    def _batch_map(self, loc):
        return np.array(list(map(self.params.map_func, self.params.data[loc])))
    
    def _batch(self, loc):
        return self.params.data[loc]
    
    def __iter__(self):
        self.params.rank_list = [i[0] for i in sorted(self.params.rank.items(), key=lambda x:x[1])]
        if 'shuffle' in self.params.rank_list and 'repeat' in self.params.rank_list:
            if self.params.rank_list.index('shuffle')<self.params.rank_list.index('repeat'):
                self.params.rank_list.remove('shuffle')
                self.params.rank_list.append('shuffle')
        if  'list' in self.params.data_mode:
            if 'map' in self.params.rank_list:
                self.batch_func = self._batch_list_map
            else:
                self.batch_func = self._batch_list
        elif 'map' in self.params.rank_list:
            self.batch_func = self._batch_map
        else:
            self.batch_func = self._batch
        for i in self.params.rank_list:
            if i == 'shuffle':
                self._shuffle()
            elif i == 'repeat':
                self._repeat()
            elif i == 'skip':
                self._skip()
            elif i == 'take':
                self._take()
            elif i == 'shard':
                self._shard()
        return self
    
    def __next__(self):
        loc = self.params.data_index[self.params.batch_size*self.params.batch:self.params.batch_size*(self.params.batch+1)]
        if len(loc)==0:
            raise StopIteration
        elif len(loc)<self.params.batch_size:
            if self.params.drop_remainder:
                raise StopIteration
            else:
                self.params.batch += 1
                return self.batch_func(loc)
        self.params.batch += 1
        return self.batch_func(loc)

class from_Dataframe(file_no):
    """Represents a potentially large set of elements from dataframe."""
    def __init__(self, data):
        super(from_Dataframe, self).__init__()
        self.params.data_mode = 'Dataframe_list' if isinstance(data, tuple) else 'Dataframe'
        self.params.data_index = data[0].index if isinstance(data, tuple) else data.index
        self.params.data = [i.values for i in data] if isinstance(data, tuple) else data.values

class from_Array(file_no):
    """Represents a potentially large set of elements from array."""
    def __init__(self, data):
        super(from_Array, self).__init__()
        self.params.data_mode = 'array_list' if isinstance(data, tuple) else 'array'
        self.params.data_index = pd.Series(range(data[0].shape[0] if isinstance(data, tuple) else data.shape[0])).index
        self.params.data = data
    
class from_Image(file_no):
    """Represents a potentially large set of elements from image file list."""
    def __init__(self, data):
        super(from_Image, self).__init__()
        self.params.data_mode = 'image_list' if isinstance(data, tuple) else 'image'
        self.params.data_index = pd.Series(range(len(data[0] if isinstance(data, tuple) else data))).index
        self.params.data = [pd.Series(i).values for i in data] if isinstance(data, tuple) else pd.Series(data).values
