import math
import random
import itertools

import pandas as pd
from linora.sample.Dataset._dataset import DataSet

__all__ = ['from_Csv']

class file_yes(DataSet):
    def __init__(self):
        super(file_yes, self).__init__()
        self.params.rank_list = None
        self.params.batch_index = 0
        self.params.batch_file = 0
        self.params.sample = 0
        
    def _repeat(self):
        self.params.data = self.params.data*(self.params.repeat_size+1)
        
    def __iter__(self):
        if self.params.rank_list is None:
            self.params.rank_list = [i[0] for i in sorted(self.params.rank.items(), key=lambda x:x[1])]
        if 'repeat' in self.params.rank_list:
            self._repeat()
            self.params.rank_list.remove('repeat')
        if 'skip' not in self.params.rank_list:
            self.params.skip = None
        self.params.shuffle_size = math.ceil(max(self.params.shuffle_size, self.params.prefetch_size, 1)/self.params.batch_size)*self.params.batch_size
        self.params.df = pd.read_csv(self.params.data[self.params.batch_file], sep=self.params.sep, 
                                     iterator=True, header=self.params.header, skiprows=self.params.skip)
        self.params.values = self.params.df.get_chunk(self.params.shuffle_size).values
        self.params.data_index = list(range(self.params.values.shape[0]))
        random.shuffle(self.params.data_index, random=lambda :(self.params.shuffle_seed+self.params.batch)%100/100)
        self.params.batch_file += 1
        return self
        
    def __next__(self):
        loc = self.params.data_index[self.params.batch_size*self.params.batch_index:self.params.batch_size*(self.params.batch_index+1)]
        if len(loc)<self.params.batch_size:
            self.params.batch_index = 0
            try:
                self.params.values = self.params.df.get_chunk(self.params.shuffle_size).values
            except StopIteration:
                if self.params.batch_file==len(self.params.data):
                    raise StopIteration
                self.params.df = pd.read_csv(self.params.data[self.params.batch_file], sep=self.params.sep, header=self.params.header, iterator=True)
                self.params.values = self.params.df.get_chunk(self.params.shuffle_size).values
                self.params.batch_file += 1
            self.params.data_index = list(range(self.params.values.shape[0]))
            random.shuffle(self.params.data_index, random=lambda :(self.params.shuffle_seed+self.params.batch)%100/100)
            loc = self.params.data_index[0:self.params.batch_size]
            
        self.params.batch += 1
        self.params.batch_index += 1
        self.params.sample += len(loc)
        if self.params.take>0:
            if self.params.sample>self.params.take:
                raise StopIteration
        return self.params.values[loc]
        
class from_Csv(file_yes):
    """Represents a potentially large set of elements from csv file."""
    def __init__(self, file, sep=',', header=None):
        super(from_Csv, self).__init__()
        self.params.sep = sep
        self.params.header = header
        self.params.data = file if isinstance(file, (tuple, list)) else [file]
    
