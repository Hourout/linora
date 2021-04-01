import random
import itertools

__all__ = ['DataSet']

class Params:
    batch = 0
    batch_size = 1
    repeat_size = 0
    shuffle_size = 0
    rank = dict()
    skip = 0
    take = -1
    shard_step = 1
    shard_index = 0

class DataSet():
    def __init__(self):
        self.params = Params()
        
#     def from_csv(self):
#         return self
    
    def from_array(self, data):
        self.params.data_mode = 'array_list' if isinstance(data, (list, tuple)) else 'array'
        self.params.data_index = pd.Series(range(data[0].shape[0] if isinstance(data, (list, tuple)) else data.shape[0])).index
        self.params.data = data
        return self
    
    def from_Dataframe(self, data):
        self.params.data_mode = 'Dataframe_list' if isinstance(data, (list, tuple)) else 'Dataframe'
        self.params.data_index = data[0].index if isinstance(data, (list, tuple)) else data.index
        self.params.data = [i.values for i in data] if isinstance(data, (list, tuple)) else data.values
        return self
    
#     def from_imagelist(self, data):
#         self.params.data_mode = 'imagelist'
#         self.params.data_index = pd.Series(range(len(data))).index
#         self.params.data = data
#         return self
    
    def batch(self, batch_size, drop_remainder=False):
        self.params.batch_size = batch_size
        self.params.drop_remainder = drop_remainder
        return self
        
    def prefetch(self, buffer_size):
        if 'prefetch' not in self.params.rank:
            self.params.rank['prefetch'] = len(self.params.rank)+1
        self.params.prefetch_size = buffer_size
        return self
        
    def repeat(self, count):
        if 'repeat' not in self.params.rank:
            self.params.rank['repeat'] = len(self.params.rank)+1
        self.params.repeat_size = count
        return self
        
    def shuffle(self, buffer_size, seed=None):
        if 'shuffle' not in self.params.rank:
            self.params.rank['shuffle'] = len(self.params.rank)+1
        self.params.shuffle_size = buffer_size
        self.params.shuffle_seed = seed if seed is not None else random.randint(1, 99)
        return self
    
    def skip(self, count):
        if 'skip' not in self.params.rank:
            self.params.rank['skip'] = len(self.params.rank)+1
        self.params.skip = count
        return self
        
    def take(self, count):
        if 'take' not in self.params.rank:
            self.params.rank['take'] = len(self.params.rank)+1
        self.params.take = count
        return self
    
    def shard(self, num_shards, index):
        if 'shard' not in self.params.rank:
            self.params.rank['shard'] = len(self.params.rank)+1
        self.params.shard_step = num_shards
        self.params.shard_index = index
        return self
        
    def _repeat(self):
        self.params.data_index = self.params.data_index.append([self.params.data_index for i in range(self.params.repeat_size)])
        
    def _shuffle(self):
        if isinstance(self.params.data_index, list):
            self.params.data_index = pd.Series(index=self.params.data_index, data=1).index
        if self.params.shuffle_size > 0:
            t = [self.params.data_index[self.params.shuffle_size*i:self.params.shuffle_size*(i+1)].to_list() for i in range(len(self.params.data_index)//self.params.shuffle_size+1)]
            [random.shuffle(i, random=lambda :self.params.shuffle_seed%100/100) for i in t]
            self.params.data_index = list(itertools.chain.from_iterable(t))
        elif self.params.shuffle_size == -1:
            self.params.data_index = self.params.data_index.to_series().sample(frac=1, random_state=self.params.shuffle_seed).tolist()
            
    def _skip(self):
        self.params.data_index = self.params.data_index[self.params.skip:] if self.params.skip != -1 else []
        
    def _take(self):
        self.params.data_index = self.params.data_index[:self.params.take] if self.params.take != -1 else self.params.data_index
    
    def _shard(self):
        self.params.data_index = [self.params.data_index[i] for i in range(self.params.shard_index, len(self.params.data_index), self.params.shard_step)]
    
    def __iter__(self):
        self.params.rank_list = [i[0] for i in sorted(self.params.rank.items(), key=lambda x:x[1])]
        if 'shuffle' in self.params.rank_list and 'repeat' in self.params.rank_list:
            if self.params.rank_list.index('shuffle')<self.params.rank_list.index('repeat'):
                self.params.rank_list.remove('shuffle')
                self.params.rank_list.append('shuffle')

        if self.params.data_mode in ['Dataframe', 'Dataframe_list', 'array_list', 'array']:
            if 'prefetch' in self.params.rank_list:
                self.params.rank_list.remove('prefetch')
            for i in self.params.rank_list:
                if i == 'skip':
                    self._skip()
                elif i == 'repeat':
                    self._repeat()
                elif i == 'take':
                    self._take()
                elif i == 'shard':
                    self._shard()
                elif i == 'shuffle':
                    self._shuffle()
#             self.params.data_index = self.params.data_index.append([self.params.data_index for i in range(self.params.repeat_size)])
#             if self.params.shuffle_size > 0:
#                 t = [self.params.data_index[self.params.shuffle_size*i:self.params.shuffle_size*(i+1)].to_list() for i in range(len(self.params.data_index)//self.params.shuffle_size+1)]
#                 [random.shuffle(i, random=lambda :self.params.shuffle_seed%100/100) for i in t]
#                 self.params.data_index = list(itertools.chain.from_iterable(t))
#             elif self.params.shuffle_size == -1:
#                 self.params.data_index = self.params.data_index.to_series().sample(frac=1, random_state=self.params.shuffle_seed).tolist()
#             self.params.data_index = self.params.data_index[self.params.skip:] if self.params.skip != -1 else []
#             self.params.data_index = self.params.data_index[:self.params.take] if self.params.take != -1 else self.params.data_index
#             self.params.data_index = [self.params.data_index[i] for i in range(self.params.shard_index, len(self.params.data_index), self.params.shard_step)]
#         print(len(self.params.data_index))
        return self
    
    def __next__(self):
        if self.params.data_mode in ['Dataframe', 'Dataframe_list', 'array_list', 'array']:
            loc = self.params.data_index[self.params.batch_size*self.params.batch:self.params.batch_size*(self.params.batch+1)]
            if len(loc)==0:
                raise StopIteration
            elif len(loc)<self.params.batch_size:
                if self.params.drop_remainder:
                    raise StopIteration
                else:
                    self.params.batch += 1
                    return [i[loc] for i in self.params.data] if 'list' in self.params.data_mode else self.params.data[loc]
            self.params.batch += 1
            return [i[loc] for i in self.params.data] if 'list' in self.params.data_mode else self.params.data[loc]
      
    
# import pandas as pd
# df = pd.DataFrame({'a':[random.choice([0,1,2,3]) for i in range(100)],
#                    'b':[random.choice([4,5,6,7]) for i in range(100)],
#                   'c':[random.choice([0,1]) for i in range(100)]})
# ds = DataSet().from_Dataframe((df,df.c)).shuffle(30).batch(4,)
# ds = DataSet().from_array((df.values,df.c.values)).skip(20).shuffle(20, 34).repeat(1000).prefetch(20).batch(5,)
# for i in ds:
#     dd = i
#     print(i)
#     if j==
#     break
# df.sample(frac=1, random_state)
