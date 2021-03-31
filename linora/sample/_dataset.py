class Params:
    batch = 0
    batch_size = 1
    repeat_size = 0
    shuffle_size = 0
    rank = dict()

class DataSet():
    def __init__(self):
        self.params = Params()
        
    def from_csv(self):
        return self
    
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
    
    def from_imagelist(self, data):
        self.params.data_mode = 'imagelist'
        self.params.data_index = pd.Series(range(len(data))).index
        self.params.data = data
        return self
    
    def batch(self, batch_size, drop_remainder=False):
        self.params.batch_size = batch_size
        self.params.drop_remainder = drop_remainder
        return self
        
    def prefetch(self, buffer_size):
#         if 'prefetch' not in self.param.rank:
#             self.params.rank['prefetch'] = len(self.params.rank)+1
        self.params.prefetch_size = buffer_size
        return self
        
    def repeat(self, count):
#         if 'repeat' not in self.params.rank:
#             self.params.rank['repeat'] = len(self.params.rank)+1
        self.params.repeat_size = count
        return self
        
    def shuffle(self, buffer_size, seed=None):
#         if 'shuffle' not in self.params.rank:
#             self.params.rank['shuffle'] = len(self.params.rank)+1
        self.params.shuffle_size = buffer_size
        self.params.shuffle_seed = seed if seed is not None else random.randint(1, 99)
        return self
        
    def __iter__(self):
#         self.params.rank_list = [i[0] for i in sorted(self.params.rank.items(), key=lambda x:x[1])]
        if self.params.data_mode in ['Dataframe', 'array']:
            self.params.data_index = self.params.data_index.append([self.params.data_index for i in range(self.params.repeat_size)])
            if self.params.shuffle_size > 0:
                t = [self.params.data_index[self.params.shuffle_size*i:self.params.shuffle_size*(i+1)].to_list() for i in range(len(self.params.data_index)//self.params.shuffle_size+1)]
                [random.shuffle(i, random=lambda :self.params.shuffle_seed%100/100) for i in t]
                self.params.data_index = list(itertools.chain.from_iterable(t))
            elif self.params.shuffle_size == -1:
                self.params.data_index = self.params.data_index.to_series().sample(frac=1, random_state=self.params.shuffle_seed).tolist()
#         raise
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
        
        
