import random

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
    
    def map(self, map_func, num_parallel_calls=None, deterministic=None):
        if 'map' not in self.params.rank:
            self.params.rank['map'] = len(self.params.rank)+1
        self.params.map_func = map_func
        self.params.num_parallel_calls = num_parallel_calls
        self.params.deterministic = deterministic
        return self
