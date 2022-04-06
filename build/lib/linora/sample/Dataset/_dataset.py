import random

class Params:
    batch = 0
    batch_size = 1
    prefetch_size = 0
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
        """Combines consecutive elements of this dataset into batches.
        Args:
            batch_size: representing the number of consecutive elements of this dataset to combine in a single batch.
            drop_remainder: representing whether the last batch should be dropped in the case it has fewer than batch_size elements; the default behavior is not to drop the smaller batch.
        Returns:
            A Dataset.
        """
        self.params.batch_size = batch_size
        self.params.drop_remainder = drop_remainder
        return self
        
    def prefetch(self, buffer_size):
        """Creates a Dataset that prefetches elements from this dataset.
        Args:
            buffer_size: representing the maximum number of elements that will be buffered when prefetching.
        Returns:
            A Dataset.
        """
        if 'prefetch' not in self.params.rank:
            self.params.rank['prefetch'] = len(self.params.rank)+1
        self.params.prefetch_size = buffer_size
        return self
        
    def repeat(self, count):
        """Repeats this dataset so each original value is seen count times.
        Args:
            count: representing the number of times the dataset should be repeated. 
        Returns:
            A Dataset.
        """
        if 'repeat' not in self.params.rank:
            self.params.rank['repeat'] = len(self.params.rank)+1
        self.params.repeat_size = count
        return self
        
    def shuffle(self, buffer_size, seed=None):
        """Randomly shuffles the elements of this dataset.
        Args:
            buffer_size: representing the number of elements from this dataset from which the new dataset will sample.
            seed: representing the random seed that will be used to create the distribution.
        Returns:
            A Dataset.
        """
        if 'shuffle' not in self.params.rank:
            self.params.rank['shuffle'] = len(self.params.rank)+1
        self.params.shuffle_size = buffer_size
        self.params.shuffle_seed = seed if seed is not None else random.randint(1, 99)
        return self
    
    def skip(self, count):
        """Creates a Dataset that skips count elements from this dataset.
        Args:
            count: representing the number of elements of this dataset that should be skipped to form the new dataset. 
                   If count is greater than the size of this dataset, 
                   the new dataset will contain no elements.
        Returns:
            A Dataset.
        """
        if 'skip' not in self.params.rank:
            self.params.rank['skip'] = len(self.params.rank)+1
        self.params.skip = count
        return self
        
    def take(self, count):
        """Creates a Dataset with at most count elements from this dataset.
        Args:
            count: representing the number of elements of this dataset that should be taken to form the new dataset. 
                   If count is -1, or if count is greater than the size of this dataset, 
                   the new dataset will contain all elements of this dataset.
        Returns:
            A Dataset.
        """
        if 'take' not in self.params.rank:
            self.params.rank['take'] = len(self.params.rank)+1
        self.params.take = count
        return self
    
    def shard(self, num_shards, index):
        """Creates a Dataset that includes only 1/num_shards of this dataset.
        Args:
            num_shards: representing the number of shards operating in parallel.
            index: representing the worker index.
        Returns:
            A Dataset.
        """
        if 'shard' not in self.params.rank:
            self.params.rank['shard'] = len(self.params.rank)+1
        self.params.shard_step = num_shards
        self.params.shard_index = index
        return self
    
    def map(self, map_func):
        """Maps map_func across the elements of this dataset.
        Args:
            map_func: A function mapping a dataset element to another dataset element.
        Returns:
            A Dataset.
        """
        if 'map' not in self.params.rank:
            self.params.rank['map'] = len(self.params.rank)+1
        self.params.map_func = map_func
        self.params.num_parallel_calls = num_parallel_calls
        self.params.deterministic = deterministic
        return self
