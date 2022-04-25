import numpy as np

from linora.data._dataset import DataSet
from linora.parallel._thread import ThreadLoom

__all__ = ['Dataset', 'ImageDataset']


class Dataset(DataSet):
    """Represents a potentially large set of elements from dataframe."""
    def __init__(self):
        super(Dataset, self).__init__()
        
    def from_tensor(self, data):
        """Creates a Dataset whose elements are slices of the given tensors.
        
        Args:
            data: numpy array or tuple of numpy array
        """
        if isinstance(data, tuple):
            for i in data:
                assert len(data[0])==len(i), 'Length needs to be consistent between data.'
        self._params_init()
        self.params.data_mode = 'list' if isinstance(data, tuple) else 'array'
        self.params.data_index = list(range(len(data[0] if isinstance(data, tuple) else data)))
        self.params.data = [np.array(i) for i in data] if isinstance(data, tuple) else np.array(data)
        return self
    
    def _batch_list_map(self, loc):
        data = list(map(self.params.map_func, *(i[loc] for i in self.params.data)))
        return [np.array(list(map(lambda x:x[i], data))) for i in range(len(data[0]))]
    
    def _batch_list(self, loc):
        return [i[loc] for i in self.params.data]
    
    def _batch_map(self, loc):
        return np.array(list(map(self.params.map_func, self.params.data[loc])))
    
    def _batch(self, loc):
        return self.params.data[loc]


class ImageDataset(DataSet):
    """Represents a potentially large set of elements from image path."""
    def __init__(self):
        super(ImageDataset, self).__init__()
        
    def from_tensor(self, data):
        """Creates a Dataset whose elements are slices of the given tensors.
        
        Args:
            data: image path or tuple of image path
        """
        if isinstance(data, tuple):
            for i in data:
                assert len(data[0])==len(i), 'Length needs to be consistent between data.'
        self._params_init()
        self.params.data_mode = 'list' if isinstance(data, tuple) else 'image'
        self.params.data_index = list(range(len(data[0] if isinstance(data, tuple) else data)))
        self.params.data = [np.array(i) for i in data] if isinstance(data, tuple) else np.array(data)
        return self
    
    def _batch_list_map(self, loc):
        loom = ThreadLoom(self.params.map_size)
        for i in loc:
            loom.add_function(self.params.map_func, [j[i] for j in self.params.data])
        t = loom.execute()
        for i in t:
            if t[i]['got_error']:
                continue
            if isinstance(t[i]['output'], (list, tuple)):
                return [np.concatenate([np.expand_dims(t[j]['output'][k], 0) for j in t if not t[j]['got_error']]) for k in range(len(t[i]['output']))]
            else:
                return np.concatenate([np.expand_dims(t[j]['output'], 0) for j in t if not t[j]['got_error']])
    
    def _batch_list(self, loc):
        return [i[loc] for i in self.params.data]
    
    def _batch_map(self, loc):
        loom = ThreadLoom(self.params.map_size)
        for i in loc:
            loom.add_function(self.params.map_func, [self.params.data[i]])
        t = loom.execute()
        print(i)
        for i in t:
            if t[i]['got_error']:
                continue
            if isinstance(t[i]['output'], (list, tuple)):
                return [np.concatenate([np.expand_dims(t[j]['output'][k], 0) for j in t if not t[j]['got_error']]) for k in range(len(t[i]['output']))]
            else:
                return np.concatenate([np.expand_dims(t[j]['output'], 0) for j in t if not t[j]['got_error']])
    
    def _batch(self, loc):
        return self.params.data[loc]