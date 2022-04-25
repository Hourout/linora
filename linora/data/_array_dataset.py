import numpy as np

from linora.data._dataset import DataSet

__all__ = ['Dataset']


class file_no(DataSet):
    def __init__(self):
        super(file_no, self).__init__()
        
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
        if 'list' in self.params.data_mode:
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

    
class Dataset(file_no):
    """Represents a potentially large set of elements from dataframe.
    
    Args:
        data: numpy array or tuple of numpy array
    """
    def __init__(self):
        super(Dataset, self).__init__()
        
    def from_tensor(self, data):
        if isinstance(data, tuple):
            for i in data:
                assert len(data[0])==len(i), 'Length needs to be consistent between data.'
        self._params_init()
        self.params.data_mode = 'list' if isinstance(data, tuple) else 'array'
        self.params.data_index = list(range(len(data[0] if isinstance(data, tuple) else data)))
        self.params.data = [np.array(i) for i in data] if isinstance(data, tuple) else np.array(data)
        return self
