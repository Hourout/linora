import os
import pathlib
from itertools import chain

import numpy as np
import pandas as pd

from linora.data._dataset import DataSet

__all__ = ['from_tensor', 'range']


class BatchFunction():
    def _batch_list_map(self, loc):
        data = list(map(self.params.map_func, *(i[loc] for i in self.params.data)))
        return [np.array(list(map(lambda x:x[i], data))) for i in range(len(data[0]))]
    
    def _batch_list(self, loc):
        return [i[loc] for i in self.params.data]
    
    def _batch_map(self, loc):
        return np.array(list(map(self.params.map_func, self.params.data[loc])))
    
    def _batch(self, loc):
        return self.params.data[loc]

    
class from_tensor(DataSet, BatchFunction):
    """Creates a Dataset whose elements are slices of the given tensors.
        
        Args:
            data: numpy array or tuple of numpy array
    """
    def __init__(self, data):
        super(from_tensor, self).__init__()
        if isinstance(data, tuple):
            for i in data:
                assert len(data[0])==len(i), 'Length needs to be consistent between data.'
        self._params_init()
        self.params.data_mode = 'list' if isinstance(data, tuple) else 'array'
        self.params.data_index = list(range(len(data[0] if isinstance(data, tuple) else data)))
        self.params.data = [np.array(i) for i in data] if isinstance(data, tuple) else np.array(data)
    
    
class range(DataSet, BatchFunction):
    """Creates a Dataset of a step-separated range of values."""
    def __init__(self, *args, **kwargs):
        super(range, self).__init__()
        self._params_init()
        self.params.data_mode = 'array'
        self.params.data = np.arange(*args, **kwargs)
        self.params.data_index = list(np.arange(len(self.params.data)))
#         self.params.options['range'].update({self.params.step: {'args':args, 'kwargs':kwargs}})
#         self.params.step += 1
