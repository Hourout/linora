import random as random1

import numpy as np

from linora.data._dataset import DataSet

__all__ = ['from_tensor', 'range', 'random', 'choose_from_datasets', 'sample_from_datasets']


class BatchFunction():
    def _batch_list_map(self, loc):
        data = list(map(self._params.map_func, *(i[loc] for i in self._params.data)))
        return [np.array(list(map(lambda x:x[i], data))) for i in np.arange(len(data[0]))]
    
    def _batch_list(self, loc):
        return [i[loc] for i in self._params.data]
    
    def _batch_map(self, loc):
        return np.array(list(map(self._params.map_func, self._params.data[loc])))
    
    def _batch(self, loc):
        return self._params.data[loc]

    
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
        if isinstance(data, (int, float, str)):
            self._params.data = np.array([data])
        elif isinstance(data, tuple):
            self._params.data = [np.array(i) for i in data]
        else:
            self._params.data = np.array(data)
        self._params.data_mode = 'list' if isinstance(data, tuple) else 'array'
        self._params.data_index = list(np.arange(len(self._params.data[0] if isinstance(data, tuple) else self._params.data)))
    
    
class range(DataSet, BatchFunction):
    """Creates a Dataset of a step-separated range of values."""
    def __init__(self, *args, **kwargs):
        super(range, self).__init__()
        self._params.data_mode = 'array'
        self._params.data = np.arange(*args, **kwargs)
        self._params.data_index = list(np.arange(len(self._params.data)))


class random(DataSet, BatchFunction):
    """Creates a Dataset of pseudorandom values. The dataset generates a sequence of uniformly distributed integer values.
        
        Args:
            size: shape of output values.
            lower: min random values.
            upper: max random values.
            seed: random seed.
        """
    def __init__(self, size, lower=0, upper=10, seed=None):
        super(random, self).__init__()
        if isinstance(size, int):
            t = (list(np.arange(lower, upper))*(size//int(upper-lower)+1))[:size]
        else:
            t = 1
            for i in size:
                t *= i
            t = (list(np.arange(lower, upper))*(t//int(upper-lower)+1))[:t]
        random1.shuffle(t, random=lambda :((seed if seed is not None else random1.randint(1, 99)))%10/10)
        self._params.data_mode = 'array'
        self._params.data = np.array(t).reshape(size)
        self._params.data_index = list(np.arange(len(self._params.data)))

        
class choose_from_datasets(DataSet, BatchFunction):
    """Creates a dataset that deterministically chooses elements from datasets.

    Args:
        datasets: A non-empty list of la.data.Dataset objects with compatible structure.
        index: A list of scalar between 0 and len(datasets) - 1.
        stop_on_empty_dataset: If True, selection stops if it encounters an empty dataset. 
                               If False, it skips empty datasets. It is recommended to set it to True. 
                               Otherwise, the selected elements start off as the user intends, 
                               but may change as input datasets become empty. 
                               This can be difficult to detect since the dataset starts off looking correct. 
                               Defaults to True.
    """
    def __init__(self, datasets, index, stop_on_empty_dataset=True):
        super(choose_from_datasets, self).__init__()
        if isinstance(datasets[0]._params.data, list):
            self._params.data = []
            for i in np.arange(len(datasets[0]._params.data)):
                self._params.data += [np.concatenate([sets._params.data[i] for sets in datasets])]
        else:
            self._params.data = np.concatenate([sets._params.data for sets in datasets])
        data_index = []
        for r, sets in enumerate(datasets):
            if r==0:
                data_index.append(sets._params.data_index)
            else:
                t = max(data_index[-1])+1
                data_index.append([i+t for i in sets._params.data_index])
        if stop_on_empty_dataset:
            self._params.data_index = []
            for i in index:
                if len(data_index[i])==0:
                    break
                self._params.data_index.append(data_index[i].pop(0))
        else:
            self._params.data_index = [data_index[i].pop(0) for i in index if len(data_index[i])>0]
        self._params.data_mode = 'array'


class sample_from_datasets(DataSet, BatchFunction):
    """Creates a dataset that not deterministically chooses elements from datasets.

    Args:
        datasets: A non-empty list of la.data.Dataset objects with compatible structure.
        weight: A list of len(datasets) floating-point values where weight[i] 
                represents the probability to sample from datasets[i].
        stop_on_empty_dataset: If True, selection stops if it encounters an empty dataset. 
                               If False, it skips empty datasets. It is recommended to set it to True. 
                               Otherwise, the selected elements start off as the user intends, 
                               but may change as input datasets become empty. 
                               This can be difficult to detect since the dataset starts off looking correct. 
    """
    def __init__(self, datasets, weight=None, stop_on_empty_dataset=False):
        super(sample_from_datasets, self).__init__()
        index = np.random.choice(np.arange(len(datasets)), size=sum([len(sets._params.data_index) for sets in datasets])*1.5, p=weight)
        choose_from_datasets(datasets, index, stop_on_empty_dataset)