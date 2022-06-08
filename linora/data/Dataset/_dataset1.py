import os
import pathlib
import random as random1
from itertools import chain

import numpy as np
import pandas as pd

from linora import gfile
from linora.data.Dataset._data import DataSet
from linora.parallel._thread import ThreadLoom

__all__ = ['from_tensor', 'from_folder', 'from_class_folder', 'range', 'random', 
           'choose_from_datasets', 'sample_from_datasets']


class BatchFunction():
    def _batch_list_map(self, loc):
        if 'array' in self._params.data_mode:
            data = list(map(self._params.map[self._params.mode][0], *(i[loc] for i in self._params.data[self._params.mode1])))
            return [np.array(list(map(lambda x:x[i], data))) for i in np.arange(len(data[0]))]
        loom = ThreadLoom(self._params.map[self._params.mode][1])
        for i in loc:
            loom.add_function(self._params.map[self._params.mode][0], [j[i] for j in self._params.data[self._params.mode1]])
        t = loom.execute()
        for i in t:
            if t[i]['got_error']:
                continue
            if isinstance(t[i]['output'], (list, tuple)):
                return [np.concatenate([np.expand_dims(t[j]['output'][k], 0) for j in t if not t[j]['got_error']]) for k in np.arange(len(t[i]['output']))]
            else:
                return np.concatenate([np.expand_dims(t[j]['output'], 0) for j in t if not t[j]['got_error']])
    
    def _batch_list(self, loc):
        return [i[loc] for i in self._params.data[self._params.mode1]]
    
    def _batch_map(self, loc):
        if 'array' in self._params.data_mode:
            return np.array(list(map(self._params.map[self._params.mode][0], self._params.data[self._params.mode1][loc])))
        loom = ThreadLoom(self._params.map[self._params.mode][1])
        for i in loc:
            loom.add_function(self._params.map[self._params.mode][0], [self._params.data[self._params.mode1][i]])
        t = loom.execute()
        for i in t:
            if t[i]['got_error']:
                continue
            if isinstance(t[i]['output'], (list, tuple)):
                return [np.concatenate([np.expand_dims(t[j]['output'][k], 0) for j in t if not t[j]['got_error']]) for k in np.arange(len(t[i]['output']))]
            else:
                return np.concatenate([np.expand_dims(t[j]['output'], 0) for j in t if not t[j]['got_error']])
    
    def _batch(self, loc):
        return self._params.data[self._params.mode1][loc]


class from_tensor(DataSet, BatchFunction):
    """Creates a Dataset whose elements are slices of the given tensors.
        
        Args:
            data: numpy array or tuple of numpy array
    """
    def __init__(self, data):
        super(from_tensor, self).__init__()
        if isinstance(data, (int, float, str)):
            self._params.data[self._params.mode1] = np.array([data])
        elif isinstance(data, tuple):
            for i in data:
                assert len(data[0])==len(i), 'Length needs to be consistent between data.'
            self._params.data[self._params.mode1] = [np.array(i) for i in data]
        else:
            self._params.data[self._params.mode1] = np.array(data)
        self._params.index[self._params.mode] = list(np.arange(len(self._params.data[self._params.mode1][0] if isinstance(data, tuple) else self._params.data[self._params.mode1])))
        self._data_mode()
    

class from_folder(DataSet, BatchFunction):
    """Construct an image dataset label index.

    Args:
        root: image dataset file root.
        image_format: list, default ['png', 'jpg', 'jpeg'].
        label_func: Function is applied to the name of each picture.
        label_encoder: whether encode labels with value between 0 and n_classes-1.
    """
    def __init__(self, root, image_format=['png', 'jpg', 'jpeg'], label_func=None, label_encoder=False):
        super(from_folder, self).__init__()
        p = pathlib.Path(root)
        dataset = pd.DataFrame(chain.from_iterable((p.rglob(f'*.{i}') for i in image_format)), columns=['image'])
        if label_func is not None:
            dataset['label'] = dataset.image.map(lambda x:label_func(x.name))
            name_label_dict = {j: i for i, j in enumerate(dataset.label.unique())}
            self.name_label_dict = {'positive':name_label_dict, 'negative':{j: i for i, j in name_label_dict.items()}}
            if label_encoder:
                dataset['label'] = dataset.label.replace(self.name_label_dict['positive'])
        dataset['image'] = dataset.image.astype(str).map(lambda x:eval(repr(x).replace("\\", '/').replace("//", '/')))
        if label_func is not None:
            self._params.data[self._params.mode1] = [dataset.image.values, dataset.label.values]
        else:
            self._params.data[self._params.mode1] = dataset.image.values
        self._params.index[self._params.mode] = dataset.index.to_list()
        self._data_mode()
        self._params.data_from = 'from_folder'
        
    
class from_class_folder(DataSet, BatchFunction):
    """Construct an image dataset label index.

    Args:
        root: image dataset file root.
        image_format: list, default ['png', 'jpg', 'jpeg'].
        label_encoder: whether encode labels with value between 0 and n_classes-1.
    """
    def __init__(self, root, image_format=['png', 'jpg', 'jpeg'], label_encoder=False):
        super(from_class_folder, self).__init__()
        file = os.listdir(root)
        file = [i for i in file if os.path.isdir(root+'/'+i) and i[0]!='.']
        data = pd.DataFrame()
        for i in file:
            data = pd.concat([data, pd.DataFrame({'image':os.listdir(root+'/'+i), 'label':i})])
        data = data.reset_index(drop=True)
        data['image'] = root+'/'+data.label+'/'+data.image
        data = data[data.image.map(lambda x: True if '.' in x.split('/')[-1] else False)]
        data = data[data.image.map(lambda x: True if x.split('/')[-1][0]!='.' else False)]
        data = data[data.image.map(lambda x: True if len(x.split('/')[-1].split('.'))==2 else False)]
        data = data[data.image.map(lambda x: True if str.lower(x.split('/')[-1].split('.')[1]) in image_format else False)]
        data = data.reset_index(drop=True)
        name_label_dict = {j: i for i, j in enumerate(data.label.unique())}
        self.name_label_dict = {'positive':name_label_dict, 'negative':{j: i for i, j in name_label_dict.items()}}
        if label_encoder:
            data['label'] = data.label.replace(self.name_label_dict['positive'])
        self._params.data[self._params.mode1] = [data.image.values, data.label.values]
        self._params.index[self._params.mode] = data.index.to_list()
        self._data_mode()
        self._params.data_from = 'from_class_folder'


class range(DataSet, BatchFunction):
    """Creates a Dataset of a step-separated range of values."""
    def __init__(self, *args, **kwargs):
        super(range, self).__init__()
        self._params.data[self._params.mode1] = np.arange(*args, **kwargs)
        self._params.index[self._params.mode] = list(np.arange(len(self._params.data[self._params.mode1])))
        self._data_mode()


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
        self._params.data[self._params.mode1] = np.array(t).reshape(size)
        self._params.index[self._params.mode] = list(np.arange(len(self._params.data[self._params.mode1])))
        self._data_mode()

        
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
        if isinstance(datasets[0]._params.data[datasets[0]._params.mode1], list):
            self._params.data[self._params.mode1] = []
            for i in np.arange(len(datasets[0]._params.data[datasets[0]._params.mode1])):
                self._params.data[self._params.mode1] += [np.concatenate([sets._params.data[sets._params.mode1][i] for sets in datasets])]
        else:
            self._params.data[self._params.mode1] = np.concatenate([sets._params.data[sets._params.mode1] for sets in datasets])
        data_dict = []
        for r, sets in enumerate(datasets):
            if r==0:
                data_dict.append(sets._params.index[sets._params.mode].copy())
            else:
                t = max(data_dict[-1])+1
                data_dict.append([i+t for i in sets._params.index[sets._params.mode].copy()])
        if stop_on_empty_dataset:
            self._params.index[self._params.mode] = []
            for i in index:
                if len(data_dict[i])==0:
                    break
                self._params.index[self._params.mode].append(data_dict[i].pop(0))
        else:
            self._params.index[self._params.mode] = [data_dict[i].pop(0) for i in index if len(data_dict[i])>0]
        self._data_mode()
        
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
        weights = weight.copy()
        if isinstance(datasets[0]._params.data[datasets[0]._params.mode1], list):
            self._params.data[self._params.mode1] = []
            for i in np.arange(len(datasets[0]._params.data[datasets[0]._params.mode1])):
                self._params.data[self._params.mode1] += [np.concatenate([sets._params.data[sets._params.mode1][i] for sets in datasets])]
        else:
            self._params.data[self._params.mode1] = np.concatenate([sets._params.data[sets._params.mode1] for sets in datasets])
        data_dict = []
        for r, sets in enumerate(datasets):
            if r==0:
                data_dict.append(sets._params.index[sets._params.mode].copy())
            else:
                t = max(data_dict[-1])+1
                data_dict.append([i+t for i in sets._params.index[sets._params.mode].copy()])
        if stop_on_empty_dataset:
            self._params.index[self._params.mode] = []
            while 1:
                index = np.random.choice(np.arange(len(data_dict)), p=weights)
                self._params.index[self._params.mode].append(data_dict[index].pop(0))
                if len(data_dict[index])==0:
                    break
        else:
            self._params.index[self._params.mode] = []
            while 1:
                if len(data_dict)==0:
                    break
                index = np.random.choice(np.arange(len(data_dict)), p=weights)
                self._params.index[self._params.mode].append(data_dict[index].pop(0))
                if len(data_dict[index])==0:
                    data_dict.pop(index)
                    if weights is not None:
                        weights.pop(index)
                        weights = [i/sum(weights) for i in weights]
        self._data_mode()