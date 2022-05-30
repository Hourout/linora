import os
import pathlib
from itertools import chain

import numpy as np
import pandas as pd

from linora.data._dataset import DataSet
from linora.parallel._thread import ThreadLoom

__all__ = ['from_class_folder', 'from_folder', 'from_tensor']


class BatchFunction():
    def _batch_list_map(self, loc):
        loom = ThreadLoom(self._params.map_size)
        for i in loc:
            loom.add_function(self._params.map_func, [j[i] for j in self._params.data])
        t = loom.execute()
        for i in t:
            if t[i]['got_error']:
                continue
            if isinstance(t[i]['output'], (list, tuple)):
                return [np.concatenate([np.expand_dims(t[j]['output'][k], 0) for j in t if not t[j]['got_error']]) for k in range(len(t[i]['output']))]
            else:
                return np.concatenate([np.expand_dims(t[j]['output'], 0) for j in t if not t[j]['got_error']])
    
    def _batch_list(self, loc):
        return [i[loc] for i in self._params.data]
    
    def _batch_map(self, loc):
        loom = ThreadLoom(self._params.map_size)
        for i in loc:
            loom.add_function(self._params.map_func, [self._params.data[i]])
        t = loom.execute()
#         print(i)
        for i in t:
            if t[i]['got_error']:
                continue
            if isinstance(t[i]['output'], (list, tuple)):
                return [np.concatenate([np.expand_dims(t[j]['output'][k], 0) for j in t if not t[j]['got_error']]) for k in range(len(t[i]['output']))]
            else:
                return np.concatenate([np.expand_dims(t[j]['output'], 0) for j in t if not t[j]['got_error']])
    
    def _batch(self, loc):
        return self._params.data[loc]
    

class from_tensor(DataSet, BatchFunction):
    """Represents a potentially large set of elements from image path.
    
    Args:
        data: image path or list of image path or tuple(list, list,...) data format.
    """
    def __init__(self, data):
        super(from_tensor, self).__init__()
        if isinstance(data, (int, float, str)):
            self._params.data = np.array([data])
        elif isinstance(data, tuple):
            for i in data:
                assert len(data[0])==len(i), 'Length needs to be consistent between data.'
            self._params.data = [np.array(i) for i in data]
        else:
            self._params.data = np.array(data)
        self._params.data_mode = 'list' if isinstance(data, tuple) else 'image'
        self._params.data_index = list(range(len(self._params.data[0] if isinstance(data, tuple) else self._params.data)))
    
    
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
            self._params.data = [dataset.image.values, dataset.label.values]
        else:
            self._params.data = dataset.image.values
        self._params.data_index = dataset.index.to_list()
        self._params.data_mode = 'list' if isinstance(self._params.data, list) else 'image'
        
    
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
        self._params.data = [data.image.values, data.label.values]
        self._params.data_index = data.index.to_list()
        self._params.data_mode = 'list'

    