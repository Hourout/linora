import os
import pathlib
from itertools import chain

import pandas as pd

__all__ = ['ImageDataset', 'ImageClassFolderDataset']


class ImageDataset():
    def __init__(self, root, image_format=['png', 'jpg', 'jpeg'], label_func=None, label_encoder=False):
        """Construct an image dataset label index.
        
        Args:
            root: image dataset file root.
            image_format: list, default ['png', 'jpg', 'jpeg'].
            label_func: if label_func is None, self.dataset['label'] is not exist; Function is applied to the name of each picture.
            label_encoder: whether encode labels with value between 0 and n_classes-1.
        Returns:
            class, self.dataset['image'] is image path, self.dataset['label'] is image label.
        """
        p = pathlib.Path(root)
        self.dataset = pd.DataFrame(chain.from_iterable((p.rglob(f'*.{i}') for i in image_format)), columns=['image'])
        if label_func is not None:
            self.dataset['label'] = self.dataset.image.map(lambda x:label_func(x.name))
            self.name_label_dict = {j: i for i, j in enumerate(self.dataset.label.unique())}
            self.name_label_dict = {'positive':self.name_label_dict, 'negative':{j: i for i, j in self.name_label_dict.items()}}
            if label_encoder:
                self.dataset['label'] = self.dataset.label.replace(self.name_label_dict['positive'])
        self.dataset['image'] = self.dataset.image.astype(str).map(lambda x:eval(repr(x).replace("\\", '/').replace("//", '/')))

        
class ImageClassFolderDataset():
    def __init__(self, root, image_format=['png', 'jpg', 'jpeg'], label_encoder=False):
        """Construct an image dataset label index.
        
        Args:
            root: image dataset file root.
            image_format: list, default ['png', 'jpg', 'jpeg'].
            label_encoder: whether encode labels with value between 0 and n_classes-1.
        Returns:
            class, self.data['image'] is image path, self.data['label'] is image label.
        """
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
        self.dataset = data.reset_index(drop=True)
        self.name_label_dict = {j: i for i, j in enumerate(data.label.unique())}
        self.name_label_dict = {'positive':self.name_label_dict, 'negative':{j: i for i, j in self.name_label_dict.items()}}
        if label_encoder:
            self.dataset['label'] = self.dataset.label.replace(self.name_label_dict['positive'])
