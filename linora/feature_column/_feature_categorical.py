from linora.feature_column._categorical import *

class FeatureCategorical(object):
    def __init__(self):
        self.pipe = {}
        
    def categorical_count(self, variable, normalize=True, abnormal_value=0, miss_value=0, name=None, keep=False):
        """Count or frequency of conversion category variables.
    
        Args:
            variable: str, feature variable name.
            normalize: bool, If True then the object returned will contain the relative frequencies of the unique values.
            abnormal_value: int or float, if feature values not in feature_scale dict, return `abnormal_value`.
            miss_value: int or float, if feature values are missing, return `miss_value`.
            name: str, output feature name, if None, name is variable.
            keep: if name is not None, variable whether to keep in the final output.
        Returns:
            return count labels and label parameters dict.
        """
        config = {'param':{'normalize':normalize, 'abnormal_value':abnormal_value, 
                           'miss_value':miss_value, 'name':variable if name is None else name},
                  'type':'categorical_count', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def categorical_crossed(self, variable, hash_bucket_size=3, name=None, keep=False):
        """Crossed categories and hash labels with value between 0 and hash_bucket_size-1.

        Args:
            variable: list, feature variable name of list.
            hash_bucket_size: int, number of categories that need hash.
            name: str, output feature name, if None, name is feature.name .
            keep: if name is not None, variable whether to keep in the final output.
        Returns:
            return hash labels.
        """
        config = {'param':{'hash_bucket_size':hash_bucket_size, 
                           'name':'_'.join(name)+'_crossed' if name is None else name}, 
                  'type':'categorical_crossed', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    