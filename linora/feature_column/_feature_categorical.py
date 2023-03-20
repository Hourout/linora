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
            keep: if `name` is not None and `variable`!=`name`, `name` whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
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
            keep: if name is not None and variable!=name, variable whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
        config = {'param':{'hash_bucket_size':hash_bucket_size, 
                           'name':'_'.join(name)+'_crossed' if name is None else name}, 
                  'type':'categorical_crossed', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def categorical_encoder(self, variable, abnormal_value=-1, miss_value=-1, name=None, keep=False):
        """Encode labels with value between 0 and n_classes-1.

        Args:
            variable: str, feature variable name.
            abnormal_value: int, if feature values not in feature_scale dict, return `abnormal_value`.
            miss_value: int or float, if feature values are missing, return `miss_value`.
            name: str, output feature name, if None, name is variable.
            keep: if name is not None and variable!=name, variable whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
        config = {'param':{'abnormal_value':abnormal_value, 'miss_value':miss_value, 
                           'name':variable if name is None else name},
                  'type':'categorical_encoder', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def categorical_hash(self, variable, hash_bucket_size=3, name=None, keep=False):
        """Hash labels with value between 0 and hash_bucket_size-1.

        Args:
            variable: str, feature variable name.
            hash_bucket_size: int, number of categories that need hash.
            name: str, output feature name, if None, name is variable.
            keep: if name is not None and variable!=name, variable whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
        config = {'param':{'hash_bucket_size':hash_bucket_size, 
                           'name':variable if name is None else name},
                  'type':'categorical_hash', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def categorical_hist(self, variable, label, abnormal_value=0, miss_value=0, name=None, keep=False):
        """Hist labels with value counts prob.

        Args:
            feature: str, feature variable name.
            label: str, label variable name.
            abnormal_value: int, if feature values not in feature_scale dict, return `abnormal_value`.
            miss_value: int or float, if feature values are missing, return `miss_value`.
            name: str, output feature name, if None, name is variable.
            keep: if name is not None and variable!=name, variable whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
        config = {'param':{'abnormal_value':abnormal_value, 'miss_value':miss_value, 
                           'name':variable if name is None else name, 'label':label},
                  'type':'categorical_hist', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def categorical_onehot_binarizer(self, variable, abnormal_value=0, miss_value=0, name=None, keep=False):
        """Transform between iterable of iterables and a multilabel format, sample is simple categories.

        Args:
            variable: str, feature variable name.
            abnormal_value: int, if feature values not in feature_scale dict, return `abnormal_value`.
            miss_value: int or float, if feature values are missing, return `miss_value`.
            name: str, output feature name, if None, name is variable.
            keep: if name is not None and variable!=name, variable whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
        config = {'param':{'abnormal_value':abnormal_value, 'miss_value':miss_value, 
                           'name':variable if name is None else name},
                  'type':'categorical_onehot_binarizer', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def categorical_onehot_multiple(self, variable, abnormal_value=0, miss_value=0, name=None, keep=False):
        """Transform between iterable of iterables and a multilabel format, sample is multiple categories.

        Args:
            variable: str, feature variable name.
            abnormal_value: int, if feature values not in feature_scale dict, return `abnormal_value`.
            miss_value: int or float, if feature values are missing, return `miss_value`.
            name: str, output feature name, if None, name is variable.
            keep: if name is not None and variable!=name, variable whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
        config = {'param':{'abnormal_value':abnormal_value, 'miss_value':miss_value, 
                           'name':variable if name is None else name},
                  'type':'categorical_onehot_multiple', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def categorical_rare(self, variable, p=0.05, min_num=None, max_num=None, abnormal_value=-1, miss_value=-1, name=None, keep=False):
        """Groups rare or infrequent categories in a new category called “Rare”, or any other name entered by the user.

        Args:
            variable: str, feature variable name.
            p: The minimum frequency a label should have to be considered frequent. 
                Categories with frequencies lower than tol will be grouped.
            min_num: The minimum number of categories a variable should have for the encoder to find frequent labels. 
                If the variable contains less categories, all of them will be considered frequent.
            max_num: The maximum number of categories that should be considered frequent. 
                If None, all categories with frequency above the tolerance (tol) will be considered frequent. 
                If you enter 5, only the 4 most frequent categories will be retained and the rest grouped.
            abnormal_value: int or float, if feature values not in feature_scale dict, return `abnormal_value`.
            miss_value: int or float, if feature values are missing, return `miss_value`.
            name: str, output feature name, if None, name is variable.
            keep: if name is not None and variable!=name, variable whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
        config = {'param':{'p':p, 'min_num':min_num, 'max_num':max_num,
                           'abnormal_value':abnormal_value, 'miss_value':miss_value, 
                           'name':variable if name is None else name},
                  'type':'categorical_rare', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def categorical_regress(self, variable, label, method='mean', abnormal_value='mean', miss_value='mean', name=None, keep=False):
        """Regress labels with value counts prob.

        Args:
            variable: str, feature variable name.
            label: str, label variable name.
            method: 'mean' or 'median'
            abnormal_value: int, if feature values not in feature_scale dict, return `abnormal_value`.
            miss_value: int or float, if feature values are missing, return `miss_value`.
            name: str, output feature name, if None, name is variable.
            keep: if name is not None and variable!=name, variable whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
        config = {'param':{'method':method,
                           'abnormal_value':abnormal_value, 'miss_value':miss_value, 
                           'name':variable if name is None else name, 'label':label},
                  'type':'categorical_regress', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def categorical_woe(self, variable, label, pos_label=1, abnormal_value=-1, miss_value=-1, name=None, keep=False):
        """Calculate series woe value

        Args:
            variable: str, feature variable name.
            label: str, label variable name.
            pos_label: int, default=1, positive label value.
            abnormal_value: int, if feature values not in feature_scale dict, return `abnormal_value`.
            miss_value: int or float, if feature values are missing, return `miss_value`.
            name: str, output feature name, if None, name is variable.
            keep: if name is not None and variable!=name, variable whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
        config = {'param':{'pos_label':pos_label,
                           'abnormal_value':abnormal_value, 'miss_value':miss_value, 
                           'name':variable if name is None else name, 'label':label},
                  'type':'categorical_woe', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    
    