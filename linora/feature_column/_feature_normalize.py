from linora.feature_column._numerical import *

class FeatureNormalize(object):
    def __init__(self):
        self.pipe = {}
        
    def normalize_max(self, variable, name=None, keep=False):
        """normalize feature with max method.

        Args:
            variable: str, feature variable name.
            name: str, output feature name, if None, name is variable.
            keep: if name is not None and variable!=name, variable whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
        config = {'param':{'name':variable if name is None else name},
                  'type':'normalize_max', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def normalize_maxabs(self, variable, name=None, keep=False):
        """normalize feature with maxabs method.

        Args:
            variable: str, feature variable name.
            name: str, output feature name, if None, name is variable.
            keep: if name is not None and variable!=name, variable whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
        config = {'param':{'name':variable if name is None else name},
                  'type':'normalize_maxabs', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def normalize_l1(self, variable, name=None, keep=False):
        """normalize feature with l1 method.

        Args:
            variable: str, feature variable name.
            name: str, output feature name, if None, name is variable.
            keep: if name is not None and variable!=name, variable whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
        config = {'param':{'name':variable if name is None else name},
                  'type':'normalize_l1', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def normalize_l2(self, variable, name=None, keep=False):
        """normalize feature with l2 method.

        Args:
            variable: str, feature variable name.
            name: str, output feature name, if None, name is variable.
            keep: if name is not None and variable!=name, variable whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
        config = {'param':{'name':variable if name is None else name},
                  'type':'normalize_l2', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def normalize_meanminmax(self, variable, name=None, keep=False):
        """normalize feature with meanminmax method.

        Args:
            variable: str, feature variable name.
            name: str, output feature name, if None, name is variable.
            keep: if name is not None and variable!=name, variable whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
        config = {'param':{'name':variable if name is None else name},
                  'type':'normalize_meanminmax', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def normalize_minmax(self, variable, feature_range=(0, 1), name=None, keep=False):
        """normalize feature with minmax method.

        Args:
            variable: str, feature variable name.
            feature_range: list or tuple, range of values after feature transformation.
            name: str, output feature name, if None, name is variable.
            keep: if name is not None and variable!=name, variable whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
        config = {'param':{'feature_range':feature_range, 'name':variable if name is None else name},
                  'type':'normalize_minmax', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def normalize_norm(self, variable, name=None, keep=False):
        """normalize feature with norm method.

        Args:
            variable: str, feature variable name.
            name: str, output feature name, if None, name is variable.
            keep: if name is not None and variable!=name, variable whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
        config = {'param':{'name':variable if name is None else name},
                  'type':'normalize_norm', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    def normalize_robust(self, variable, feature_scale=(0.5, 0.5), name=None, keep=False):
        """normalize feature with robust method.

        Args:
            variable: str, feature variable name.
            feature_scale: list or tuple, each element is in the [0,1] interval.
                (feature_scale[0], feature.quantile(0.5+feature_scale[1]/2)-feature.quantile(0.5-feature_scale[1]/2)).
            name: str, output feature name, if None, name is variable.
            keep: if name is not None and variable!=name, variable whether to keep in the final output.
        """
        keep = keep if name is not None and variable!=name else False
        config = {'param':{'feature_scale':feature_scale, 'name':variable if name is None else name},
                  'type':'normalize_robust', 'variable':variable, 'keep':keep}
        self.pipe[len(self.pipe)] = config
        return self
    
    
    
    
        