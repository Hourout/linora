import numpy as np

__all__ = ['Metrics']


class Metrics:
    """Generic metrics update class
    
    Args:
        function: function name, use la.mertics.*, function(y_true, y_pred)
    """
    def __init__(self, function):
        self.function = function
        self.result = []
    
    def compute(self, y_true, y_pred, **kwargs):
        """compute function value
        
        Args:
            y_true: pd.Series or array or list, ground truth (correct) labels.
            y_pred: pd.Series or array or list, predicted values
            **kwargs: function other params
        Return:
            function value
        """
        return self.function(y_true, y_pred, **kwargs)
        
    def update(self, y_true, y_pred, **kwargs):
        """update function value
        
        Args:
            y_true: pd.Series or array or list, ground truth (correct) labels.
            y_pred: pd.Series or array or list, predicted values
            **kwargs: function other params
        """
        self.result += [self.function(y_true, y_pred, **kwargs)]
        
    def accumulate(self, latest_num=-1, method='mean'):
        """compute function history value
        
        Args:
            latest_num: select the latest number of samples.
            method: samples compute method.['mean', 'sum', 'max', 'min']
        """
        if method=='mean':
            score = np.mean(self.result[-max(latest_num, 0):])
        elif method=='sum':
            score = np.sum(self.result[-max(latest_num, 0):])
        elif method=='max':
            score = np.max(self.result[-max(latest_num, 0):])
        elif method=='min':
            score = np.min(self.result[-max(latest_num, 0):])
        else:
            raise ValueError("`method` must one of `mean`, `sum`, `max`, `min`.")
        return score
    
    def reset():
        self.result = []
