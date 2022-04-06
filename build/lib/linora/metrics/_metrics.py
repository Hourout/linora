import numpy as np

__all__ = ['Metrics']

class Metrics:
    def __init__(self, function):
        self.function = function
        self.result = []
    
    def compute(self, y_true, y_pred):
        return self.function(y_true, y_pred)
        
    def update(self, y_true, y_pred):
        self.result += [self.function(y_true, y_pred)]
        
    def accumulate(self, latest_num=-1, method='mean'):
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
