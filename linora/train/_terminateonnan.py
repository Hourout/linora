import numpy as np

from linora.utils._config import Config

__all__ = ['TerminateOnNaN', 'TerminateOverfitted']


class TerminateOnNaN():
    """Callback that terminates training when a NaN loss is encountered.
    
    Args:
        monitor: Quantity to be monitored.
    """
    def __init__(self, monitor):
        self._params = Config()
        self._params.monitor = monitor
        self._params.state = False
        self._params.name = 'TerminateOnNaN'
        
    def _update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        if self._params.monitor in log:
            if np.isnan(log[self._params.monitor]).any():
                self._params.state = True
                

class TerminateOverfitted():
    '''Terminates training when the running loss is smaller than the theoretical best, 
    which is strong indication that the model will end up overfitting.
    
    Args:
        monitor: Quantity to be monitored.
        theory_best : float, The theoretical-smallest loss achievable without overfiting.
        mode: One of {"min", "max"}. In min mode, training will stop when the quantity monitored has stopped decreasing; 
            in "max" mode it will stop when the quantity monitored has stopped increasing.
    '''
    def __init__(self, monitor, theory_best, mode='min'):
        self._params = Config()
        self._params.monitor = monitor
        self._params.theory_best = theory_best
        self._params.state = False
        self._params.name = 'TerminateOverfitted'

    def _update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        if self._params.monitor in log:
            if self._params.mode=='min':
                if log[self._params.monitor]<self._params.theory_best:
                    self._params.state = True
            else:
                if log[self._params.monitor]>self._params.theory_best:
                    self._params.state = True
        
