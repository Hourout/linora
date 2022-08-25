import numpy as np
from linora.utils._config import Config

__all__ = ['Chain', 'Sequential']


class Chain():
    """Chains list of learning rate schedulers. 

    Args:
        schedulers: list, List of chained schedulers.
    """
    def __init__(self, schedulers):
        self._params = Config()
        self._params.lr = schedulers[0]._params.lr
        self._params.schedulers = schedulers
        self._params.name = 'LRChain'
    
    def _update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        for schedulers in self._params.schedulers:
            schedulers._update(batch, log)
        self._params.lr = np.cumprod([i._params.lr for i in self._params.schedulers])[-1]
        

class Sequential():
    """Receives the list of schedulers that is expected to be called sequentially 
    during optimization process and milestone points that provides exact intervals 
    to reflect which scheduler is supposed to be called at a given batch.

    Args:
        schedulers: list, List of chained schedulers.
        batch_list: list, List of batch indices.
    """
    def __init__(self, schedulers, batch_list):
        self._params = Config()
        self._params.lr = schedulers[0]._params.lr
        self._params.schedulers = schedulers
        self._params.batch_list = sorted(batch_list)
        self._params.name = 'LRSequential'
    
    def _update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        if self._params.batch_list:
            if batch>=self._params.batch_list[0]:
                self._params.batch_list.pop(0)
                self._params.schedulers.pop(0)
        self._params.schedulers[0]._update(batch, log)
        self._params.lr = self._params.schedulers[0]._params.lr
        