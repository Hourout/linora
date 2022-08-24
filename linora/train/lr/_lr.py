from linora.utils._config import Config

__all__ = ['LRStep', 'LRConstant', 'LRStepMulti']


class LRConstant():
    """Decays the learning rate of each parameter group by a small constant 
    factor until the number of epoch reaches a pre-defined milestone: total_iters. 
    
    >>> # Assuming optimizer uses lr = 0.05
    >>> # lr = 0.025   if batch == 0
    >>> # lr = 0.025   if batch == 1
    >>> # lr = 0.025   if batch == 2
    >>> # lr = 0.025   if batch == 3
    >>> # lr = 0.05    if batch >= 4

    Args:
        lr_initial: lr initial value.
        factor: The number we multiply learning rate until the milestone.
        batch: The number of steps that the scheduler decays the learning rate. 
    """
    def __init__(self, lr_initial, factor, batch):
        self._params = Config()
        self._params.lr = lr_initial*factor
        self._params.factor = lr_initial
        self._params.batch = batch
        self._params.name = 'LRConstant'
    
    def _update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        if self._params.batch<=batch:
            self._params.lr = self._params.factor


class LRStep():
    """Decays the learning rate of each parameter group by gamma every step_size epochs. 
    
    >>> # Assuming optimizer uses lr = 0.05 
    >>> # lr = 0.05     if batch < 30
    >>> # lr = 0.005    if 30 <= batch < 60
    >>> # lr = 0.0005   if 60 <= batch < 90

    Args:
        lr_initial: lr initial value.
        step_size: int, Period of learning rate decay.
        gamma: float, Multiplicative factor of learning rate decay. Default: 0.1.
    """
    def __init__(self, lr_initial, step_size, gamma=0.1):
        self._params = Config()
        self._params.lr = lr_initial
        self._params.step_size = step_size
        self._params.gamma = gamma
        self._params.name = 'LRStep'
        self._params.step_num = 0
        self._params.batch = -1
    
    def _update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        if self._params.batch!=batch:
            self._params.batch = batch
            self._params.step_num += 1
            if self._params.step_num%self._params.step_size==0:
                self._params.lr = self._params.lr*self._params.gamma


class LRStepMulti():
    """Decays the learning rate of each parameter group by 
    gamma once the number of epoch reaches one of the batch_list.  
    
    >>> # Assuming optimizer uses lr = 0.05
    >>> # lr = 0.05     if batch < 30
    >>> # lr = 0.005    if 30 <= batch < 80
    >>> # lr = 0.0005   if batch >= 80

    Args:
        lr_initial: lr initial value.
        batch_list: list, List of epoch indices. Must be increasing.
        gamma: float, Multiplicative factor of learning rate decay. Default: 0.1.
    """
    def __init__(self, lr_initial, batch_list, gamma=0.1):
        self._params = Config()
        self._params.lr = lr_initial
        self._params.batch_list = sorted(batch_list)
        self._params.gamma = gamma
        self._params.name = 'LRStepMulti'
        self._params.step_num = 0
        self._params.batch = -1
    
    def _update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        for r, i in enumerate(self._params.batch_list.copy()):
            if batch>i:
                self._params.lr = self._params.lr*self._params.gamma
                self._params.batch_list.pop(r)

