from linora.utils._config import Config

__all__ = ['LRStep', 'LRConstant', 'LRStepMulti', 'LRScheduler', 'LRReduceOnPlateau']


class LRConstant():
    """Decays the learning rate of each parameter group by a small constant 
    factor until the number of batch reaches a pre-defined batch. 
    
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


class LRReduceOnPlateau():
    """Reduce learning rate when a metric has stopped improving.

    Args:
        lr_initial: lr initial value.
        scheduler: a function that takes current learning rate (float)  
            and an batch index (integer, indexed from 0) and log (dict) 
            as inputs and returns a new learning rate as output (float).
        monitor: quantity to be monitored.
        patience: number of batch with no improvement after which learning rate will be reduced.
        mode: one of {'min', 'max'}. 
            In 'min' mode, the learning rate will be reduced when the quantity monitored has stopped decreasing; 
            in 'max' mode it will be reduced when the quantity monitored has stopped increasing; 
        min_delta: Minimum change in the monitored quantity to qualify as an improvement, 
            i.e. an absolute change of less than min_delta, will count as no improvement.
        lr_min: lower bound on the learning rate.
    """
    def __init__(self, lr_initial, scheduler, monitor, patience=10, mode='min', min_delta=0.0001, lr_min=0):
        self._params = Config()
        self._params.lr = lr_initial
        self._params.scheduler = scheduler
        self._params.monitor = monitor
        self._params.patience = patience
        self._params.mode = mode
        self._params.min_delta = min_delta
        self._params.lr_min = lr_min
        self._params.history = []
        self._params.name = 'LRReduceOnPlateau'
    
    def _update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        if self._params.monitor in log:
            self._params.history = self._params.history[-self._params.patience:]+[log[self._params.monitor]]
            if self._params.mode=='min':
                if min(self._params.history[-self._params.patience:])+self._params.min_delta>self._params.history[0]:
                    self._params.lr = max(self._params.lr_min, self._params.scheduler(self._params.lr, batch, log))
            else:
                if max(self._params.history[-self._params.patience:])-self._params.min_delta<self._params.history[0]:
                    self._params.lr = max(self._params.lr_min, self._params.scheduler(self._params.lr, batch, log))


class LRScheduler():
    """Custom learning rate scheduler.

    Args:
        lr_initial: lr initial value.
        scheduler: a function that takes current learning rate (float)  
            and an batch index (integer, indexed from 0) and log (dict) 
            as inputs and returns a new learning rate as output (float).
    """
    def __init__(self, lr_initial, scheduler):
        self._params = Config()
        self._params.lr = lr_initial
        self._params.scheduler = scheduler
        self._params.name = 'LRScheduler'
    
    def _update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        self._params.lr = self._params.scheduler(self._params.lr, batch, log)
        
        
class LRStep():
    """Decays the learning rate of each parameter group by gamma every step_size batch. 
    
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
    gamma once the number of batch reaches one of the batch_list.  
    
    >>> # Assuming optimizer uses lr = 0.05
    >>> # lr = 0.05     if batch < 30
    >>> # lr = 0.005    if 30 <= batch < 80
    >>> # lr = 0.0005   if batch >= 80

    Args:
        lr_initial: lr initial value.
        batch_list: list, List of batch indices.
        gamma: float, Multiplicative factor of learning rate decay. Default: 0.1.
    """
    def __init__(self, lr_initial, batch_list, gamma=0.1):
        self._params = Config()
        self._params.lr = lr_initial
        self._params.batch_list = sorted(batch_list)
        self._params.gamma = gamma
        self._params.name = 'LRStepMulti'
    
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

