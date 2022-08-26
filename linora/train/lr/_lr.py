import numpy as np
from linora.utils._config import Config

__all__ = ['LRStep', 'LRConstant', 'LRCyclic', 'LRSGDR', 'LRStepMulti', 'LRScheduler', 'LRReduceOnPlateau']


class LRCyclic():
    """
    https://www.kaggle.com/hireme/fun-api-keras-f1-metric-cyclical-learning-rate/code
    
    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    Args:
        lr_min: initial learning rate which is the
            lower boundary in the cycle.
        lr_max: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (lr_max - lr_min).
            The lr at any cycle is the sum of lr_min
            and some scaling of the amplitude; therefore 
            lr_max may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """
    def __init__(self, lr_min=0.001, lr_max=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        self._params = Config()
        self._params.lr = lr_min
        self._params.lr_min = lr_min
        self._params.lr_diff = lr_max-lr_min
        self._params.step_size = step_size
        self._params.mode = mode
        self._params.gamma = gamma
        if scale_fn is None:
            if self._params.mode == 'triangular':
                self._params.scale_fn = lambda x: 1.
                self._params.scale_mode = 'cycle'
            elif self._params.mode == 'triangular2':
                self._params.scale_fn = lambda x: 1/(2.**(x-1))
                self._params.scale_mode = 'cycle'
            elif self._params.mode == 'exp_range':
                self._params.scale_fn = lambda x: gamma**(x)
                self._params.scale_mode = 'iterations'
        else:
            self._params.scale_fn = scale_fn
            self._params.scale_mode = scale_mode
        self._params.name = 'LRCyclic'
        
    def _update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        cycle = np.floor(1+batch/(2*self._params.step_size))
        x = np.abs(batch/self._params.step_size - 2*cycle + 1)
        if self._params.scale_mode == 'cycle':
            self._params.lr = self._params.lr_min + self._params.lr_diff*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            self._params.lr = self._params.lr_min + self._params.lr_diff*np.maximum(0, (1-x))*self.scale_fn(batch)


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


class LRSGDR():
    """Cosine annealing learning rate scheduler with periodic restarts.

    Args:
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: The number of batches per epoch
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
        
    References
        Original paper: http://arxiv.org/abs/1608.03983
    """
    def __init__(self, lr_min, lr_max, steps_per_epoch, lr_decay=1, cycle_length=10, mult_factor=2):
        self._params = Config()
        self._params.lr = lr_max
        self._params.lr_min = lr_min
        self._params.lr_max = lr_max
        
        self._params.lr_decay = lr_decay

        self._params.batch_since_restart = 0
        self._params.next_restart = cycle_length

        self._params.steps_per_epoch = steps_per_epoch

        self._params.cycle_length = cycle_length
        self._params.mult_factor = mult_factor
        self._params.name = 'LRSGDR'

    def _update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        self._params.batch_since_restart += 1
        restart = self._params.batch_since_restart / (self._params.steps_per_epoch * self._params.cycle_length)
        self._params.lr = self._params.lr_min+0.5*(self._params.lr_max-self._params.lr_min)*(1+np.cos(restart*np.pi))

        if np.ceil(batch/self._params.steps_per_epoch) + 1 == self._params.next_restart:
            self._params.batch_since_restart = 0
            self._params.cycle_length = np.ceil(self._params.cycle_length * self._params.mult_factor)
            self._params.next_restart += self._params.cycle_length
            self._params.lr_max *= self._params.lr_decay


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

