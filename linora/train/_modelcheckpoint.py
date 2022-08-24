from linora.utils._config import Config

__all__ = ['ModelCheckpoint']


class ModelCheckpoint():
    """Callback to save model at some frequency.
    
    Args:
        monitor: Quantity to be monitored.
        patience: The number of batches for the training monitoring interval.
        mode: One of {"min", "max"}. the current save file is made based on either the 
            maximization or the minimization of the monitored quantity.
    """
    def __init__(self, monitor, patience=10, mode='min'):
        self._params = Config()
        self._params.monitor = monitor
        self._params.patience = patience
        self._params.mode = mode
        self._params.history = []
        self._params.checkpoint = False
        self._params.polt_num = 0
        self._params.best = None
        self._params.name = 'ModelCheckpoint'
        
    def _update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        if self._params.monitor in log:
            self._params.history += [log[self._params.monitor]]
            if self._params.polt_num%self._params.patience==0:
                if self._params.best is None:
                    self._params.checkpoint = True
                    self._params.best = self._params.history[0]
                elif self._params.mode=='min':
                    self._params.checkpoint = min(self._params.history)<self._params.best
                else:
                    self._params.checkpoint = max(self._params.history)>self._params.best
                self._params.history = []
            else:
                self._params.checkpoint = False
            self._params.polt_num += 1