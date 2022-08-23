from linora.utils._config import Config

__all__ = ['ModelCheckpoint']


class ModelCheckpoint():
    """Callback to save model at some frequency.
    
    Args:
        monitor: Quantity to be monitored.
        patience: Number of epochs with no improvement after which training will be stopped.
        mode: One of {"min", "max"}. In min mode, training will stop when the quantity monitored has stopped decreasing; 
            in "max" mode it will stop when the quantity monitored has stopped increasing.
    """
    def __init__(self, monitor, patience=0, mode='min'):
        self._params = Config()
        self._params.monitor = monitor
        self._params.patience = patience
        self._params.mode = mode
        self._params.history = []
        self.state = False
        self._params.polt_num = 0
        self._params.best = None
        
    def update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        if self._params.monitor in log:
            self._params.history += [log[self._params.monitor]]
            if self._params.polt_num%self._params.patience==0:
                if self._params.best is None:
                    self.state = True
                    self._params.best = self._params.history[0]
                elif self._params.mode=='min':
                    self.state = min(self._params.history)<self._params.best
                else:
                    self.state = max(self._params.history)>self._params.best
                self._params.history = []
            else:
                self.state = False
            self._params.polt_num += 1