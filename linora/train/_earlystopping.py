from linora.utils._config import Config

__all__ = ['EarlyStopping']


class EarlyStopping():
    """Stop training when a monitored metric has stopped improving.
    
    Args:
        monitor: Quantity to be monitored.
        min_delta: Minimum change in the monitored quantity to qualify as an improvement, 
            i.e. an absolute change of less than min_delta, will count as no improvement.
        patience: Number of batch with no improvement after which training will be stopped.
        mode: One of {"min", "max"}. In min mode, training will stop when the quantity monitored has stopped decreasing; 
            in "max" mode it will stop when the quantity monitored has stopped increasing.
        baseline: Baseline value for the monitored quantity. 
            Training will stop if the model doesn't show improvement over the baseline.
    """
    def __init__(self, monitor, min_delta=0, patience=0, mode='min', baseline=None):
        self._params = Config()
        self._params.monitor = monitor
        self._params.min_delta = min_delta
        self._params.patience = patience
        self._params.mode = mode
        self._params.baseline = baseline
        self._params.history = []
        self._params.state = False
        self._params.name = 'EarlyStopping'
        
    def _update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        if self._params.monitor in log:
            self._params.history = self._params.history[-self._params.patience:]+[log[self._params.monitor]]
            if self._params.mode=='min':
                self._params.state = min(self._params.history[-self._params.patience:])+self._params.min_delta>self._params.history[0]
                if not self._params.state and self._params.baseline is not None:
                    self._params.state = min(self._params.history[-self._params.patience:])>self._params.baseline
            else:
                self._params.state = max(self._params.history[-self._params.patience:])-self._params.min_delta<self._params.history[0]
                if not self._params.state and self._params.baseline is not None:
                    self._params.state = max(self._params.history[-self._params.patience:])<self._params.baseline
        