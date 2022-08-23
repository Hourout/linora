from linora.utils._config import Config

__all__ = ['TerminateOnNaN']


class TerminateOnNaN():
    """Callback that terminates training when a NaN loss is encountered.
    
    Args:
        monitor: Quantity to be monitored.
    """
    def __init__(self, monitor):
        self._params = Config()
        self._params.monitor = monitor
        self.state = False
        
    def update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        if self._params.monitor in log:
            if np.isnan(log[self._params.monitor]).any():
                self.state = True