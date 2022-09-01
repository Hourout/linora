from linora.utils._config import Config

__all__ = ['Callback', 'CallbackList']


class Callback():
    """Abstract base class used to build new callbacks."""
    def __init__(self):
        self._params = Config()
        self.state = False
        self.checkpoint = False
        self.lr = None
        self._params.name = 'Callback'
        
    def update(self, batch, log=None):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        pass
    
    def _update(self, batch, log):
        self.update(batch, log)
        
        
class CallbackList():
    """Container abstracting a list of callbacks.
    
    Args:
        callback: List of Callback instances.
    """
    def __init__(self, callbacks=None):
        self._params = Config()
        self._params.callbacks = []
        if callbacks is not None:
            self._params.callbacks += callbacks if isinstance(callbacks, list) else [callbacks]
        self.state = False
        self.checkpoint = False
        self.lr = None
        for callback in self._params.callbacks:
            if 'LR' in callback._params.name:
                self.lr = callback._params.lr
            elif callback._params.name=='Callback':
                self.state = self.state or callback.state
                self.checkpoint = self.checkpoint or callback.checkpoint
                if callback.lr is not None:
                    self.lr = callback.lr
        self._params.name_list_state = ['EarlyStopping', 'TerminateOnNaN', 'TerminateOverfitted']
        self._params.name_list_checkpoint = ['ModelCheckpoint']
        
    def append(self, callback):
        """append callback.
        
        Args:
            callback: Callback instances.
        """
        self._params.callbacks.append(callback)
        if 'LR' in callback._params.name:
            self.lr = callback._params.lr
        
    def update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        for callback in self._params.callbacks:
            callback._update(batch, log)
            if callback._params.name in self._params.name_list_state:
                self.state = self.state or callback._params.state
            elif callback._params.name in self._params.name_list_checkpoint:
                self.checkpoint = self.checkpoint or callback._params.checkpoint
            elif 'LR' in callback._params.name:
                self.lr = callback._params.lr
            elif callback._params.name=='Callback':
                self.state = self.state or callback.state
                self.checkpoint = self.checkpoint or callback.checkpoint
                if callback.lr is not None:
                    self.lr = callback.lr
                