__all__ = ['CallbackList']


class CallbackList():
    """Container abstracting a list of callbacks.
    
    Args:
        callback: List of Callback instances.
    """
    def __init__(self, callbacks=None):
        self._callbacks = []
        if callbacks is not None:
            self._callbacks += callbacks if isinstance(callbacks, list) else [callbacks]
        
    def append(self, callback):
        """append callback.
        
        Args:
            callback: Callback instances.
        """
        self._callbacks.append(callback)
        
    def update(self, batch, log):
        """update log.
        
        Args:
            batch: Integer, index of batch.
            log: dict, name and value of loss or metrics;
        """
        for callback in self._callbacks:
            callback.update(batch, log)