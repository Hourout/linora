import multiprocessing

__all__ = ['ProcessLoom']

class Params:
    pass
    
class ProcessLoom():
    """ProcessLoom class: executes runners using multi-processing."""
    def __init__(self, max_runner=None):
        """max_runner: int, The total number of runners that are allowed to be running at any given time."""
        self.params = Params()
        self.params.max_runner = multiprocessing.cpu_count() if max_runner is None else max_runner
        self.params.runners = list()
        self.params.started = list()
        self.params.time_pause = 0.1
        manager = multiprocessing.Manager()
        self.params.tracker_dict = manager.dict()

    def add_function(self, func, args=None, kwargs=None, key=None):
        """ Adds function in the Loom
        Args:
            func (reference): reference to the function
            args (list): function args
            kwargs (dict): function kwargs
            key (str): ket to store the function output in dictionary
        """
        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()
        if key is None:
            key = len(self.runners)
        
        run_id = len(self.runners)
        self.params.tracker_dict[key] = dict()
        self.params.runners.append((func, args, kwargs, key, run_id))
        
    def add_work(self, works):
        """ Adds work to the loom
        Args:
            works (list): list of works [(func, args, kwargs, key), (func2, args2, kwargs2), ...]
        """
        for work in works:
            if len(work) > 4 or len(work) == 0:
                raise ValueError('Need 1 to 4 values to unpack')
            args = work[1] if len(work) > 1 else None
            kwargs = work[2] if len(work) > 2 else None
            key = work[3] if len(work) == 4 else None
            self.add_function(work[0], args, kwargs, key)
